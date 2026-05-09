import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
from sklearn.metrics import f1_score
from thop import profile

from src.data import get_classification_dataloaders, get_metric_dataloader, get_retrieval_eval_dataloader
from src.models import build_model
from src.visualise import visualize_retrieval, visualize_tsne


def profile_model(model: nn.Module, device: torch.device):
    # FashionMNIST shape: (batch_size=1, channels=1, height=28, width=28)
    dummy_input = torch.randn(1, 1, 28, 28).to(device)

    # calc FLOPs and params
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    return flops, params


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # fwd pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # bwd pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_metric_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for anchor, positive, negative in loader:
        anchor = anchor.to(device)
        pos = positive.to(device)
        neg = negative.to(device)

        optimizer.zero_grad()

        # fwd pass all three through the network
        emb_a = model(anchor)
        emb_p = model(pos)
        emb_n = model(neg)

        loss = criterion(emb_a, emb_p, emb_n)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def train_distillation_epoch(student, teacher, loader, optimizer, criterion, device):
    """Trains the student to mimic the teacher's embeddings using MSE."""
    student.train()
    teacher.eval()  # Teacher is frozen
    total_loss = 0

    # We can use the classification loader here because we just need single
    # images to pass through both networks, not pairs/triplets.
    for images, _ in loader:
        images = images.to(device)

        optimizer.zero_grad()

        # Get target embeddings from the frozen teacher
        with torch.no_grad():
            teacher_embs = teacher(images)

        # Get predictions from the student
        student_embs = student(images)

        # The objective is to make student embeddings identical to the teacher's
        loss = criterion(student_embs, teacher_embs)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # get preds
            _, preds = torch.max(outputs, 1)

            # accumulate
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # calc metrics
    acc = correct / total
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, macro_f1


def evaluate_retrieval(model, loader, device):
    model.eval()
    embeddings = []
    labels_list = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            embs = model(images)

            # normalize embeddings for stable distance calculation
            embs = F.normalize(embs, p=2, dim=1)

            embeddings.append(embs.cpu())
            labels_list.append(labels)

    embeddings = torch.cat(embeddings)
    labels = torch.cat(labels_list)

    # calc pairwise euclidean distances
    distances = torch.cdist(embeddings, embeddings)

    # mask self-distances so an image doesnt retrieve itself
    distances.fill_diagonal_(float('inf'))

    # get index of closest embedding
    nearest_idx = torch.argmin(distances, dim=1)
    nearest_labels = labels[nearest_idx]

    # calc ratio of correct top-1 retrievals
    recall_at_1 = (nearest_labels == labels).float().mean().item()
    return recall_at_1


def run_part_a(device):
    print("\n--- Starting Part A: Backbone Comparison ---")

    # fetch shared data loaders
    train_loader, val_loader, test_loader = get_classification_dataloaders(batch_size=64)

    models_to_test = ["cnn", "separable_cnn"]

    for model_name in models_to_test:
        print(f"\nEvaluating {model_name}...")
        model = build_model(model_name).to(device)

        flops, params = profile_model(model, device)
        print(f"Parameter count: {params:,}")
        print(f"FLOPs (single input): {flops:,}")

        # setup opt
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # dummy train loop
        epochs = 5
        for epoch in range(epochs):
            train_epoch(model, train_loader, optimizer, criterion, device)
            val_acc, val_f1 = evaluate(model, val_loader, device)
            print(f"Epoch {epoch + 1} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        # test eval
        test_acc, test_f1 = evaluate(model, test_loader, device)
        print(f"Final Test Acc: {test_acc:.4f} | Final Test F1: {test_f1:.4f}")

        # SAVE THE WEIGHTS
        torch.save(model.state_dict(), f"{model_name}_partA.pt")
        print(f"Saved checkpoint to {model_name}_partA.pt")


def run_part_b(device):
    print("\n--- Starting Part B: Metric Learning ---")
    train_loader = get_metric_dataloader(loss_name="triplet", batch_size=64)
    eval_loader = get_retrieval_eval_dataloader(batch_size=256)

    # build model with 64-dim embedding head
    model = build_model("separable_cnn", embedding_dim=64).to(device)

    # load part A weights
    state_dict = torch.load("separable_cnn_partA.pt", map_location=device, weights_only=True)

    # filter out old classification head weights
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head")}
    model.load_state_dict(state_dict, strict=False)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.TripletMarginLoss(margin=1.0)

    epochs = 5
    for epoch in range(epochs):
        loss = train_metric_epoch(model, train_loader, optimizer, criterion, device)
        recall = evaluate_retrieval(model, eval_loader, device)
        print(f"Epoch {epoch + 1} | Triplet Loss: {loss:.4f} | Recall@1: {recall:.4f}")

    torch.save(model.state_dict(), "separable_cnn_partB.pt")
    print("Saved checkpoint to separable_cnn_partB.pt")

    print("Generating visualizations...")
    visualize_retrieval(model, eval_loader, device)
    visualize_tsne(model, eval_loader, device)


def apply_pruning(model, amount=0.5):
    """
    Applies L1 Unstructured pruning to all Conv2d and Linear layers.
    Removes the specified percentage of the lowest-magnitude weights.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # Apply pruning mask
            prune.l1_unstructured(module, name='weight', amount=amount)
            # Make the pruning permanent (bakes the zeros into the weights)
            prune.remove(module, 'weight')
    return model


def run_part_c(device):
    print("\n--- Starting Part C: Compression ---")

    # Load Data and the Part B Teacher Model
    train_loader, _, _ = get_classification_dataloaders(batch_size=64)
    eval_loader = get_retrieval_eval_dataloader(batch_size=256)

    teacher = build_model("separable_cnn", embedding_dim=64).to(device)
    teacher.load_state_dict(torch.load("separable_cnn_partB.pt", map_location=device, weights_only=True))

    teacher_flops, teacher_params = profile_model(teacher, device)
    teacher_recall = evaluate_retrieval(teacher, eval_loader, device)

    print("\n--- Baseline (Part B Teacher) ---")
    print(f"Params: {teacher_params:,} | FLOPs: {teacher_flops:,} | Recall@1: {teacher_recall:.4f}")

    # Pruning
    print("\n--- Pruning (50% Sparsity) ---")
    pruned_model = build_model("separable_cnn", embedding_dim=64).to(device)
    pruned_model.load_state_dict(torch.load("separable_cnn_partB.pt", map_location=device, weights_only=True))

    pruned_model = apply_pruning(pruned_model, amount=0.5)
    pruned_recall = evaluate_retrieval(pruned_model, eval_loader, device)

    print(f"\tPruned Recall@1: {pruned_recall:.4f}")
    # Unstructured pruning zeros out weights but doesn't change the tensor shape,
    # so standard FLOP/Param counters will show the same numbers, addressed in report.

    # Knowledge Distillation
    print("\n--- Knowledge Distillation ---")
    student = build_model("compact_separable_cnn", embedding_dim=64).to(device)
    student_flops, student_params = profile_model(student, device)

    print(f"Student Params: {student_params:,} | Student FLOPs: {student_flops:,}")

    optimizer = optim.Adam(student.parameters(), lr=1e-3)
    criterion = nn.MSELoss()  # Distillation loss

    epochs = 5
    for epoch in range(epochs):
        loss = train_distillation_epoch(student, teacher, train_loader, optimizer, criterion, device)
        recall = evaluate_retrieval(student, eval_loader, device)
        print(f"Epoch {epoch + 1} | Distillation MSE: {loss:.4f} | Student Recall@1: {recall:.4f}")

    torch.save(student.state_dict(), "compact_student_partC.pt")
    print("Saved checkpoint to compact_student_partC.pt")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # run classification training
    run_part_a(device)

    # run retrieval training
    run_part_b(device)

    run_part_c(device)


if __name__ == "__main__":
    main()