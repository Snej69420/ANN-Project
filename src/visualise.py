import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.nn.functional as F

FASHION_MNIST_CLASSES = {
    0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
}

def visualize_retrieval(model, loader, device, num_queries=3, top_k=4):
    """Visualizes query images alongside their top-K nearest neighbors."""
    model.eval()
    embeddings_list, images_list, labels_list = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            embs = F.normalize(model(images), p=2, dim=1)

            embeddings_list.append(embs.cpu())
            images_list.append(images.cpu())
            labels_list.append(labels)

    embeddings = torch.cat(embeddings_list)
    images = torch.cat(images_list)
    labels = torch.cat(labels_list)

    distances = torch.cdist(embeddings, embeddings)
    distances.fill_diagonal_(float('inf'))

    np.random.seed(420)
    query_indices = np.random.choice(len(images), num_queries, replace=False)

    fig, axes = plt.subplots(num_queries, top_k + 1, figsize=(12, 3 * num_queries))
    if num_queries == 1:
        axes = [axes]

    for i, idx in enumerate(query_indices):
        query_class = FASHION_MNIST_CLASSES[labels[idx].item()]
        axes[i][0].imshow(images[idx].squeeze(), cmap='gray')
        axes[i][0].set_title(f"Query:\n{query_class}")
        axes[i][0].axis('off')

        top_k_idx = torch.topk(distances[idx], top_k, largest=False).indices
        for j, match_idx in enumerate(top_k_idx):
            match_class = FASHION_MNIST_CLASSES[labels[match_idx].item()]
            axes[i][j + 1].imshow(images[match_idx].squeeze(), cmap='gray')
            axes[i][j + 1].set_title(f"Match {j + 1}:\n{match_class}")
            axes[i][j + 1].axis('off')

    plt.tight_layout()
    plt.savefig("retrieval_examples.png")
    print("Saved retrieval visualization to retrieval_examples.png")
    plt.close()


def visualize_tsne(model, loader, device):
    """Projects 64D embeddings down to 2D for visualization."""
    model.eval()
    embeddings_list, labels_list = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            embs = F.normalize(model(images), p=2, dim=1)
            embeddings_list.append(embs.cpu())
            labels_list.append(labels)

            if sum(len(e) for e in embeddings_list) > 1000:
                break

    embeddings = torch.cat(embeddings_list)[:1000].numpy()
    labels = torch.cat(labels_list)[:1000].numpy()

    tsne = TSNE(n_components=2, random_state=42)
    reduced_embs = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embs[:, 0], reduced_embs[:, 1], c=labels, cmap='tab10', alpha=0.7)

    # Use actual names for the legend
    class_names = [FASHION_MNIST_CLASSES[i] for i in range(10)]
    plt.legend(handles=scatter.legend_elements()[0], labels=class_names, title="Classes")
    plt.title("t-SNE Visualization of Embedding Space")

    plt.savefig("tsne_visualization.png")
    print("Saved t-SNE visualization to tsne_visualization.png")
    plt.close()