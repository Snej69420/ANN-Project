# Report
## Part A: Backbone Comparison

The objective of Part A was to isolate the effect of replacing standard convolutions with depthwise separable convolutions under a constrained architecture budget. The models were trained and evaluated on the FashionMNIST dataset.  

The empirical results for a single forward pass are summarized in the table below:

| Metric               | Standard CNN | Separable CNN |
|----------------------|--------------|---------------|
| Parameter Count      | 93,962       | 31,338        |
| FLOPs (Single Input) | 7,458,816    | 2,143,296     |
| Test Accuracy        | 76.98%       | 84.55%        |
| Test Macro F1        | 75.31%       | 84.45%        |

The Separable CNN was selected as the backbone for the subsequent retrieval tasks. 
The evaluation metrics demonstrate that it is vastly superior for a compact computer vision system. 
The standard CNN requires over three times the amount of mathematical operations (FLOPs) and stores three times as many parameters, yet yields worse predictive performance. 
Because the depthwise separable block decouples spatial filtering from channel-wise feature mixing, it extracts representations much more efficiently. 
It achieved an 84.55% test accuracy compared to the standard CNN's 76.98%, proving that model quality does not need to be sacrificed to achieve a lightweight deployment footprint. 
Given the strict hardware constraints of the target fashion platform, the Separable CNN is the obvious choice.

## Part B: Learn Embeddings for Retrieval
### Training and Loss Selection

The classification head of the Separable CNN was replaced with a 64-dimensional embedding head, and the network was fine-tuned for image retrieval. 

To train the embedding space, Triplet Loss was selected over Contrastive Loss. 
Contrastive loss forces absolute constraints: it attempts to push all items of the same class to the exact same point while pushing dissimilar items apart by a fixed margin. 
This can be overly restrictive and distort the embedding space. 
Triplet loss, on the other hand, enforces a *relative* constraint: it only requires that an anchor image is closer to a positive match than it is to a negative match by a given margin. 
This creates a much more flexible and continuous space that is better optimized for nearest-neighbour retrieval tasks.

### Retrieval Performance

After 5 epochs of training, the model achieved a **Recall@1 of 80.75%**. 

**Question: Did your retrieval model learn meaningful similarity?**

Yes, the embedding space is highly useful for retrieval. With 10 classes in FashionMNIST, a random baseline model would achieve a Recall@1 of approximately 10%. Achieving ~81% indicates that the network has learned a deeply meaningful representation of clothing features, successfully clustering visually similar items together despite the removal of the explicit classification objective.

### Embedding Space Visualizations

To further validate the quality of the learned embedding space, the network's representations were visualized both qualitatively (via nearest-neighbor retrieval) and structurally (via t-SNE dimensionality reduction).

#### Nearest-Neighbor Retrieval

![Retrieval Examples](retrieval_examples.png)

The retrieval examples demonstrate that the model has successfully transitioned from strict categorical classification to understanding continuous visual similarity. 
* The first query, a **T-shirt/top**, perfectly retrieves highly similar T-shirts with matching sleeve lengths and collar styles. 
* The second query, a **Coat**, retrieves a mix of coats and long-sleeved dresses. 
* The third query, a **Shirt**, retrieves other shirts as well as a cropped coat. 

#### t-SNE Projection

![t-SNE Visualization](tsne_visualization.png)

Projecting the 64-dimensional embeddings down to 2D using t-SNE reveals a logical and highly structured topology:
* **Distinct Clusters:** Visually unique categories like **Trousers** (orange) and **Bags** (yellow-green) form tight, highly isolated clusters far away from the rest of the dataset. 
* **Semantic Groupings:** The footwear classes—**Sandals**, **Sneakers**, and **Ankle boots**—group together in the same general region on the left side of the space, clearly separated from the upper-body clothing classes.
* **Expected Overlap:** The "Tops" and full-body clothing items (**T-shirts/tops, Pullovers, Dresses, Coats, and Shirts**) show significant overlap on the right side of the plot. This is an accurate reflection of the dataset's visual properties, as these categories share many foundational geometric features at this resolution. 

Overall, the visualizations confirm that the embedding space accurately maps the real-world semantic relationships between the clothing items.

## Part C: Model Compression 
To ensure the retrieval system can run efficiently on the fashion platform's limited hardware, the baseline embedding model (the Part B teacher) was subjected to two distinct compression techniques: 50% unstructured L1 pruning and knowledge distillation. 
For the distillation process, a fixed, smaller student architecture (CompactSeparableCNN) was trained to mimic the continuous embedding space of the teacher using Mean Squared Error (MSE) loss. 
The empirical tradeoffs between model size, computational cost, and retrieval quality are summarized in the table below:

| Model                 | Parameter | Count FLOPs (Single Input) | Test Recall@1 |
|-----------------------|-----------|----------------------------|---------------|
| Baseline (Teacher)    | 38,304    | 2,150,208                  | 79.90%        | 
| Pruned (50% Sparsity) | 38,304*   | 2,150,208*                 | 64.85%        |
| Distilled (Student)   | 12,528    | 673,696                    | 77.45%        |

*Unstructured L1 pruning zeros out weights but does not alter the underlying tensor dimensions. 

Consequently, standard parameter and FLOP counters reflect the original dense architecture unless specialized sparse-execution hardware or libraries are utilized.

Applying a one-shot, 50% unstructured prune to the baseline model resulted in a severe degradation of retrieval quality. 
The Recall@1 dropped by roughly 15 absolute percentage points (from 79.90% down to 64.85%). 
Without an iterative pruning and fine-tuning schedule, instantaneously removing half of the network's capacity caused catastrophic forgetting of the embedding space's learned topology.
Furthermore, because unstructured pruning does not inherently shrink the model's footprint in standard memory matrices, this method offered no immediate speed or storage benefits for a standard deployment pipeline.
Conversely, knowledge distillation proved vastly superior in preserving model performance. 
The compact student model successfully learned to mimic the teacher's embedding space, achieving a 77.45% Recall@1. 
This represents a retention of nearly 97% of the teacher's original retrieval capability despite a massive reduction in architectural complexity.

## Final Deployment Recommendation
Based on the empirical evidence, the Distilled Student Model is the optimal choice for deployment. 
The compression tradeoff here is highly favorable. By sacrificing only ~2.45% in top-1 retrieval recall, the resulting system is roughly three times smaller (storing only 12,528 parameters) and requires three times less computational power (reducing inference to 673,696 FLOPs) compared to the Part B baseline. 
Given the explicit requirement to deploy a lightweight system on limited hardware, the distilled model provides the absolute best balance between semantic retrieval usefulness and operational efficiency.
