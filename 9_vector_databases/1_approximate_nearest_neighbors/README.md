# 🔍 Approximate Nearest Neighbors (ANN)

Approximate Nearest Neighbors (ANN) algorithms are essential in modern machine learning and information retrieval systems. They enable **efficient similarity search** in high-dimensional datasets—where exact search is often computationally expensive or infeasible.

---

## 🧠 What is ANN?

In the **Nearest Neighbor Search** problem, the goal is to find the most similar data points to a given **query point**. This similarity is usually measured using distance metrics like **Euclidean distance**, **cosine similarity**, or **Manhattan distance**.

- **Exact Nearest Neighbor** algorithms scan the entire dataset to return the closest points with 100% accuracy.
- **Approximate Nearest Neighbor** algorithms relax the requirement for exactness, returning results that are *approximately* the closest—often orders of magnitude faster.

### ⚖️ Why Approximate Instead of Exact?

| Feature                      | Exact NN                | Approximate NN (ANN)   |
|-----------------------------|--------------------------|-------------------------|
| Accuracy                    | ✅ 100%                  | ⚠️ ~90–99% (configurable) |
| Speed (Large Datasets)      | ❌ Slow                 | ✅ Fast                |
| Scalability                 | ❌ Limited               | ✅ High                |
| Use Cases                   | Small datasets, critical accuracy | Web search, embeddings, recommender systems |

ANN is particularly useful when:

- Working with **large-scale vector datasets**
- Building **real-time** systems (e.g., semantic search)
- Performing **high-dimensional** similarity matching

---

## 🚀 Core ANN Algorithms

Several ANN algorithms have been developed to optimize both performance and accuracy:

- **Hierarchical Navigable Small Worlds (HNSW)**  
  Builds a graph structure for logarithmic search time in high-dimensional spaces.

- **Locality Sensitive Hashing (LSH)**  
  Hashes input data such that similar items fall into the same bucket with high probability.

- **Product Quantization (PQ)**  
  Compresses vectors into smaller codebooks for faster distance approximation.

- **Annoy (Approximate Nearest Neighbors Oh Yeah)**  
  Developed by Spotify, uses random projection trees.

---

## 🧪 Brute-Force Baseline (Python + NumPy)

To understand ANN, it's helpful to first implement a **brute-force k-Nearest Neighbors** search, which compares the query to every point in the dataset.

```python
import numpy as np

def brute_force_knn(data, query, k=1):
    """
    Find the k nearest neighbors of a query point using brute-force search.

    Parameters:
    - data: np.ndarray of shape (n_samples, n_features)
        The dataset containing all points.
    - query: np.ndarray of shape (n_features,)
        The query point.
    - k: int
        The number of nearest neighbors to find.

    Returns:
    - indices: np.ndarray of shape (k,)
        Indices of the k nearest neighbors in the dataset.
    - distances: np.ndarray of shape (k,)
        Distances of the k nearest neighbors from the query point.
    """
    distances = np.linalg.norm(data - query, axis=1)  # Euclidean distance
    indices = np.argsort(distances)[:k]               # Indices of closest points
    k_distances = distances[indices]                  # Corresponding distances
    return indices, k_distances

# Example dataset: 5 vectors in 3D space
data = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [2.0, 3.0, 4.0],
    [6.0, 7.0, 8.0]
])

# Define a query point
query = np.array([3.0, 3.5, 4.5])

# Find 2 nearest neighbors
neighbors, distances = brute_force_knn(data, query, k=2)

print("Indices of nearest neighbors:", neighbors)
print("Distances to nearest neighbors:", distances)
````

### 🔍 Explanation of Key Steps

* **Data Representation**: Each row in `data` is a vector in multi-dimensional space.
* **Distance Metric**: `np.linalg.norm(data - query, axis=1)` calculates the Euclidean distance from the query to each point.
* **Nearest Neighbor Selection**: `np.argsort(distances)[:k]` finds the indices of the top-`k` closest vectors.

---

## 💡 When to Use ANN

ANN methods are ideal for use cases involving **similarity**, **ranking**, or **semantic retrieval**, such as:

* 🔎 **Search engines** — e.g., matching queries with documents or images
* 🎧 **Music or content recommendation** — e.g., finding songs or products similar to a user’s interest
* 📦 **eCommerce** — "Customers who viewed this also liked..."
* 🧠 **LLMs and embeddings** — used to index large vector stores in RAG systems

---

## 🧰 Popular ANN Libraries in Python

| Library     | Description                                              |
| ----------- | -------------------------------------------------------- |
| **FAISS**   | Facebook’s library for efficient similarity search       |
| **Annoy**   | Spotify’s tree-based method optimized for disk usage     |
| **ScaNN**   | Google’s efficient ANN for high recall + high speed      |
| **HNSWLib** | High-performance implementation of HNSW                  |
| **NMSLIB**  | Versatile and fast, supports multiple distance functions |

---

## 📚 Further Reading & Resources

* [Understanding Approximate Nearest Neighbor Search](https://towardsdatascience.com/approximate-nearest-neighbor-search-e6b2ee08b7df)
* [FAISS GitHub Repository](https://github.com/facebookresearch/faiss)
* [Annoy GitHub Repository](https://github.com/spotify/annoy)
* [HNSW Paper](https://arxiv.org/abs/1603.09320)
* [ScaNN by Google](https://github.com/google-research/google-research/tree/master/scann)
