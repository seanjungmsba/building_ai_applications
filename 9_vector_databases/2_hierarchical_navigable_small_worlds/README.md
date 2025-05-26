# 🕸️ Hierarchical Navigable Small Worlds (HNSW)

**Hierarchical Navigable Small Worlds (HNSW)** is a high-performance algorithm for **Approximate Nearest Neighbor (ANN)** search. It is widely adopted in real-world vector search engines thanks to its remarkable balance of **accuracy**, **efficiency**, and **scalability**.

---

## 🔍 What is HNSW?

HNSW constructs a **multi-layered, navigable graph** structure to organize data points in high-dimensional space. Each node represents a data vector, and edges connect it to other semantically close vectors.

### 🔗 Key Concepts

- **Hierarchical Levels**:
  - The top layers contain a *sparse* graph—broadly representing the data space.
  - Lower layers provide *denser*, fine-grained local connections.
- **Greedy Search**:
  - The algorithm starts at the top layer and greedily navigates closer to the query point as it descends.
- **Logarithmic Complexity**:
  - In practice, HNSW achieves **sub-linear** search time for large datasets.

<p align="center">
  <img src="https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/479e126f46fd816e1e4c690e529874b2a9f1d679/3-Figure1-1.png" width="600" />
  <br />
  <em>Figure: Hierarchical structure with multiple levels (Credits: Pinecone)</em>
</p>

---

## ⚙️ Why HNSW?

| Feature                  | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| 🔄 **Dynamic Indexing**  | Allows insertion of new points post-construction                           |
| 🧠 **Memory Efficient**   | Stores only a small number of connections per node                         |
| 🚀 **Fast & Accurate**   | Often outperforms other ANN algorithms like LSH, Annoy, and PQ              |
| 🔍 **Recall@k**           | Achieves >90% recall with much faster search time than brute-force methods |

HNSW is especially effective in:

- Semantic search
- Vector database indexing
- Image/audio/text similarity
- Large-scale ML systems

---

## 🧪 Python Implementation (Custom)

This project provides a simplified educational version of the HNSW algorithm using only standard Python and `numpy`.

### 🧱 Core Classes

- `Node`: Represents a data point with vector and layer-specific neighbors.
- `HNSW`: Manages the hierarchical graph, node insertion, and search routines.

### 📦 Features Implemented

✅ Multi-layered graph  
✅ Probabilistic layer assignment  
✅ Greedy search per layer  
✅ Layer-specific neighbor management  
✅ Bidirectional edge connections  
✅ Recall validation with brute-force

---

## 🧑‍💻 Example Usage

```python
import numpy as np
from hnsw_class import HNSW

# Parameters
vector_dim = 10
num_vectors = 1000
k = 5

# Generate random data
vectors = np.random.rand(num_vectors, vector_dim)

# Initialize and build the index
hnsw_index = HNSW(max_layers=4, M=8, ef_construction=100)
for i, vector in enumerate(vectors):
    hnsw_index.add_node(vector, id=i)

# Query the index
query = np.random.rand(vector_dim)
results = hnsw_index.search(query, k=k, ef_search=50)

# Print results
for rank, (node_id, node_vector) in enumerate(results):
    dist = np.linalg.norm(query - node_vector)
    print(f"Rank {rank+1}: ID={node_id}, Distance={dist:.4f}")
````

---

## 🔬 Benchmarking with Recall

To validate the quality of the search results, this implementation supports **Recall\@K** comparison with brute-force results.

```text
Recall@5: 0.80
True nearest IDs: [102, 50, 78, 9, 233]
HNSW found IDs:   [50, 78, 233, 182, 301]
```

> Recall is the fraction of true nearest neighbors found by the ANN algorithm. In many real-world applications, even 70–90% recall is acceptable given the massive speedup over brute-force.

---

## 📁 File Structure

```
├── hnsw_class.py          # HNSW algorithm and data structures
├── README.md              # This documentation
├── main.py                # Sample script to build, query, and benchmark
```

---

## 🧠 How the Algorithm Works

### Node Insertion

1. A new node chooses a random highest layer via a logarithmic probability.
2. Starting from the entry point at the top, the algorithm searches down to the bottom layer.
3. At each layer, it connects the node to its M nearest neighbors using a heap-based candidate selection.

### Query Search

1. Begin from the topmost layer and navigate down using greedy distance descent.
2. On the bottom layer, a wider search is conducted using an **ef\_search** parameter.
3. The k closest results are returned.

---

## 📚 Further Reading

* [Original HNSW Paper (arXiv)](https://arxiv.org/abs/1603.09320)
* [HNSWLib GitHub](https://github.com/nmslib/hnswlib)
* [Pinecone's Guide to HNSW](https://www.pinecone.io/learn/hnsw/)
* [ScaNN (Google)](https://github.com/google-research/google-research/tree/master/scann)

---

## 🧰 Tips

* Tune `M` (max connections) and `ef_construction` for indexing quality.
* Tune `ef_search` during querying to trade off between speed and recall.
* The more layers, the faster the coarse search — but more memory is used.

---

## 🧑‍🎓 Ideal For:

* Educational purposes (understanding ANN search)
* Building minimal custom vector indexes
* Prototyping HNSW logic before using libraries like FAISS or HNSWLib

---

## 🤝 Contributing

Pull requests are welcome! Whether you're improving efficiency, adding unit tests, or optimizing distance metrics, your contributions help others learn and build better.

---

## 🛠️ License

This implementation is intended for educational use. For production-level deployments, consider using optimized C++ libraries like [hnswlib](https://github.com/nmslib/hnswlib) or [faiss](https://github.com/facebookresearch/faiss).
