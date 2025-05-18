# 🧠 Grouped Query Attention (GQA)

---

## 📖 Description

Grouped Query Attention (**GQA**) is a scalable variant of the traditional multi-head attention mechanism. Rather than giving each query vector its own dedicated set of keys and values, GQA **shares key/value heads across multiple queries**, reducing redundancy and computational overhead.

It is especially beneficial for **large language models**, where the **number of attention heads increases linearly with model size**. GQA helps **reduce memory usage and improve throughput** — without significantly degrading performance.

---

## 💡 Why GQA?

Traditional multi-head attention requires each query head to have its own unique key and value heads:

* This leads to **high memory and compute cost**, especially as models scale (e.g., GPT-3, LLaMA).
* Empirically, not all query heads need their own keys and values — **some can share them**.

**GQA** addresses this by:

* Using **fewer key-value heads** than query heads.
* **Grouping multiple query heads** to attend over **shared key-value pairs**.
* Striking a **balance between expressiveness and efficiency**.

---

## 🔬 Formula: Grouped Query Attention

For each group $g$, grouped attention is computed as:

$$
\text{GQA}(Q_g, K_g, V_g) = \text{softmax}\left(\frac{Q_g K_g^\top}{\sqrt{d_k}}\right) V_g
$$

Where:

* $Q_g$: Query matrix for group $g$
* $K_g, V_g$: Shared Key and Value matrices for group $g$
* $d_k$: Dimensionality of keys (for scaling the dot product)

> ✅ All queries in the same group use the **same keys and values** during attention computation.

---

## 🧠 Multi-Head Grouped Query Attention (MHGQA)

Multi-head GQA extends this to multiple groups (or heads) in parallel.

$$
\text{MHGQA}(Q, K, V) = \text{Concat}\left(\text{Attention}_1, \text{Attention}_2, \dots, \text{Attention}_G\right)
$$

Where:

* The total number of **query heads** may be **greater than** the number of **key/value heads**.
* For example: 32 query heads and 8 shared key/value heads → 4 query heads per group.

---

## 📈 How Query Grouping Works

To group query vectors, models can use several strategies:

| Grouping Strategy             | Description                                                           |
| ----------------------------- | --------------------------------------------------------------------- |
| **Random Grouping**           | Assign queries to groups arbitrarily. Simple but not optimal.         |
| **Fixed/Manual Grouping**     | Each query head is statically assigned to a specific key/value group. |
| **Learned Grouping**          | The model learns optimal groupings during training.                   |
| **Clustering (e.g. K-Means)** | Group queries based on similarity in the representation space.        |

> 🔍 In LLaMA 2, query heads are grouped *uniformly and statically*, typically using a fixed ratio (e.g., 8:1 query to kv heads).

---

## 🚀 Benefits in LLaMA and GPT-4

Grouped Query Attention has been **adopted in high-performance models** due to its strong efficiency-to-performance tradeoff:

| Benefit                    | Description                                                                    |
| -------------------------- | ------------------------------------------------------------------------------ |
| 💾 **Memory Efficient**    | Reduces the number of key/value projections (fewer parameters).                |
| ⚡ **Faster Inference**     | Reduces FLOPs and attention computation during inference.                      |
| 🧠 **Good Generalization** | Maintains accuracy comparable to full attention in many benchmarks.            |
| 🤖 **Scaling-Friendly**    | Particularly helpful for 7B+ parameter models where attention is a bottleneck. |

---

## 🔁 Comparison with Standard Multi-Head Attention

| Feature           | Standard Multi-Head Attention | Grouped Query Attention (GQA) |
| ----------------- | ----------------------------- | ----------------------------- |
| Query Heads       | 1 per head                    | 1 per head                    |
| Key/Value Heads   | 1 per head                    | Shared across groups          |
| Total Projections | q + k + v per head            | q per head, fewer k/v         |
| Memory Cost       | High                          | Lower                         |
| Typical Use       | BERT, GPT-2                   | LLaMA-2, Falcon, GPT-4        |

---

## 🧩 Visual Intuition

```
Traditional Attention (Full)
[Q1] -> [K1, V1]
[Q2] -> [K2, V2]
[Q3] -> [K3, V3]
...

Grouped Query Attention
[Q1] ┐
[Q2] ├──> [K_shared, V_shared]
[Q3] ┘
...
```

> Instead of learning a separate K/V for every Q, GQA **shares them across grouped queries**, reducing the number of expensive matrix multiplications.

---

## ✅ Summary Table

| Component       | Grouped Query Attention     |
| --------------- | --------------------------- |
| Type            | Attention optimization      |
| Origin          | First used in PaLM, LLaMA-2 |
| Query Heads     | Per-head                    |
| Key/Value Heads | Fewer, shared across Q      |
| Memory Usage    | Lower                       |
| Speed           | Faster than vanilla MHA     |
| Use Cases       | LLaMA-2, GPT-4, Falcon      |

---

## 📚 Further Reading

* [Efficient Transformers: Grouped Query Attention](https://arxiv.org/abs/2305.13245)
* [LLaMA 2 Model Card (Meta)](https://ai.meta.com/llama/)
* [PaLM Architecture (Google)](https://arxiv.org/abs/2204.02311)
