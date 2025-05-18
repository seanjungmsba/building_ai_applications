# 🦙 LLaMA Decoder-Only Transformer (Mini Implementation)

This project provides a simplified, educational implementation of a **decoder-only Transformer architecture** inspired by **Meta's LLaMA model**. It supports modern design features like **Rotary Positional Embedding (RoPE)**, **Grouped Query Attention (GQA)**, **SwiGLU feedforward layers**, and **RMSNorm**.

---

## 📁 Project Structure

```text
.
├── model.py           # Core model implementation (LLaMA + components)
├── model_test.py      # Script to test the model on a toy sentence
└── README.md          # You are here!
```

---

## 🧠 `model.py` — LLaMA-like Model Definition

### 🔧 Components

| Module                      | Description                                                                           |
| --------------------------- | ------------------------------------------------------------------------------------- |
| `Llama`                     | Top-level language model class. Composes embedding, decoder blocks, and output layer. |
| `Decoder`                   | Contains self-attention + feedforward + normalization + residual connections.         |
| `GroupedQueryAttention`     | GQA implementation with shared key/value heads for efficiency.                        |
| `RotaryPositionalEmbedding` | Implements RoPE to encode relative positions via trig. rotation.                      |
| `FeedForward`               | Two-layer MLP with **SwiGLU** activation and dropout.                                 |
| `SwiGLU`                    | Efficient gated activation: `SwiGLU(a, b) = silu(a) * b`.                             |
| `RMSNorm`                   | Simpler normalization than LayerNorm, used in LLaMA for better efficiency.            |

### 🧱 Key Innovations

* **RoPE** injects relative positional info into queries/keys *without trainable embeddings*.
* **GQA** reduces memory and compute by grouping query heads to share fewer key/value heads.
* **SwiGLU** improves performance and stability over GeLU in feedforward blocks.
* **RMSNorm** normalizes based on Root Mean Square — efficient and stable.

---

## 🧪 `model_test.py` — Model Evaluation Script

This script runs a forward pass using the model defined in `model.py` and prints out **top-k predictions** for the **next word** in a given input sequence.

### 🔄 Steps

1. **Define a toy vocabulary** with a few words.
2. **Tokenize a short input sentence** (`"The brown fox"`).
3. **Pass tokens into the model** to generate logits.
4. **Apply softmax** to get a probability distribution.
5. **Extract and print top-k predicted tokens**.

### 💡 Sample Output

```bash
Input Sequence: 'The brown fox'
Top predictions for the next word:
  crosses: 0.2513
  the:     0.2321
  road:    0.2005
  <pad>:   0.1800
  <unk>:   0.1361
```

---

## 📚 Concepts Illustrated

| Concept                               | Explanation                                                                                        |
| ------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **Decoder-Only Transformer**          | Only uses the decoder portion of the Transformer architecture (suitable for autoregressive tasks). |
| **Self-Attention with Causal Mask**   | Ensures that tokens only attend to previous positions.                                             |
| **Rotary Positional Encoding (RoPE)** | Adds relative position info to queries and keys through rotation.                                  |
| **Grouped Query Attention (GQA)**     | Shares key/value projections across groups of query heads, reducing overhead.                      |
| **SwiGLU FeedForward**                | Provides efficient non-linearity and richer transformations.                                       |
| **RMSNorm**                           | Applies root mean square normalization for stable training.                                        |

---

## 🛠️ Requirements

Make sure you have the following libraries installed:

```bash
pip install torch
```

---

## 🚀 Run the Test Script

```bash
python model_test.py
```

---

## 📌 Credits

This is a simplified educational reproduction of architectural components used in modern Transformer architectures like:

* [Meta AI's LLaMA](https://arxiv.org/abs/2302.13971)
* [GPT-4 architecture variants](https://platform.openai.com/docs/)
* [RoPE (Rotary Positional Encoding)](https://arxiv.org/abs/2104.09864)