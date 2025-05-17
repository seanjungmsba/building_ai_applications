# 🧩 Rotated Positional Embeddings (RoPE)

## 📚 Description

* In transformer models like **LLaMA**, **Rotary Positional Embeddings (RoPE)** are used to inject positional information **directly into the attention mechanism** without adding extra trainable parameters like traditional positional embeddings.
* Instead of adding fixed vectors to token embeddings, RoPE **rotates** the **query and key vectors** in multi-head attention according to their position index — allowing the model to encode **relative and absolute positions** implicitly.
* This technique enables **better generalization** to **longer sequences**, as it decouples position modeling from fixed embeddings.

---

## 🧠 How RoPE Works

1. RoPE modifies the **attention mechanism**, not the embeddings directly.
2. At each position, it **rotates each vector subspace** (typically of size 2) using a fixed frequency-based sinusoidal pattern.
3. The result is that token relationships are modeled via relative distance — enabling **long context support** without loss of meaning.

---

## 🧮 Key Concepts Summary

| Concept               | Description                                                                                         |
| --------------------- | --------------------------------------------------------------------------------------------------- |
| **Rotation**          | Instead of positional vectors being added, RoPE applies a sinusoidal rotation to queries and keys   |
| **Parameterless**     | No learnable parameters are needed — uses mathematical frequency functions                          |
| **Relative Encoding** | Enables attention to focus on *how far* tokens are apart, not just where they are                   |
| **Interpretable**     | Acts like a "positional phase shift" in embedding space — making position effects easy to interpret |

---

## 🔢 Mathematical Formula

RoPE transformation applied to a vector $x$ at position $p$:

$$
\text{RoPE}(x, p) = x \cdot \cos(p) + \text{rotate}(x) \cdot \sin(p)
$$

Where:

* $x$ is the embedding vector (query or key),
* $p$ is the frequency-adjusted positional scalar (based on sinusoidal functions),
* $\text{rotate}(x)$ flips 2D sub-vectors:

  $$
  \text{rotate}([x_1, x_2]) = [-x_2, x_1]
  $$

This process rotates the vector in its embedding subspace (like a 2D rotation matrix), encoding the position into its orientation.

---

### 📘 Example Calculation

Given:

* Input vector $x = [1, 2]$
* Position scalar $p = 1$

Then:

$$
\text{RoPE}([1, 2], 1) = [1, 2] \cdot \cos(1) + [-2, 1] \cdot \sin(1)
$$

Result: A new embedding that encodes position **implicitly via direction**.

---

## 🧠 Behind the Code (PyTorch RoPE)

Here's a minimal implementation of RoPE in PyTorch:

```python
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        seq_len = x.size(1)
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_emb = emb.cos().unsqueeze(0).unsqueeze(0)
        sin_emb = emb.sin().unsqueeze(0).unsqueeze(0)
        return (x * cos_emb) + (rotate_half(x) * sin_emb)
```

And the `rotate_half` function:

```python
def rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat((-x2, x1), dim=-1)
```

---

## 🧩 Visual Intuition

Imagine each token’s vector as an arrow in 2D space.
RoPE rotates that arrow based on its position. The **angle of rotation is derived from sine/cosine waves** and varies per dimension.
When two tokens are compared in attention, their **relative rotation encodes their distance**.

```
                       ↑
           Token A     |    Token B (rotated more)
             │         |        /
             │         |       /
             │         |      /
             │         |     /
             │         |    /
─────────────┼─────────┼───/────────────→ Position Axis (Token Index)
             │         |
             │         |
             │         |
             ▼         ▼
       Origin       (rotated by θ_B)
```
### 🔍 Explanation:
- Each token has a vector (arrow) pointing in a direction — these represent embeddings in 2D for simplicity.
- RoPE rotates each vector by an angle proportional to its position in the sequence. The deeper the token is in the sequence, the larger the rotation.

    #### For example:
    - Token A (early position) is rotated slightly (almost vertical).
    - Token B (later position) is rotated more significantly.
    - Dot product in attention between rotated queries and keys captures their relative position naturally.

---

## 🚀 Why RoPE is Powerful

| Feature           | Traditional Positional Embeddings | RoPE                                     |
| ----------------- | --------------------------------- | ---------------------------------------- |
| Operation         | Added to embeddings               | Applied to attention vectors (query/key) |
| Generalization    | Poor on unseen sequence lengths   | Strong extrapolation                     |
| Trainable         | Often yes                         | No                                       |
| Relative encoding | ❌                                 | ✅                                        |
| Memory-efficient  | ❌                                 | ✅                                        |

---

## 📈 Summary Table

| Attribute | RoPE                               |
| --------- | ---------------------------------- |
| Type      | Relative Positional Encoding       |
| Params    | None                               |
| Operation | Sinusoidal Rotation                |
| Stability | High                               |
| Used In   | LLaMA, GPT-NeoX, ChatGLM, GLM, MPT |

---

## 📚 Further Reading

* 📄 [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
* 🔍 [LLaMA Model Card](https://huggingface.co/docs/transformers/model_doc/llama)
* 🔬 [Illustrated Explanation](https://kexue.fm/archives/8265)