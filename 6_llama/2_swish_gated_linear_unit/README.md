# ⚡ SwiGLU (Swish-Gated Linear Unit)

---

## 📖 Description

* **SwiGLU** is an advanced **activation function** used in modern transformer architectures such as **LLaMA**, **GPT-4**, and **PaLM**.
* It is derived from **Swish** and **Gated Linear Units (GLU)**, combining the best of **non-linear expressiveness** and **gating control** to allow models to learn more complex and stable patterns.
* Compared to ReLU or even GeLU, SwiGLU improves training **speed**, **memory efficiency**, and **numerical stability** without sacrificing accuracy.

---

## 🔍 Core Idea: Gating with Smooth Activation

SwiGLU introduces **gating** into the activation layer by splitting the input tensor and **modulating one half using the sigmoid of the other**:

* ✅ **Gating mechanism** enables control over information flow
* ✅ **Sigmoid** adds a soft decision boundary
* ✅ **Element-wise product** introduces nonlinear transformation with multiplicative interaction

This makes SwiGLU **richer than ReLU/GeLU** in terms of expressive capacity.

---

## 🧮 Mathematical Formula

SwiGLU is defined in three key steps:

### 1. **Chunking**

Split input tensor $x$ into two equal parts along the last dimension:

$$
\text{Chunk}(x) \rightarrow (a, b)
$$

### 2. **Sigmoid Activation**

Apply the sigmoid function to chunk $a$:

$$
\sigma(a) = \frac{1}{1 + e^{-a}}
$$

### 3. **Gated Multiplication**

Compute the final activation:

$$
\text{SwiGLU}(a, b) = b \cdot \sigma(a)
$$

---

## 🔬 Example: Step-by-Step Walkthrough

Suppose:

```python
x = torch.tensor([4.0, -1.0, 5.0, 2.0])  # shape = (4,)
```

### Step 1: Chunk

```python
a = [4.0, -1.0]
b = [5.0, 2.0]
```

### Step 2: Apply sigmoid to a

```python
sigmoid(a) ≈ [0.982, 0.269]
```

### Step 3: Multiply elementwise

```python
output = [5.0 * 0.982, 2.0 * 0.269] ≈ [4.91, 0.538]
```

➡️ Final SwiGLU output: **`[4.91, 0.538]`**

---

## 🧠 Why SwiGLU Is Used in Transformers (like LLaMA)

In the feedforward layer of a transformer block, the sequence typically looks like:

```python
x → Linear → Activation → Linear
```

For models using **SwiGLU**, the middle activation is implemented as:

```python
x → Linear(2d) → SwiGLU → Linear(d)
```

The **first linear layer doubles the hidden size**, producing two tensors (`a` and `b`). SwiGLU **gates the values** before projecting them back to the original dimensionality.

This offers:

| Benefit          | Description                                                |
| ---------------- | ---------------------------------------------------------- |
| 🚀 **Speed**     | Faster than GeLU due to simpler math (sigmoid vs erf)      |
| 🧠 **Stability** | Bounded sigmoid avoids overflow/underflow issues           |
| 🪶 **Memory**    | Reduces intermediate tensor creation                       |
| 🎯 **Accuracy**  | Often yields better or equal performance on NLP benchmarks |

---

## 🖼 Visual Intuition

```
        ┌────────────┐
        │  Input x   │
        └────┬───────┘
             ▼
   ┌─────────────────────┐
   │ Chunk into (a, b)   │  ← split x along last dim
   └────┬────────┬───────┘
        ▼        ▼
   Sigmoid(a)    b
        │        │
        ▼        ▼
   ┌───────────────┐
   │ a × sigmoid(b)│  ← element-wise product (gating)
   └──────┬────────┘
          ▼
      SwiGLU Output
```

---

## 📊 SwiGLU vs GeLU: Comparison Table

| Property              | GeLU                       | SwiGLU                   |
| --------------------- | -------------------------- | ------------------------ |
| Core Function         | Gaussian Error Linear Unit | Swish + Gating           |
| Speed                 | Slower due to erf          | Faster (sigmoid only)    |
| Expression Power      | High                       | Higher (due to gating)   |
| Numerical Stability   | Moderate                   | Excellent                |
| Usage in Transformers | Common                     | LLaMA, GPT-4, PaLM, GLaM |

---

## ⚙️ Implementation in PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split input tensor along last dimension into two chunks
        a, b = x.chunk(2, dim=-1)
        # Apply sigmoid to first chunk, and multiply with second chunk
        return F.silu(a) * b  # silu is numerically stable swish variant
```

Used after a linear layer that expands the input dim from `d` to `2d`.

---

## 🧩 Summary

| Feature         | Details                                |
| --------------- | -------------------------------------- |
| Full Name       | Swish-Gated Linear Unit                |
| Formula         | `SwiGLU(x) = b * sigmoid(a)`           |
| Activation Type | Smooth, Gated                          |
| Speed           | Faster than GeLU                       |
| Found In        | LLaMA, GPT-4, PaLM                     |
| Key Strength    | Expressive gating with stable training |

---

## 📚 Further Reading

* 🔬 [Gated Linear Units (GLU) - Facebook](https://arxiv.org/abs/1612.08083)
* 🧠 [SwiGLU Activation (Official Code Reference)](https://github.com/facebookresearch/llama)
* 📝 [Silicon-Valley-friendly Explanation of SwiGLU](https://sebastianraschka.com/blog/2023/swiglu-explained.html)