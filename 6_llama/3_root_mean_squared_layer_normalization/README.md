# 🧮 Root Mean Square Layer Normalization (RMSNorm)

---

## 📘 Description

* **RMSNorm** (Root Mean Square Normalization) is a simplified variant of **Layer Normalization (LayerNorm)** that normalizes inputs using the **root mean square (RMS)** instead of full standard deviation.
* Unlike LayerNorm, which subtracts the mean and scales by the variance, RMSNorm **only scales the input** — it does **not center it**. This reduces computational complexity and improves numerical stability while still enabling effective training.

---

## 🔬 Formula

The RMSNorm transformation is defined as:

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}} \cdot \gamma
$$

Where:

* $x$: input vector of shape `[batch, hidden_dim]`
* $N$: number of features (hidden dimension)
* $\epsilon$: small constant to avoid division by zero
* $\gamma$: learnable scaling parameter (same shape as $x$)

---

### 🧠 Intuition

* Instead of computing **mean** and **variance**, RMSNorm computes just the **root mean square (RMS)** of the input.
* Think of RMS as a way to **measure the average magnitude** of the vector without worrying about direction or shift.
* This technique allows the model to **retain the raw structure of the signal** while keeping the magnitude under control.

---

## 🔍 Key Difference from LayerNorm

| Feature              | **LayerNorm**              | **RMSNorm**                     |
| -------------------- | -------------------------- | ------------------------------- |
| Mean subtraction     | ✅ Yes                      | ❌ No                            |
| Division by variance | ✅ Yes (std)                | ✅ Yes (but only RMS)            |
| Learnable scaling    | ✅ Yes ($\gamma$)           | ✅ Yes ($\gamma$)                |
| Learnable bias       | ✅ Optional ($\beta$)       | ❌ Usually omitted               |
| Computation cost     | ❌ Higher (mean + variance) | ✅ Lower (only squared mean)     |
| Stability            | ✅ Good                     | ✅ Often more numerically stable |
| Use in Transformers  | BERT, GPT, etc.            | **LLaMA**, PaLM, GLM, etc.      |

> **Summary**: RMSNorm simplifies the math of LayerNorm while preserving its benefits for deep learning stability and convergence.

---

## 🚀 Why LLaMA Uses RMSNorm

LLaMA replaces LayerNorm with RMSNorm because:

1. **Efficiency**: No need to compute mean — fewer operations.
2. **Scalability**: More numerically stable for very large models.
3. **No Centering Required**: Works well in attention-heavy architectures where zero-centered inputs aren't necessary.
4. **Compatible with Pre-Norm** Transformers: RMSNorm is well-suited for **pre-norm transformer blocks**, where normalization is applied before attention/FFN.

---

## ⚙️ Practical PyTorch Snippet

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-8):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.gamma * x / rms
```

---

## 📊 Visual Intuition

```
Before RMSNorm:
    x = [1.2, -0.8, 0.5, -1.7]  → varies in both direction and scale

After RMSNorm:
    x ≈ [0.61, -0.41, 0.25, -0.86] → same "shape", but consistent magnitude
```

Unlike LayerNorm, the **center (mean) of the vector is not shifted** — it simply scales everything to have unit RMS magnitude.

---

## 🧩 Summary Table

| Attribute            | RMSNorm                          |
| -------------------- | -------------------------------- |
| Centering (mean sub) | ❌ No                             |
| Scale adjustment     | ✅ Yes (via RMS)                  |
| Learnable scale γ    | ✅ Yes                            |
| Bias parameter β     | ❌ Usually not included           |
| Used In              | LLaMA, PaLM, GLM, RWKV           |
| Benefits             | Simpler, faster, and more stable |

---

## 📚 Further Reading

* [RMSNorm Paper (Zhang et al., 2019)](https://arxiv.org/abs/1910.07467)
* [LLaMA Architecture Insights](https://huggingface.co/blog/llama2)