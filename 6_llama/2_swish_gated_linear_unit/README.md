# ⚡ SwiGLU (Swish-Gated Linear Unit)

---

## 📖 Description

- **SwiGLU** is an activation function variant used in modern transformer architectures like **Llama** to **speed up training** and **improve model quality**.
- It is a **gated version** of the **Swish** activation and serves as a **more efficient replacement** for functions like **GeLU**.
- SwiGLU is smooth and non-monotonic, meaning it introduces gentle non-linearities that help models **learn more expressive representations** without abrupt changes (like ReLU's hard zeroing).

---
<br />

## 🔥 Why Use SwiGLU Over GeLU?

| Aspect                    | GeLU                                      | SwiGLU                                 |
|----------------------------|------------------------------------------|----------------------------------------|
| Speed                     | Involves approximating error functions (erf) | Simple sigmoid and element-wise multiplication |
| Memory Efficiency         | Slightly higher, due to intermediate computations | Lower memory footprint |
| Numerical Stability       | Risk of instability with extreme inputs | More stable due to bounded sigmoid |
| Training Impact           | Smooth activation, good results | Similar or better results, faster training |

✅ **Summary**: SwiGLU achieves **comparable or better accuracy** while **reducing computational overhead**, making it ideal for large-scale transformer models like Llama.

---
<br />

## 🧮 Mathematical Formula

The SwiGLU operation can be broken into **three main steps**:

1. **Chunking**:  
   The input tensor \( x \) is **split into two halves** along the hidden dimension:  
   $$
   \text{Chunk}(x) \rightarrow (a, b)
   $$
   where \( a \) and \( b \) are equal-sized partitions of \( x \).

2. **Sigmoid Activation**:  
   The **first chunk** \( a \) passes through a **sigmoid activation function** \( \sigma(x) \), which smoothly squashes values between 0 and 1:
   $$
   \sigma(a) = \frac{1}{1 + e^{-a}}
   $$

3. **Element-wise Multiplication**:  
   Finally, the second chunk \( b \) is **element-wise multiplied** by the sigmoid-activated first chunk:
   $$
   \text{SwiGLU}(a, b) = b \times \sigma(a)
   $$

---
<br />

## 🔥 How It Works (Step-by-Step Example)

Let's say:

- Input tensor \( x = [4, -1, 5, 2] \)

### Step 1: Chunking

Split into two halves:
- \( a = [4, -1] \)
- \( b = [5, 2] \)

### Step 2: Sigmoid Activation

Apply sigmoid to \( a \):
- \( \sigma(4) \approx 0.982 \)
- \( \sigma(-1) \approx 0.269 \)

So:
- \( \sigma(a) = [0.982, 0.269] \)

### Step 3: Multiplication

Multiply element-wise:
- \( [5 \times 0.982, 2 \times 0.269] = [4.91, 0.538] \)

Thus, **SwiGLU output** = `[4.91, 0.538]`

---
<br />

## ⚙️ How It's Used in Transformers (e.g., Llama)

- In transformer blocks, after the multi-head attention and before applying the output layer, a **feedforward network** (FFN) uses SwiGLU instead of GeLU.
- This allows **faster training**, **lower memory usage**, and often **better generalization** without sacrificing performance.
- SwiGLU **preserves the gating mechanism** similar to Gated Linear Units (GLUs), offering **richer transformations** for each token’s embedding.

---
<br />

## 📊 Quick Visual Comparison: SwiGLU vs GeLU

| Feature | GeLU | SwiGLU |
|--------|------|--------|
| Shape | Smooth but based on Gaussian | Smooth based on sigmoid |
| Speed | Slower | Faster |
| Stability | Moderate | High |
| Training Effect | Very good | Equally good, sometimes better |

---
<br />

## ✍️ Summary

| Feature | Details |
|---------|---------|
| Name | SwiGLU (Swish-Gated Linear Unit) |
| Purpose | Fast, stable, efficient non-linearity for transformers |
| Mathematical Trick | Split tensor → Apply sigmoid → Multiply |
| Model Usage | Llama, GPT-4, modern transformers |
| Benefit | Speed and memory efficiency |
