# 🧠 Attention Mechanisms

## 🎯 What is Attention?

**Attention** is a technique that allows models to **focus on relevant parts** of the input when making predictions. In natural language, not every word is equally important for understanding meaning — attention helps prioritize what matters.

---

## 1. 🔄 Self-Attention

### ❓ What

**Self-attention** allows a token to "look at" other tokens in the same sequence to understand their relationships. It outputs a weighted combination of the tokens based on how relevant they are to each other.

### 🤔 Why

- Words often depend on other words for meaning (e.g., "he" might refer to "John" earlier in the sentence).
- We need to **model long-range dependencies** — even across many words.

### ⚙️ How

Self-attention is computed using this formula:

$$ 
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

- **Q** (Query): The word we're evaluating (e.g., "he").
- **K** (Key): Words it compares against (e.g., "John", "dog").
- **V** (Value): The actual word info we want to pull from.
- **dₖ**: Scaling factor to prevent large dot products from dominating softmax.

### 📦 Analogy

Imagine you're in a meeting. You’re trying to understand your task (query), so you look around (keys) to see who is speaking (values). The more relevant someone’s message is to your task, the more attention you give it.

---

## 2. 🧠 Multi-Head Attention

### ❓ What

**Multi-head attention** means running **multiple self-attention mechanisms in parallel**, each with different learned parameters (called "heads").

### 🤔 Why

- Different attention heads can focus on **different types of relationships**:
  - One head may track syntax (e.g., subject-verb pairs),
  - Another may follow semantics (e.g., pronoun references).

### ⚙️ How

Each head computes:

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

Then, all heads are concatenated and projected back to the original dimension:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$

Where:
- Wᵢᑫ, Wᵢᵏ, Wᵢᵛ: learned projection matrices for each head  
- Wᵒ: final output projection

### 📦 Analogy

Think of multiple **spotlights on a stage**, each focused on something different — dialogue, props, lighting. Combining them gives a complete understanding of the scene.

---

## 3. 🙈 Masked Attention

### ❓ What

**Masked attention** blocks access to future tokens during training in **autoregressive tasks** (like text generation).

### 🤔 Why

- In generation, you shouldn't be able to "see the future".
- The model should only use the current and previous words to predict the next one.

### ⚙️ How

Add a **mask matrix M** to the attention formula:

$$
\text{MaskedAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

Where:
- \( M \): A matrix with `-∞` in future positions (to zero them out after softmax).
- Only past/present positions get valid attention weights.

### 📦 Analogy

Like a student writing a sentence one word at a time, with **blinders** on — they can only see the words they’ve already written.

---

# 🧪 Self-Attention: Step-by-Step Example

### 📝 Sentence:
**"The brown fox crossed the road"**  
Tokens: `"The"`, `"brown"`, `"fox"`

---

### Step 1: Token Embeddings


Each token is converted into a vector representation:

$$
$$
X = \begin{bmatrix}
1 & 0 & 1 & 0 \\\\
0 & 2 & 0 & 2 \\\\
1 & 1 & 1 & 1
1 & 0 & 1 & 0 \\\\
0 & 2 & 0 & 2 \\\\
1 & 1 & 1 & 1
\end{bmatrix}
$$
$$

---

### Step 2: Generate Q, K, V

All equal to \( X \):
### Step 2: Generate Q, K, V

All equal to \( X \):

$$
$$
Q = K = V = X
$$
$$

---

### Step 3: Compute Attention Scores

$$
QK^\top =
$$
QK^\top =
\begin{bmatrix}
2 & 0 & 2 \\\\
0 & 8 & 4 \\\\
2 & 0 & 2 \\\\
0 & 8 & 4 \\\\
2 & 4 & 4
\end{bmatrix}
$$
$$

---

### Step 4: Scale Scores

$$
\frac{QK^\top}{\sqrt{4}} = \frac{QK^\top}{2} =
$$
\frac{QK^\top}{\sqrt{4}} = \frac{QK^\top}{2} =
\begin{bmatrix}
1 & 0 & 1 \\\\
0 & 4 & 2 \\\\
1 & 0 & 1 \\\\
0 & 4 & 2 \\\\
1 & 2 & 2
\end{bmatrix}
$$
$$

---

### Step 5: Apply Softmax

Softmax turns scores into probabilities:

For token 1:

$$

$$
\text{Softmax}([1, 0, 1]) \approx [0.422, 0.155, 0.422]
$$
$$

---

### Step 6: Weighted Sum of Values

$$
$$
\text{Output}_{\text{The}} = [0.422, 0.155, 0.422] \times V \approx [0.844, 0.577, 0.844, 0.577]
$$
$$

---

### ✅ Final Outputs

| **Token** | **Self-Attention Output**               |
|-----------|-----------------------------------------|
| The       | [0.844, 0.577, 0.844, 0.577]            |
| brown     | [0.134, 1.865, 0.134, 2.015]            |
| fox       | [0.577, 1.422, 0.577, 1.422]            |

---

# 🧪 Masked Attention: Step-by-Step Example

Same input: `"The brown fox"`  
Same token embeddings and Q = K = V

---

### Step 5: Apply Mask

$$
$$
M = \begin{bmatrix}
0 & -\infty & -\infty \\\\
0 & 0 & -\infty \\\\
0 & -\infty & -\infty \\\\
0 & 0 & -\infty \\\\
0 & 0 & 0
\end{bmatrix}
$$
$$

$$
$$
\text{Masked Scores} = \text{Scaled Scores} + M =
\begin{bmatrix}
1 & -\infty & -\infty \\\\
0 & 4 & -\infty \\\\
1 & -\infty & -\infty \\\\
0 & 4 & -\infty \\\\
1 & 2 & 2
\end{bmatrix}
$$
$$

---

### Step 6: Apply Softmax

$$
\text{Softmax}([1, -\infty, -\infty]) = [1, 0, 0]
$$
### Step 6: Apply Softmax

$$
\text{Softmax}([1, -\infty, -\infty]) = [1, 0, 0]
$$

---

### Step 7: Weighted Sum of Values

For Token `"The"`:

$$

$$
\text{Output}_{\text{The}} = [1, 0, 0] \times V = [1, 0, 1, 0]
$$
$$

---

### ✅ Final Outputs

| **Token** | **Masked Attention Output**             |
|-----------|-----------------------------------------|
| The       | [1, 0, 1, 0]                            |
| brown     | [0.018, 1.964, 0.018, 1.964]            |
| fox       | [0.577, 1.422, 0.577, 1.422]            |

---

# ✅ Summary Table

| Mechanism           | What it Does                                      | Why it's Useful                                      | Analogy                              |
|---------------------|---------------------------------------------------|------------------------------------------------------|--------------------------------------|
| Self-Attention       | Each word attends to all others in the sequence   | Captures context from the entire sentence            | You listen to everyone in the room   |
| Multi-Head Attention | Multiple attention layers in parallel             | Captures multiple relationships and nuances          | Spotlights on different scene parts  |
| Masked Attention     | Prevents attention to future tokens               | Enables sequential prediction like language modeling | You write with blinders on           |
