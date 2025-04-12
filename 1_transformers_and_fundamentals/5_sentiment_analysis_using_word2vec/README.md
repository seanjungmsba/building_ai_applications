## 🧠 Word2Vec and Gensim

### 📌 What is Word2Vec?

- **Word2Vec** is a shallow, two-layer neural network that learns to represent words as **dense vectors** in a continuous vector space.
- These vectors (called **word embeddings**) capture **semantic relationships** — for example:
  - `vector("king") - vector("man") + vector("woman") ≈ vector("queen")`

### 🔍 Why Use Word2Vec?

- Traditional one-hot encoding doesn’t capture meaning, similarity, or context.
- Word2Vec embeddings:
  - Place **similar words closer together** in vector space.
  - Encode **relationships and analogies**.
  - Serve as input to downstream tasks like **sentiment analysis, text classification,** and **chatbots**.

### ⚙️ How It Works

There are two main architectures:
1. **CBOW (Continuous Bag of Words):** Predicts a word based on its context.
2. **Skip-Gram:** Predicts surrounding words given a target word.

These models are trained on large corpora and learn from **co-occurrence patterns** in text.

---

## 🚀 Using Word2Vec with Gensim

### 🧰 Gensim

- [**Gensim**](https://radimrehurek.com/gensim/) is a Python library for unsupervised learning on text, especially useful for loading **pretrained Word2Vec models**.

### 💾 What We'll Do

- Load a **pretrained Word2Vec model** trained on the Google News corpus.
- Use this model to embed words from our own sentences.
- Use the embeddings in tasks like **sentiment analysis**.

---

## 📦 Setup Instructions

```bash
pip install torch
pip install gensim
pip install numpy
```

---

### 📂 Load Pretrained Word2Vec

```python
from gensim.models import KeyedVectors

# Load binary format model (300-dimensional vectors)
model_path = "googlenews-vectors-negative300.bin"
model = KeyedVectors.load_word2vec_format(model_path, binary=True)
```

> 💡 **Note:** This pretrained model is ~3.5GB and was trained on **100 billion words** from Google News.

---

### 🧪 Example: Get a Word’s Vector

```python
vector = model['happy']
print(vector.shape)  # (300,)
print(vector[:10])   # First 10 values of the embedding
```

---

### 🔍 Example: Measure Similarity

```python
similarity = model.similarity('happy', 'joyful')
print(f"Similarity between 'happy' and 'joyful': {similarity:.2f}")
```

---

### 🔁 Example: Find Most Similar Words

```python
model.most_similar('coffee')
# Output might include: 'espresso', 'caffeine', 'latte', etc.
```

---

## 📚 Additional Resources

- 🧠 [Word2Vec - Inner Workings (Medium)](https://towardsdatascience.com/word2vec-with-pytorch-implementing-original-paper-2cd7040120b0/)
- 📦 Download pretrained model:  
  [Google News Word2Vec (binary format)](https://data-engineering-academy.s3.us-east-1.amazonaws.com/ai-course/assets/section-one/GoogleNewsvectorsnegative300.bin)

---

## 🧠 TL;DR

| Concept      | Description                                               |
|--------------|-----------------------------------------------------------|
| Word2Vec     | Neural net that converts words into meaningful vectors    |
| Embeddings   | 300-dim vectors capturing word similarity & meaning       |
| Gensim       | Python lib for loading and querying Word2Vec models       |
| Pretrained   | Google News corpus vectors (100 billion words)            |

