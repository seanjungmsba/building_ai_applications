## 🔢 Embeddings

### 📌 What Are Embeddings?

Embeddings are **dense vector representations** of data — usually **tokens like words or subwords** — used as numerical input for neural networks.

- Each token (e.g. a word like "cat") is mapped to a fixed-length vector, such as `[0.12, -0.98, 0.44, ...]`.
- These vectors **capture semantic meaning**, allowing the model to understand how words relate to each other.
- They serve as a **bridge between raw text and mathematical operations** used in deep learning.

✅ **Why are embeddings important?**
- They let models work with complex data (like language) in a form that's learnable.
- Similar words (e.g. "king", "queen") have similar vector representations, making it easier to model relationships.

---

### 🧠 How Are Embeddings Used?

Embeddings are applied at various stages of natural language processing (NLP) pipelines:

- **In Transformers**: The first step is converting tokens into embeddings before feeding them into encoders or decoders.
- **In Search Engines**: To measure similarity between queries and documents.
- **In Recommender Systems**: For mapping users, items, and even text descriptions into the same space.
- **Cross-modal applications**: Like aligning text and image data (e.g., CLIP by OpenAI).

---

### 📚 Types of Embeddings

Depending on what you’re embedding and how much context you want, there are multiple levels:

#### 1. **Word Embeddings**
- Represent each word as a single vector.
- Do **not consider context** (e.g., “bank” will have the same vector whether it means a financial institution or river bank).

#### 2. **Subword / Character Embeddings**
- Break words into smaller units (e.g., "playing" → "play" + "##ing").
- Helpful for handling unknown or rare words.

#### 3. **Sentence / Document Embeddings**
- Represent an entire sentence, paragraph, or document as a single vector.
- Capture **context and structure** beyond individual words.

---

### 🛠️ Example Word Embedding Algorithms

These are foundational methods used to train word-level embeddings:

- **Word2Vec**: Learns embeddings by predicting context words (Skip-gram) or predicting a word from its context (CBOW).
- **GloVe (Global Vectors)**: Uses word co-occurrence statistics to learn embeddings.
- **FastText**: Enhances Word2Vec by using subword information (great for morphology).
- **ELMo**: Contextual word embeddings generated using bidirectional LSTMs (captures meaning based on sentence context).
- **BERT Embeddings**: Outputs embeddings that are **context-aware** and change depending on usage in a sentence.

---

### 🧪 Bonus: Embeddings in Practice

You can visualize embeddings using techniques like **t-SNE** or **UMAP** to reduce them to 2D/3D and cluster similar meanings.

You can also **fine-tune** embeddings during training, or use **pre-trained** ones (like GloVe or BERT) depending on your needs and compute budget.

---
