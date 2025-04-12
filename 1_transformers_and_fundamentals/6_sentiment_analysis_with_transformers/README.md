# 📘 Project 1 — Sentiment Analysis with Transformers

## 🧠 Overview

This project demonstrates how to perform **sentiment analysis** using a lightweight Transformer-based model powered by **Word2Vec embeddings**. The model is designed to classify a sentence as **positive** or **negative** based on its semantic structure and word composition.

Instead of training from scratch on large datasets, we leverage **pretrained Word2Vec vectors** to embed tokens and feed them through a **Self-Attention mechanism**, mimicking the core ideas behind Transformer encoders.

---

## 🎯 Objective

- Convert a sentence into Word2Vec-based embeddings.
- Apply a **self-attention mechanism** to capture contextual relationships between words.
- Use the final representation to perform **binary classification** (positive or negative sentiment).

---

## 🧱 Model Architecture

### 1. 🔡 **Word2Vec Embeddings**
- We begin by converting each word in a sentence into a **300-dimensional vector** using a pretrained Word2Vec model.
- Out-of-vocabulary tokens are handled using zero vectors.
- These embeddings form the **input matrix** of shape `[batch_size, sequence_length, embed_dim]`.

### 2. 🧲 **Self-Attention Layer**
Implemented via a custom `SelfAttention` class:
- Inspired by the Transformer encoder mechanism.
- Computes `Query`, `Key`, and `Value` matrices from the word embeddings.
- Uses **multi-head attention** to capture multiple perspectives of the token relationships.
- Produces an attention-enhanced representation for each word in context.

### 3. 📊 **Mean Pooling**
- After attention is applied, the outputs are mean-pooled across the sequence dimension to summarize the sentence into a single embedding vector.

### 4. 🎯 **Classification Layer**
- A fully connected layer maps the pooled embedding to a 2-dimensional output:
  - `logits[0]` → Negative
  - `logits[1]` → Positive

---

## 🧪 Sentiment Classification

### ✨ Why Sentiment Analysis?
- It's one of the most accessible and practical NLP tasks.
- Helps in customer feedback analysis, product reviews, social media monitoring, etc.

### 📊 Task Type
- **Binary classification**: We aim to detect **positive** or **negative** sentiment in each input sentence.

### 🏷️ Target Classes
- `0` → Negative
- `1` → Positive

---

## 🧩 Code Components

### 🔸 `SelfAttention(embed_size, heads)`
- Breaks embeddings into `heads` and projects them to `Q`, `K`, and `V`.
- Uses **Einstein summation (`einsum`)** to compute attention scores and outputs efficiently.
- Output shape: `[batch_size, sequence_length, embed_size]`.

### 🔸 `SentimentAnalysisModel(embed_size, heads, num_classes)`
- Wraps the `SelfAttention` block and applies **mean pooling** over the sequence.
- Final `Linear(embed_size, num_classes)` projects pooled vector to class logits.

---

## 💡 Usage Example

```python
# Input: tensor of Word2Vec embeddings shaped [batch_size, sequence_len, 300]
model = SentimentAnalysisModel(embed_size=300, heads=6, num_classes=2)
output = model(x)  # output shape: [batch_size, 2]
