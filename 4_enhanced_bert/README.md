# 🧠 RoBERTa: A Robustly Optimized BERT Pretraining Approach

## 📌 Overview

**RoBERTa** (Robustly Optimized BERT Approach) is a **transformer-based encoder model** introduced by Facebook AI, designed to push the limits of BERT by improving the training methodology while keeping the underlying architecture intact.

This repository demonstrates RoBERTa’s usage for **abstractive text summarization**, including a local Flask API and Streamlit interface for generating summaries from real articles.

---

## 🔄 How RoBERTa is Similar to BERT

RoBERTa and BERT share key architectural foundations:

- 🧱 **Identical Transformer Encoder Layers**  
  Both use multi-head self-attention + position-wise feedforward networks.

- 📚 **Masked Language Modeling (MLM)**  
  Pretraining objective involves predicting randomly masked words.

- 🧩 **WordPiece Tokenization**  
  Inputs are tokenized into subword units using a shared vocabulary.

---

## 🔧 How RoBERTa Improves on BERT

RoBERTa introduces a number of **training-level optimizations**:

1. **Larger Batch Sizes**  
   Trained with larger mini-batches for better gradient estimates.

2. **Dynamic Learning Rates**  
   Implements warmup schedules and decay for stable convergence.

3. **Dynamic Masking**  
   Random masks change on every epoch to improve generalization.

4. **No Next Sentence Prediction (NSP)**  
   NSP task is removed as it was found unhelpful in practice.

5. **More Data, Longer Training**  
   Trained on over 160GB of uncompressed text vs BERT’s 16GB.

---

## 🧪 Project Structure & Implementation

### 🔍 `summarizer_util.py`

This script loads a **pretrained RoBERTa encoder-decoder model** (`google/roberta2roberta_L-24_bbc`) using Hugging Face’s `transformers` library:

- Loads tokenizer + `AutoModelForSeq2SeqLM`.
- Encodes raw text input into token IDs.
- Generates summaries using `.generate()` (greedy decoding).
- Converts output token IDs back into human-readable summaries.

> 📎 Try it in isolation with `run_summarizer.py` (see below).

---

### 🌐 `model_api.py`

Implements a RESTful API using **Flask**, exposing a `/summary` endpoint.

- Accepts POST requests with article text.
- Calls `generate_summary()` from `summarizer_util.py`.
- Returns a JSON response with model metadata and the summary.

This is intended for use by both command-line clients and the Streamlit frontend.

---

### 📺 `client.py` (Streamlit UI)

Lightweight web app interface for summarization:

- Input a block of text via a `text_area`.
- Sends the content to `http://localhost:8080/summary`.
- Renders the result inside the Streamlit app.

> 🚀 Run it with:  
> `streamlit run client.py`

---

### 🧪 `run_summarizer.py`

Offline test script that:

- Downloads a long article from S3.
- Feeds it to `generate_summary()` locally (no API involved).
- Prints both full text and summarized output.

Useful for quick model experiments without launching the API.

---

### 📦 `summarizer_api_test.py`

API testing script for verifying Flask functionality.

- Sends a test POST request to the running server.
- Parses and prints the JSON response.

---

## 🔍 Static vs Dynamic Masking

### 🧱 BERT: Static Masking

In traditional BERT, once a token is masked, it **remains masked in every epoch**. This results in overfitting to specific tokens and **poorer generalization**.

#### 💬 Example

Given: `"The cat sat on the mat."`

| Epoch   | Masked Sentence                   |
|---------|-----------------------------------|
| 1       | "The cat **[MASK]** on the mat." |
| 2       | "The cat **[MASK]** on the mat." |
| 3       | "The cat **[MASK]** on the mat." |

---

### 🚀 RoBERTa: Dynamic Masking

RoBERTa dynamically reselects mask positions every time a sentence is seen during training.

#### 💬 Example

| Epoch   | Masked Sentence                           |
|---------|-------------------------------------------|
| 1       | "**[MASK]** sat on the mat."              |
| 2       | "The cat sat **[MASK]** the mat."         |
| 3       | "The cat **[MASK]** on the **[MASK]**."   |

✅ This strategy forces the model to **learn contextual representations** instead of memorizing patterns.

---

## 🧠 Summary Table: BERT vs RoBERTa

| Feature                     | BERT                         | RoBERTa                         |
|-----------------------------|------------------------------|----------------------------------|
| Pretraining Objective       | MLM + NSP                    | MLM only                        |
| Masking Strategy            | Static                       | Dynamic                         |
| Batch Size                  | ~256                         | >8K                             |
| Training Data               | ~16GB                        | >160GB                          |
| Tokenization                | WordPiece                    | WordPiece                       |
| Learning Rate Schedule      | Fixed/Warm-up                | Tuned decay                     |
| Input Format                | Sentence pairs               | Single/contiguous sequences     |
| Use Case in This Project    | Not used                     | Summarization (via encoder-decoder) |

---

## 🚀 Example Usage

```bash
# Run the API
python model_api.py

# Open the Streamlit UI in another terminal
streamlit run client.py
```

Or test the pipeline without a UI:
```bash
python run_summarizer.py
```

---

## 📚 Further Reading

- [RoBERTa Paper (arXiv)](https://arxiv.org/abs/1907.11692)
- [Illustrated BERT](https://jalammar.github.io/illustrated-bert/)
- [RoBERTa on HuggingFace](https://huggingface.co/roberta-base)
- [Google RoBERTa2RoBERTa Model](https://huggingface.co/google/roberta2roberta_L-24_bbc)

---
