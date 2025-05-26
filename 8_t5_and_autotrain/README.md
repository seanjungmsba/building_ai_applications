# 🧠 T5: Text-to-Text Transfer Transformer

---

## 📖 Overview

**T5** (Text-to-Text Transfer Transformer), introduced by Google Research, is a powerful encoder-decoder model that unifies **all** natural language processing (NLP) tasks under the format of **text-to-text**. This means:

> “Every task, whether classification, translation, or summarization, is cast as feeding a text input and expecting a text output.”

This unified approach allows a single model architecture to handle a wide range of tasks without needing task-specific changes.

---

## ✨ Key Highlights

* Reframes every NLP task as a sequence-to-sequence task.
* Based on the Transformer architecture (encoder + decoder).
* Trained using a **span corruption** denoising objective.
* Achieves state-of-the-art results across multiple benchmarks.
* Easily extensible to multilingual and instruction-following tasks.

---

## 🔄 Unified Format for All Tasks

| Task          | Input Format                                 | Output                    |
| ------------- | -------------------------------------------- | ------------------------- |
| Translation   | `translate English to German: That is good.` | `Das ist gut.`            |
| Sentiment     | `sst2 sentence: I love this movie.`          | `positive`                |
| Summarization | `summarize: The article discusses...`        | `Key points are...`       |
| QA            | `question: What is T5? context: ...`         | `A text-to-text model...` |

---

## 🏗️ Architecture

T5 is built on the **standard Transformer encoder-decoder** architecture.

### 🔹 Encoder

* **Input**: Corrupted span-masked text.
* **Uses**:

  * **Relative position embeddings** (instead of absolute).
  * **Layer normalization** and **GELU** activation.
* **Objective**: Encode meaningful representations of partially masked text.

### 🔸 Decoder

* **Input**: Previously generated tokens.
* **Uses**:

  * **Causal (masked) self-attention**.
  * **Cross-attention to encoder outputs**.
  * **Autoregressive token generation** from left to right.
* **Goal**: Reconstruct the masked spans by generating the correct tokens one at a time.

---

## 🧪 Example: Span Corruption and Reconstruction

**Original Sentence**:
`The quick brown fox jumps over the lazy dog.`

**Corrupted Input to Encoder**:
`The <extra_id_0> fox jumps <extra_id_1> dog.`
(Masked spans: `quick brown` and `over the lazy`)

**Target Output to Decoder**:
`<extra_id_0> quick brown <extra_id_1> over the lazy`

**Generation Process**:

| Step | Prediction     |
| ---- | -------------- |
| t=1  | `<extra_id_0>` |
| t=2  | `quick`        |
| t=3  | `brown`        |
| t=4  | `<extra_id_1>` |
| t=5  | `over`         |
| t=6  | `the`          |
| t=7  | `lazy`         |

---

## 🔍 T5 vs BERT vs GPT

| Feature                 | **T5**                                                                | **BERT**                                     | **GPT**                                  |
| ----------------------- | --------------------------------------------------------------------- | -------------------------------------------- | ---------------------------------------- |
| Architecture            | Encoder-Decoder                                                       | Encoder-Only                                 | Decoder-Only                             |
| Input Format            | Text-to-Text                                                          | Classification/MLM                           | Autoregressive Text Generation           |
| Pretraining Objective   | Span Corruption (Text-to-Text Denoising)                              | Masked Language Modeling (MLM)               | Left-to-right Language Modeling          |
| Output Generation       | Text (seq2seq)                                                        | Classification / Embedding-based             | Text (unidirectional)                    |
| Use Cases               | Translation, QA, Summarization, Classification                        | Embeddings, Classification, QA               | Story Generation, Chatbots, QA           |
| Autoregressive?         | ✅ (Decoder is autoregressive)                                         | ❌                                            | ✅                                        |
| Pretraining Example     | `Input: The <extra_id_0> fox...` → `Output: <extra_id_0> quick brown` | `Input: The [MASK] fox...` → `Output: quick` | `Input: The quick brown` → `Output: fox` |
| Fine-Tuning Flexibility | Very High (all tasks as text-to-text)                                 | Task-specific heads often required           | Very High (for generation tasks)         |

---

## 🧬 Variants

| Variant     | Description                                                             |
| ----------- | ----------------------------------------------------------------------- |
| **mT5**     | Multilingual T5 trained on 101 languages.                               |
| **FLAN-T5** | Instruction-tuned version of T5, excels in zero-shot/few-shot learning. |
| **LongT5**  | Uses a reformulated encoder to handle very long inputs efficiently.     |

---

## 📚 Further Reading

* [📄 T5 Paper (2020)](https://arxiv.org/abs/1910.10683) — *"Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"*
* [🤗 HuggingFace T5 Models](https://huggingface.co/models?search=t5)
* [🌍 mT5 Paper](https://arxiv.org/abs/2010.11934)
* [📘 FLAN-T5 Paper](https://arxiv.org/abs/2210.11416)
