---
marp: true
theme: gaia
paginate: true
---

<center>

<br />
<br />
<br />
<br />

# Transformers

</center>

---
<!-- header: Agenda -->

## Agenda

1. Use Cases
2. High-Level Architecture
3. Tokenization
4. Attention Mechanisms
5. Position Embeddings
6. Project 1 - Sentiment Analysis with Transformer from Scratch via PyTorch
7. Popular Variants - BERT and RoBERTa
8. Project 2 - Text Summarization with BERT Models from HuggingFace
9. Knowledge Distillation
10. Popular Variants - DistilBERT
11. Increasing Context Windows - RoPE and Flash Attention
12. Fine-tuning
13. Efficient Fine-tuning with Low Rank Adaptation (LoRa)
14. Popular Variants - T5
14. Project 3 - Fine-tuning T5 for Named Entity Recognition (NER) with Autotrain
16. Generalized Pretrained Transformer (GPT)
17. Project 4 - Building GPT from Scratch with PyTorch and Lightning AI
18. Alignment - RLHF and DPO
19. Project 5 - Improving GPT Responses with DPO
20. Large Language Models (LLMs)
21. Popular Variants - Llama Architecture
22. Project 6 - Fine-tuning Llama 3.3 8B for Medical Question-Answering with LitGPT
23. Popular Variants - DeepSeek
24. Alignment Variant - GRPO
25. Project 7 - Conducting Local Inference with DeepSeek via Ollama 

<!-- header: Use Cases -->

<center>

## Use Cases

</center>

1. Machine Translation
    - Translating spoken / written languages.
    - Converting one programming language to another.

2. Question Answering
    - Answering questions based on a given context.

3. Text Generation
    - Generating text based on a given prompt.
    - Summarizing long texts.

4. Classification
    - Classifying text into categories.
    - Sentiment analysis.

5. Named Entity Recognition
    - Identifying and classifying entities in text.

---

<!-- header: Transformer Architecture -->

<center>

## Transformer Architecture

![Credits: Dive into Deep Learning](https://d2l.ai/_images/transformer.svg)

# 🧠 Understanding Transformer Architecture (For Software Engineers)

The Transformer is a machine learning architecture designed for tasks like **language translation**, **text summarization**, or even **code generation**. It consists of two major parts: the **Encoder** and the **Decoder**.

Let’s use a real-world analogy you might appreciate:

---

## 🔄 High-Level Analogy

Imagine you're leading two teams:

- 🏗️ **Encoder Team** – Reads a document (e.g. Korean tech spec) and deeply understands it.
- ✍️ **Decoder Team** – Takes that understanding and writes a clean English version.

Each team uses a communication system that includes **attention mechanisms** to know which parts of the document are most relevant at each step.

---

## 🧱 1. Encoder – “The Analyst”

**Goal:** Understand the input and turn it into a rich, context-aware representation.

- **Input:** A sequence of tokens (e.g. words from a sentence or JSON keys).
- **Process:**
  - Convert tokens to embeddings (numerical vectors).
  - Add positional encoding (to maintain word order).
  - Use multi-head attention to understand relationships between words.
  - Normalize and refine with a feed-forward layer.
- **Output:** A sequence of context-rich embeddings sent to the decoder.

🧠 **Analogy:**  
Like parsing a config file and annotating each key-value pair with rich metadata and relationships, preparing it for someone else to use.

---

## ✍️ 2. Decoder – “The Developer”

**Goal:** Generate meaningful output (like translated text or predicted code) one token at a time.

- **Input:**
  - The previous output tokens (initially just `<start>`),
  - The encoder’s output (context embeddings).
- **Process:**
  - Masked multi-head attention prevents peeking ahead.
  - Another attention layer looks at the encoder’s context.
  - Output is refined with a feed-forward layer.
- **Output:** A sequence of tokens forming the final result (e.g. translated sentence or next line of code).

🧠 **Analogy:**  
Like generating documentation or test cases based on a config file with rich metadata.

---

## 🎯 Bonus: Probability Score Output

Instead of directly outputting words, the decoder produces a **probability distribution** over the entire vocabulary. For example:

```json
{
  "the": 0.45,
  "a": 0.30,
  "an": 0.25
}
```
The model picks or samples from these probabilities to produce the most likely next word.

</center>

---

1. Encoder
    - **Input**: Sequence of tokens from your data.
    - **Output**: Sequence of embeddings to provide context to the decoder.

2. Decoder
    - **Input**: Sequence of tokens from your data.
    - **Output**: Sequence of tokens to generate text or probability score for a given task.

---

<!-- header: Tokenization -->

<br />
<br />
<br />
<br />

<center>

## Tokenization

</center>

---

1. Tokenization
    - Splitting text into tokens.
    - Converting tokens into embeddings.

2. What is a token?
    - A token is a word or a subword or a character in language modeling.

3. What is an embedding?
    - An embedding is a vector representation of a token in language modeling.

---

<center>

![](https://miro.medium.com/v2/resize:fit:1050/1*6ttE_KrYJ9iPdBEfa-Qj2g.png)

</center>

---

<center>

![](https://i0.wp.com/eastgate-software.com/wp-content/uploads/2024/08/Blogs-photos-2.jpg?resize=978%2C551&ssl=1)

</center>

---

4. Tokenizer
    - A tokenizer is a function that converts text into tokens.

5. Popular Tokenizers
    - Word Tokenizers
    - Sentence Tokenizers
    - Byte Pair Encoding (BPE)

---

## Python Examples

1. `nltk` tokenization
    - Word Tokenizer
    - Sentence Tokenizer
2. `PyTorch` implementation of `Byte Pair Encoding`

---

<!-- header: Attention Mechanisms  -->

<br />
<br />
<br />
<br />

<center>

## Attention Mechanisms

</center>

---

1. Self Attention
    - A mechanism to allow the model to focus on different positions of the input sequence.

2. Multi-head Attention
    - A mechanism to allow the model to focus on different positions of the input sequence, but with multiple attention heads for parallelization.

3. Masked Multi-head Attention
    - Contrasts multi-head attention by introducing a mask to prevent the model from attending to future tokens.
