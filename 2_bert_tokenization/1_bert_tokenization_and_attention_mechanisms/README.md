# 🔤 BERT Tokenizer: How It Works and Why It Matters

## 📘 Description

In this section, we explore how the **BERT tokenizer** processes raw text into a format suitable for Transformer models. Specifically, we'll examine:

- How BERT handles tokenization using **WordPiece**
- How special tokens like `[CLS]`, `[SEP]`, and `[MASK]` are used
- How tokenized inputs are transformed into numerical IDs
- How to implement and visualize these steps in practice

---

## 🧠 Understanding BERT Tokenization

### 🧾 What Is Tokenization in BERT?

- BERT (Bidirectional Encoder Representations from Transformers) does not work with raw text.
- Input text must first be **tokenized** and then **converted into numerical IDs**.
- This is achieved using a specialized tokenizer built on **WordPiece**, a subword tokenization algorithm.

### 💡 Why Use WordPiece?

- Standard word-based tokenizers fail with **out-of-vocabulary (OOV)** words.
- WordPiece solves this by breaking rare or unseen words into **smaller known subword units**.
- This helps BERT handle typos, new words, or domain-specific language.

---

## ⚙️ Tokenization Steps in BERT

### 1. ✂️ Basic Tokenization

- Text is **split into words and punctuation**.
- Lowercased (for `bert-base-uncased`).
- Handles unicode normalization and separates punctuation (e.g., `can't` → `can` and `'t`).

### 2. 🧩 WordPiece Tokenization

- Each word is **decomposed into subwords** using a fixed vocabulary.
- Prefix `##` is used to mark **continuation subwords**.

**Example:**

```
Input: "unbelievable"
Output Tokens: ["un", "##bel", "##iev", "##able"]
```

### 3. 🔢 Token ID Conversion

- Each token is mapped to a **unique index** in the BERT vocabulary (e.g., `bert-base-uncased` has ~30,000 tokens).
- These token IDs are used as **input to the model**.

### 4. 🔖 Adding Special Tokens

| Token  | Purpose |
|--------|---------|
| `[CLS]` | Marks the beginning of a sequence; used for classification tasks |
| `[SEP]` | Separates multiple sentences or marks end of input |
| `[PAD]` | Used to fill empty slots in sequences to match fixed input length |
| `[MASK]` | Used during training for masked word prediction (MLM) |

### 5. 🧱 Padding and Attention Masks

- Inputs are padded to match max length (e.g., 128 tokens).
- An **attention mask** is used to distinguish **real tokens (1)** vs **padding (0)**.

---

## 🔠 Special Tokens in Detail

### 🟪 `[CLS]` – Classification Token

- Always the **first token** in every input.
- Its final hidden state represents the **entire sequence**.
- Used as the embedding for tasks like:
  - Sentiment Analysis
  - Text Classification
  - Next Sentence Prediction

**Example:**

```
Input Sentence: "I love this movie."
Tokenized: [CLS] I love this movie . [SEP]
```

### 🟨 `[SEP]` – Separator Token

- Used to indicate the **end of a sentence** or separate **two sentences**.

**Single Sentence Example:**

```
[CLS] BERT is amazing . [SEP]
```

**Sentence Pair Example (NSP Task):**

```
[CLS] BERT is amazing . [SEP] It powers many chatbots . [SEP]
```

### 🟥 `[MASK]` – Masking Token for MLM

- Used during pretraining to randomly hide 15% of words.
- BERT learns to predict masked tokens using context.

**Example:**

```
Input: [CLS] The cat sat on the [MASK] . [SEP]
Goal: Predict \"mat\"
```

---

## 📌 Summary Table

| Stage                  | Description                                                  |
|------------------------|--------------------------------------------------------------|
| Basic Tokenization     | Lowercases, splits text and punctuation                      |
| WordPiece Tokenization | Breaks rare words into subwords with `##` prefix            |
| Token ID Conversion    | Maps each token to an integer ID from vocab                  |
| Special Tokens         | Adds `[CLS]`, `[SEP]`, `[MASK]`, `[PAD]`                     |
| Padding & Masking      | Pads sequences and applies attention mask                    |

---

## 📎 Resources

### 📚 Theory & Concepts

- [🔗 BERT Explained - Medium](https://medium.com/@samia.khalid/bert-explained-a-complete-guide-with-theory-and-tutorial-3ac9ebc8fa7c)
- [🔗 WordPiece Tokenization Explained](https://towardsdatascience.com/wordpiece-subword-based-tokenization-algorithm-1fbd14394ed7/)
- [🔗 Trie Data Structure for WordPiece](https://www.geeksforgeeks.org/trie-insert-and-search/)
- [🔗 Unicode Normalization (NFKC)](https://unicode.org/reports/tr15/#Norm_Forms)

---

## 📁 Vocabulary Files

- [BERT Vocab Text File (from HuggingFace)](https://huggingface.co/google-bert/bert-base-uncased/blob/main/vocab.txt)
- [JSON version for easier inspection](https://data-engineer-academy.s3.us-east-1.amazonaws.com/ai-course/assets/section-two/bert_vocab.json)

### 🔄 Convert `.txt` to `.json`

```python
from os import getcwd
from json import dump
from requests import get

# Path to save vocabulary JSON
vocab_json_path = getcwd() + "/path_to_output_file"

# Download vocab.txt from HuggingFace repo
vocab: str = get("https://huggingface.co/google-bert/bert-base-uncased/resolve/main/vocab.txt").text

# Convert to list of tokens
vocab_json: list[str] = vocab.strip().split(\"\\n\")

# Save as JSON
with open(f\"{vocab_json_path}/bert_vocab.json\", \"w\") as f:
    dump(vocab_json, f, indent=2)
```

## 🧪 Practical Usage: Hugging Face's `BertTokenizer`

The `transformers` library by Hugging Face provides a high-level interface for using pretrained BERT tokenizers out of the box. Below is an end-to-end example.

---

### ✅ Installation (if not already installed)

```bash
pip install transformers
```

---

### 🧾 Example: Tokenizing a Sentence

```python
from transformers import BertTokenizer

# Load the pretrained BERT tokenizer (uncased version)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Input text
text = "I love building NLP applications with BERT!"

# Tokenize and encode the input
encoded = tokenizer(
    text,
    add_special_tokens=True,        # Adds [CLS] and [SEP]
    padding='max_length',           # Pads to max_length (optional)
    truncation=True,                # Truncates if text is too long
    max_length=16,                  # Maximum sequence length
    return_tensors='pt',            # Return PyTorch tensors
    return_attention_mask=True      # Generate attention mask
)

# View results
print("Tokens:", tokenizer.tokenize(text))
print("Token IDs:", encoded['input_ids'])
print("Attention Mask:", encoded['attention_mask'])
```

---

### 🔄 Decode Back to Text

```python
# Decode token IDs back into human-readable text
decoded_text = tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=True)
print("Decoded:", decoded_text)
```

---

### 📌 Output Explanation

- **`input_ids`**: Tensor of token IDs corresponding to `[CLS]`, tokens, and `[SEP]`
- **`attention_mask`**: Binary mask where `1` indicates actual tokens and `0` indicates padding
- **`tokenizer.tokenize()`**: Shows how the sentence was split using WordPiece
- **`tokenizer.decode()`**: Reconstructs the sentence from the token IDs

---

### 🧪 Sample Output

```txt
Tokens: ['i', 'love', 'building', 'nl', '##p', 'applications', 'with', 'bert', '!']
Token IDs: tensor([[  101,  1045,  2293,  2311,  17953,  2361,  2003,  2007,  14324,   999,
            102,     0,     0,     0,     0,     0]])
Attention Mask: tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
Decoded: i love building nlp applications with bert!
```

---

### 📘 Summary

| Function                      | Purpose                                      |
|-------------------------------|----------------------------------------------|
| `tokenizer.tokenize(text)`    | Tokenizes text into subword tokens          |
| `tokenizer.encode(text)`      | Tokenizes and converts to token IDs         |
| `tokenizer()` (with args)     | Complete preprocessing for model input      |
| `tokenizer.decode(ids)`       | Converts token IDs back to readable string  |

---

### ✅ Wrap-Up

BERT’s tokenizer is a crucial component in preparing input for Transformer-based models. Its use of **subword tokenization** via WordPiece, combined with **special tokens** and **attention-aware formatting**, makes it extremely flexible for real-world NLP tasks.



