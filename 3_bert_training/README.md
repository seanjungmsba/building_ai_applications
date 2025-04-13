# 🏷️ BERT - Named Entity Recognition (NER)

## 📌 What Is NER?

Named Entity Recognition (NER) is a **sequence labeling** task in Natural Language Processing (NLP) where we aim to identify **real-world entities** in text and categorize them into predefined classes.

### 🧠 Example:  
Sentence:  
> `"Barack Obama was born in Hawaii and served as the president of the United States."`  

NER Output:  
> `["B-PER", "I-PER", "O", "O", "O", "B-LOC", "O", "O", "O", "O", "O", "B-ORG", "I-ORG", "O"]`  

This process supports applications such as:
- **Information Extraction**
- **Question Answering**
- **Clinical Text Analysis**
- **Search Relevance Enhancement**

---

## 🧬 BIO Labeling Schema

The **BIO** schema is the most common way to annotate named entities. Each token is labeled as:

| Tag      | Meaning                                 |
|----------|------------------------------------------|
| `B-XXX`  | Beginning of entity type `XXX`           |
| `I-XXX`  | Inside (continuation) of entity type `XXX` |
| `O`      | Outside any named entity                 |

### 📚 Common BIO Tags:

| Label   | Description                  |
|---------|------------------------------|
| B-PER   | Beginning of a person's name |
| I-PER   | Inside of a person's name    |
| B-LOC   | Beginning of a location name |
| I-LOC   | Inside of a location name    |
| B-ORG   | Beginning of an organization |
| I-ORG   | Inside of an organization    |
| O       | Non-entity token             |

### 🧮 How Many Labels Do We Need?

You need one output class per unique tag in your dataset. For example, if you're recognizing 3 entity types (PER, LOC, ORG), you'll typically need:

```
(Number of entity types * 2) + 1
[B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, O] → 7 output labels
```

> Each token in the input sequence is assigned one of these labels during training and inference.

---

## 🧪 Project: Medical Named Entity Recognition with BERT

In this project, we apply **BERT** to extract **medical entities** from clinical or radiology notes. We fine-tune BERT to detect the following medical entity types:

### 🏥 Entity Types:
- `B-CONDITION` / `I-CONDITION`: Medical diagnoses or diseases
- `B-SYMPTOM` / `I-SYMPTOM`: Patient-reported or observed symptoms
- `B-PROCEDURE` / `I-PROCEDURE`: Clinical or imaging procedures

### 📌 Example:

**Sentence**: `"MRI scan showed a brain tumor."`  
**Tokens**: `["MRI", "scan", "showed", "a", "brain", "tumor", "."]`  
**NER Tags**: `["B-PROCEDURE", "I-PROCEDURE", "O", "O", "B-CONDITION", "I-CONDITION", "O"]`

---

## ⚙️ Technology Stack

We leverage modern Python NLP tools for training and evaluation.

### 🔗 HuggingFace Transformers
- `AutoTokenizer`: Automatically loads the appropriate tokenizer for BERT
- `AutoModelForTokenClassification`: Loads BERT for NER task
- `AdamW`: Optimizer designed for weight decay regularization

### 🔥 PyTorch
- `DataLoader`: Efficient batching and shuffling of token-label pairs

### ⚡ PyTorch Lightning
- `LightningModule`: Modular structure for training/validation logic
- `LightningDataModule`: Handles dataset splits and loading
- `Trainer`: Engine that automates training/validation/testing
- `TensorBoardLogger`: For logging and visualization

### 📊 Torchmetrics
- Metrics such as `Accuracy`, `F1-score`, and `Precision` are used to evaluate model performance

---

## 🧱 Sample Input and Output Pipeline

1. **Raw Sentence**: `"Patient experienced chest pain during exercise."`  
2. **Tokenized Input**: `["Patient", "experienced", "chest", "pain", "during", "exercise", "."]`  
3. **NER Tags**: `["O", "O", "B-SYMPTOM", "I-SYMPTOM", "O", "O", "O"]`  
4. **Model Output**: Logits → Softmax → Predicted Tags

---

## 🧰 Useful Resources

### 📄 Theory and Examples

- [📚 Medical NER Training Data - Python Script](https://data-engineer-academy.s3.us-east-1.amazonaws.com/ai-course/assets/section-two/bert-training/medical_ner_data.py)
- [🧠 What is NER?](https://nlp.stanford.edu/software/CRF-NER.html)
- [🧬 BIO Tagging Explained](https://huggingface.co/transformers/task_summary.html#named-entity-recognition)

### 🧪 Related Tools

- [scispacy](https://allenai.github.io/scispacy/): Pretrained models and pipelines for biomedical NER
- [spaCy NER](https://spacy.io/usage/linguistic-features#named-entities): Lightweight alternative to HuggingFace for some use cases

---

## ✅ Summary Table

| Component               | Description                                       |
|-------------------------|---------------------------------------------------|
| BERT Token Classification | Fine-tunes BERT for sequence tagging           |
| BIO Schema              | Standard labeling scheme for NER                 |
| PyTorch Lightning       | Modular and scalable training framework          |
| HuggingFace Transformers | Plug-and-play BERT model and tokenizer support |
| Metrics                 | Precision, Recall, F1-score for evaluation       |

---
