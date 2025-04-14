# 🧠 Named Entity Recognition with BERT (Medical Domain)

This project demonstrates how to fine-tune a BERT model for **Named Entity Recognition (NER)** in the **medical domain**, identifying key entities such as medical conditions, symptoms, and procedures.

---

## 📌 Project Overview

Named Entity Recognition (NER) is the task of identifying and classifying named entities in text into predefined categories. In this project:

- We fine-tune `bert-base-uncased` using **BIO tagging format**.
- Our entities include:
  - `condition`
  - `symptom`
  - `procedure`

We use Hugging Face Transformers, PyTorch Lightning, and TorchMetrics for robust model training and evaluation.

---

## 📁 Directory Structure

```
named_entity_recognition/
├── train_model.py           # Main training script
├── ner_model.py             # BERT model wrapper + LightningModule
├── ner_dataset.py           # Dataset and DataModule definitions
├── medical_ner_data.py      # Sample training/validation data & label mapping
├── metrics.csv              # Training output file
└── README.md                # This file
```

---

## 🔤 Labels and BIO Schema

We follow the **BIO (Beginning, Inside, Outside)** tagging format to label tokens:

| Tag           | Meaning                            |
|---------------|-------------------------------------|
| B-condition   | Beginning of a condition entity     |
| I-condition   | Inside of a condition entity        |
| B-symptom     | Beginning of a symptom entity       |
| I-symptom     | Inside of a symptom entity          |
| B-procedure   | Beginning of a procedure entity     |
| I-procedure   | Inside of a procedure entity        |
| O             | Token outside of any entity         |

### Example

Sentence: `"MRI scan showed a tumor."`

Labels: `[B-procedure, I-procedure, O, O, B-condition, O]`

---

## 🏗️ How It Works

### Step 1: Data Preparation

- Defined in `medical_ner_data.py`
- Each sentence is annotated using BIO schema.
- Sentences are wrapped in lists for compatibility with Hugging Face tokenizers.

### Step 2: Dataset & DataModule (`ner_dataset.py`)

- `NERDataset` aligns subword tokens with labels using `word_ids()`.
- `NERDataModule` handles batching, padding, and loading splits for training and validation.

### Step 3: Model Definition (`ner_model.py`)

- `NERModel` wraps Hugging Face's `BertForTokenClassification`.
- `LitModule` defines:
  - `training_step`, `validation_step`, and `configure_optimizers`
  - Logs metrics using `torchmetrics.Accuracy` (ignores padding)

### Step 4: Training Script (`train_model.py`)

- Loads tokenizer and datasets.
- Initializes model and PyTorch Lightning `Trainer`.
- Runs training for 3 epochs.

---

## 🧪 Sample Training Output

```text
Epoch 0: train_loss = 2.020, train_acc = 0.000
Epoch 1: train_loss = 1.540, train_acc = 0.714, val_loss = 1.850, val_acc = 0.233
Epoch 2: train_loss = 1.220, train_acc = 0.714, val_loss = 1.400, val_acc = 0.700
```

- Accuracy improves significantly by Epoch 2.
- Indicates model is learning to recognize medical entities.

---

## 📦 Technologies Used

- 🤗 Transformers
  - `AutoTokenizer`, `BertForTokenClassification`
- ⚡ PyTorch Lightning
  - `LightningModule`, `Trainer`, `DataModule`
- 📊 Torchmetrics
  - `MulticlassAccuracy`

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install transformers torch pytorch-lightning torchmetrics nltk

# Run training
python train_model.py
```

---

## 🧠 Next Steps

- Add a test set and evaluation script
- Integrate F1-score and confusion matrix
- Add entity-level span extraction during inference
- Deploy model using FastAPI or Streamlit

---

## 📚 References

- [BERT - Devlin et al.](https://arxiv.org/abs/1810.04805)
- [Hugging Face Token Classification](https://huggingface.co/docs/transformers/tasks/token_classification)
- [BIO Tagging Format](https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging))
