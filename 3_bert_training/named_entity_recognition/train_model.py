import lightning as pl
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# ---------------------------------------------------------
# 🔧 Custom Modules (Local files)
# ---------------------------------------------------------
# These modules handle dataset preparation and model logic.
# - ner_dataset.py: Contains the NERDataModule class
# - ner_model.py: Contains NERModel (wrapper around BERT) and LitModule (Lightning logic)
# - medical_ner_data.py: Provides training and validation examples

from ner_dataset import NERDataModule
from ner_model import NERModel, LitModule
from medical_ner_data import train_texts, train_labels, val_texts, val_labels, label_to_id

# ---------------------------------------------------------
# 🔤 Step 1: Load Tokenizer
# ---------------------------------------------------------
# Load the pretrained BERT tokenizer for 'bert-base-uncased'.
# This tokenizer will be used to convert raw text into tokens + token IDs.
tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained('bert-base-uncased')

# ---------------------------------------------------------
# 📦 Step 2: Initialize DataModule
# ---------------------------------------------------------
# The NERDataModule handles:
# - Tokenization of text and label alignment
# - Batching and padding sequences
# - Creating PyTorch DataLoaders for training and validation
#
# Args:
# - tokenizer: the BERT tokenizer
# - train/val_texts: input sentences
# - train/val_labels: BIO-formatted labels
# - label_map: dictionary mapping entity tags to numeric IDs
# - batch_size, max_length: tokenization and training settings
# - num_workers: parallelism for DataLoader

data_module = NERDataModule(
    tokenizer=tokenizer,
    train_sentences=train_texts,
    train_labels=train_labels,
    val_sentences=val_texts,
    val_labels=val_labels,
    label_map=label_to_id,
    batch_size=2,
    max_length=128,
    num_workers=1
)

# ---------------------------------------------------------
# 🧠 Step 3: Initialize NER Model (BERT + Classification Head)
# ---------------------------------------------------------
# The NERModel wraps BERT (e.g., AutoModelForTokenClassification)
# and outputs per-token logits for entity prediction.
#
# Args:
# - model_name: BERT variant (must match tokenizer)
# - num_labels: number of distinct BIO tags

model = NERModel(
    model_name='bert-base-uncased',
    num_labels=len(label_to_id)
)

# ---------------------------------------------------------
# ⚡ Step 4: Wrap with PyTorch LightningModule
# ---------------------------------------------------------
# The LitModule defines:
# - forward pass
# - loss calculation
# - training/validation steps and metrics
#
# You can also customize learning rate or optimization logic.

model = LitModule(model=model, learning_rate=1e-5)

# ---------------------------------------------------------
# 🚀 Step 5: Set Up Trainer
# ---------------------------------------------------------
# Trainer coordinates training loops, callbacks, logging, etc.
# - max_epochs: how many full passes over the training data
# - accelerator='auto': uses GPU if available

trainer = pl.Trainer(
    max_epochs=3,
    accelerator='auto'
)

# ---------------------------------------------------------
# 🏁 Step 6: Start Training
# ---------------------------------------------------------
# Run training using the Trainer and prepared model/data.
# This will tokenize, feed into BERT, compute loss, and update weights.

if __name__ == '__main__':
    trainer.fit(model=model, datamodule=data_module)

"""
---------------------------------------------------------
🧪 Training Output Summary (Run Log)
---------------------------------------------------------

✅ MODEL LOADING
- Successfully loaded 'bert-base-uncased'.
- Classification head (`classifier.weight` and `classifier.bias`) is randomly initialized — expected for downstream NER task.

✅ TRAINING LOGS
- Epoch 0: train_loss = 2.020, train_acc = 0.000
- Epoch 1: train_loss = 1.540, train_acc = 0.714, val_loss = 1.850, val_acc = 0.233
- Epoch 2: train_loss = 1.220, train_acc = 0.714, val_loss = 1.400, val_acc = 0.700

✅ OBSERVATIONS
- Training accuracy improved sharply after the first epoch.
- Validation accuracy rose significantly, reaching 70% after 3 epochs.
- Suggests the model is learning the NER task effectively.

⚠️ NOTES
- GPU not detected (training on CPU, slower).
- `tensorboardX` is not installed, so training logs are stored in CSV format.
- Consider using `LitModelCheckpoint` for saving best models.
- The `hf_xet` warning is related to GitHub model versioning and can be ignored unless using XetHub.

📈 NEXT STEPS
- Evaluate on test data or unseen sentences.
- Visualize using TensorBoard (`pip install lightning[extra] tensorboard`).
- Add F1-score for better sequence tagging evaluation.

---------------------------------------------------------
"""
