"""
NER Model using Hugging Face Transformers and PyTorch Lightning

This module defines:
1. `NERModel`: A wrapper around `BertForTokenClassification` for performing Named Entity Recognition (NER).
2. `LitModule`: A PyTorch Lightning module that manages the training and evaluation loop, metrics, and optimization.

It integrates Hugging Face's pretrained BERT model with Lightning for clean, scalable training.
"""

import torch
import torch.nn as nn
import torchmetrics  # Metrics library compatible with PyTorch Lightning
import lightning as pl  # PyTorch Lightning for organized model training
from transformers import BertForTokenClassification  # Hugging Face components
from torch.optim import AdamW  # AdamW optimizer for transformer models

# --------------------------------------------------------------------------------
# Class: NERModel
# --------------------------------------------------------------------------------
# A simple wrapper around Hugging Face's BertForTokenClassification.
# This is used to fine-tune BERT for token-level classification tasks like NER.

class NERModel(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        """
        Args:
            model_name (str): Name of the pretrained BERT model (e.g., 'bert-base-uncased').
            num_labels (int): Total number of unique NER labels including O, B-*, I-* tags.
        """
        super(NERModel, self).__init__()

        # Load a pre-trained BERT model configured for token classification
        # This automatically adds a classification head on top of the encoder
        self.model = BertForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels  # Adjusts the output layer for the number of classes
        )

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward method for model inference or training.
        Args:
            input_ids (torch.Tensor): Token indices from tokenizer
            attention_mask (torch.Tensor): 1s for real tokens, 0s for padding
            labels (torch.Tensor, optional): Ground-truth labels for loss computation

        Returns:
            A model output containing loss (if labels are provided), logits, etc.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels  # If labels are given, model returns loss as well
        )

# --------------------------------------------------------------------------------
# Class: LitModule
# --------------------------------------------------------------------------------
# PyTorch Lightning module to handle training loop, validation loop, metrics, and optimizer config.

class LitModule(pl.LightningModule):
    def __init__(self, model: NERModel, learning_rate: float = 2e-5):
        """
        Args:
            model (NERModel): An instance of the NERModel class.
            learning_rate (float): Learning rate for optimizer.
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

        # TorchMetrics Accuracy: handles masking and correct averaging across batches
        self.accuracy = torchmetrics.Accuracy(
            task='multiclass',
            num_classes=self.model.model.config.num_labels,
            ignore_index=-100  # Ensure special tokens and padding are excluded
        )

    def training_step(self, batch, batch_idx):
        """
        Executes one training step:
        - Runs forward pass
        - Computes loss and accuracy
        - Logs metrics

        Args:
            batch (dict): Contains input_ids, attention_mask, and labels
            batch_idx (int): Index of the current batch (unused)
        """
        outputs = self.model(batch['input_ids'], batch['attention_mask'], batch['labels'])
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=-1)
        labels = batch['labels']
        acc = self.accuracy(preds, labels)

        # Logging to Lightning's logger; prog_bar=True enables display in progress bar
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Executes one validation step:
        - Runs forward pass
        - Computes loss and accuracy
        - Logs validation metrics

        Args:
            batch (dict): Contains input_ids, attention_mask, and labels
            batch_idx (int): Index of the current batch (unused)
        """
        outputs = self.model(batch['input_ids'], batch['attention_mask'], batch['labels'])
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=-1)
        labels = batch['labels']
        acc = self.accuracy(preds, labels)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """
        Sets up the optimizer for training.
        Using AdamW, which is recommended for transformer-based models.

        Returns:
            torch.optim.Optimizer: The configured optimizer instance.
        """
        return AdamW(params=self.model.parameters(), lr=self.learning_rate)
