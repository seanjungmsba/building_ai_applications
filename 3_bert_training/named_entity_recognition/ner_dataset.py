"""
NER Dataset and DataModule for PyTorch Lightning

This module defines two classes:
1. `NERDataset`: A custom PyTorch Dataset that handles tokenization, label alignment,
   and tensor preparation for a BIO-tagged Named Entity Recognition task.
2. `NERDataModule`: A PyTorch Lightning DataModule that encapsulates all logic needed 
   to train and validate an NER model using the `NERDataset`.

The code is compatible with Hugging Face tokenizers and designed for transformer-based 
NER pipelines.
"""

import torch
import lightning as pl
from torch.utils.data import Dataset, DataLoader


class NERDataset(Dataset):
    """
    A custom PyTorch Dataset for BIO-tagged Named Entity Recognition (NER) tasks.
    
    This class:
    - Takes tokenized sentences and their BIO labels.
    - Uses a tokenizer (like BERT's) to encode the input text.
    - Aligns the word-level labels with the subword tokenized output.
    - Pads and truncates sequences to a fixed max_length.
    - Outputs `input_ids`, `attention_mask`, and `labels` for each sample.
    """
    def __init__(self, tokenizer, sentences, labels, label_map, max_length=128):
        """
        Args:
            tokenizer (PreTrainedTokenizer): Hugging Face tokenizer.
            sentences (List[List[str]]): List of tokenized sentences (word-level).
            labels (List[List[str]]): BIO-tagged labels for each token.
            label_map (Dict[str, int]): Mapping from label string to integer.
            max_length (int): Maximum token length for padding/truncation.
        """
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.labels = labels
        self.label_map = label_map
        self.max_length = max_length

    def __len__(self):
        """
        Returns the total number of samples.
        """
        return len(self.sentences)

    def __getitem__(self, idx):
        """
        For a given index, returns a dictionary with:
            - input_ids: Tensor of token IDs
            - attention_mask: Tensor indicating which tokens are padding
            - labels: Tensor of label IDs aligned with input_ids

        Special tokens (e.g., [CLS], [SEP]) are ignored in loss calculation via -100 labels.
        """
        # Retrieve tokens and their associated BIO labels
        tokens = self.sentences[idx]
        labels = [self.label_map[label] for label in self.labels[idx]]

        # Tokenize using the Hugging Face tokenizer
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,  # Treat each item in the list as a separate word
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Remove batch dimension since this is a single example
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # Initialize label tensor with -100 (ignored in loss)
        label_ids = torch.full(
            (self.max_length,),
            -100,
            dtype=torch.long
        )

        # Map token positions to word positions to align labels correctly
        word_ids = encoding.word_ids()

        label_index = 0  # Pointer to iterate over our actual label list
        for i, word_id in enumerate(word_ids):
            if word_id is not None and label_index < len(labels):
                label_ids[i] = labels[label_index]
                label_index += 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids
        }


class NERDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for managing NER datasets.

    This class:
    - Wraps the `NERDataset` class for train/val splits.
    - Provides easy integration with Lightning’s Trainer.
    - Creates efficient DataLoaders for training and validation phases.
    """
    def __init__(
        self,
        tokenizer,
        train_sentences,
        train_labels,
        val_sentences,
        val_labels,
        label_map,
        batch_size=16,
        max_length=128,
        num_workers=2
    ):
        """
        Args:
            tokenizer (PreTrainedTokenizer): Hugging Face tokenizer.
            train_sentences (List[List[str]]): Training tokenized sentences.
            train_labels (List[List[str]]): Training label sequences.
            val_sentences (List[List[str]]): Validation tokenized sentences.
            val_labels (List[List[str]]): Validation label sequences.
            label_map (Dict[str, int]): Mapping of label strings to integer IDs.
            batch_size (int): Batch size for data loading.
            max_length (int): Max sequence length for inputs.
            num_workers (int): Number of subprocesses for data loading.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.train_sentences = train_sentences
        self.train_labels = train_labels
        self.val_sentences = val_sentences
        self.val_labels = val_labels
        self.label_map = label_map
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers

    def setup(self, stage=None):
        """
        Initializes datasets for training and validation.

        Called automatically by PyTorch Lightning before training or evaluating.
        """
        self.train_dataset = NERDataset(
            tokenizer=self.tokenizer,
            sentences=self.train_sentences,
            labels=self.train_labels,
            label_map=self.label_map,
            max_length=self.max_length
        )

        self.val_dataset = NERDataset(
            tokenizer=self.tokenizer,
            sentences=self.val_sentences,
            labels=self.val_labels,
            label_map=self.label_map,
            max_length=self.max_length
        )

    def train_dataloader(self):
        """
        Returns a DataLoader for the training dataset.
        Shuffles data for robustness during training.
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        """
        Returns a DataLoader for the validation dataset.
        No shuffling is applied to keep evaluation deterministic.
        """
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
