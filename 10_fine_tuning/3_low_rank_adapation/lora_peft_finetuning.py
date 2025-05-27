"""
LoRA Fine-Tuning Script Using HuggingFace PEFT on LLaMA 3.1

This script demonstrates how to fine-tune Meta's LLaMA 3.1 model using
Low-Rank Adaptation (LoRA) via HuggingFace's PEFT library.
It highlights a modular and production-friendly way to fine-tune large
language models efficiently with LoRA adapters.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd
from datasets import Dataset

# Step 1: Create a simple instruction-response dataset
# ----------------------------------------------------
instruction_dataset = [
    {
        "instruction": "Explain the difference between an atom and a molecule.",
        "response": (
            "An atom is the smallest unit of an element that retains its properties."
            " A molecule is composed of two or more atoms bonded together."
        )
    }
    # ➕ Add more training examples here for improved generalization
]

# Step 2: Convert to HuggingFace Dataset
# --------------------------------------
df = pd.DataFrame(instruction_dataset)
dataset = Dataset.from_pandas(df)

# Step 3: Format the dataset into text-style examples for instruction tuning
# --------------------------------------------------------------------------
def format_instruction(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    }

formatted_dataset = dataset.map(format_instruction)

# Tokenize the formatted dataset
# ------------------------------
# This prepares the text for model input and labels
model_name = "meta-llama/Meta-Llama-3.1-8B"
print(f"Loading tokenizer for: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize using padding and truncation for uniform batch processing
def tokenize(example):
    tokens = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=2048,
        return_tensors="pt"
    )
    return {
        "input_ids": tokens["input_ids"][0],
        "attention_mask": tokens["attention_mask"][0],
    }

print("Tokenizing dataset...")
tokenized_dataset = formatted_dataset.map(tokenize)

# Step 4: Load base model and apply LoRA
# --------------------------------------
print("Loading base LLaMA model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Configure LoRA adapter with PEFT
print("Configuring LoRA adapters...")
lora_config = LoraConfig(
    r=16,                                # Low-rank adaptation matrix rank
    lora_alpha=16,                       # Scaling factor
    lora_dropout=0.05,                   # Dropout for regularization
    bias="none",                         # Do not update any bias parameters
    task_type=TaskType.CAUSAL_LM,        # Language modeling objective
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Apply to attention projections
)

# Apply the adapter configuration to the base model
model = get_peft_model(model, lora_config)

# Print number of trainable parameters
print("Trainable parameters:")
model.print_trainable_parameters()

# Step 5: Define training arguments
# ----------------------------------
print("Setting up training configuration...")
training_args = TrainingArguments(
    output_dir="./lora-llama",
    learning_rate=2e-4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    warmup_ratio=0.03,
    logging_steps=50,
    save_strategy="epoch",
    bf16=True                           # Enable bfloat16 for efficiency
)

# Step 6: Create Trainer
# ----------------------
print("Initializing Trainer with tokenized dataset...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=lambda data: {
        'input_ids': torch.stack([f['input_ids'] for f in data]),
        'attention_mask': torch.stack([f['attention_mask'] for f in data]),
        'labels': torch.stack([f['input_ids'] for f in data])  # Labels match input for causal LM
    }
)

# Step 7: Train and Save
# ----------------------
print("Starting training...")
trainer.train()

# Save the adapter weights only (modular deployment)
print("Saving trained LoRA adapters...")
model.save_pretrained("./lora-llama-adapter")
tokenizer.save_pretrained("./lora-llama-adapter")
