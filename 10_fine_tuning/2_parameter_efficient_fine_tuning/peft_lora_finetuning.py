"""
PEFT Finetuning Script using LoRA on Meta LLaMA 3.1 8B

This script demonstrates parameter-efficient finetuning (PEFT) using LoRA
on Meta's LLaMA 3.1 8B model. PEFT allows fine-tuning by injecting small,
trainable adapters into a frozen base model, significantly reducing the
number of trainable parameters and required compute resources.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

# ----------------------------------------
# Step 1: Load the base model and tokenizer
# ----------------------------------------
model_name = "meta-llama/Meta-Llama-3.1-8B"
print(f"Loading model and tokenizer: {model_name}")

# Load the tokenizer that matches the LLaMA model architecture
# This tokenizer is used to tokenize text input into model-readable tensors
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the pre-trained LLaMA model with float16/bfloat16 precision
# and place it on available GPU(s) automatically
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# --------------------------------------------------
# Step 2: Define PEFT Configuration for LoRA adapters
# --------------------------------------------------
print("Defining PEFT LoRA configuration...")

# Create a LoRAConfig that describes how LoRA adapters should be applied
# - task_type: Causal Language Modeling (for models like LLaMA)
# - r: Rank of low-rank adaptation matrices
# - lora_alpha: Scaling factor for the adapted weights
# - lora_dropout: Dropout to regularize the adapter during training
# - target_modules: Parts of the transformer to inject LoRA (usually attention projections)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none"
)

# --------------------------------------------------
# Step 3: Apply the LoRA adapters to the base model
# --------------------------------------------------
print("Wrapping base model with PEFT adapters using LoRA...")

# Wrap the base model with PEFT's adapter logic
# Only LoRA-injected parameters will be marked trainable
peft_model = get_peft_model(model, peft_config)

# --------------------------------------------------
# Step 4: Display trainable parameters
# --------------------------------------------------
print("\nTrainable Parameters Summary:")

# Print out how many parameters will be updated during training
# Useful for debugging and confirmation that LoRA is active
peft_model.print_trainable_parameters()

# --------------------------------------------------
# Step 5: Define Hugging Face training arguments
# --------------------------------------------------
print("Setting up TrainingArguments...")

# Configure training hyperparameters and logging behavior
# This configuration works well for typical consumer-grade GPUs
training_args = TrainingArguments(
    output_dir="./peft-llama-finetuned",            # Where to save checkpoints
    per_device_train_batch_size=4,                  # Adjust based on your GPU memory
    gradient_accumulation_steps=4,                  # Accumulate gradients to simulate larger batch size
    learning_rate=2e-4,
    num_train_epochs=3,
    warmup_ratio=0.03,
    logging_steps=50,                               # Print logs every 50 steps
    save_strategy="epoch",                         # Save model once per epoch
    fp16=True                                       # Enable mixed precision training (optional for performance)
)

# --------------------------------------------------
# Step 6: Load your preprocessed dataset
# --------------------------------------------------
print("Loading your dataset...")

# Replace this placeholder with your actual preprocessed dataset object
# Dataset should be tokenized with keys: input_ids, attention_mask
# Example: HuggingFace `datasets.Dataset` object
# Example loading logic (commented):
# from datasets import load_from_disk
# your_dataset = load_from_disk("./path/to/your/dataset")

your_dataset = None  # TODO: Replace with actual dataset

# --------------------------------------------------
# Step 7: Initialize Trainer with LoRA model and dataset
# --------------------------------------------------
print("Initializing Hugging Face Trainer...")

# Define a basic data collator for Causal LM
# Ensures each batch is formatted correctly with input_ids, attention_mask, and labels
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=your_dataset,
    data_collator=lambda data: {
        'input_ids': torch.stack([f['input_ids'] for f in data]),
        'attention_mask': torch.stack([f['attention_mask'] for f in data]),
        'labels': torch.stack([f['input_ids'] for f in data])  # For causal LM, labels are the same as input_ids
    }
)

# --------------------------------------------------
# Step 8: Begin fine-tuning the model
# --------------------------------------------------
print("Starting training...")

# Uncomment the line below when dataset is ready
# trainer.train()

# --------------------------------------------------
# Step 9: Save trained LoRA adapters
# --------------------------------------------------
print("Saving fine-tuned LoRA adapters...")

# Saves ONLY the adapter weights (not the base model)
# These can be reloaded using `PeftModel.from_pretrained(...)`
# Useful for modular deployment or sharing fine-tuned skills
# Uncomment the line below when ready to save
# peft_model.save_pretrained("./peft-llama-finetuned")
