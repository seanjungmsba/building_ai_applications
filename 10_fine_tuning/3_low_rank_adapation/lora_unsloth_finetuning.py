"""
LoRA Fine-Tuning Script Using Unsloth on LLaMA 3.1

This script demonstrates how to fine-tune Meta's LLaMA 3.1 model using
Low-Rank Adaptation (LoRA) via the Unsloth framework. It emphasizes
parameter-efficient training by injecting trainable low-rank matrices into
frozen model layers, significantly reducing resource consumption while
maintaining performance.
"""

import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel

# Step 1: Prepare a sample instruction-response dataset
# -----------------------------------------------------
# This dataset serves as input for instruction fine-tuning.
instruction_dataset = [
    {
        "instruction": "Explain the difference between an atom and a molecule.",
        "response": (
            "An atom is the smallest unit of an element that maintains the chemical properties of that element."
            " A molecule is formed when two or more atoms bond together chemically."
            " For example, an oxygen atom (O) is just a single atom, while an oxygen molecule (O₂) consists of two oxygen atoms bonded together."
        )
    },
    # ➕ Add more diverse instruction-response pairs here for better generalization
]

# Step 2: Convert to Hugging Face Dataset
# ---------------------------------------
# Transform list of dictionaries into a DataFrame, then convert to Dataset format
print("Preparing dataset for training...")
df = pd.DataFrame(instruction_dataset)
dataset = Dataset.from_pandas(df)

# Step 3: Format the dataset into a unified prompt structure
# -----------------------------------------------------------
# Each example will follow the format:
# ### Instruction:
# <instruction>
# ### Response:
# <response>

def format_instruction(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    }

formatted_dataset = dataset.map(format_instruction)

# Step 4: Load LLaMA model with Unsloth
# --------------------------------------
# load_in_4bit=False ensures use of full-precision weights suitable for LoRA
print("Loading LLaMA 3.1 8B model using Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Meta-Llama-3.1-8B",
    max_seq_length=2048,
    dtype=torch.bfloat16,           # bfloat16 for efficiency and stability
    load_in_4bit=False              # Standard full precision mode for clean LoRA application
)

# Step 5: Apply LoRA to the model
# -------------------------------
# This wraps target modules with trainable LoRA adapters
model = FastLanguageModel.get_peft_model(
    model=model,
    r=16,                           # Low-rank dimension (controls expressive power)
    lora_alpha=16,                  # Scaling factor for LoRA updates
    lora_dropout=0.05,              # Dropout for regularization
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target attention projections
    modules_to_save=None            # Save all modified modules by default
)

# Step 6: Define the training configuration
# -----------------------------------------
# FastLanguageModel.get_trainer returns a Hugging Face-compatible trainer
print("Configuring trainer...")
trainer = FastLanguageModel.get_trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    output_dir="./llama-3.1-lora-finetuned"
)

# Step 7: Train the model
# ------------------------
print("Starting fine-tuning...")
trainer.train()

# Step 8: Save the trained model and tokenizer
# ---------------------------------------------
print("Saving fine-tuned model and tokenizer...")
model.save_pretrained("./llama-3.1-lora-finetuned")
tokenizer.save_pretrained("./llama-3.1-lora-finetuned")

# Step 9: Run an inference test on the fine-tuned model
# ------------------------------------------------------
# This verifies that the model responds appropriately to instruction prompts
print("Testing model inference...")
test_prompt = "### Instruction:\nWhat is the difference between covalent and ionic bonds?\n\n### Response:"
inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
