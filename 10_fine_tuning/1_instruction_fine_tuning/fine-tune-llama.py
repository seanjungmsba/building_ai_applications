"""
Instruction Finetuning Script for LLaMA 3.1 with Unsloth

This script demonstrates how to fine-tune Meta's LLaMA 3.1 8B model using instruction-response data.
It utilizes the Unsloth library for accelerated model loading and training, with parameter-efficient fine-tuning via LoRA.

Requirements:
- Access to meta-llama/Meta-Llama-3.1-8B via HuggingFace
- GPU with bfloat16 or 4-bit support
- Properly formatted dataset of instruction-response pairs

Author: Sean
"""

import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel

# Define a sample instruction dataset
instruction_dataset = [
    {
        "instruction": "Explain the difference between an atom and a molecule.",
        "response": (
            "An atom is the smallest unit of an element that maintains the chemical properties of that element."
            " A molecule is formed when two or more atoms bond together chemically. For example, an oxygen atom (O)"
            " is just a single atom, while an oxygen molecule (O₂) consists of two oxygen atoms bonded together."
        )
    },
    # Additional examples should be added here for better training performance
]

# Step 1: Convert dataset to HuggingFace-compatible format
print("Converting dataset to HuggingFace Dataset format...")
df = pd.DataFrame(instruction_dataset)
dataset = Dataset.from_pandas(df)

# Step 2: Format each instruction-response into a unified text string
def format_instruction(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    }

print("Formatting dataset for training...")
formatted_dataset = dataset.map(format_instruction)

# Step 3: Load LLaMA 3.1 model with Unsloth
print("Loading LLaMA 3.1 8B model using Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Meta-Llama-3.1-8B",
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

# Step 4: Enable Parameter-Efficient Fine-Tuning with LoRA
print("Applying LoRA configuration for PEFT...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    modules_to_save=None
)

# Step 5: Define training configuration
print("Configuring training parameters...")
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
    output_dir="./llama-3.1-8b-finetuned"
)

# Step 6: Begin fine-tuning
print("Starting model fine-tuning...")
trainer.train()

# Step 7: Save the trained model and tokenizer
print("Saving fine-tuned model to ./llama-3.1-8b-finetuned")
model.save_pretrained("./llama-3.1-8b-finetuned")
tokenizer.save_pretrained("./llama-3.1-8b-finetuned")

# Step 8: Inference test
print("Testing the fine-tuned model with a sample prompt...")
test_prompt = "### Instruction:\nWhat is the difference between covalent and ionic bonds?\n\n### Response:"
inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
