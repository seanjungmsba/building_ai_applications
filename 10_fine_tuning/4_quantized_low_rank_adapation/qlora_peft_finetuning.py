"""
QLoRA Fine-Tuning Script Using HuggingFace PEFT and BitsAndBytes

This script demonstrates fine-tuning Meta's LLaMA 3.1 model using Quantized Low-Rank Adaptation (QLoRA).
We leverage HuggingFace Transformers, PEFT (parameter-efficient fine-tuning), and BitsAndBytes for
NF4 quantization. This allows training massive models (e.g., 13B or 65B) with extremely low memory overhead.
"""

import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

# Step 1: Create an instruction-tuned dataset
instruction_dataset = [
    {
        "instruction": "Explain the difference between an atom and a molecule.",
        "response": "An atom is the smallest indivisible unit of matter. A molecule consists of two or more atoms bonded together."
    },
    # ➕ Add more examples for generalization
]

# Step 2: Convert to HF Dataset format
df = pd.DataFrame(instruction_dataset)
dataset = Dataset.from_pandas(df)

# Step 3: Format into prompt-response style
def format_instruction(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    }

formatted_dataset = dataset.map(format_instruction)

# Step 4: Load 4-bit quantized model and tokenizer
model_name = "meta-llama/Meta-Llama-3.1-8B"
print(f"Loading 4-bit quantized base model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",             # NormalFloat 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16, # Adapter compute precision
    bnb_4bit_use_double_quant=True,        # Enable double quantization
    device_map="auto"
)

# Step 5: Prepare model for k-bit fine-tuning
print("Preparing model for k-bit LoRA fine-tuning...")
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True  # Saves memory by re-computing activations
)

# Step 6: Define LoRA adapter configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# Step 7: Inject LoRA into quantized model
print("Applying LoRA configuration to quantized model...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Step 8: Tokenize dataset for training
def tokenize(example):
    output = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=2048,
        return_tensors="pt"
    )
    return {
        "input_ids": output["input_ids"][0],
        "attention_mask": output["attention_mask"][0]
    }

tokenized_dataset = formatted_dataset.map(tokenize)

# Step 9: Define training arguments
training_args = TrainingArguments(
    output_dir="./qlora-llama-finetuned",
    learning_rate=2e-4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    warmup_ratio=0.03,
    logging_steps=50,
    save_strategy="epoch",
    bf16=True  # Enable bfloat16 for faster training
)

# Step 10: Initialize Trainer
print("Starting HuggingFace Trainer with tokenized dataset...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=lambda data: {
        "input_ids": torch.stack([f["input_ids"] for f in data]),
        "attention_mask": torch.stack([f["attention_mask"] for f in data]),
        "labels": torch.stack([f["input_ids"] for f in data])
    }
)

# Step 11: Begin training
trainer.train()

# Step 12: Save final adapter model
print("Saving trained QLoRA adapter...")
model.save_pretrained("./qlora-llama-finetuned")
tokenizer.save_pretrained("./qlora-llama-finetuned")
