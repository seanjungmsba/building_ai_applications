"""
QLoRA Fine-Tuning Script Using Unsloth on LLaMA 3.1

This script demonstrates how to fine-tune Meta's LLaMA 3.1 model using
Quantized Low-Rank Adaptation (QLoRA) through the Unsloth framework.
QLoRA enables highly efficient training by combining 4-bit quantized model weights
with full-precision LoRA adapters, allowing even large models to be fine-tuned
on consumer-grade GPUs.
"""

import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel

# Step 1: Prepare a sample instruction-response dataset
# -----------------------------------------------------
instruction_dataset = [
    {
        "instruction": "Explain the difference between an atom and a molecule.",
        "response": (
            "An atom is the smallest unit of an element that retains its properties."
            " A molecule is formed when two or more atoms bond together chemically."
        )
    },
    # ➕ Add additional instruction-response examples for more robust training
]

# Step 2: Convert to Hugging Face Dataset format
# ----------------------------------------------
df = pd.DataFrame(instruction_dataset)
dataset = Dataset.from_pandas(df)

# Step 3: Format dataset into instruction-tuned prompts
# ------------------------------------------------------
def format_instruction(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    }

formatted_dataset = dataset.map(format_instruction)

# Step 4: Load quantized LLaMA model using Unsloth (QLoRA-enabled)
# ------------------------------------------------------------------
print("Loading LLaMA 3.1 model with 4-bit quantization via Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Meta-Llama-3.1-8B",
    max_seq_length=2048,
    dtype=torch.bfloat16,         # Full-precision adapters
    load_in_4bit=True,            # Enable 4-bit base model weights (NF4)
    use_bf16_immediately=True     # Ensure adapter side uses bfloat16
)

# Step 5: Inject LoRA adapters for QLoRA
# --------------------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                           # Rank of low-rank decomposition
    lora_alpha=16,                  # Scaling factor
    lora_dropout=0.05,              # Regularization
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# Step 6: Set up training arguments and Trainer
# ---------------------------------------------
print("Setting up training configuration...")
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
    output_dir="./llama-3.1-qlora-finetuned"
)

# Step 7: Start training loop
# ---------------------------
print("Beginning fine-tuning...")
trainer.train()

# Step 8: Save final model and tokenizer
# --------------------------------------
print("Saving adapter-trained model and tokenizer...")
model.save_pretrained("./llama-3.1-qlora-finetuned")
tokenizer.save_pretrained("./llama-3.1-qlora-finetuned")

# Step 9: Inference sanity check
# ------------------------------
print("Running inference on sample prompt...")
test_prompt = "### Instruction:\nWhat is the difference between covalent and ionic bonds?\n\n### Response:"
inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
