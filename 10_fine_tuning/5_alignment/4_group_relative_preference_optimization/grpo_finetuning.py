"""
grpo_finetuning.py

This script demonstrates how to fine-tune a Meta LLaMA model using Group Relative Preference Optimization (GRPO).
GRPO is a technique for aligning large language models with human preferences by learning from grouped rankings
rather than binary choices. This script uses LoRA for parameter-efficient training and the TRL library for GRPO.
"""

import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig


def create_grouped_dataset():
    """
    Create a toy dataset for GRPO training, where each prompt has multiple responses ranked by preference.
    The dataset is transformed into pairwise comparisons for GRPOTrainer to consume.

    Returns:
        Dataset: HuggingFace Dataset object containing prompt, chosen, rejected responses and their ranks.
    """
    prompts = [
        {
            "prompt": "What is reinforcement learning?",
            "responses": [
                "Reinforcement learning is a type of ML where agents learn by reward.",
                "I don't know.",
                "Reinforcement learning involves trial-and-error and policy optimization."
            ],
            "rankings": [0, 2, 1]  # Lower is better (0 = best)
        }
    ]

    # Flatten ranked responses into all valid (chosen, rejected) preference pairs
    pairs = []
    for item in prompts:
        p = item["prompt"]
        responses = item["responses"]
        ranks = item["rankings"]

        # For every pair of responses, include if one is preferred over the other
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                if ranks[i] < ranks[j]:  # response[i] is preferred over response[j]
                    pairs.append({
                        "prompt": p,
                        "chosen": responses[i],
                        "rejected": responses[j],
                        "chosen_rank": ranks[i],
                        "rejected_rank": ranks[j]
                    })

    # Convert to HuggingFace Dataset
    return Dataset.from_pandas(pd.DataFrame(pairs))


def train_grpo_model():
    """
    Loads the tokenizer and model, applies LoRA adapters, initializes the GRPO trainer,
    and runs the training process. The final model is saved to disk.
    """
    model_name = "meta-llama/Meta-Llama-3.1-8B"

    # Load tokenizer and ensure it has a pad token
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set EOS token as padding token if pad_token is None

    # Load base model in 4-bit precision (saves memory and enables efficient fine-tuning)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True
    )

    # Prepare model for 4-bit training (adds normalization and gradient checkpointing)
    model = prepare_model_for_kbit_training(model)

    # Define LoRA configuration for injecting trainable adapters into attention layers
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none"
    )

    # Apply LoRA adapters to the model
    model = get_peft_model(model, lora_config)

    # Load a reference model (used in KL-divergence regularization during GRPO loss computation)
    reference_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True
    )

    # Create dataset with grouped preference information
    dataset = create_grouped_dataset()

    # Split into training and evaluation subsets
    dataset_split = dataset.train_test_split(test_size=0.2)
    train_ds = dataset_split["train"]
    eval_ds = dataset_split["test"]

    # Define HuggingFace TrainingArguments for optimizer, batch sizes, logging, etc.
    training_args = TrainingArguments(
        output_dir="./grpo-aligned-llama",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        num_train_epochs=3,
        bf16=True,  # Use bfloat16 for faster training with reduced memory
        logging_steps=5,
        save_steps=50,
        eval_steps=50,
        evaluation_strategy="steps"
    )

    # GRPO-specific configuration for preference loss
    grpo_config = GRPOConfig(
        beta=0.1,                # KL penalty coefficient
        delta=0.05,              # Margin for log-odds comparison
        rank_weight_method="linear"  # Method for weighting preference strength by rank difference
    )

    # Create the GRPOTrainer with all configuration and datasets
    trainer = GRPOTrainer(
        model=model,
        ref_model=reference_model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
        grpo_config=grpo_config,
        max_length=512,           # Max total sequence length
        max_prompt_length=256     # Max length for the prompt section only
    )

    # Run the training loop
    trainer.train()

    # Save the fine-tuned model and tokenizer to disk
    model.save_pretrained("./grpo-aligned-llama")
    tokenizer.save_pretrained("./grpo-aligned-llama")


# Main script entry point
if __name__ == "__main__":
    train_grpo_model()
