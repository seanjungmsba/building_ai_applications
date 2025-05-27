# ⚙️ Parameter Efficient Fine-Tuning (PEFT) with LLaMA 3.1 + LoRA

## Overview

As large language models (LLMs) grow into billions of parameters, the cost of fine-tuning them becomes prohibitive. Enter **Parameter Efficient Fine-Tuning (PEFT)** — a powerful set of techniques that dramatically reduce the number of trainable parameters and hardware requirements while achieving near-parity with full-model fine-tuning.

This project demonstrates how to implement PEFT with **LoRA (Low-Rank Adaptation)** on **Meta's LLaMA 3.1-8B** model using the Hugging Face `transformers` and `peft` libraries.

---

## 📘 What is PEFT?

Traditional fine-tuning updates all weights in a pre-trained LLM. This is memory-intensive, computationally expensive, and often impractical for individuals or smaller teams.

**PEFT methods**, on the other hand, **freeze most of the original model** and inject a small number of **trainable parameters** (adapters, prompts, or matrices). These components learn to "nudge" the base model's behavior while maintaining its general knowledge.

---

## 💡 Why Use PEFT?

| Benefit                                  | Explanation                                                                     |
| ---------------------------------------- | ------------------------------------------------------------------------------- |
| 🧠 **Preserve Pretrained Knowledge**     | The original model weights remain untouched, preventing catastrophic forgetting |
| ⚡ **Drastically Reduced Training Costs** | PEFT adapters represent <3% of the full model’s parameters                      |
| 💾 **Smaller Storage Footprint**         | Only the adapter weights (\~50MB) need to be saved and deployed                 |
| 🔄 **Flexible Model Switching**          | Load one base model, swap in multiple adapters on demand                        |
| 🛠️ **Modular Architecture**             | Fine-tune adapters separately for different tasks, domains, or users            |

---

## 🔬 Common PEFT Methods

| Method            | Mechanism                                                          | Use Case                                       |
| ----------------- | ------------------------------------------------------------------ | ---------------------------------------------- |
| **LoRA**          | Injects low-rank trainable matrices into key attention projections | General-purpose fine-tuning                    |
| **QLoRA**         | Combines LoRA with 4-bit quantization for extreme memory savings   | Fine-tuning very large models on consumer GPUs |
| **Prefix Tuning** | Adds continuous trainable vectors to the input context             | Control or guide generation tasks              |
| **Prompt Tuning** | Learns soft prompt embeddings                                      | Lightweight task adaptation                    |
| **Adapters**      | Inserts small trainable networks between model layers              | Modular extensions to model behavior           |

---

## ⚙️ How PEFT Works

### Traditional Fine-Tuning

* All weights in the model are updated.
* Requires hundreds of GBs of memory and storage.
* Often results in **catastrophic forgetting** of pretraining knowledge.

### PEFT Approach

* Base model weights are **frozen**.
* Tiny modules (e.g. LoRA adapters) are inserted into attention or MLP layers.
* Only those adapters are trained, requiring far less compute.
* The base model can be reused across tasks with different adapters.

---

## 🧑‍💻 Implementation

We use **LoRA** with Hugging Face’s PEFT library on Meta's LLaMA 3.1 8B model. This is demonstrated in the Python script: [`peft_lora_finetuning.py`](#canvas:peft_lora_finetuning).

### 🧱 Requirements

Ensure the following packages are installed:

```bash
pip install -r requirements.txt
```

Also ensure you have access to:

* A **GPU-enabled machine** (preferably with bfloat16/4-bit support)
* HuggingFace access to the [Meta-LLaMA 3.1](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B) checkpoint

---

### 🧾 Code Walkthrough

The core steps in the script are:

1. **Load base model** using `AutoModelForCausalLM` and `AutoTokenizer`
2. **Define LoRA configuration** using `LoraConfig`
3. **Inject adapters** using `get_peft_model(...)`
4. **Configure training** via Hugging Face `TrainingArguments`
5. **Run training** using the `Trainer` API
6. **Save adapters** using `peft_model.save_pretrained(...)`

> 🔧 Note: You must replace the placeholder `your_dataset` with a tokenized `Dataset` object formatted for causal language modeling.

---

### Example: LoRA Configuration

```python
LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none"
)
```

* `r`: Rank of low-rank matrices (tradeoff between accuracy and memory)
* `lora_alpha`: Scales the update; acts as a learning rate modifier
* `target_modules`: Apply adapters to key attention projections

---

## 🧪 Training a Model

1. Prepare your dataset using Hugging Face’s `datasets.Dataset` format
2. Tokenize with the appropriate `AutoTokenizer`
3. Format each sample with `input_ids`, `attention_mask`, and `labels`
4. Pass it into `Trainer` as shown in the script

Training with LoRA often requires **1–10 hours**, even on consumer GPUs.

---

## 🧠 Deployment Tips

* **Store adapters separately** from the base model.
* You only need to load the base LLaMA model once and **swap adapters** dynamically.
* Keep a registry of adapters per domain or application.
* For inference, use:

```python
from peft import PeftModel
model = AutoModelForCausalLM.from_pretrained(...)
model = PeftModel.from_pretrained(model, "path_to_adapter")
```

---

## 📊 Example Use Cases

| Domain     | Use Case                             | Adapter Reuse     |
| ---------- | ------------------------------------ | ----------------- |
| Healthcare | Clinical question answering          | ✅ Per specialty   |
| Legal      | Case law summarization               | ✅ By jurisdiction |
| Finance    | ESG report generation                | ✅ By firm type    |
| Retail     | Product Q\&A + Inventory Chatbot     | ✅ By catalog type |
| Education  | Subject-specific tutoring assistants | ✅ By course       |

---

## 📁 Project Structure

```
.
├── peft_lora_finetuning.py         # PEFT fine-tuning script using LoRA
├── peft-llama-finetuned/           # Saved adapter weights
├── requirements.txt                # Dependencies
└── README.md                       # This documentation
```

---

## 🎯 Choosing the Right PEFT Technique

| Method          | Best For                                    | Memory Usage | Adapter Size |
| --------------- | ------------------------------------------- | ------------ | ------------ |
| LoRA            | General fine-tuning                         | Moderate     | \~40MB       |
| QLoRA           | Memory-constrained fine-tuning              | Very Low     | \~40MB       |
| Prefix Tuning   | High-control generation tasks               | Low          | \~20MB       |
| Prompt Tuning   | Classification / Low-data scenarios         | Very Low     | \~1–10MB     |
| Adapter Modules | Multi-tasking, architecture-level extension | Moderate     | \~100MB      |

---

## 📚 Further Reading

* [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
* [PEFT GitHub Repository](https://github.com/huggingface/peft)
* [Meta LLaMA 3.1 Access](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)
