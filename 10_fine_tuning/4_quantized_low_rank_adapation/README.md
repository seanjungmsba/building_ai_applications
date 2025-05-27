# 🧠 Quantized Low-Rank Adaptation (QLoRA) for Efficient LLM Fine-Tuning

**QLoRA** represents a transformative leap in large language model (LLM) fine-tuning. By combining **4-bit quantization** with **Low-Rank Adaptation (LoRA)**, QLoRA enables researchers and practitioners to train powerful models with **minimal hardware**, **maximum efficiency**, and **near-state-of-the-art performance**.

Whether you're working on a laptop with limited GPU memory or running high-throughput experiments in the cloud, QLoRA brings the power of frontier LLMs within reach.

---

## 🔍 What is QLoRA?

**QLoRA (Quantized Low-Rank Adaptation)** is a technique introduced to make large-scale language model fine-tuning accessible on modest hardware by fusing:

* **Quantization**: Reducing the memory footprint of the model by representing weights in 4-bit precision
* **LoRA Adapters**: Training small, task-specific low-rank matrices on top of the frozen, quantized model

Instead of updating billions of parameters, you update a few million **LoRA adapter weights** — all while the core model is compressed and frozen.

---

## 🔬 How QLoRA Works — Under the Hood

### 🔧 Architectural Breakdown

| Component               | Role                                                                  |
| ----------------------- | --------------------------------------------------------------------- |
| **NF4 Quantization**    | Converts model weights to NormalFloat 4-bit values for memory savings |
| **Double Quantization** | Compresses quantization constants for further reduction (\~0.4–3%)    |
| **Paged Optimizers**    | Offload optimizer states to CPU or disk to free GPU memory            |
| **LoRA Adapters**       | Inserted into attention modules and trained in full precision         |

### 🧠 Forward Pass

Mathematically, QLoRA builds on LoRA's core idea:

```text
x → Quantized W (frozen) + BA (LoRA adapters) → y
```

Where:

* `W` is the frozen quantized weight matrix (4-bit)
* `B` and `A` are low-rank trainable matrices
* Only `BA` is updated during fine-tuning

---

## 🎯 Why Use QLoRA? Key Advantages

| Capability                       | Impact                                                              |
| -------------------------------- | ------------------------------------------------------------------- |
| 🧠 **Radical Memory Savings**    | Train 65B+ models on <16GB GPUs                                     |
| 📉 **Low Compute Requirements**  | Enables training on consumer laptops and affordable cloud instances |
| 📦 **Compact Storage Footprint** | Just \~1–2GB of adapter weights, not hundreds of GBs                |
| 🔁 **Adapter Modularity**        | Swap in task-specific adapters without touching the base model      |
| 🎯 **Near-FP32 Performance**     | Only \~0.2–0.5% degradation compared to full fine-tuning            |

QLoRA makes the dream of personal LLM fine-tuning a reality.

---

## 🧮 Technical Insights

### 🧬 NormalFloat (NF4)

* 4-bit floating-point format inspired by weight distribution patterns in LLMs
* Offers higher granularity near zero (where most weights lie)
* Outperforms uniform INT4 quantization in empirical tests

### 🔁 Double Quantization

* Compresses the quantization lookup table (scaling constants)
* Saves 0.4%–3% additional memory without performance loss

### 🧠 Paged Optimizers & Attention

* Paged optimizers offload gradients and optimizer states to CPU RAM
* Paged attention prevents memory blow-up from large sequence activations

Together, these components unlock **high performance at minimal cost**.

---

## ⚙️ Implementation Paths

### ✅ Option 1: Using [Unsloth](https://github.com/unslothai/unsloth) (Fastest & Simplest)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Meta-Llama-3.1-8B",
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=True,
    use_bf16_immediately=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
```

➡️ Full pipeline: [`qlora_unsloth_finetuning.py`](./qlora_unsloth_finetuning.py)

---

### 🧰 Option 2: HuggingFace PEFT + BitsAndBytes (More Configurable)

```python
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B",
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    device_map="auto",
)

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
model = get_peft_model(model, lora_config)
```

➡️ Full pipeline: [`qlora_peft_finetuning.py`](./qlora_peft_finetuning.py)

---

## 🧠 QLoRA vs. LoRA: Which One to Choose?

| Feature              | LoRA (Full Precision) | QLoRA (Quantized)                  |
| -------------------- | --------------------- | ---------------------------------- |
| Precision            | float16 / bfloat16    | NF4 (4-bit) + float16 adapters     |
| GPU Memory Usage     | Medium–High           | Extremely Low                      |
| Training Speed       | Slightly faster       | Slightly slower (due to quant ops) |
| Accuracy             | Very High             | \~98% of full fine-tuning quality  |
| Deployment Footprint | 20–200GB              | 2–8GB                              |
| Best Use Case        | Ample VRAM (24GB+)    | 8–16GB VRAM scenarios              |

---

## 🧩 Best Practices for Training with QLoRA

### ✅ Configuration

* Use `nf4` with `double_quant=True`
* Use `bfloat16` for compute (or fallback to `float16` on older GPUs)
* Set `r=16`, `alpha=16`, `dropout=0.05`

### 🚀 Optimizing Training

* Use **gradient checkpointing** to save VRAM
* Leverage **gradient accumulation** to simulate large batches
* Offload optimizer state if training on <16GB GPU

### 📊 Monitoring

* Use `nvidia-smi` to profile usage
* Monitor loss curve — QLoRA training should behave like LoRA
* Periodically evaluate on held-out samples to verify fidelity

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
.
├── qlora_unsloth_finetuning.py     # QLoRA with Unsloth (simplified, fast)
├── qlora_peft_finetuning.py        # QLoRA with HF PEFT + BitsAndBytes
├── llama-3.1-qlora-finetuned/      # Output directory for adapter weights
├── requirements.txt                # Dependencies
└── README.md                       # This documentation
```

---

## 📚 Further Reading

* [📄 QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
* [🤗 PEFT GitHub](https://github.com/huggingface/peft)
* [🔧 BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
* [🌀 Unsloth](https://github.com/unslothai/unsloth)
* [🦙 Meta LLaMA 3.1 Card](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)
