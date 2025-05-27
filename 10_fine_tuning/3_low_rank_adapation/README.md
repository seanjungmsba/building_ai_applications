# 📘 Low-Rank Adaptation (LoRA) for LLM Fine-Tuning

Low-Rank Adaptation (LoRA) is a breakthrough technique in the field of parameter-efficient fine-tuning (PEFT), tailored specifically for adapting large language models (LLMs) with minimal computational and memory overhead. Rather than retraining or fine-tuning the entirety of a pre-trained model's parameters, LoRA allows for elegant and modular adaptation by injecting lightweight, trainable low-rank matrices into targeted components of the model — all while keeping the original parameters frozen.

---

## 🔬 What is LoRA?

LoRA, short for **Low-Rank Adaptation**, is designed to make fine-tuning large models — like Meta's LLaMA-3.1 series — feasible even on modest hardware. It achieves this by expressing weight updates through low-rank matrix decompositions, thereby reducing the number of trainable parameters by orders of magnitude.

### ✅ Core Concept:

> Given a frozen weight matrix `W ∈ ℝ^{d × k}`, LoRA proposes learning a low-rank update `ΔW = BA` such that:
>
> * `B ∈ ℝ^{d × r}` (trainable matrix)
> * `A ∈ ℝ^{r × k}` (trainable matrix)
> * `r` (the rank) is much smaller than `d` or `k`

The effective weight during inference becomes:

```
    W_eff = W + BA
```

Only `A` and `B` are optimized during training. The original weight matrix `W` remains untouched — making the update process lightweight, reversible, and modular.

---

## ⚙️ How LoRA Works Under the Hood

- **Forward Pass**:
  - Let `W` be a fixed (frozen) weight from the base model.
  - During training, LoRA computes: `y = Wx + BAx` instead of updating `W` directly.

- **Low-Rank Matrix (r)**:
  - Governs the expressivity of the parameter update.
  - Typical values range between `4` and `128`, depending on task complexity.

- **Alpha Scaling Factor (α)**:
  - LoRA usually scales the update `BA` by `α / r` to regulate its influence.
  - This helps stabilize training and gradient magnitudes.

---

## 🎯 Why Use LoRA? Benefits at a Glance

| Feature               | Benefit                                                                 |
|----------------------|-------------------------------------------------------------------------|
| 🧠 Memory Efficient   | Updates <1% of model parameters; avoids full backward pass through `W` |
| 🚀 Training Speed     | Smaller parameter set = faster updates, fewer compute cycles           |
| 💾 Compact Storage    | Save adapter modules as small (~20–100MB) checkpoints                  |
| 🔄 Swappable Modules  | Easily inject task-specific adapters into the same base model          |
| 📈 Comparable Quality | Retains or even exceeds performance of full fine-tuning                |

---

## 🔍 Where to Inject LoRA: Target Modules

To be effective, LoRA should be applied to highly influential components in transformer-based architectures:

- **Attention Projections**:
  - `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **Feedforward Layers (MLP)**:
  - `gate_proj`, `up_proj`, `down_proj`
- **Other Modules**:
  - `embed_tokens`, `attention_blocks`

> 🧠 Tip: Start with attention heads — then expand to MLPs if needed for additional expressivity.

---

## 🧪 LoRA in Action: Implementation Options

### ✳️ Option 1: Using [Unsloth](https://github.com/unslothai/unsloth)

Unsloth offers an optimized and intuitive interface for fast LLaMA fine-tuning.

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Meta-Llama-3.1-8B",
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=False,  # Use full precision
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
```

➡️ See the full script in [`lora_unsloth_finetuning.py`](./lora_unsloth_finetuning.py)

---

### ✳️ Option 2: Using HuggingFace PEFT Library

For maximum compatibility with the HuggingFace ecosystem:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
model = get_peft_model(model, lora_config)
```

➡️ See the full script in [`lora_peft_finetuning.py`](./lora_peft_finetuning.py)

---

## 🧠 Best Practices and Optimization Strategies

### 🔧 Hyperparameter Suggestions

- Start simple: `r=16`, `alpha=16`, `lora_dropout=0.05`
- For larger tasks or multilingual data, increase `r`
- Monitor model loss — adjust upward if underfitting

### 🎯 Module Targeting Heuristics

- Prioritize attention projections (`q_proj`, `v_proj`) for general improvements
- Expand to `up_proj`/`down_proj` in FFN if convergence slows

### 🚀 Scaling Guidance

- LoRA is most effective for models ≥ 7B parameters
- For <1B models, the advantage of LoRA over full fine-tuning is less significant

### 🔄 LoRA vs. QLoRA: When to Choose What

| Aspect       | LoRA                  | QLoRA                        |
|--------------|-----------------------|------------------------------|
| Precision    | bfloat16 / float16    | int4 (quantized) + bfloat16 |
| Speed        | Fast                  | Faster, lower VRAM usage     |
| Model Quality| Slight edge in quality| Nearly equivalent            |
| Ideal Usage  | When memory is ample  | When memory is constrained   |

---

## 📦 Installation Requirements

Install all necessary dependencies via pip:

```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
.
├── lora_unsloth_finetuning.py     # LoRA fine-tuning via Unsloth
├── lora_peft_finetuning.py        # LoRA fine-tuning via HuggingFace PEFT
├── llama-3.1-lora-finetuned/      # Directory to store trained adapter weights
└── README.md                      # This documentation file
```

---

## 📚 Further Reading and References

- [🔗 LoRA Paper (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [📘 HuggingFace PEFT Docs](https://github.com/huggingface/peft)
- [⚡ Unsloth Framework](https://unsloth.ai/)
- [🦙 Meta LLaMA 3.1 Model Card](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)
