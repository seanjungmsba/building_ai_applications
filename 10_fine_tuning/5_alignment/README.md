# 🧭 Alignment in Large Language Models (LLMs)

Alignment is the discipline of ensuring that language models not only generate coherent responses but do so in a way that is consistent with human values, ethical standards, and user intent. This is crucial for building safe, helpful, and responsible AI systems.

---

## 🧠 Why Alignment Matters

Raw pre-trained models are trained to imitate text distributions, not necessarily to be helpful or truthful. Without proper alignment, they may:

- Produce harmful, unethical, or biased content
- Hallucinate facts or mislead users
- Misinterpret instructions or subtly disobey user intent
- Fail to account for social and cultural context
- Act in unpredictable or unsafe ways when prompted in edge cases

Alignment serves as the final bridge between model capabilities and real-world deployment.

---

## 🧪 Core Alignment Techniques

Below are the most common strategies used to align LLMs with human expectations:

### ⚙️ Reinforcement Learning from Human Feedback (RLHF)

- **Overview**: RLHF is a multi-stage pipeline where models are fine-tuned using reinforcement learning (typically PPO) based on a reward model trained from human preferences.
- **Workflow**:
  1. Collect human feedback on pairs of model outputs
  2. Train a **reward model** to score outputs
  3. Fine-tune the base model via PPO to maximize this score
- **Pros**:
  - High expressivity and nuance
  - Fine-grained control
- **Cons**:
  - High infrastructure cost
  - Sensitive to reward hacking

### 🎯 Direct Preference Optimization (DPO)

- **Overview**: DPO simplifies the RLHF pipeline by turning preference learning into a classification problem between preferred and rejected responses.
- **Mathematical Foundation**:
  - Minimizes a binary classification loss between preferred and dispreferred outputs
  - Inspired by the KL-regularized RL objective
- **Pros**:
  - Simple to implement
  - Stable and sample efficient
- **Cons**:
  - Lacks long-term credit assignment

### ⚖️ Odds Ratio Preference Optimization (ORPO)

- **Overview**: ORPO refines DPO by adjusting the loss function to account for **odds ratios** between preferences.
- **Key Insight**:
  - Replaces binary indicators with calibrated preference probabilities
- **Benefits**:
  - More robust to noisy or subtle preferences
  - Smooths gradients, leading to more stable convergence
- **Ideal Use Case**:
  - Instruction-tuned models where user preferences are nuanced but not binary

### 🧮 Group Relative Preference Optimization (GRPO)

- **Overview**: GRPO expands preference learning from pairwise (DPO/ORPO) to **groupwise** comparisons.
- **How It Works**:
  - Compares multiple responses (n-way ranking)
  - Optimizes the model to favor higher-ranked outputs within a group
- **Strengths**:
  - More expressive signal per training step
  - Captures finer semantic distinctions across multiple candidates
- **Challenges**:
  - Requires more complex datasets with group annotations
  - Computationally more intensive

---

## 📊 Comparison Table

| Technique | Description | Complexity | Strengths | Drawbacks |
|----------|-------------|------------|-----------|-----------|
| **RLHF** | Reinforcement via reward model and PPO | High | Maximum flexibility and alignment capacity | Hard to tune, expensive to run |
| **DPO** | Binary preference classification | Medium | Simple, fast, stable | Less nuanced than RLHF |
| **ORPO** | Weighted preference via odds ratios | Medium | Smooth training, robust to noise | Requires probabilistic labels |
| **GRPO** | Ranking over groups of responses | Medium-High | Better semantic resolution | Needs groupwise labeled data |

---

## 📦 Building Alignment Datasets

- **Dataset Types**:
  - **Binary**: Pairs of outputs (chosen vs rejected)
  - **Groupwise**: Ranked sets of outputs for a given prompt

- **Best Practices**:
  - Ensure diversity of prompt types (factual, creative, moral)
  - Document clear labeling guidelines for raters
  - Include edge cases and counterfactuals
  - Sample across cultures, demographics, and edge intents

```json
[
  {
    "prompt": "Explain the difference between a star and a planet.",
    "chosen": "A star emits light from nuclear fusion; a planet reflects light from its star.",
    "rejected": "Stars and planets are both round."
  }
]
```

---

## 🔧 Python Workflow

```bash
pip install -r requirements.txt
```

**Steps:**
1. Load pre-trained model
2. Format preference dataset (JSON, Hugging Face Dataset, etc.)
3. Choose alignment algorithm (DPO, ORPO, etc.)
4. Fine-tune with `trl` or custom training loop
5. Evaluate on safety and instruction-following metrics

---

## 🧠 Choosing the Right Technique

| Use Case | Best Technique |
|----------|----------------|
| Maximum alignment control, large budget | RLHF |
| Fast training with pairwise preferences | DPO |
| Stable training with noisy feedback | ORPO |
| Multi-output comparison (e.g., 4-shot) | GRPO |

For implementation guides:
- [`RLHF.md`](./RLHF.md)
- [`DPO.md`](./DPO.md)
- [`ORPO.md`](./ORPO.md)
- [`GRPO.md`](./GRPO.md)

---

## 📚 References

- OpenAI InstructGPT (Ouyang et al., 2022)
- Anthropic Constitutional AI (2023)
- Direct Preference Optimization (Rafailov et al., 2023)
- ORPO (Zhou et al., 2024)
- GRPO (Open-Rank LLM paper, 2024)
- HuggingFace TRL library
- [RLHF Papers List](https://github.com/AlignmentResearch/awesome-rlhf)
