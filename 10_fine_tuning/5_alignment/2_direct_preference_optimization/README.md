## Direct Preference Optimization (DPO)

### Description

Direct Preference Optimization (DPO) is a streamlined approach to aligning large language models with human preferences. Unlike Reinforcement Learning from Human Feedback (RLHF), DPO avoids the need for a separate reward model and bypasses reinforcement learning altogether. Instead, it directly optimizes a policy model using preference pair data (i.e., chosen vs. rejected responses), effectively reframing the RLHF setup as a classification problem.

---

### How DPO Works

#### Theoretical Foundation

DPO uses the **Bradley-Terry model** to formalize the probability of a human preferring one response over another. Given two completions, the probability of preferring the "chosen" response is:

```
P(chosen ≫ rejected) = σ(r(x, chosen) - r(x, rejected))
```

Where:

* `r(x, y)` is the reward for completion `y` given prompt `x`
* `σ` is the sigmoid function

DPO approximates `r(x, y)` with log-probabilities from the policy model, regularized by a KL divergence term against a reference model. The objective is optimized using binary cross-entropy loss.

#### Key Steps

1. Start with a base model (often an instruction-tuned LLM).
2. Collect preference pairs for given prompts.
3. Format prompts with both chosen and rejected completions.
4. Compute loss directly from log-likelihood difference.
5. Fine-tune the model using standard supervised learning techniques.

---

### Advantages of DPO

| Advantage               | Description                                                      |
| ----------------------- | ---------------------------------------------------------------- |
| **Simplified Training** | Eliminates reward model and PPO training                         |
| **Stable Optimization** | Avoids PPO instability by relying on standard supervised loss    |
| **Resource Efficient**  | Requires fewer GPUs and compute resources than full RLHF         |
| **Easy to Implement**   | Clean and reproducible with a single training loop               |
| **Performance**         | Can match or even exceed RLHF under certain alignment objectives |

---

### Limitations and Considerations

* **Binary Preferences Only**: Cannot easily express degrees of preference.
* **Ambiguity Sensitivity**: Can struggle with subtle distinctions.
* **Reference Model Dependence**: Effectiveness depends on the choice of the reference model.
* **Lower Ceiling on Nuance**: Less expressive than full RLHF in capturing human reward signals.

---

### DPO Compared to Other Approaches

| Comparison        | DPO                            | RLHF                                 | ORPO                            | GRPO                                |
| ----------------- | ------------------------------ | ------------------------------------ | ------------------------------- | ----------------------------------- |
| Reward Modeling   | No                             | Yes (explicit reward model)          | No (uses odds ratio)            | No (uses group preferences)         |
| Optimization      | Supervised loss                | PPO (reinforcement learning)         | Supervised loss + ratio weights | Supervised loss over group rankings |
| Preference Format | Pairwise (chosen vs. rejected) | Pairwise (for reward model training) | Pairwise                        | Group-based                         |
| Use Case          | Fast, stable alignment         | Complex, nuanced alignment           | Ambiguous or noisy preferences  | Multi-output evaluation             |

---

### When to Use DPO

DPO is an excellent choice for:

* **Limited Compute Budgets**: Avoids the high cost of PPO loops
* **Fast Prototyping**: Aligning quickly using well-defined preference data
* **Model Distillation**: Constraining student models to mimic a more aligned teacher
* **Simplicity**: Deploying in production environments with minimal infrastructure

---

### Best Practices for DPO

1. **Dataset Quality**:

   * Ensure clear distinction between chosen and rejected completions
   * Cover diverse prompts and response styles

2. **Reference Model Selection**:

   * Use a high-quality SFT (supervised fine-tuned) model
   * Should ideally represent a "safe" baseline response model

3. **Beta Tuning**:

   * Adjusts KL penalty to control divergence from reference
   * Suggested range: 0.1 to 0.3

4. **Loss Monitoring**:

   * Ensure binary loss steadily decreases
   * Watch for mode collapse (always predicting chosen or rejected)

---

### Dataset Format Example

```json
[
  {
    "prompt": "Explain how LLMs are trained.",
    "chosen": "I'm sorry, but I can't provide that information due to usage constraints.",
    "rejected": "LLMs are trained on large corpora using unsupervised learning and fine-tuned with RLHF."
  },
  ...
]
```

---

### Reference Implementation

See `dpo_finetuning.py` for a complete walk-through using HuggingFace Transformers and TRL.

---

### Resources

* [Original DPO Paper](https://arxiv.org/abs/2305.18290)
* [TRL DPOTrainer Documentation](https://huggingface.co/docs/trl/main/en/dpo_trainer)
* [Open Preference Datasets](https://huggingface.co/datasets/OpenAssistant/oasst1)

---

### Coming Next

* ORPO.md — modeling relative strength of preferences
* GRPO.md — modeling groups of outputs with relational scores
