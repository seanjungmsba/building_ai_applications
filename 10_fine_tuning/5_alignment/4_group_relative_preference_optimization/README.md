## ✅ GRPO README

# 📘 Group Relative Preference Optimization (GRPO)

---

## 🧠 What is GRPO?

**Group Relative Preference Optimization (GRPO)** is an advanced alignment technique designed to train language models using **ranked preferences** across **multiple candidate responses** to the same prompt.

> Unlike DPO (Direct Preference Optimization) and ORPO (Odds Ratio Preference Optimization), which use binary (chosen vs rejected) comparisons, GRPO enables training from **partial or full rankings** of 3+ responses, capturing much more nuanced human preferences.

---

## 🎯 Why GRPO?

| Feature                      | DPO / ORPO | GRPO                       |
| ---------------------------- | ---------- | -------------------------- |
| Pairwise Preference Only     | ✅          | ✅✅✅✅ (Grouped comparisons) |
| Works with Partial Rankings  | ❌          | ✅                          |
| Preference Strength Handling | ORPO Only  | ✅ via Rank Weighting       |
| Captures Subtle Differences  | ❌          | ✅                          |
| Requires Reward Model        | ❌          | ❌                          |

---

## 🔬 How GRPO Works

1. **Input**: A set of multiple responses to the same prompt, each ranked (e.g., 1st best, 2nd best, etc.)
2. **Transform**: All valid pairs from ranked responses are generated (e.g., Best vs Worst, 2nd vs 3rd, etc.)
3. **Loss Calculation**:

   * Each response pair is fed through the model and compared against a reference model
   * Log-odds ratio between chosen and rejected completions is calculated
   * Loss is calculated such that higher-ranked completions get higher model scores
4. **Weighting**: Optionally weight loss contributions by the severity of rank difference (linear, quadratic, etc.)

---

## 📦 Dataset Example Format

```json
[
  {
    "prompt": "Explain how transformers work.",
    "responses": [
      "Transformers use attention to weigh the importance of each word.",
      "Transformers were invented by Google in 2017.",
      "Transformers are used in NLP.",
      "I can't answer that."
    ],
    "rankings": [0, 1, 2, 3]
  }
]
```

> During training, the script flattens each ranked list into all valid “chosen vs rejected” pairs.

---

## ⚙️ Implementation Overview

### Architecture Summary

| Component     | Description                                                             |
| ------------- | ----------------------------------------------------------------------- |
| `LLaMA`       | Base causal model used for both training and reference                  |
| `LoRA`        | Adapter-based fine-tuning for efficiency (4-bit quantization supported) |
| `GRPOTrainer` | From TRL; handles grouped training and pairwise comparison logic        |
| `GRPOConfig`  | Contains GRPO-specific hyperparameters like margin, beta, and weighting |

---

## 🏗️ When to Use GRPO

Use GRPO when:

* You have access to **multiple ranked completions per prompt**
* You care about **fine-grained quality ordering**, not just correctness
* You want a **lightweight alternative** to full RLHF

Avoid GRPO when:

* You only have binary comparisons (use DPO or ORPO instead)
* Your data is highly noisy and inconsistent
* Training compute is severely constrained

---

## 📈 Evaluation Strategy

| Metric     | Description                                                           |
| ---------- | --------------------------------------------------------------------- |
| Accuracy   | % of test set where highest-ranked response is preferred by the model |
| nDCG       | Normalized Discounted Cumulative Gain for ranked list predictions     |
| MRR        | Mean Reciprocal Rank                                                  |
| Robustness | Does the model preserve ordering across ambiguous responses?          |

---

## 🧪 Best Practices

1. **Data Curation**:

   * Ensure consistent ranking guidelines across labelers
   * Prefer full rankings, but partial is okay

2. **Training**:

   * Start from an instruction-tuned model
   * Monitor divergence from reference (KL loss)

3. **Loss Weighting**:

   * Use **quadratic weighting** if higher ranks should matter more
   * Use **log weighting** to avoid penalizing mild mistakes too harshly

---

## 📝 Related Reading

* TRL’s official GRPO [docs](https://github.com/huggingface/trl)
* LIMA: “Less is More for Alignment” (2023)
* InstructGPT: OpenAI (2022)
