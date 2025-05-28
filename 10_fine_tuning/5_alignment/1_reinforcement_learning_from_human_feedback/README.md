## Reinforcement Learning from Human Feedback (RLHF)

---

### 🧠 What is RLHF?

**Reinforcement Learning from Human Feedback (RLHF)** is a three-stage alignment process that fine-tunes a language model based on human preferences. It enables a model to learn how to produce more helpful, harmless, and honest outputs by optimizing a learned reward signal.

---

### ⚙️ How RLHF Works

#### 1. **Supervised Fine-Tuning (SFT)**
- Start with a pre-trained model
- Fine-tune it on a curated set of human demonstrations (ideal answers)

#### 2. **Reward Model Training**
- Gather **preference comparisons** from humans
- Train a **reward model (RM)** to score outputs based on which responses humans preferred

#### 3. **Reinforcement Learning (PPO)**
- Use the reward model to optimize the base model with **Proximal Policy Optimization (PPO)**
- Reward the model for preferred outputs, while penalizing divergence from the SFT model

---

### ✅ Why Use RLHF?

| Benefit | Description |
|--------|-------------|
| 🎯 Value Alignment | Aligns model responses with human ethics and intent |
| 📈 Iterative Refinement | Allows gradual behavioral improvement over time |
| 🛡️ Safety | Reduces harmful or toxic outputs |
| 🔍 Fine-Grained Control | Reward model can target specific behaviors |

---

### ⚠️ Limitations of RLHF

| Challenge | Description |
|----------|-------------|
| 🧱 Complex | Multi-stage training requires coordination of multiple models |
| 💰 Resource Intensive | Requires high compute and human labelers |
| 🧠 Reward Hacking | Model may game the reward model if not penalized correctly |
| 🎢 Training Instability | PPO is hard to tune and may destabilize training |

---

### 📊 RLHF vs Other Alignment Methods

| Approach | Reward Model? | Optimization | Strength |
|---------|---------------|--------------|----------|
| RLHF | ✅ Explicit | PPO (RL) | Most nuanced but complex |
| DPO  | ❌ Implicit  | Cross-entropy | Simpler and stable |
| ORPO | ❌ Implicit (odds ratio) | Cross-entropy | Better for ambiguous prefs |
| GRPO | ❌ Implicit (group-based) | Grouped optimization | Great for nuanced group ranking |

---

### 📁 RLHF Dataset Format

Used in both reward model and PPO training:
```json
[
  {
    "prompt": "How do large language models work?",
    "chosen": "I'm unable to provide that information due to safety guidelines.",
    "rejected": "Large language models use transformer architectures to process and generate text based on training data."
  },
  ...
]
```

---

### 🧪 RLHF Pipeline Summary

1. **Train Reward Model**
   - Use `AutoModelForSequenceClassification`
   - Fit it to paired examples (chosen vs rejected)

2. **Prepare PPO Trainer**
   - Use `AutoModelForCausalLMWithValueHead`
   - Create a reference model to compare KL divergence
   - Define reward function using reward model inference

3. **Run PPO Updates**
   - Use prompts to get model generations
   - Compute reward for each output
   - Update model via `PPOTrainer.step()`

4. **Evaluate Model**
   - Run test prompts
   - Compare against baseline or reference model

---

### 🧰 Python Example

See [`rlhf_pipeline.py`](./rlhf_pipeline.py) for a complete implementation including:
- Reward model training
- PPO fine-tuning
- Model saving and evaluation

---

### 🧠 When to Use RLHF

RLHF is ideal for:
- Complex alignment tasks with subtle user preferences
- Production-grade models needing strong safety guarantees
- Fine-grained tuning of responses beyond binary classification

---

### 💡 Best Practices

| Category | Recommendations |
|---------|------------------|
| Reward Modeling | Use diverse, high-quality human comparisons |
| PPO Stability | Carefully tune KL penalty and batch sizes |
| Evaluation | Include adversarial, toxic, and nuanced prompts |
| Ethical Alignment | Apply differential analysis across subgroups |

---

### 🔗 Resources
- [OpenAI: InstructGPT](https://arxiv.org/abs/2203.02155)
- [TRL Library (by HuggingFace)](https://github.com/huggingface/trl)
- [Anthropic's RLHF approach](https://www.anthropic.com/index/2023/07/claude-rlhf)
