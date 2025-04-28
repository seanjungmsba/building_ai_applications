import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Generate x values by creating 500 points between -5 and 5
x = torch.linspace(-5, 5, 500)

# Compute ReLU and GeLU outputs
relu = F.relu(x)
gelu = F.gelu(x)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(x.numpy(), relu.numpy(), label='ReLU', linestyle='--')
plt.plot(x.numpy(), gelu.numpy(), label='GeLU', linestyle='-')
plt.title('Comparison of ReLU vs GeLU Activation Functions')
plt.xlabel('Input (x)')
plt.ylabel('Activation Output')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

'''
---------------------------------------------------------------------------------------------------------------------------
Activation Function	| Behavior
--------------------|------------------------------------------------------------------------------------------------------
ReLU	            | Sharp cutoff at 0 — all negative inputs are mapped to zero, positives are linear
GeLU	            | Smooth transition — gradually increases from zero based on input's magnitude (probabilistic behavior)
---------------------------------------------------------------------------------------------------------------------------

🔵 Key takeaway:
- GeLU behaves more like a smooth soft "gate", while ReLU behaves like a hard switch.
- This smoothness helps GPT models converge better during training by avoiding sharp non-linearities.
'''
