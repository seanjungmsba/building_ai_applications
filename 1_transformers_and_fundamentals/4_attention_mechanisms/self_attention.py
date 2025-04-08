"""
self_attention.py

This script defines a PyTorch module for self-attention — the core building block
of the Transformer architecture. Self-attention allows the model to weigh
the importance of different tokens in a sequence when encoding each token.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------
# 🧠 Self-Attention Layer Definition
# -----------------------------------
class SelfAttention(nn.Module):
    """
    Implements a single multi-head self-attention mechanism.

    Args:
        embed_size (int): The total embedding size (e.g., 64, 128).
        heads (int): The number of attention heads to split the embedding into.
    """
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()

        # Store input parameters
        self.embed_size = embed_size
        self.heads = heads

        # Each head handles a fraction of the embedding space
        self.head_dim = embed_size // heads
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size must be divisible by the number of heads"

        # Create linear layers for generating keys, queries, and values
        # from each chunk (head) of the input embeddings.
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # After concatenating the heads, this layer projects the combined output back
        # to the original embedding size.
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        """
        Forward pass for self-attention.

        Args:
            values (Tensor): Value embeddings [batch_size, value_len, embed_size]
            keys (Tensor): Key embeddings [batch_size, key_len, embed_size]
            query (Tensor): Query embeddings [batch_size, query_len, embed_size]
            mask (Tensor or None): Attention mask [batch_size, 1, query_len, key_len]

        Returns:
            Tensor: Self-attention output of shape [batch_size, query_len, embed_size]
        """
        N = query.shape[0]  # Batch size

        # Lengths of each input sequence
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # ------------------------------
        # 🪓 Step 1: Split into heads
        # ------------------------------
        # Reshape input tensors into multiple heads:
        # Shape becomes [batch_size, seq_len, heads, head_dim]
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # ------------------------------
        # 🧮 Step 2: Linear Projections
        # ------------------------------
        # Apply learned linear transformations to each head independently
        values = self.values(values)     # V projection
        keys = self.keys(keys)           # K projection
        queries = self.queries(queries)  # Q projection

        # ------------------------------
        # 🔍 Step 3: Calculate Scaled Dot-Product Attention Scores
        # ------------------------------
        # energy: [batch_size, heads, query_len, key_len]
        # Uses Einstein summation to perform batch matrix multiplication between Q and K^T
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # ------------------------------
        # 🙈 Step 4: Apply Mask (Optional)
        # ------------------------------
        if mask is not None:
            # Replace masked positions with a very negative value (acts as -∞ in softmax)
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # ------------------------------
        # 🧠 Step 5: Softmax to get Attention Weights
        # ------------------------------
        # Scale scores before softmax to prevent gradient issues
        attention = torch.softmax(energy / (self.embed_size ** 0.5), dim=3)

        # ------------------------------
        # 🎯 Step 6: Weighted Sum of Values
        # ------------------------------
        # attention: [batch_size, heads, query_len, key_len]
        # values:    [batch_size, key_len, heads, head_dim]
        # output:    [batch_size, query_len, heads, head_dim]
        out = torch.einsum("nhql, nlhd->nqhd", [attention, values])

        # Merge heads back into a single embedding
        out = out.reshape(N, query_len, self.heads * self.head_dim)

        # Final linear transformation to match original embed size
        out = self.fc_out(out)

        return out

# -----------------------------------
# 🚀 Example Usage
# -----------------------------------
embed_size = 64  # Total embedding size
heads = 8        # Number of attention heads

# Create a self-attention layer
self_attention = SelfAttention(embed_size, heads)

# Create dummy input: batch size 1, sequence length 5, embedding size 64
x = torch.rand(1, 5, embed_size)

# Without a mask
output = self_attention(x, x, x, mask=None)

print("Output without mask:")
print(output)
print("Output shape:", output.shape)  # Expected: [1, 5, 64]

'''

Output without mask:
tensor([[[-0.2681, -0.1415,  0.1290,  0.1401,  0.0254,  0.4232, -0.0823,
           0.3906,  0.1530,  0.1404, -0.0246,  0.1153, -0.1129, -0.0911,
           0.0617,  0.0459,  0.0226,  0.1901, -0.2138, -0.0231, -0.2005,
           0.3752,  0.0681,  0.0295, -0.1137,  0.1161, -0.0825,  0.0798,
           0.2593, -0.1551, -0.2672,  0.1486,  0.4015,  0.1838,  0.3372,
          -0.2485, -0.0763,  0.2185, -0.0481, -0.0878, -0.4780,  0.1916,
          -0.0967,  0.1507, -0.1665, -0.1916, -0.2028, -0.0284, -0.3517,
          -0.0742,  0.5779,  0.2873,  0.0998,  0.2394,  0.2294, -0.0765,
           0.5665,  0.0762,  0.0489, -0.0425, -0.1206,  0.0943,  0.2850,
          -0.2104],
         [-0.2687, -0.1421,  0.1291,  0.1402,  0.0250,  0.4229, -0.0823,
           0.3908,  0.1537,  0.1403, -0.0249,  0.1151, -0.1124, -0.0905,
           0.0611,  0.0460,  0.0229,  0.1897, -0.2136, -0.0227, -0.2014,
           0.3753,  0.0683,  0.0302, -0.1140,  0.1162, -0.0824,  0.0795,
           0.2595, -0.1555, -0.2672,  0.1489,  0.4021,  0.1841,  0.3371,
          -0.2490, -0.0762,  0.2186, -0.0486, -0.0878, -0.4779,  0.1915,
          -0.0972,  0.1499, -0.1677, -0.1913, -0.2035, -0.0277, -0.3514,
          -0.0745,  0.5786,  0.2879,  0.0996,  0.2389,  0.2296, -0.0764,
           0.5661,  0.0756,  0.0487, -0.0422, -0.1209,  0.0952,  0.2852,
          -0.2108],
         [-0.2674, -0.1410,  0.1290,  0.1393,  0.0253,  0.4230, -0.0823,
           0.3904,  0.1526,  0.1410, -0.0240,  0.1146, -0.1130, -0.0916,
           0.0611,  0.0448,  0.0226,  0.1897, -0.2135, -0.0234, -0.1991,
           0.3744,  0.0690,  0.0296, -0.1128,  0.1159, -0.0823,  0.0792,
           0.2591, -0.1547, -0.2673,  0.1478,  0.4014,  0.1842,  0.3372,
          -0.2485, -0.0750,  0.2191, -0.0475, -0.0882, -0.4775,  0.1920,
          -0.0978,  0.1498, -0.1661, -0.1911, -0.2018, -0.0288, -0.3513,
          -0.0740,  0.5780,  0.2872,  0.0992,  0.2400,  0.2301, -0.0751,
           0.5665,  0.0770,  0.0486, -0.0417, -0.1209,  0.0943,  0.2846,
          -0.2100],
         [-0.2673, -0.1411,  0.1292,  0.1388,  0.0254,  0.4226, -0.0822,
           0.3911,  0.1537,  0.1404, -0.0243,  0.1143, -0.1125, -0.0905,
           0.0603,  0.0450,  0.0222,  0.1896, -0.2143, -0.0232, -0.1996,
           0.3744,  0.0694,  0.0299, -0.1130,  0.1169, -0.0825,  0.0789,
           0.2582, -0.1541, -0.2676,  0.1480,  0.4010,  0.1840,  0.3371,
          -0.2476, -0.0746,  0.2188, -0.0481, -0.0883, -0.4768,  0.1916,
          -0.0980,  0.1496, -0.1660, -0.1911, -0.2022, -0.0287, -0.3517,
          -0.0739,  0.5780,  0.2877,  0.0994,  0.2405,  0.2307, -0.0757,
           0.5662,  0.0757,  0.0495, -0.0420, -0.1216,  0.0942,  0.2840,
          -0.2104],
         [-0.2680, -0.1417,  0.1294,  0.1392,  0.0248,  0.4224, -0.0821,
           0.3906,  0.1536,  0.1409, -0.0247,  0.1147, -0.1133, -0.0907,
           0.0605,  0.0456,  0.0225,  0.1895, -0.2133, -0.0235, -0.2003,
           0.3744,  0.0694,  0.0301, -0.1137,  0.1155, -0.0820,  0.0789,
           0.2598, -0.1545, -0.2672,  0.1478,  0.4015,  0.1841,  0.3367,
          -0.2488, -0.0747,  0.2189, -0.0474, -0.0881, -0.4773,  0.1915,
          -0.0972,  0.1498, -0.1669, -0.1913, -0.2031, -0.0284, -0.3510,
          -0.0741,  0.5780,  0.2872,  0.0986,  0.2394,  0.2299, -0.0755,
           0.5662,  0.0759,  0.0490, -0.0412, -0.1212,  0.0950,  0.2843,
          -0.2102]]], grad_fn=<ViewBackward0>)

Output shape: torch.Size([1, 5, 64])

'''
# -----------------------------------
# 🛡️ With Masking (Autoregressive)
# -----------------------------------
def create_mask(seq_length):
    """
    Creates a lower triangular matrix used for autoregressive masking.

    Args:
        seq_length (int): Length of the sequence

    Returns:
        Tensor: Mask of shape [1, 1, seq_len, seq_len]
    """
    # tril creates a lower triangle of 1s (including the diagonal)
    # unsqueeze twice to match attention energy shape
    return torch.tril(torch.ones((seq_length, seq_length))).unsqueeze(0).unsqueeze(0)

# Apply self-attention with a mask to simulate causal (left-to-right) behavior
masked_output = self_attention(x, x, x, mask=create_mask(5))

print("\nOutput with mask:")
print(masked_output)
print("Masked output shape:", masked_output.shape)  # Expected: [1, 5, 64]

'''

Output with mask:
tensor([[[-2.9867e-01, -1.9059e-01,  2.1555e-01,  5.5862e-02, -1.2442e-02,
           5.3448e-01,  7.3574e-05,  3.2280e-01,  1.0360e-01,  5.3489e-02,
          -1.1723e-01,  7.0012e-02, -7.9898e-02, -2.0058e-01,  1.5826e-01,
           8.2268e-02,  7.7589e-03,  1.9911e-01, -1.9210e-01,  8.5225e-03,
          -1.7315e-01,  4.3492e-01, -1.0698e-01,  1.2240e-02, -2.3558e-01,
           1.6961e-01, -1.5851e-01,  1.2918e-01,  3.0156e-01, -3.3017e-01,
          -2.0001e-01,  4.6684e-02,  4.0214e-01,  1.2587e-01,  3.2171e-01,
          -3.2872e-01, -2.0064e-01,  9.9563e-02, -8.0177e-03, -3.8663e-02,
          -6.1646e-01,  1.6098e-01, -3.2582e-02,  1.3704e-01, -2.1498e-01,
          -2.0921e-01, -2.8022e-01, -2.7247e-02, -3.8385e-01, -1.0041e-01,
           6.8968e-01,  2.3744e-01,  1.7163e-01,  1.1802e-01,  1.8603e-01,
          -6.4335e-02,  6.5406e-01,  1.5101e-01,  3.9300e-02, -1.6887e-01,
          -7.7417e-02,  1.3379e-01,  3.9712e-01, -2.6386e-01],
         [-3.5660e-01, -1.0962e-01,  9.6501e-02,  1.6374e-01,  6.3351e-02,
           4.8503e-01, -9.2432e-02,  3.5220e-01,  9.5866e-02,  1.7490e-01,
          -8.9103e-02,  5.6328e-02, -1.6309e-01, -1.0849e-01,  8.0479e-02,
          -5.8477e-03,  5.3278e-02,  1.7154e-01, -2.7678e-01, -1.2366e-02,
          -2.0376e-01,  5.5654e-01, -7.5206e-03, -6.5074e-02, -1.4071e-01,
           1.3276e-01, -1.4435e-01,  1.7336e-01,  2.5445e-01, -2.2234e-01,
          -2.6666e-01,  1.8623e-01,  4.0615e-01,  1.7699e-01,  3.0533e-01,
          -1.6830e-01, -1.9391e-01,  1.6226e-01, -8.8999e-02, -8.2388e-02,
          -5.8753e-01,  7.1143e-02, -8.9659e-03,  2.0880e-01, -1.7355e-01,
          -2.8108e-01, -1.9467e-01, -1.5849e-03, -4.8024e-01, -1.4687e-01,
           6.2475e-01,  2.5500e-01,  2.0464e-01,  2.3789e-01,  1.0349e-01,
          -1.5269e-01,  6.0017e-01,  8.8709e-02,  8.7462e-02, -1.0578e-01,
          -7.6493e-02,  2.2199e-04,  2.7597e-01, -1.9631e-01],
         [-2.9203e-01, -1.2649e-01,  1.3757e-01,  1.9337e-01,  5.8200e-02,
           4.5682e-01, -9.5199e-02,  3.5847e-01,  9.1196e-02,  1.3423e-01,
          -4.6512e-02,  8.0177e-02, -1.1966e-01, -1.0986e-01,  7.9655e-02,
           2.7128e-02,  4.3814e-02,  2.0216e-01, -2.4553e-01, -3.5662e-02,
          -1.9749e-01,  4.7701e-01, -1.0233e-02, -1.2278e-02, -1.0099e-01,
           1.2404e-01, -1.0780e-01,  1.3189e-01,  2.3966e-01, -1.9119e-01,
          -2.6191e-01,  2.0618e-01,  4.2569e-01,  1.8212e-01,  3.3201e-01,
          -2.0193e-01, -1.3926e-01,  1.8954e-01, -7.9470e-02, -1.0269e-01,
          -5.3337e-01,  1.6938e-01, -7.4609e-02,  2.1186e-01, -1.2947e-01,
          -2.0727e-01, -1.9286e-01, -9.4408e-03, -3.9955e-01, -1.0083e-01,
           5.7846e-01,  2.4195e-01,  1.6554e-01,  2.2442e-01,  1.7145e-01,
          -9.9589e-02,  5.6913e-01,  1.0932e-01,  4.7471e-02, -8.5351e-02,
          -7.3051e-02,  4.4852e-02,  3.0433e-01, -1.9819e-01],
         [-2.7070e-01, -1.1101e-01,  1.1831e-01,  1.6902e-01,  6.1771e-02,
           4.5038e-01, -9.0721e-02,  4.0775e-01,  1.1461e-01,  1.5987e-01,
          -3.8612e-02,  1.1075e-01, -1.2192e-01, -1.1194e-01,  7.3862e-02,
           4.5723e-02,  3.1030e-02,  2.0706e-01, -2.3719e-01, -4.2384e-02,
          -1.5768e-01,  4.0900e-01,  4.8115e-02, -2.4050e-03, -1.1561e-01,
           1.2418e-01, -1.1177e-01,  1.0356e-01,  2.4528e-01, -1.8638e-01,
          -2.6854e-01,  1.7544e-01,  4.0915e-01,  1.9102e-01,  3.3686e-01,
          -2.4461e-01, -9.5016e-02,  2.2206e-01, -3.5606e-02, -8.9865e-02,
          -5.2191e-01,  1.7859e-01, -7.3850e-02,  1.7641e-01, -1.2767e-01,
          -2.0136e-01, -1.6580e-01, -4.2845e-02, -3.7537e-01, -9.8677e-02,
           5.6790e-01,  2.6331e-01,  1.2184e-01,  2.4543e-01,  2.2439e-01,
          -9.2584e-02,  5.6999e-01,  1.0333e-01,  4.4215e-02, -8.4537e-02,
          -9.3985e-02,  6.6453e-02,  2.8042e-01, -1.9419e-01],
         [-2.6798e-01, -1.4169e-01,  1.2938e-01,  1.3924e-01,  2.4792e-02,
           4.2241e-01, -8.2135e-02,  3.9055e-01,  1.5359e-01,  1.4086e-01,
          -2.4734e-02,  1.1473e-01, -1.1329e-01, -9.0702e-02,  6.0511e-02,
           4.5584e-02,  2.2484e-02,  1.8953e-01, -2.1326e-01, -2.3472e-02,
          -2.0031e-01,  3.7437e-01,  6.9449e-02,  3.0145e-02, -1.1375e-01,
           1.1551e-01, -8.1991e-02,  7.8902e-02,  2.5979e-01, -1.5451e-01,
          -2.6719e-01,  1.4785e-01,  4.0150e-01,  1.8408e-01,  3.3673e-01,
          -2.4883e-01, -7.4673e-02,  2.1894e-01, -4.7429e-02, -8.8111e-02,
          -4.7732e-01,  1.9153e-01, -9.7201e-02,  1.4984e-01, -1.6688e-01,
          -1.9129e-01, -2.0313e-01, -2.8386e-02, -3.5098e-01, -7.4145e-02,
           5.7795e-01,  2.8724e-01,  9.8615e-02,  2.3939e-01,  2.2988e-01,
          -7.5544e-02,  5.6618e-01,  7.5898e-02,  4.9030e-02, -4.1212e-02,
          -1.2117e-01,  9.5037e-02,  2.8428e-01, -2.1019e-01]]],
       grad_fn=<ViewBackward0>)

Masked output shape: torch.Size([1, 5, 64])
'''