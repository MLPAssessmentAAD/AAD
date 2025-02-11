import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim

        self.query = nn.Conv1d(in_channels=input_dim, out_channels=embed_dim, kernel_size=1)
        self.key = nn.Conv1d(in_channels=input_dim, out_channels=embed_dim, kernel_size=1)
        self.value = nn.Conv1d(in_channels=input_dim, out_channels=embed_dim, kernel_size=1)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (B, C, T)
        Q = self.query(x)  # (B, C, T)
        K = self.key(x)  # (B, C, T)
        V = self.value(x)  # (B, C, T)

        # Compute attention scores
        attn_scores = torch.matmul(Q.transpose(1, 2), K) / (self.embed_dim ** 0.5)  # (B, T, T)
        attn_probs = self.softmax(attn_scores)  # (B, T, T)

        # Apply attention to values
        output = torch.matmul(attn_probs, V.transpose(1, 2)).transpose(1, 2)  # (B, C, T)
        return output
