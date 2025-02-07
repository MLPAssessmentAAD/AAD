import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        return self.norm(x + attn_output)  # 残差连接 + 归一化

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        ff_output = self.fc2(self.activation(self.fc1(x)))
        return self.norm(x + ff_output)  # 残差连接 + 归一化

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, hidden_dim)

    def forward(self, x):
        x = self.attention(x)
        x = self.ffn(x)
        return x
