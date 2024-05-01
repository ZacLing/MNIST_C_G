from torch import nn
import torch
import math

class DiTBlock(nn.Module):
    def __init__(self, emb_size, nhead):
        super().__init__()
        
        self.emb_size = emb_size
        self.nhead = nhead
        
        # Conditioning layers
        self.gamma1 = nn.Linear(emb_size, emb_size)
        self.beta1 = nn.Linear(emb_size, emb_size)
        self.alpha1 = nn.Linear(emb_size, emb_size)
        self.gamma2 = nn.Linear(emb_size, emb_size)
        self.beta2 = nn.Linear(emb_size, emb_size)
        self.alpha2 = nn.Linear(emb_size, emb_size)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)
        
        # Multi-head self-attention layers
        self.wq = nn.Linear(emb_size, nhead * emb_size)
        self.wk = nn.Linear(emb_size, nhead * emb_size)
        self.wv = nn.Linear(emb_size, nhead * emb_size)
        self.lv = nn.Linear(nhead * emb_size, emb_size)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            nn.ReLU(),
            nn.Linear(emb_size * 4, emb_size)
        )

    def forward(self, x, cond):
        # Apply conditioning
        gamma1_val = self.gamma1(cond)
        beta1_val = self.beta1(cond)
        alpha1_val = self.alpha1(cond)
        gamma2_val = self.gamma2(cond)
        beta2_val = self.beta2(cond)
        alpha2_val = self.alpha2(cond)
        
        # First layer normalization
        y = self.ln1(x)
        
        # Scale and shift
        y = y * (1 + gamma1_val.unsqueeze(1)) + beta1_val.unsqueeze(1)

        # Attention mechanism
        q = self.wq(y)
        k = self.wk(y)
        v = self.wv(y)
        q = q.view(q.size(0), q.size(1), self.nhead, self.emb_size).permute(0, 2, 1, 3)
        k = k.view(k.size(0), k.size(1), self.nhead, self.emb_size).permute(0, 2, 3, 1)
        v = v.view(v.size(0), v.size(1), self.nhead, self.emb_size).permute(0, 2, 1, 3)
        attn = q @ k / math.sqrt(q.size(2))
        attn = torch.softmax(attn, dim=-1)
        y = attn @ v
        y = y.permute(0, 2, 1, 3)
        y = y.reshape(y.size(0), y.size(1), y.size(2) * y.size(3))
        y = self.lv(y)

        # Scale and add residual connection
        y = y * alpha1_val.unsqueeze(1)
        y = x + y
        
        # Second layer normalization and scale & shift
        z = self.ln2(y)
        z = z * (1 + gamma2_val.unsqueeze(1)) + beta2_val.unsqueeze(1)
        
        # Feed-forward
        z = self.ff(z)
        
        # Scale and add final residual
        z = z * alpha2_val.unsqueeze(1)
        return y + z
