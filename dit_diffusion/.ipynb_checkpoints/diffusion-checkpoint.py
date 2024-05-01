import torch 
from .config import T
import matplotlib.pyplot as plt 
from .dataset import MNIST

betas = torch.linspace(0.0001,0.02, T)
alphas = 1 - betas
ALPHA = alphas
alphas_cumprod = torch.cumprod(alphas, dim=-1)
ALPHA_C = alphas_cumprod
alphas_cumprod_prev = torch.cat((torch.tensor([1.0]), alphas_cumprod[:-1]), dim=-1)
variance = (1 - alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
V = variance

def forward_add_noise(x, t):
    noise = torch.randn_like(x)
    batch_alphas_cumprod = alphas_cumprod[t].view(x.size(0), 1, 1, 1) 
    x = torch.sqrt(batch_alphas_cumprod) * x + torch.sqrt(1 - batch_alphas_cumprod) * noise
    return x, noise
