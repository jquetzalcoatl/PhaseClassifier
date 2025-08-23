"""
Module: transformer
-------------------
This module defines neural network components for the PhaseClassifier project, including MLP, normalization, self-attention, multi-head attention, and a classifier architecture.
Classes:
--------
MLP(cfg):
    A simple multi-layer perceptron with two linear layers.
    Args:
        cfg: Configuration object containing model parameters.
    Methods:
        forward(x): Forward pass through the MLP.
LNorm(dim, fn):
    Layer normalization wrapper around a given function.
    Args:
        dim: Dimension for normalization.
        fn: Function to apply before normalization.
    Methods:
        forward(x): Applies the function and then layer normalization.
Classifier(cfg=None):
    Main classifier model using convolutional, linear, and attention layers.
    Args:
        cfg: Configuration object containing model parameters.
    Methods:
        forward(x): Forward pass through the classifier.
        loss(pred, target): Computes cross-entropy loss.
Head(dim, head_size=16):
    Self-attention block for sequence modeling.
    Args:
        dim: Input dimension.
        head_size: Size of each attention head.
    Methods:
        forward(x): Computes self-attention output.
Multihead(dim, num=1):
    Multi-head attention block using multiple Headv2 modules.
    Args:
        dim: Input dimension.
        num: Number of attention heads.
    Methods:
        forward(x): Computes multi-head attention output.
Notes:
------
- Uses einops for tensor rearrangement.
- Designed for phase classification tasks with configurable architecture.
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange

from PhaseClassifier import logging
logger = logging.getLogger(__name__)


class MLP(nn.Module):
    def __init__(self, cfg):
        super(MLP, self).__init__()
        self._config = cfg
        self.seq = nn.Sequential(
            nn.Linear(self._config.model.input_size,100),
            nn.Linear(100,self._config.model.input_size),
        )
    def forward(self,x):
        return self.seq(x)

class LNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(self.fn(x))
        return x
    
class Classifier(nn.Module):
    def __init__(self, cfg=None):
        super(Classifier, self).__init__()
        self._config = cfg
        self.seq = nn.Sequential(
            nn.Conv2d(1, self._config.model.channels, kernel_size=2, stride=2, padding=0),
        )
        self.lm_head1 = nn.Linear(self._config.model.channels, self._config.model.n_embd)
        self.sa_head = Head(self._config.model.n_embd, self._config.model.head_size)
        self.lm_head2 = nn.Linear(self._config.model.head_size, self._config.model.n_embd)
        self.conv = nn.Conv2d(self._config.model.n_embd,
                              self._config.model.labels,
                              self._config.model.n_embd,1,0)

    def forward(self, x):
        x = self.seq(x)
        B,C,H,W = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.lm_head1(x)
        x = self.sa_head(x)
        x = self.lm_head2(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.conv(x).squeeze(3).squeeze(2)
        return x

    def loss(self, pred, target):
        return F.cross_entropy(pred, target.view(-1))

class Head(nn.Module):
    '''
    Self-attention block
    '''
    def __init__(self, dim, head_size=16):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(dim,head_size, bias=False)
        self.query = nn.Linear(dim,head_size, bias=False)
        self.value = nn.Linear(dim,head_size, bias=False)
        

    def forward(self, x):
        # b, c, l, h, w = x.shape
        # x = rearrange(x, "b c l h w -> b (l h w) c")
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * self.head_size**-0.5
        wei = F.softmax(wei,dim=-1)
        v = self.value(x)
        out = wei @ v
        
        return out
    
class Multihead(nn.Module):
    '''
        Multi-head attention
    '''
    def __init__(self, dim, num=1):
        super().__init__()
        head_size = dim // num
        self.heads = nn.ModuleList([Headv2(dim,head_size) for _ in range(num)])
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        b, c, l, h, w = x.shape
        x = rearrange(x, "b c l h w -> b (l h w) c")
        x = torch.cat([h(self.ln1(x)) for h in self.heads], dim=2)
        x = self.ln2(x)
        return rearrange(x, "b (l h w) c -> b c l h w",l=l,h=h,w=w)