import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import logging

# Configure enterprise-grade logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DistributedConfig:
    """Configuration for distributed training environment."""
    def __init__(self, backend: str = 'nccl', master_addr: str = 'localhost', master_port: str = '12355'):
        self.backend = backend
        self.master_addr = master_addr
        self.master_port = master_port
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.rank = int(os.environ.get('RANK', 0))

    def init_process_group(self):
        """Initialize the distributed process group."""
        if self.world_size > 1:
            os.environ['MASTER_ADDR'] = self.master_addr
            os.environ['MASTER_PORT'] = self.master_port
            dist.init_process_group(self.backend, rank=self.rank, world_size=self.world_size)
            logger.info(f"Initialized distributed process group: Rank {self.rank}/{self.world_size}")
        else:
            logger.info("Running in standalone mode (no distributed context detected).")

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with optimized tensor operations.
    Implements Scaled Dot-Product Attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    def __init__(self, d_model: int, n_head: int, context_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        
        self.d_head = d_model // n_head
        self.n_head = n_head
        self.d_model = d_model
        
        # Key, Query, Value projections combined for efficiency
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Causal mask to ensure autoregressive property
        self.register_buffer("bias", torch.tril(torch.ones(context_len, context_len))
                                     .view(1, 1, context_len, context_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for attention mechanism.
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        B, T, C = x.size() # Batch, Time (seq_len), Channels (d_model)
        
        # Calculate Query, Key, Value
        # q, k, v shape: (B, T, n_head, d_head) -> permute to (B, n_head, T, d_head)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # Scaled Dot-Product Attention
        # (B, n_head, T, d_head) @ (B, n_head, d_head, T) -> (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply causal mask
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        # Softmax and Dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Aggregate values
        # (B, n_head, T, T) @ (B, n_head, T, d_head) -> (B, n_head, T, d_head)
        y = att @ v
        
        # Reassemble heads
        # (B, n_head, T, d_head) -> (B, T, n_head, d_head) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.resid_dropout(self.c_proj(y))

class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""
    def __init__(self, d_model: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expansion_factor),
            nn.GELU(),
            nn.Linear(d_model * expansion_factor, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerBlock(nn.Module):
    """Standard Transformer Block: Pre-LN architecture."""
    def __init__(self, d_model: int, n_head: int, context_len: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, context_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffwd = FeedForward(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class AdvancedTransformer(nn.Module):
    """
    Full Transformer Decoder model designed for scalability.
    Supports gradient checkpointing for memory efficiency.
    """
    def __init__(self, vocab_size: int, d_model: int, n_head: int, n_layer: int, context_len: int, dropout: float = 0.1):
        super().__init__()
        self.context_len = context_len
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_head, context_len, dropout)
            for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying: https://paperswithcode.com/method/weight-tying
        self.token_embedding.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        Args:
            idx: (B, T) tensor of token indices
            targets: (B, T) tensor of target token indices (optional)
        """
        B, T = idx.shape
        assert T <= self.context_len, f"Sequence length {T} exceeds context length {self.context_len}"
        
        # Embeddings
        tok_emb = self.token_embedding(idx) # (B, T, C)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device)) # (T, C)
        x = self.dropout(tok_emb + pos_emb)
        
        # Transformer Blocks
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x)
        
        # Logits
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            # Flatten for CrossEntropyLoss
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)
            
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """
        Autoregressive generation.
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx[:, -self.context_len:]
            
            # Forward
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-K Sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

def prepare_distributed_model(model: nn.Module, device_id: int) -> nn.Module:
    """Wraps model in DDP if distributed environment is active."""
    if dist.is_initialized():
        model = model.to(device_id)
        model = DDP(model, device_ids=[device_id])
        logger.info("Model wrapped in DistributedDataParallel")
    else:
        model = model.to(device_id if torch.cuda.is_available() else 'cpu')
    return model

if __name__ == "__main__":
    # Simulation of a training loop
    dist_config = DistributedConfig()
    dist_config.init_process_group()
    
    # Hyperparameters
    VOCAB_SIZE = 50257 # GPT-2 size
    D_MODEL = 768
    N_HEAD = 12
    N_LAYER = 12
    CONTEXT_LEN = 1024
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = AdvancedTransformer(VOCAB_SIZE, D_MODEL, N_HEAD, N_LAYER, CONTEXT_LEN)
    model = prepare_distributed_model(model, device_id=0 if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Dummy input
    x = torch.randint(0, VOCAB_SIZE, (4, 128)).to(device)
    logits, loss = model(x, targets=x)
    logger.info(f"Forward pass successful. Loss: {loss.item():.4f}")
