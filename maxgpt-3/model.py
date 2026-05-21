import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

class MLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Expand to 4x, then back to hidden_dim
        self.fc1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * hidden_dim, hidden_dim)
    
    def forward(self, x):
        # x shape: (B, T, C)
        x = self.fc1(x)       # (B, T, 4C)
        x = self.gelu(x)      # (B, T, 4C)
        x = self.fc2(x)       # (B, T, C)
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, context_window, use_flash=True):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads     # e.g., 384 / 6 = 64
        self.hidden_dim = hidden_dim
        self.use_flash = use_flash 
        
        # One linear layer that produces Q, K, V all at once
        # Input C, output 3C — then we'll split into Q, K, V
        # (More efficient than three separate linear layers)
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        
        # After attention, project back to hidden_dim
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Precompute the causal mask: a (T, T) lower-triangular matrix of 1s.
        # Position i can attend to positions 0..i but not i+1, i+2, ..., T-1.
        # We use register_buffer because we want this tensor to move with the model
        # (e.g., to GPU) but NOT be a learnable parameter.
        mask = torch.tril(torch.ones(context_window, context_window))
        self.register_buffer("causal_mask", mask)
    
    def forward(self, x):
        # x shape: (B, T, C)
        B, T, C = x.shape
        
        # Project to Q, K, V all at once
        qkv = self.qkv_proj(x)                      # (B, T, 3C)
        
        # Split the last dimension into three pieces: Q, K, V
        # Each will be shape (B, T, C)
        q, k, v = qkv.split(C, dim=-1)
        # or equivalently: q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention:
        # We want to split C into (num_heads, head_dim) and move num_heads
        # to the second dimension so heads are processed in parallel.
        # 
        # (B, T, C) → (B, T, num_heads, head_dim) → (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # Now all three have shape (B, num_heads, T, head_dim)
        
        # Compute attention scores: Q @ K^T
        # (B, num_heads, T, head_dim) @ (B, num_heads, head_dim, T)
        # = (B, num_heads, T, T)
        # 
        # Divide by sqrt(head_dim) to stabilize gradients
        # (Without scaling, dot products grow large with head_dim, pushing
        # softmax into regions where gradients vanish)
        # Branch: use fused attention OR manual computation
        if self.use_flash:
            # Try Flash first, fall back to Efficient if unavailable. PyTorch picks
            # the first kernel in the list that supports the current hardware/build.
            # WSL2 Linux wheel → Flash. Windows wheel → falls back to Efficient.
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    is_causal=True   # auto-applies causal mask, no need for our custom one
                )
        else:
            # Manual attention (for presentations — shows what's actually happening)
            scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            scores = scores.masked_fill(self.causal_mask[:T, :T] == 0, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            out = attn @ v

        # Continue with existing reshape and output projection
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, context_window, use_flash=True):
        super().__init__()
        # Pre-norm style: normalize BEFORE the sub-layer, not after
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, num_heads, context_window, use_flash=use_flash)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim)
    
    def forward(self, x):
        # x shape: (B, T, C), stays (B, T, C) throughout
        # Each sub-layer is wrapped: residual + pre-norm
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, context_window, hidden_dim, num_heads, num_blocks, use_flash=True):
        super().__init__()
        
        self.context_window = context_window
        
        # Token embedding: vocab_size → hidden_dim
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        
        # Positional embedding: context_window → hidden_dim
        # We learn a separate embedding for each position 0, 1, ..., T-1
        self.pos_embed = nn.Embedding(context_window, hidden_dim)
        
        # Stack of N transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, context_window, use_flash=use_flash)
            for _ in range(num_blocks)
        ])
        
        # Final layer norm before the output projection
        self.ln_f = nn.LayerNorm(hidden_dim)
        
        # Output projection: hidden_dim → vocab_size
        # bias=False is standard practice for the lm_head
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
    
    def forward(self, token_ids, targets=None):
        # token_ids shape: (B, T)
        # targets shape: (B, T) — same shape, shifted by one position
        # If targets is provided, we compute loss; otherwise just return logits.
        
        B, T = token_ids.shape
        assert T <= self.context_window, "sequence longer than context window"
        
        # Position indices: [0, 1, 2, ..., T-1]
        positions = torch.arange(0, T, device=token_ids.device)
        
        # Look up embeddings
        tok_emb = self.token_embed(token_ids)      # (B, T, C)
        pos_emb = self.pos_embed(positions)        # (T, C)
        
        # Add them. pos_emb is (T, C); it broadcasts across the B dimension.
        x = tok_emb + pos_emb                       # (B, T, C)
        
        # Pass through all transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)                            # (B, T, C)
        
        # Project to vocabulary logits
        logits = self.lm_head(x)                    # (B, T, vocab_size)
        
        # Compute loss if targets provided (during training)
        if targets is not None:
            # Cross-entropy needs logits flattened to (B*T, vocab_size)
            # and targets flattened to (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            return logits, loss
        else:
            return logits, None

if __name__ == "__main__":
    # Quick shape + forward-pass test
    model = Transformer(
        vocab_size=16000,
        context_window=256,
        hidden_dim=384,
        num_heads=6,
        num_blocks=6,
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
    
    # Fake input batch: 2 sequences of 64 tokens, random IDs in vocab range
    token_ids = torch.randint(0, 16000, (2, 64))
    targets = torch.randint(0, 16000, (2, 64))
    
    # Test forward pass without targets (inference mode)
    logits, loss = model(token_ids)
    print(f"Logits shape: {logits.shape}     # should be (2, 64, 16000)")
    print(f"Loss (no targets): {loss}        # should be None")
    
    # Test forward pass with targets (training mode)
    logits, loss = model(token_ids, targets)
    print(f"Loss (with targets): {loss.item():.4f}   # should be around log(16000) ≈ 9.68")
