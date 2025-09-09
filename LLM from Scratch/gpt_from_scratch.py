import torch.nn.functional as F
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


# GELU
class GELU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x, approximate="tanh")


# Layer Norm
class LayerNorm(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.scale = nn.Parameter(torch.ones(cfg["embed_dim"]))
        self.shift = nn.Parameter(torch.zeros(cfg["embed_dim"]))
        self.eps = 1e-5

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm = (x - mean) / torch.sqrt(var + self.eps)

        return norm * self.scale + self.shift


# Feedforward Network (Position Wise)
class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(cfg["embed_dim"], 4 * cfg["embed_dim"]),  ## Expansion layer
            GELU(),
            nn.Linear(4 * cfg["embed_dim"], cfg["embed_dim"]),  ## Contraction layer
        )

    def forward(self, x):
        return self.layers(x)

# Multi-head Self Attention
class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_len, n_heads, dropout, qkv_bias):
        super().__init__()

        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = self.d_out // n_heads
        self.scale = self.head_dim**-0.5  # scaling factor
        
        # Linear projections for queries, keys, and values
        self.w_Q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_K = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_V = nn.Linear(d_in, d_out, bias=qkv_bias)

        # attention mask (autoregressive)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_len, context_len), diagonal=1).bool()
        )

        self.out_proj = nn.Linear(d_out, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n_tokens, e = x.shape

        # Linear projections
        q_MAT = self.w_Q(x)
        k_MAT = self.w_K(x)
        v_MAT = self.w_V(x)

        # batch ,n_tokens,emd -> batch, n_tokens, n_head, head_dim
        q_MAT = q_MAT.view(b, n_tokens, self.n_heads, self.head_dim)
        k_MAT = k_MAT.view(b, n_tokens, self.n_heads, self.head_dim)
        v_MAT = v_MAT.view(b, n_tokens, self.n_heads, self.head_dim)

        # batch, n_tokens, n_head, head_dim -> batch, n_head, n_tokens, head_dim
        q_MAT = q_MAT.transpose(1, 2)
        k_MAT = k_MAT.transpose(1, 2)
        v_MAT = v_MAT.transpose(1, 2)

        attn = (q_MAT @ k_MAT.transpose(-2, -1)) * self.scale

        # batch, n_head, n_tokens, n_tokens
        attn = attn.masked_fill(self.mask[:n_tokens, :n_tokens], float("-inf"))
        attn_weights = torch.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # batch, n_head, n_tokens, n_tokens -> batch, n_head, n_tokens, head_dim
        attn_scores = attn_weights @ v_MAT

        # batch, n_head, n_tokens, head_dim -> batch, n_tokens, n_head, head_dim
        attn_scores = attn_scores.transpose(1, 2)
        attn_scores = attn_scores.contiguous().view(b, n_tokens, self.d_out)
        attn_scores = self.out_proj(attn_scores)

        return attn_scores


# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.norm1 = LayerNorm(cfg)
        self.norm2 = LayerNorm(cfg)

        self.drop1 = nn.Dropout(cfg["drop_rate"])
        self.drop2 = nn.Dropout(cfg["drop_rate"])

        self.attn = CausalAttention(
            d_in=cfg["embed_dim"],
            d_out=cfg["embed_dim"],
            context_len=cfg["context_len"],
            n_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )

        self.ff = MLP(cfg)

    def forward(self, x):

        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop1(x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop2(x)
        x = residual + x

        return x

# GPT-2 Model
class GPT2(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.out_head = nn.Linear(cfg["embed_dim"], cfg["vocab_size"], bias=False)
        self.final_norm = LayerNorm(cfg)
        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layer"])]
        )

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["embed_dim"])
        self.pos_emb = nn.Embedding(cfg["context_len"], cfg["embed_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        b, n_tokens = x.shape

        tok = self.tok_emb(x)
        pos = self.pos_emb(torch.arange(n_tokens, device=x.device))

        x = tok + pos

        x = self.drop_emb(x)

        for tf_block in self.transformer_layers:
            x = tf_block(x)

        x = self.final_norm(x)

        logits = self.out_head(x)
        return logits
    
    # Text Generation (Autoregressive)  
    @torch.no_grad()
    def generate(
        self,
        tokens,
        max_new_tokens,
        context_size,
        temperature=1.0,
        top_k=None,
        eos_id=None,
    ):

        self.eval()
        for _ in range(max_new_tokens):

            toks = (
                tokens[:, -context_size:] if tokens.size(1) > context_size else tokens
            )

            logits = self(toks)
            logits = logits[:, -1, :]

            if top_k is not None:
                topk, _ = torch.topk(logits, top_k)
                min_val = topk[:, -1]
                logits = torch.where(
                    logits < min_val, torch.tensor(float("-inf")), logits
                )

            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)

            else:
                next_tok = torch.argmax(logits, dim=-1, keepdim=True)

            if next_tok == eos_id:
                break

            tokens = torch.cat([tokens, next_tok], dim=1)

        return tokens
