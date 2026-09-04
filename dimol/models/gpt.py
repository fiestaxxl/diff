"""Small autoregressive SMILES model — used as a grammar teacher for the
AR-guided diffusion sampling scheme. Trained independently from the diffusion
model on the same tokenized corpus, same tokenizer, same <bos>...<eos><pad> convention.
"""
from __future__ import annotations

import math
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
@dataclass
class SmilesARConfig:
    vocab_size: int = 512
    model_dim: int = 256
    n_heads:   int = 8
    n_layers:  int = 4
    max_pos:   int = 220          # >= max seq_len of any input
    pad_idx:   int = 0
    attn_dropout: float = 0.0
    tie_lm_head: bool = True      # LM head shares weights with token embedding

    def to_dict(self) -> dict: 
        return asdict(self)
        
    def save(self, path) -> None: 
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path) -> "SmilesARConfig":
        return cls(**json.loads(Path(path).read_text()))


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: SmilesARConfig):
        super().__init__()
        assert cfg.model_dim % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.model_dim // cfg.n_heads
        self.qkv = nn.Linear(cfg.model_dim, 3 * cfg.model_dim, bias=True)
        self.out_proj = nn.Linear(cfg.model_dim, cfg.model_dim, bias=True)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.attn_dropout = cfg.attn_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)   # (B, nh, L, hd)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q.float()).type_as(q)
        k = self.k_norm(k.float()).type_as(k)
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, L, C))


class FeedForward(nn.Module):
    def __init__(self, cfg: SmilesARConfig, mult: int = 4):
        super().__init__()
        h = mult * cfg.model_dim
        self.fc1 = nn.Linear(cfg.model_dim, h, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(h, cfg.model_dim, bias=False)
    def forward(self, x): return self.fc2(self.act(self.fc1(x)))


class GPTBlock(nn.Module):
    def __init__(self, cfg: SmilesARConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.model_dim)
        self.attn  = CausalSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.model_dim)
        self.ff    = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class SmilesAR(nn.Module):
    CONFIG_NAME  = "config.json"
    WEIGHTS_NAME = "model.safetensors"

    def __init__(self, cfg: SmilesARConfig):
        super().__init__()
        self.config = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.model_dim, padding_idx=cfg.pad_idx)
        self.pos_embedding   = nn.Embedding(cfg.max_pos,    cfg.model_dim)
        self.blocks  = nn.ModuleList([GPTBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm    = nn.LayerNorm(cfg.model_dim)
        self.lm_head = nn.Linear(cfg.model_dim, cfg.vocab_size, bias=False)
        if cfg.tie_lm_head:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_params)
        # residual-stream init (GPT-2 trick)
        n = cfg.n_layers
        for blk in self.blocks:
            nn.init.normal_(blk.attn.out_proj.weight, std=0.02 / math.sqrt(2 * n))
            nn.init.normal_(blk.ff.fc2.weight,        std=0.02 / math.sqrt(2 * n))

    def _init_params(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
            if m.padding_idx is not None:
                with torch.no_grad(): m.weight[m.padding_idx].zero_()
        elif isinstance(m, (nn.LayerNorm, nn.RMSNorm)):
            if getattr(m, "weight", None) is not None: nn.init.ones_(m.weight)
            if getattr(m, "bias",   None) is not None: nn.init.zeros_(m.bias)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (B, L) -> logits: (B, L, V).
        logits[:, i, :] = p(token at position i+1 | tokens[:, :i+1]) via causal mask.
        For grammar-guidance use, the prediction *for* position i is logits[:, i-1, :].
        """
        B, L = token_ids.shape
        pos = torch.arange(L, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.pos_embedding(pos)[None]
        for blk in self.blocks: 
            x = blk(x)
        x = self.norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=200, temperature=1.0,
                 top_k=None, eos_id=None) -> torch.Tensor:
        """Sequential sampling — useful for sanity-checking validity post-training."""
        self.eval()
        for _ in tqdm(range(max_new_tokens)):
            ids = prompt_ids[:, -self.config.max_pos:]
            logits = self(ids)[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            next_id = torch.multinomial(logits.softmax(-1), num_samples=1)
            prompt_ids = torch.cat([prompt_ids, next_id], dim=-1)
            if eos_id is not None and (next_id == eos_id).all():
                break
        return prompt_ids

    # ---------- save / load ----------
    def save_model(self, save_dir) -> None:
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        self.config.save(save_dir / self.CONFIG_NAME)
        try:
            from safetensors.torch import save_file
            state = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
            if self.config.tie_lm_head:
                state.pop("lm_head.weight", None)         # shared with token_embedding.weight
            save_file(state, str(save_dir / self.WEIGHTS_NAME))
        except ImportError:
            torch.save(self.state_dict(), save_dir / "model.pt")

    @classmethod
    def from_pretrained(cls, load_dir, map_location="cpu") -> "SmilesAR":
        load_dir = Path(load_dir)
        config = SmilesARConfig.load(load_dir / cls.CONFIG_NAME)
        model = cls(config)
        st = load_dir / cls.WEIGHTS_NAME
        pt = load_dir / "model.pt"
        if st.exists():
            from safetensors.torch import load_file
            state = load_file(str(st), device=str(map_location))
        else:
            state = torch.load(pt, map_location=map_location)
        model.load_state_dict(state, strict=False)
        if config.tie_lm_head:
            model.lm_head.weight = model.token_embedding.weight
        model.to(map_location)
        return model