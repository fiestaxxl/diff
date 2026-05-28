
from dimol.models.nn import TimeEmbeddings, Modulation, TextEmbeddings, RoPE1D, TransformerEncoderBlock, TransformerDecoderBlock, OutLayer, FeedForward, NormalizedLinear

import json
import math
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict, field
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class TransformerConfig:
    # ---- core dims ----
    model_dim: int = 768          # main residual stream width 
    emb_dim: int = 32
    time_dim: int = 768           # time embedding dim (often == model_dim)
    num_heads: int = 12
    num_text_blocks: int = 12
 
    # ---- vocab / sequence ----
    vocab_size: int = 32000
    pad_idx: int = 0
    max_pos: int = 1024
 
    # ---- regularization ----
    attn_dropout: float = 0.0
 
    # ---- alias kept for backward compat with code using emb_dim ----
    # @property
    # def emb_dim(self) -> int:
    #     return self.emb_dim
 
    def to_dict(self) -> dict:
        d = asdict(self)
        return d
 
    @classmethod
    def from_dict(cls, d: dict) -> "TransformerConfig":
        # filter unknown keys defensively
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})
 
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
 
    @classmethod
    def load(cls, path: str | Path) -> "TransformerConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


class DiffusionTransformer(nn.Module):
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "model.safetensors"          # falls back to .pt if safetensors not installed

    def __init__(
        self,
        config: TransformerConfig
    ):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.emb_dim, padding_idx=config.pad_idx)
        self.up_proj = nn.Linear(config.emb_dim, config.model_dim)

        self.time_embeddings = TimeEmbeddings(config)

        self.text_rope_embeddings = RoPE1D(config, dim=config.model_dim // config.num_heads)
        self.text_transformer_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(config)
                for _ in range(config.num_text_blocks)
            ]
        )

        self.noise_head = OutLayer(config)

        # self.out_proj = nn.Linear(config.emb_dim, config.vocab_size, bias=True)
        self.out_proj = NormalizedLinear(config.emb_dim, config.vocab_size, bias=True)
        # self.out_proj.weight = self.token_embedding.weight

        self.apply(self._init_params)
        self._init_special_layers() 

    def _init_params(self, module):
        if isinstance(module, nn.Linear):
            # GPT-2 style: Normal(0, 0.02). Works well for transformer-scale models.
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            # Most LayerNorms in your model are elementwise_affine=False, so they have no params.
            # The few that do (TextEmbeddings.norm) get the standard 1/0 init.
            if getattr(module, "weight", None) is not None:
                nn.init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def _init_special_layers(self):
        n = self.config.num_text_blocks
        residual_std = 0.02 / math.sqrt(2 * n)
        for block in self.text_transformer_blocks:
            nn.init.normal_(block.self_attention.out_layer.weight, std=residual_std)
            nn.init.normal_(block.feed_forward.out_layer.weight,   std=residual_std)
            nn.init.zeros_(block.self_attention.out_layer.bias)   # bias still zero
        for m in self.modules():
            if isinstance(m, Modulation):
                nn.init.zeros_(m.out_layer.weight)

    def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process=True):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        seen = set()
        decay_params, nodecay_params = [], []
        for n, p in param_dict.items():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if "token_embedding" in n:
                nodecay_params.append(p)
            elif p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)
            # (decay_params if p.dim() >= 2 else nodecay_params).append(p)

        optim_groups = [
            {"params": decay_params,   "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        if master_process:
            nd = sum(p.numel() for p in decay_params)
            nn_ = sum(p.numel() for p in nodecay_params)
            print(f"decayed: {len(decay_params)} tensors, {nd:,} params")
            print(f"non-decayed: {len(nodecay_params)} tensors, {nn_:,} params")

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")

        return torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )

    def forward(
        self,
        input_embeddings: torch.Tensor,        # (B, T, C)
        time: torch.Tensor,             # (B,) or (B, 1) in [0, 1]
        attention_mask: Optional[torch.Tensor] = None,
    ):

        input_embeddings = self.up_proj(input_embeddings)

        B, T, C = input_embeddings.shape
        device = input_embeddings.device
 
        # text_embed = self.token_embedding(input_ids)
        text_embed = input_embeddings
        time_embed = self.time_embeddings(time)
        rope_pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        # rope = self.text_rope_embeddings(rope_pos)
        cos, sin = self.text_rope_embeddings(rope_pos)
 
        for block in self.text_transformer_blocks:
            # text_embed = block(text_embed, time_embed, rope, attention_mask)
            text_embed = block(text_embed, time_embed, (cos, sin), attention_mask)
 

        return self.noise_head(text_embed, time_embed)

    # -------- save / load --------
    def save_model(self, save_dir: str | Path) -> None:
        """Save config + weights to a directory."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
 
        # Save config
        self.config.save(save_dir / self.CONFIG_NAME)
 
        # Save weights (prefer safetensors, fall back to torch.save)
        try:
            from safetensors.torch import save_file
            # safetensors does not support shared (tied) tensors -> need contiguous, no shared ref
            state = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
            # drop the duplicate (lm_head shares storage with token_embedding)
            state.pop("out_layer.lm_head.weight", None)
            save_file(state, str(save_dir / self.WEIGHTS_NAME))
        except ImportError:
            torch.save(self.state_dict(), save_dir / "model.pt")
 
    @classmethod
    def from_pretrained(
        cls,
        load_dir: str | Path,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> "DiffusionTransformer":
        """Load config + weights from a directory."""
        load_dir = Path(load_dir)
        config = TransformerConfig.load(load_dir / cls.CONFIG_NAME)
        model = cls(config)
 
        safetensors_path = load_dir / cls.WEIGHTS_NAME
        pt_path = load_dir / "model.pt"
 
        if safetensors_path.exists():
            from safetensors.torch import load_file
            state = load_file(str(safetensors_path), device=str(map_location))
            # lm_head was dropped on save; re-tie after load
            missing_ok = {"out_proj.weight"}
            missing, unexpected = model.load_state_dict(state, strict=False)
            unexpected_unwanted = [k for k in unexpected]
            missing_unwanted = [k for k in missing if k not in missing_ok]
            if strict and (unexpected_unwanted or missing_unwanted):
                raise RuntimeError(
                    f"State dict mismatch. Missing: {missing_unwanted}, "
                    f"Unexpected: {unexpected_unwanted}"
                )
            # re-tie
            # model.out_proj.weight = model.token_embedding.weight
        elif pt_path.exists():
            state = torch.load(pt_path, map_location=map_location)
            model.load_state_dict(state, strict=strict)
            # model.out_proj.weight = model.token_embedding.weight
        else:
            raise FileNotFoundError(
                f"No weights found in {load_dir} "
                f"(looked for {cls.WEIGHTS_NAME} and model.pt)"
            )
 
        model.to(map_location)
        return model

# class ConditionalDiffusionTransformer(nn.Module):
#     def __init__(
#         self,
#         in_text_dim=3584,
#         text_cond_dim = 768,
#         time_dim=512,
#         model_dim=2048,
#         ff_dim=5120,
#         num_text_blocks=2,
#         head_dim=512,
#         out_dim = 124,
#         attention_engine="auto",
#         vocab_size = 123,
#         pad_idx = 0
#     ):
#         super().__init__()
#         self.head_dim = head_dim
#         self.model_dim = model_dim

#         self.init_params = {
#             'in_text_dim': in_text_dim,
#             'text_cond_dim': text_cond_dim,
#             'time_dim': time_dim,
#             'model_dim': model_dim,
#             'ff_dim': ff_dim,
#             'num_text_blocks': num_text_blocks,
#             'head_dim': head_dim,
#             'out_dim': out_dim,
#             'attention_engine': attention_engine,
#             'vocab_size': vocab_size,
#             'pad_idx': pad_idx
#         }

#         self.token_embedding = nn.Embedding(vocab_size, in_text_dim, padding_idx=pad_idx)
#         self.time_embeddings = TimeEmbeddings(model_dim, time_dim)
#         self.text_embeddings = TextEmbeddings(in_text_dim, model_dim)
#         self.text_cond_proj = TextEmbeddings(text_cond_dim, model_dim)

#         self.lm_head = nn.Linear(in_text_dim, vocab_size, bias=False)
#         self.lm_head.weight = self.token_embedding.weight

#         self.null_text = nn.Parameter(
#             torch.zeros(1, 1, text_cond_dim)
#         )
#         nn.init.normal_(self.null_text, std=0.02)

#         # assert self.lm_head.weight.data_ptr() == self.token_embedding.weight.data_ptr()
#         # assert self.lm_head.bias is None

#         self.text_rope_embeddings = RoPE1D(head_dim)
#         self.text_transformer_blocks = nn.ModuleList(
#             [
#                 TransformerDecoderBlock(model_dim, time_dim, ff_dim, head_dim, attention_engine)
#                 for _ in range(num_text_blocks)
#             ]
#         )

#         self.out_layer = OutLayer(model_dim, time_dim, out_dim)

#     # @torch.compile()
#     def before_text_transformer_blocks(self, 
#                                        text_embed, 
#                                        time, 
#                                        text_rope_pos,
#                                        text_cond
#                                        ):
#         text_embed = self.text_embeddings(text_embed)
#         time_embed = self.time_embeddings(time)
#         text_rope = self.text_rope_embeddings(text_rope_pos)
#         text_cond = self.text_cond_proj(text_cond)
#         return text_embed, time_embed, text_rope, text_cond

#     def after_blocks(self, text_embed, time_embed):
#         x = self.out_layer(text_embed, time_embed)
#         return x

#     def embed_tokens(self, x):
#         return self.token_embedding(x)

#     def get_logits(self, hidden_repr):

#         return self.lm_head(hidden_repr)


#     def decode_latent(self, x):
#         emb_table = self.token_embedding.weight   # (V, D)

#         # [B, T, 1, D] - [1, 1, V, D]
#         diff = x.unsqueeze(2) - emb_table.unsqueeze(0).unsqueeze(0)
#         dist = (diff ** 2).sum(dim=-1)  # [B, T, V]

#         token_ids = dist.argmin(dim=-1)  # [B, T]
#         return token_ids

#     def forward(
#         self,
#         text_embed,
#         time,
#         text_cond,
#         attention_mask=None,
#         key_padding_mask=None
#     ):
#         # if text_embed.dim() == 2:
#         #     text_embed = text_embed.unsqueeze(1)

#         if time.dim()>2:
#             time = time.squeeze(2)
            
#         seq_len = text_embed.shape[1]   # NOW seq_len = 1
#         text_rope_pos = torch.arange(seq_len).unsqueeze(0)

#         text_embed, time_embed, text_rope, text_cond = self.before_text_transformer_blocks(
#             text_embed, time, text_rope_pos, text_cond)

#         for text_transformer_block in self.text_transformer_blocks:
#             text_embed = text_transformer_block(text_embed, time_embed, text_cond, text_rope, attention_mask, key_padding_mask)
            
#         x = self.after_blocks(text_embed, time_embed)
#         return x


#     @classmethod
#     def from_trained(cls, model_path, device=None, compile_model=True):
#         """
#         Load a trained model instance from a single file.
        
#         Args:
#             model_path: Path to the single model file (.pt or .pth)
#             device: Device to load model to (e.g., 'cuda', 'cpu')
#             compile_model: Whether to compile the model with torch.compile
            
#         Returns:
#             Loaded model instance with trained weights
#         """
#         # Set device
#         if device is None:
#             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         # Load the single file
#         model_path = Path(model_path)
#         if not model_path.exists():
#             raise FileNotFoundError(f"Model file not found: {model_path}")
        
#         # Load the checkpoint
#         checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
#         # Extract components
#         if 'model_data' in checkpoint:
#             # New format with packed data
#             model_data = checkpoint['model_data']
#             init_params = model_data['init_params']
#             state_dict = model_data['state_dict']
#             metadata = model_data.get('metadata', {})
#         else:
#             # Legacy format
#             init_params = checkpoint['init_params']
#             state_dict = checkpoint['state_dict']
#             metadata = checkpoint.get('metadata', {})
        
#         # Create model instance with the saved parameters
#         model = cls(**init_params)
        
#         # Load weights
#         model.load_state_dict(state_dict)
        
#         # Move to device
#         model = model.to(device)
        
#         # Set to eval mode
#         model.eval()
        
#         # Compile if requested
#         if compile_model and hasattr(torch, 'compile'):
#             model = torch.compile(model)
        
#         print(f"✓ Model loaded successfully from: {model_path}")
#         print(f"  Device: {device}")
#         print(f"  Compiled: {compile_model}")
#         print(f"  Model info: {init_params}")
        
#         # Return metadata along with model if needed
#         setattr(model, 'loaded_metadata', metadata)
        
#         return model

#     def save_trained(self, save_path, metadata=None):
#         """
#         Save the trained model to a single file containing everything needed.
        
#         Args:
#             save_path: Path where to save the single model file (.pt or .pth)
#             metadata: Optional dictionary with additional metadata to save
            
#         Returns:
#             Path to the saved file
#         """
#         save_path = Path(save_path+'/checkpoint.pth')
#         save_path.parent.mkdir(parents=True, exist_ok=True)
        
#         # Prepare metadata
#         if metadata is None:
#             metadata = {}
        
#         # Add automatic metadata
#         auto_metadata = {
#             'save_timestamp': torch.datetime.now().isoformat() if hasattr(torch, 'datetime') else 'N/A',
#             'pytorch_version': torch.__version__,
#             'model_class': self.__class__.__name__,
#             'model_info': {
#                 'total_params': sum(p.numel() for p in self.parameters()),
#                 'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
#                 'model_dim': self.model_dim,
#                 'head_dim': self.head_dim,
#             }
#         }
        
#         # Merge with user metadata
#         full_metadata = {**auto_metadata, **metadata}
        
#         # Create the packed model data
#         model_data = {
#             'init_params': self.init_params,
#             'state_dict': self.state_dict(),
#             'metadata': full_metadata,
#             'model_class': self.__class__.__name__,
#         }
        
#         # Pack into final checkpoint
#         checkpoint = {
#             'model_data': model_data,
#             'version': '1.0',
#             'format': 'single_file'
#         }
        
#         # Save to single file
#         torch.save(checkpoint, save_path)
        
#         print(f"✓ Model saved successfully to single file: {save_path}")
#         # print(f"  File size: {save_path.stat().st_size / (1024*1024):.2f} MB")
#         # print(f"  Contains: init_params + state_dict + metadata")
        
#         return save_path





class DenoiserModel(nn.Module):
    def __init__(self, eps_model, path, regime='epsilon'):
        super().__init__()
        self.eps_model = eps_model
        self.path = path
        self.regime = regime

    def forward(self, x, t, **kwargs):
        alpha_t = torch.clamp(self.path.alpha(t), min=1e-3)
        beta_t  = torch.clamp(self.path.beta(t),  min=1e-3)
        t_in    = t.squeeze(-1)
        pred    = self.eps_model(x, t_in, **kwargs)

        if self.regime == 'epsilon':
            x0_pred = (x - beta_t * pred) / alpha_t
        elif self.regime == 'x':
            x0_pred = pred
        else:
            raise ValueError(f"Expected regime to be 'epsilon' or 'x', got {self.regime}")

        score = (alpha_t * x0_pred - x) / (beta_t ** 2)
        return score

    # def get_logits(self, x):
    #     return self.eps_model.get_logits(x)

    # def embed_tokens(self, x):
    #     return self.eps_model.embed_tokens(x)

# class SmilesEnergyModel(nn.Module):
#     def __init__(self, D_diff, checkpoint="unikei/bert-base-smiles"):
#         super().__init__()

#         # store init params explicitly
#         self.init_params = {
#             "checkpoint": checkpoint,
#             'D_diff': D_diff
#         }

#         self.encoder = BertModel.from_pretrained(checkpoint)
#         self.encoder.embeddings.word_embeddings.requires_grad_(False)
#         self.encoder.embeddings.token_type_embeddings.requires_grad_(False)
#         self.projector = nn.Linear(D_diff, self.encoder.config.hidden_size)
#         self.energy_head = nn.Linear(self.encoder.config.hidden_size, 1)

#     def forward(self, x, attention_mask=None):
#         B, L, _ = x.shape
#         device = x.device
#         x_bert = self.projector(x)
#         position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
#         out = self.encoder(
#             inputs_embeds=x_bert,
#             attention_mask=attention_mask,
#             position_ids=position_ids
#         )
#         pooled = out.pooler_output          # (B, H)
#         energy = self.energy_head(pooled)   # (B, 1)
#         return energy.squeeze(-1)           # (B,)

#     # ------------------------------------------------------------------
#     # Saving
#     # ------------------------------------------------------------------
#     def save_trained(self, save_path, metadata=None):
#         """
#         Save encoder + energy head in a single file.
#         """
#         save_path = Path(save_path)
#         if save_path.is_dir():
#             save_path = save_path / "checkpoint.pth"
#         save_path.parent.mkdir(parents=True, exist_ok=True)

#         if metadata is None:
#             metadata = {}

#         auto_metadata = {
#             "pytorch_version": torch.__version__,
#             "model_class": self.__class__.__name__,
#             "total_params": sum(p.numel() for p in self.parameters()),
#             "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
#         }

#         checkpoint = {
#             "init_params": self.init_params,
#             "state_dict": self.state_dict(),   # encoder + head
#             "metadata": {**auto_metadata, **metadata},
#             "format_version": "1.0",
#         }

#         torch.save(checkpoint, save_path)
#         print(f"✓ Model saved to {save_path}")
#         return save_path

#     # ------------------------------------------------------------------
#     # Loading
#     # ------------------------------------------------------------------
#     @classmethod
#     def from_trained(cls, model_path, device=None, compile_model=False):
#         """
#         Restore a fully trained energy model (encoder included).
#         """
#         model_path = Path(model_path)
#         if not model_path.exists():
#             raise FileNotFoundError(model_path)

#         if device is None:
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         checkpoint = torch.load(model_path, map_location=device)

#         init_params = checkpoint["init_params"]
#         state_dict = checkpoint["state_dict"]
#         metadata = checkpoint.get("metadata", {})

#         model = cls(**init_params)
#         model.load_state_dict(state_dict, strict=True)
#         model.to(device)
#         model.eval()

#         if compile_model and hasattr(torch, "compile"):
#             model = torch.compile(model)

#         model.loaded_metadata = metadata

#         print(f"✓ Model loaded from {model_path}")
#         print(f"  Device: {device}")
#         print(f"  Compiled: {compile_model}")

#         return model

#     # @staticmethod
#     # def gamma_schedule(t, gamma_max=0.5, t0=0.6):
#     #     return gamma_max * ((t - t0).clamp(min=0))
        
#     @staticmethod
#     def gamma_schedule(t, gamma_max=0.5, t0=0.6):
#         return torch.where(t < t0, torch.tensor(0.0), torch.tensor(gamma_max))


# class EmbedProjector(nn.Module):
#     def __init__(self, D_diff, D_bert):
#         super().__init__()
#         self.proj = nn.Linear(D_diff, D_bert)

#     def forward(self, x_diff):
#         return self.proj(x_diff)

# class GrammarCorrector(nn.Module):
#     def __init__(self, emb_dim, num_layers=2, ff_dim=512, num_heads=4, max_len=128):
#         super().__init__()
#         self.layers = nn.ModuleList([
#             nn.TransformerEncoderLayer(
#                 d_model=emb_dim,
#                 nhead=num_heads,
#                 dim_feedforward=ff_dim,
#                 batch_first=True
#             ) for _ in range(num_layers)
#         ])
#         # self.out_proj = nn.Linear(emb_dim, emb_dim)
#         self.out_proj = nn.Linear(emb_dim, 1)
#         self.init_params = {
#             'emb_dim': emb_dim,
#             'num_layers': num_layers,
#             'ff_dim': ff_dim,
#             'num_heads': num_heads,
#             "max_len": max_len,
#         }
#         self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
#         self.pos_emb = nn.Parameter(
#             torch.randn(1, max_len + 1, emb_dim) * 0.02
#         )


#     def forward(self, x, mask=None):

#         B, L, D = x.shape
#         cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
#         x = torch.cat([cls_token, x], dim=1)       # (B, L+1, D)

#         cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
#         if mask is not None:
#             pad_mask = torch.cat([cls_mask, mask], dim=1)  # (B, L+1)
#         else:
#             pad_mask = None
        
#         # residual = x
#         x = x + self.pos_emb[:, : L + 1]
#         for layer in self.layers:
#             x = layer(x, src_key_padding_mask=pad_mask)

#         cls_out = x[:, 0] 

#         return self.out_proj(cls_out)


#     @classmethod
#     def from_trained(cls, model_path, device=None, compile_model=True):
#         """
#         Load a trained model instance from a single file.
        
#         Args:
#             model_path: Path to the single model file (.pt or .pth)
#             device: Device to load model to (e.g., 'cuda', 'cpu')
#             compile_model: Whether to compile the model with torch.compile
            
#         Returns:
#             Loaded model instance with trained weights
#         """
#         # Set device
#         if device is None:
#             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         # Load the single file
#         model_path = Path(model_path)
#         if not model_path.exists():
#             raise FileNotFoundError(f"Model file not found: {model_path}")
        
#         # Load the checkpoint
#         checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
#         # Extract components
#         if 'model_data' in checkpoint:
#             # New format with packed data
#             model_data = checkpoint['model_data']
#             init_params = model_data['init_params']
#             state_dict = model_data['state_dict']
#             metadata = model_data.get('metadata', {})
#         else:
#             # Legacy format
#             init_params = checkpoint['init_params']
#             state_dict = checkpoint['state_dict']
#             metadata = checkpoint.get('metadata', {})
        
#         # Create model instance with the saved parameters
#         model = cls(**init_params)
        
#         # Load weights
#         model.load_state_dict(state_dict)
        
#         # Move to device
#         model = model.to(device)
        
#         # Set to eval mode
#         model.eval()
        
#         # Compile if requested
#         if compile_model and hasattr(torch, 'compile'):
#             model = torch.compile(model)
        
#         print(f"✓ Model loaded successfully from: {model_path}")
#         print(f"  Device: {device}")
#         print(f"  Compiled: {compile_model}")
#         print(f"  Model info: {init_params}")
        
#         # Return metadata along with model if needed
#         setattr(model, 'loaded_metadata', metadata)
        
#         return model

#     def save_trained(self, save_path, metadata=None):
#         """
#         Save the trained model to a single file containing everything needed.
        
#         Args:
#             save_path: Path where to save the single model file (.pt or .pth)
#             metadata: Optional dictionary with additional metadata to save
            
#         Returns:
#             Path to the saved file
#         """
#         save_path = Path(save_path+'/checkpoint.pth')
#         save_path.parent.mkdir(parents=True, exist_ok=True)
        
#         # Prepare metadata
#         if metadata is None:
#             metadata = {}
        
#         # Add automatic metadata
#         auto_metadata = {
#             'save_timestamp': torch.datetime.now().isoformat() if hasattr(torch, 'datetime') else 'N/A',
#             'pytorch_version': torch.__version__,
#             'model_class': self.__class__.__name__,
#             'model_info': {
#                 'total_params': sum(p.numel() for p in self.parameters()),
#                 'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
#             }
#         }
        
#         # Merge with user metadata
#         full_metadata = {**auto_metadata, **metadata}
        
#         # Create the packed model data
#         model_data = {
#             'init_params': self.init_params,
#             'state_dict': self.state_dict(),
#             'metadata': full_metadata,
#             'model_class': self.__class__.__name__,
#         }
        
#         # Pack into final checkpoint
#         checkpoint = {
#             'model_data': model_data,
#             'version': '1.0',
#             'format': 'single_file'
#         }
        
#         # Save to single file
#         torch.save(checkpoint, save_path)
        
#         print(f"✓ Model saved successfully to single file: {save_path}")
#         # print(f"  File size: {save_path.stat().st_size / (1024*1024):.2f} MB")
#         # print(f"  Contains: init_params + state_dict + metadata")
        
#         return save_path



# class SymbolicGrammarEnergy(nn.Module):
#     def __init__(
#         self,
#         get_logits_fn,
#         vocab,
#         lambda_paren=1.0,
#         lambda_ring=1.0,
#         lambda_trans=1.0,
#         transition_mask=None,  # (V, V) binary
#     ):
#         super().__init__()
#         self.get_logits = get_logits_fn
#         self.vocab = vocab

#         self.lambda_paren = lambda_paren
#         self.lambda_ring = lambda_ring
#         self.lambda_trans = lambda_trans

#         self.transition_mask = transition_mask

#         # cache token ids
#         self.id_open  = vocab["("]
#         self.id_close = vocab[")"]

#         self.ring_ids = [vocab[str(i)] for i in range(1, 10) if str(i) in vocab]

#     def paren_energy(self, probs):
#         """
#         probs: (B, L, V)
#         """
#         p_open  = probs[..., self.id_open]
#         p_close = probs[..., self.id_close]

#         balance = torch.cumsum(p_open - p_close, dim=1)
#         penalty = nn.functional.relu(-balance)

#         return penalty.mean()


#     def ring_energy(self, probs):
#         energy = torch.zeros(1, device=probs.device, dtype=probs.dtype)
#         for rid in self.ring_ids:
#             count = probs[..., rid].sum(dim=1)
#             soft_parity = torch.sin(torch.pi * count)  # 0 when even, ±1 when odd
#             energy += soft_parity.abs().mean()  
#             #energy += ((count % 2.0) - 0.0).abs().mean()
#         return energy

#     def transition_energy(self, probs):
#         """
#         probs: (B, L, V)
#         """
#         if self.transition_mask is None:
#             return torch.zeros(1, device=probs.device, dtype=probs.dtype)

#         p_i = probs[:, :-1, :, None]   # (B,L-1,V,1)
#         p_j = probs[:, 1:, None, :]    # (B,L-1,1,V)

#         illegal = 1.0 - self.transition_mask
#         viol = (p_i * p_j * illegal).sum(dim=(-1, -2))

#         return viol.mean()

#     @staticmethod
#     def gamma_schedule(t, gamma_max=0.5, t0=0.7):
#         return gamma_max * ((t - t0).clamp(min=0) / (1 - t0))
        
#     def forward(self, logits):
#         #logits = self.get_logits(x)
#         # probs = nn.functional.log_softmax(logits, dim=-1)

#         probs = nn.functional.gumbel_softmax(
#             logits, tau=0.3, hard=False, dim=-1
#         )

#         E = torch.zeros(1, device=probs.device, dtype=probs.dtype)
#         E += self.lambda_paren * self.paren_energy(probs)
#         E += self.lambda_ring  * self.ring_energy(probs)
#         E += self.lambda_trans * self.transition_energy(probs)

#         return E

#     def converge(self, x_T, k):


#         #num_samples = x_T.shape[0]
#         #t = torch.ones(num_samples,1,1).to(x_T.device)
#         #gamma_t = self.gamma_schedule(t)
#         #x = x_T.clone()

#         logits_0 = self.get_logits(x_T).detach()
#         logits = logits_0.clone().detach().requires_grad_(True)
#         lambda_anchor = 1e-4
#         lr = 1.0

#         with torch.enable_grad():
#             for i in range(k):
#                 #x.requires_grad_(True)
#                 #E = self(x)
#                 E = self(logits)

#                 loss = E + lambda_anchor * (logits - logits_0).pow(2).mean()
#                 grad = torch.autograd.grad(loss, logits)[0]
#                 grad = grad / (grad.norm(dim=-1, keepdim=True) + 1e-6)
#                 logits = logits - lr * grad

#                 if i % 10 == 0:
#                     print(
#                         f"iter {i} | loss={loss.item():.4f} | "
#                         f"grad_max={grad.abs().max().item():.3e}"
#                     )
#                 logits = logits.detach().requires_grad_(True)

#                 # grad_E = torch.autograd.grad(E, x)[0]   # E must be scalar
#                 # grad_E = grad_E / (grad_E.norm(dim=-1, keepdim=True) + 1e-6)
#                 # # print(grad_E)
#                 # # print(grad_E.max())
#                 # x =  x - 1 * grad_E
        
#         #return x
#         return logits

