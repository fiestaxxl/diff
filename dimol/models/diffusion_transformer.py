
from dimol.models.layers import (
    Modulation,
    OutLayer,
    RoPE1D,
    TimeEmbeddings,
    TransformerEncoderBlock,
)

import json
import math
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict, field

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

    # ---- self-conditioning ----
    # When true the denoiser takes a second input of the same shape as the latent: its
    # own previous estimate of x0. Training feeds it on half the steps and zeros on the
    # rest, sampling feeds the estimate from the previous solver step. The extra half of
    # up_proj is initialised to zero, so a fresh model behaves exactly as before.
    self_conditioning: bool = False

    # ---- readout ----
    # The readout maps a latent back to token logits, and the latents it has to invert
    # are the embedding table's own rows, which move throughout training: the table's
    # norm grows about fiftyfold over a run. Tying the readout to the table makes it a
    # dot product against the current embeddings instead of a separate matrix chasing
    # them, which is the standard fix in language models and a candidate explanation for
    # the run-to-run spread measured here.
    tie_readout: bool = False
 
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
        self.self_conditioning = bool(getattr(config, "self_conditioning", False))
        in_dim = config.emb_dim * (2 if self.self_conditioning else 1)
        self.up_proj = nn.Linear(in_dim, config.model_dim)

        self.time_embeddings = TimeEmbeddings(config)

        self.text_rope_embeddings = RoPE1D(config, dim=config.model_dim // config.num_heads)
        self.text_transformer_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(config)
                for _ in range(config.num_text_blocks)
            ]
        )

        self.noise_head = OutLayer(config)

        self.out_proj = nn.Linear(config.emb_dim, config.vocab_size, bias=True)
        #self.out_proj = NormalizedLinear(config.emb_dim, config.vocab_size, bias=True)

        self.apply(self._init_params)
        self._init_special_layers()
        if bool(getattr(config, "tie_readout", False)):
            # one parameter, two uses: the readout is now a similarity to the embeddings
            self.out_proj.weight = self.token_embedding.weight
        if self.self_conditioning:
            # the second input starts with no influence at all
            with torch.no_grad():
                self.up_proj.weight[:, config.emb_dim:].zero_()

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

    def forward(
        self,
        input_embeddings: torch.Tensor,        # (B, T, C)
        time: torch.Tensor,             # (B,) or (B, 1) in [0, 1]
        attention_mask: Optional[torch.Tensor] = None,
        x0_self: Optional[torch.Tensor] = None,  # previous x0 estimate, (B, T, C)
    ):
        if self.self_conditioning:
            if x0_self is None:
                x0_self = torch.zeros_like(input_embeddings)
            input_embeddings = torch.cat((input_embeddings, x0_self.detach()), dim=-1)
        elif x0_self is not None:
            raise ValueError(
                "x0_self was passed but model.self_conditioning is off; "
                "the model has no input for it"
            )

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
