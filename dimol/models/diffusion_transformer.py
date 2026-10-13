
from dimol.models.layers import (
    MultiheadCrossAttention,
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

    # ---- length conditioning ----
    # On a padded canvas the objective has a degenerate optimum at short sequences: two
    # thirds of the training positions are padding, so ending early lowers the loss, and
    # the runs that collapse that way score high on validity and badly on every
    # distribution metric. Conditioning at sampling time alone does not work, because the
    # model's plan for the canvas never depended on the length. This gives the model the
    # length as an input, embedded like the timestep and added to it, so the plan is
    # conditioned on it. The embedding starts at zero, so a fresh model is unchanged.
    length_conditioning: bool = False

    # ---- text conditioning ----
    # Width of the frozen text encoding, 768 for SciBERT. Zero means no text path and no
    # extra parameters at all. When set, every block gains a cross-attention whose output
    # projection is zeroed, so a checkpoint grown into this shape behaves exactly as it
    # did before the layers existed.
    text_dim: int = 0

    # Global caption conditioning. The caption currently reaches the model only through
    # per-position cross-attention, which is the right shape for local detail and the
    # wrong one for global facts: how big the molecule is, how many rings it has, what
    # charge it carries. This pools the caption once and adds it to the adaLN vector
    # beside the timestep, which is where every other global quantity already enters.
    # The projection is zero-initialised, so a model that has it behaves exactly as one
    # that does not until training moves it.
    text_global: bool = False

    # ---- caption encoder inside the loop ----
    # Path to a huggingface encoder to run inside the forward pass, instead of reading
    # states that were computed offline and frozen. Empty is the frozen path: no encoder,
    # no transformers import, and a state dict identical to the one every number in this
    # project came from.
    #
    # The motivation is measured rather than architectural. A caption-to-length head on
    # the frozen states plateaus at 6.4 tokens of error while its own training loss keeps
    # falling — limited by its inputs — and fine-tuning the same encoder on the same task
    # reaches 3.3. The states have lost information the task needs, and nothing downstream
    # of them can put it back.
    text_encoder: str = ""

    # How many of the encoder's top blocks take gradients. 0 freezes it completely, which
    # is what makes a live encoder a control against the cached states rather than a
    # second intervention; -1 trains every block. The encoder is 110M parameters against
    # the backbone's 47.9M, so this is the knob that sets both the step time and the
    # overfitting exposure on 25,574 pairs.
    text_encoder_trainable_blocks: int = 0

    # The embedding table is a fifth of the encoder and sees every caption token, so it is
    # what overfits first on a corpus this size. Frozen unless asked for.
    text_encoder_train_embeddings: bool = False

    # ---- content channel ----
    # The diffusion carries no representation of where the molecule ends. Measured
    # directly: with the canvas left unpinned, the tail latent sits about 0.08 from the
    # padding embedding and the same distance from the nearest real token, and that
    # profile is identical whether or not the readout was trained on padding. The model
    # appears to stop only because the length reaches the readout through adaLN.
    #
    # This adds one channel to the latent, diffused jointly with the embedding channels:
    # positive where content continues, negative where the molecule has ended. Termination
    # becomes something the model generates and can revise at every solver step, rather
    # than a number handed to it. The embedding table, the readout and the vocabulary are
    # untouched, so no checkpoint surgery is needed: the extra input column is zeroed, and
    # a model with the flag off is bit-identical to one from before it existed.
    content_channel: bool = False

    # ---- distinct padding ----
    # Every padding position currently carries the SAME target embedding, so the diffusion
    # must produce one identical value at ~70% of the canvas. Measured consequence: the
    # tail latent sits about 0.08 from the padding embedding and the same distance from the
    # nearest real token, with no structure to it, whether or not the readout was trained
    # on padding. Rainbow Padding (ICLR 2026) and VoidPadding reached the same conclusion
    # from the masked-diffusion side.
    #
    # pad_variants > 1 spreads padding over that many consecutive ids starting at
    # pad_variant_base, assigned cyclically by position, so consecutive padding positions
    # have different targets. 1 reproduces the original behaviour exactly.
    pad_variants: int = 1
    pad_variant_base: int = 0     # first id of the variant block; 0 means "pick the top of
                                  # the vocabulary automatically"
 
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
        self.content_channel = bool(getattr(config, "content_channel", False))
        # The latent is the embedding channels plus, optionally, the content channel;
        # self-conditioning doubles whatever that is.
        self.latent_dim = config.emb_dim + (1 if self.content_channel else 0)
        in_dim = self.latent_dim * (2 if self.self_conditioning else 1)
        self.up_proj = nn.Linear(in_dim, config.model_dim)
        if self.content_channel:
            self.content_head = nn.Linear(config.model_dim, 1)

        self.time_embeddings = TimeEmbeddings(config)
        self.length_conditioning = bool(getattr(config, "length_conditioning", False))
        if self.length_conditioning:
            self.length_embedding = nn.Embedding(config.max_pos + 1, config.time_dim)

        self.text_global = bool(getattr(config, "text_global", False))
        if self.text_global:
            if not config.text_dim:
                raise ValueError(
                    "model.text_global needs a text path; set model.text_dim"
                )
            self.text_global_proj = nn.Linear(config.text_dim, config.time_dim)

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
        if self.text_global:
            # Zero, so the global path contributes nothing until it earns it, the same
            # discipline as the cross-attention output projection and the length table.
            with torch.no_grad():
                self.text_global_proj.weight.zero_()
                self.text_global_proj.bias.zero_()
        if self.content_channel:
            with torch.no_grad():
                # Zero the columns that read the content channel, so a checkpoint grown
                # into this shape starts out ignoring it. Gradients still reach them, and
                # the head predicting the channel is zeroed for the same reason: its
                # target is real noise, so it learns from the first step regardless.
                self._zero_content_input_columns()
                self.content_head.weight.zero_()
                self.content_head.bias.zero_()
        if self.length_conditioning:
            with torch.no_grad():
                self.length_embedding.weight.zero_()
        if bool(getattr(config, "tie_readout", False)):
            # one parameter, two uses: the readout is now a similarity to the embeddings
            self.out_proj.weight = self.token_embedding.weight
        if self.self_conditioning:
            # the second input starts with no influence at all
            with torch.no_grad():
                self.up_proj.weight[:, config.emb_dim:].zero_()

        # Built last, and the ordering is load-bearing: self.apply(self._init_params)
        # above walks every submodule and would redraw the encoder's weights from
        # Normal(0, 0.02), undoing the pretraining that is the entire reason to load one.
        self.text_encoder = None
        if config.text_encoder:
            self.text_encoder = self._build_text_encoder(config)

    def _build_text_encoder(self, config: TransformerConfig) -> nn.Module:
        """Load the caption encoder and freeze all of it but the top blocks."""
        # Imported here and not at module scope, so a model without an encoder carries no
        # transformers dependency and the frozen path imports exactly what it did before.
        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(
            config.text_encoder,
            add_pooling_layer=False,
            # Dropout off. What this model is for is a deterministic export of caption
            # states, and trainable_blocks=0 is the control that prices a live encoder
            # against the cached one. With dropout active in train() mode neither holds,
            # and the project would quietly acquire stochastic conditioning it has never
            # used and never measured.
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        if int(encoder.config.hidden_size) != int(config.text_dim):
            raise ValueError(
                f"model.text_encoder has hidden size {encoder.config.hidden_size} but "
                f"model.text_dim is {config.text_dim}; the cross-attention reads the "
                "encoder's last hidden state directly, so the two have to agree"
            )

        # Frozen here, in the constructor, because build_optimizer runs after it:
        # param_groups skips tensors with requires_grad=False, so a frozen block never
        # enters the optimizer and never becomes a parameter DDP waits for a gradient on
        # at find_unused_parameters=false.
        encoder.requires_grad_(False)
        blocks = int(config.text_encoder_trainable_blocks)
        layers = encoder.encoder.layer
        if blocks < 0:
            thawed = list(layers)
        elif blocks > 0:
            thawed = list(layers[-blocks:])
        else:
            thawed = []
        for layer in thawed:
            layer.requires_grad_(True)
        if config.text_encoder_train_embeddings:
            encoder.embeddings.requires_grad_(True)
        return encoder

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
        # the text path starts inert, which is what makes grafting it onto a pretrained
        # model safe: the first forward is identical with and without a caption
        for m in self.modules():
            if isinstance(m, MultiheadCrossAttention):
                nn.init.zeros_(m.out_layer.weight)
                nn.init.zeros_(m.out_layer.bias)

    def _zero_content_input_columns(self) -> None:
        """Zero the up_proj columns fed by the content channel, in both latent copies."""
        emb = self.config.emb_dim
        self.up_proj.weight[:, emb].zero_()
        if self.self_conditioning:
            self.up_proj.weight[:, self.latent_dim + emb].zero_()

    def forward(
        self,
        input_embeddings: torch.Tensor,        # (B, T, C)
        time: torch.Tensor,             # (B,) or (B, 1) in [0, 1]
        attention_mask: Optional[torch.Tensor] = None,
        x0_self: Optional[torch.Tensor] = None,  # previous x0 estimate, (B, T, C)
        length: Optional[torch.Tensor] = None,   # non-padding token count, (B,)
        text: Optional[torch.Tensor] = None,     # frozen caption encoding, (B, S, text_dim)
        text_mask: Optional[torch.Tensor] = None,  # (B, S), True where a token is real
        caption_ids: Optional[torch.Tensor] = None,   # caption token ids, (B, S)
        caption_attn: Optional[torch.Tensor] = None,  # (B, S), True where a token is real
    ):
        # The caption encoded here instead of read from disk. This block produces the same
        # two tensors the frozen path loads, and every line below it then runs unchanged —
        # text_global, the per-block cross-attention, the zeroed output projection and the
        # readout cannot tell which path filled them in.
        if caption_ids is not None:
            if self.text_encoder is None:
                raise ValueError(
                    "caption_ids were passed but model.text_encoder is unset; this model "
                    "conditions on precomputed states, so pass text= instead"
                )
            if text is not None:
                raise ValueError(
                    "text and caption_ids were both passed; the caption would be encoded "
                    "twice and only one of the two would reach the cross-attention"
                )
            if caption_attn is None:
                caption_attn = torch.ones_like(caption_ids, dtype=torch.bool)
            text = self.text_encoder(
                input_ids=caption_ids, attention_mask=caption_attn.long()
            ).last_hidden_state
            if text_mask is None:
                text_mask = caption_attn.bool()
        elif caption_attn is not None:
            raise ValueError(
                "caption_attn was passed without caption_ids; nothing would read it, and "
                "the caption would silently reach the model unmasked"
            )

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
        if self.length_conditioning:
            if length is None:
                raise ValueError(
                    "model.length_conditioning is on but no length was passed; "
                    "training takes it from the attention mask and sampling draws it"
                )
            clamped = length.to(text_embed.device).long().clamp(0, self.config.max_pos)
            time_embed = time_embed + self.length_embedding(clamped)
        elif length is not None:
            raise ValueError(
                "a length was passed but model.length_conditioning is off"
            )

        if self.text_global and text is not None:
            # Masked mean over caption tokens: padding must not dilute the summary.
            if text_mask is None:
                pooled = text.mean(1)
            else:
                keep = text_mask.to(text.dtype).unsqueeze(-1)
                pooled = (text * keep).sum(1) / keep.sum(1).clamp(min=1.0)
            time_embed = time_embed + self.text_global_proj(pooled.to(time_embed.dtype))
        rope_pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        # rope = self.text_rope_embeddings(rope_pos)
        cos, sin = self.text_rope_embeddings(rope_pos)
 
        if text is not None and not self.config.text_dim:
            raise ValueError(
                "a caption was passed but model.text_dim is 0; the model has no text path"
            )

        for block in self.text_transformer_blocks:
            text_embed = block(text_embed, time_embed, (cos, sin), attention_mask,
                               text=text, text_mask=text_mask)
 

        prediction = self.noise_head(text_embed, time_embed)
        if self.content_channel:
            prediction = torch.cat((prediction, self.content_head(text_embed)), dim=-1)
        return prediction

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
