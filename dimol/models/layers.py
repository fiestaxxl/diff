"""Denoiser building blocks. Moved from dimol/models/nn.py with the computations
unchanged; only dead classes (TextEmbeddings, MultiheadCrossAttention,
TransformerDecoderBlock) and commented-out alternative implementations were removed.
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import math


class NormalizedLinear(nn.Module):
    """
    Linear with L2-normalized rows of W and a learnable scalar scale.
        logits = scale * (x @ normalize(W).T) + bias

    Each output's contribution is determined by the *direction* of W[i], not its
    magnitude, so frequent tokens can no longer win argmax by ballooning their
    embedding norm. The scalar `scale` (learned via log-parameterization to keep
    it positive) sets softmax sharpness; init=10 gives reasonable confidence.
    """

    def __init__(self, in_features, out_features, bias=True, init_scale=10.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, std=0.02)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.log_logit_scale = nn.Parameter(torch.tensor(math.log(init_scale)))

    def forward(self, x):
        w = F.normalize(self.weight, dim=-1)
        out = F.linear(x, w) * self.log_logit_scale.exp()
        if self.bias is not None:
            out = out + self.bias
        return out


class TimeEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_layer = nn.Linear(config.model_dim, config.time_dim, bias=True)
        self.act = nn.SiLU()
        self.out_layer = nn.Linear(config.time_dim, config.time_dim, bias=True)

        half = config.model_dim // 2
        self.register_buffer(
            "freqs", torch.exp(-math.log(10000.0) * torch.arange(half) / (half)), persistent=False
        )

    def forward(self, t):
        """
        # t: (B,) or (B, 1) in [0, 1]
        returns: (B, dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        t = t * 1000.0  # scale to [0, 1000] before encoding
        args = t * self.freqs #(B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1) #(B, self.model_dim)
        return self.out_layer(self.act(self.in_layer(emb))) #(B, self.time_dim)


class RoPE1D(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.half = dim//2
        freq = torch.exp(-math.log(10000.0) * torch.arange(self.half) / (self.half)) # (half,)
        pos = torch.arange(config.max_pos, dtype=freq.dtype) # (max_pos,)
        args = torch.outer(pos, freq)  # (max_pos, half)
        self.register_buffer("cos", torch.cos(args), persistent=False)
        self.register_buffer("sin", torch.sin(args), persistent=False)

    def forward(self, pos):
        # pos: (B, T) -> each (B, 1, T, half) for broadcasting over heads
        return self.cos[pos].unsqueeze(1), self.sin[pos].unsqueeze(1)


class Modulation(nn.Module):
    def __init__(self, config, num_params):
        super().__init__()
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(config.time_dim, num_params * config.model_dim)

    def forward(self, x):
        return self.out_layer(self.activation(x))


class MultiheadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.model_dim % config.num_heads == 0, f"Model dim: {config.model_dim} must be divisible by number of heads: {config.num_heads}"
        self.config = config
        self.n_head = config.num_heads
        self.head_dim = config.model_dim // config.num_heads
        self.c_attn = nn.Linear(config.model_dim, 3 * config.model_dim, bias=True)

        self.query_norm = nn.RMSNorm(self.head_dim)
        self.key_norm = nn.RMSNorm(self.head_dim)

        self.out_layer = nn.Linear(config.model_dim, config.model_dim, bias=True)


    def norm_qk(self, q, k):
        q = self.query_norm(q.float()).type_as(q)
        k = self.key_norm(k.float()).type_as(k)
        return q, k

    def forward(self, x, rope, attention_mask=None):
        '''
        attention_mask (optional Tensor) – Attention mask; shape must be broadcastable to the shape of attention weights, which is 
        (N,..., T, C). Two types of masks are supported. A boolean mask where a value of True indicates that the element should take part in attention. A float mask of the same type as query, key, value that is added to the attention score.
        '''
        # import code; code.interact(local=dict(globals(), **locals()))
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

        q, k = self.norm_qk(q, k)

        q = self.apply_rotary(q, rope)
        k = self.apply_rotary(k, rope)

        # broadcast a (B, T) padding mask to (B, 1, 1, T)
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :]

        out = F.scaled_dot_product_attention(q, k, v, 
                                            attn_mask=attention_mask, 
                                            dropout_p=(self.config.attn_dropout if self.training else 0.0)) # (B, nh, T, hs)
        out = out.transpose(1, 2).contiguous()  # (B, T, nh, hs)
        out = out.view(B, T, C) # (B, T, C)
        out = self.out_layer(out)
        return out

    def apply_rotary(self, x, rope):
        cos, sin = rope                          # each (B, 1, T, half)
        x_pairs = x.unflatten(-1, (-1, 2))       # (B, nh, T, half, 2)
        a, b = x_pairs.unbind(-1)                # each (B, nh, T, half)
        out_a = a * cos - b * sin
        out_b = a * sin + b * cos
        return torch.stack([out_a, out_b], dim=-1).flatten(-2)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_layer = nn.Linear(config.model_dim, 4*config.model_dim, bias=False)
        self.activation = nn.GELU()
        self.out_layer = nn.Linear(4*config.model_dim, config.model_dim, bias=False)

    # @torch.compile()
    def forward(self, x):
        return self.out_layer(self.activation(self.in_layer(x)))


class OutLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.modulation = Modulation(config, num_params=2) 
        self.norm = nn.LayerNorm(config.model_dim, elementwise_affine=False)

        self.lm_head = nn.Linear(config.model_dim, config.emb_dim, bias=False)

    def forward(self, 
                text_embed, 
                time_embed):
        '''
        text_embed (B, T, C)
        time_embed (B, time_dim)
        '''
        shift, scale = torch.chunk(self.modulation(time_embed), 2, dim=-1) #(B,model_dim)


        text_embed = self.apply_scale_shift_norm(
            self.norm,
            text_embed,
            scale[:, None, :],
            shift[:, None, :],
        )

        x = self.lm_head(text_embed)  
        return x
    
    def apply_scale_shift_norm(self, norm, x, scale, shift):
        return (norm(x) * (scale + 1.0) + shift)
    

class MultiheadCrossAttention(nn.Module):
    """Attention from the molecule canvas into a frozen text encoding.

    Text conditioning cannot ride on the adaLN vector the way the timestep and the length
    do: a caption names specific substructures, so the canvas has to attend to individual
    caption tokens rather than to one pooled summary. Queries come from the canvas, keys
    and values from the caption; there is no RoPE on either side, because the two
    sequences have no shared coordinate.

    The output projection is zeroed at construction, exactly like the adaLN modulations,
    so a model that gains these layers is bit-for-bit the model it was before until they
    are trained. That is what makes it safe to graft them onto a pretrained checkpoint.
    """

    def __init__(self, config, text_dim: int):
        super().__init__()
        assert config.model_dim % config.num_heads == 0
        self.config = config
        self.n_head = config.num_heads
        self.head_dim = config.model_dim // config.num_heads
        self.to_q = nn.Linear(config.model_dim, config.model_dim, bias=True)
        self.to_kv = nn.Linear(text_dim, 2 * config.model_dim, bias=True)
        self.query_norm = nn.RMSNorm(self.head_dim)
        self.key_norm = nn.RMSNorm(self.head_dim)
        self.out_layer = nn.Linear(config.model_dim, config.model_dim, bias=True)

    def forward(self, x, text, text_mask=None):
        B, T, C = x.size()
        S = text.size(1)
        q = self.to_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k, v = self.to_kv(text).split(C, dim=2)
        k = k.view(B, S, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_head, self.head_dim).transpose(1, 2)
        q = self.query_norm(q.float()).type_as(q)
        k = self.key_norm(k.float()).type_as(k)

        mask = None
        if text_mask is not None:
            # (B, S) of True where a caption token is real -> (B, 1, 1, S)
            mask = text_mask.bool()[:, None, None, :]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=(self.config.attn_dropout if self.training else 0.0))
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_layer(out)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_modulation = Modulation(config, num_params=6)

        self.self_attention_norm = nn.LayerNorm(config.model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttention(config)

        text_dim = int(getattr(config, "text_dim", 0) or 0)
        self.cross_attention = None
        if text_dim:
            self.cross_attention_norm = nn.LayerNorm(config.model_dim,
                                                     elementwise_affine=False)
            self.cross_attention = MultiheadCrossAttention(config, text_dim)

        self.feed_forward_norm = nn.LayerNorm(config.model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(config)

    @staticmethod
    def _gate_sum(x, out, gate):
        return x + gate[:, None, :] * out

    def forward(self, x, time_embed, rope, attention_mask=None,
                text=None, text_mask=None):
        self_attn_params, ff_params = torch.chunk(self.text_modulation(time_embed), 2, dim=-1)

        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        out = self.apply_scale_shift_norm(self.self_attention_norm, x, scale[:, None, :], shift[:, None, :])
        out = self.self_attention(out, rope, attention_mask)
        x = self._gate_sum(x, out, gate)

        if self.cross_attention is not None and text is not None:
            # ungated: the output projection starts at zero, which is the gate
            x = x + self.cross_attention(self.cross_attention_norm(x), text, text_mask)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        out = self.apply_scale_shift_norm(self.feed_forward_norm, x, scale[:, None, :], shift[:, None, :])
        out = self.feed_forward(out)
        x = self._gate_sum(x, out, gate)
        return x

    def apply_scale_shift_norm(self, norm, x, scale, shift):
        return (norm(x) * (scale + 1.0) + shift)
