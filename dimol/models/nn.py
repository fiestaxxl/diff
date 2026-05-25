import torch
import torch.nn.functional as F
import torch.nn as nn
import math



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


class TextEmbeddings(nn.Module):
    def __init__(self, text_dim, model_dim):
        super().__init__()
        self.in_layer = nn.Linear(text_dim, model_dim, bias=True)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=True)

    def forward(self, text_embed):
        text_embed = self.in_layer(text_embed)
        return self.norm(text_embed).type_as(text_embed)



class RoPE1D(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.half = dim//2
        freq = torch.exp(-math.log(10000.0) * torch.arange(self.half) / (self.half)) # (half,)
        pos = torch.arange(config.max_pos, dtype=freq.dtype) # (max_pos,)
        # self.register_buffer(f"args", torch.outer(pos, freq), persistent=False) #(max_pos, half)

        args = torch.outer(pos, freq)  # (max_pos, half)
        self.register_buffer("cos", torch.cos(args), persistent=False)
        self.register_buffer("sin", torch.sin(args), persistent=False)

    def forward(self, pos):
        # pos: (B, T) -> each (B, 1, T, half) for broadcasting over heads
        return self.cos[pos].unsqueeze(1), self.sin[pos].unsqueeze(1)

    # def forward(self, pos):
    #     '''
    #     pos (B, T) - ids of positions
    #     returns: (B, T, half, 2, 2)
    #     '''
    #     args = self.args[pos] # (B, T, half)
    #     cosine = torch.cos(args)
    #     sine = torch.sin(args)
    #     rope = torch.stack([cosine, -sine, sine, cosine], dim=-1) # (B, T, half, 4)
    #     rope = rope.view(*rope.shape[:-2], self.half, 2, 2) # (B, T, half, 2, 2)
    #     return rope.unsqueeze(1)  # (B, 1, T, half, 2, 2)


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

    @torch._dynamo.disable
    def apply_rotary(self, x, rope):
        cos, sin = rope                          # each (B, 1, T, half)
        x_pairs = x.unflatten(-1, (-1, 2))       # (B, nh, T, half, 2)
        a, b = x_pairs.unbind(-1)                # each (B, nh, T, half)
        out_a = a * cos - b * sin
        out_b = a * sin + b * cos
        return torch.stack([out_a, out_b], dim=-1).flatten(-2)
    # def apply_rotary(self, x, rope):
    #      # x: (B, nh, T, hs), rope: (B, 1, T, half, 2, 2)
    #     x_ = x.reshape(*x.shape[:-1], -1, 2).unsqueeze(-1) # (B, nh, T, half, 2, 1)
    #     x_out = (rope * x_).sum(dim=-2) # (B, nh, T, half, 2)
    #     return x_out.reshape(*x.shape)#.to(torch.bfloat16)



class MultiheadCrossAttention(nn.Module):
    def __init__(self, num_channels, head_dim, attention_engine="auto"):
        super().__init__()
        assert num_channels % head_dim == 0
        self.num_heads = num_channels // head_dim

        self.to_query = nn.Linear(num_channels, num_channels, bias=True)
        self.to_key = nn.Linear(num_channels, num_channels, bias=True)
        self.to_value = nn.Linear(num_channels, num_channels, bias=True)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)

        self.out_layer = nn.Linear(num_channels, num_channels, bias=True)
        self.attn_engine = SelfAttentionEngine(attention_engine)

    # @torch.compile()
    def get_qkv(self, x, cond):
        query = self.to_query(x)
        key = self.to_key(cond)
        value = self.to_value(cond)

        shape = query.shape[:-1]
        query = query.reshape(*shape, self.num_heads, -1)

        key_shape = key.shape[:-1]
        key = key.reshape(*key_shape, self.num_heads, -1)
        value = value.reshape(*key_shape, self.num_heads, -1)

        return query, key, value

    # @torch.compile()
    def norm_qk(self, q, k):
        q = self.query_norm(q.float()).type_as(q)
        k = self.key_norm(k.float()).type_as(k)
        return q, k

    # @torch.compile()
    def scaled_dot_product_attention(self, query, key, value, attention_mask=None):
        args = {"q": query, "k": key, "v": value}
        if attention_mask is not None:
            args["attn_mask"] = attention_mask

        out = self.attn_engine.get_attention()(**args).flatten(-2, -1) #TODO CHECK THIS
        return out

    # @torch.compile()
    def out_l(self, x):
        return self.out_layer(x)

    def forward(self, x, cond, key_padding_mask=None):
        query, key, value = self.get_qkv(x, cond)
        query, key = self.norm_qk(query, key)

        out = self.scaled_dot_product_attention(query, key, value, key_padding_mask)
        out = self.out_l(out)
        return out

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
    

class TransformerEncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_modulation = Modulation(config, num_params=6)

        self.self_attention_norm = nn.LayerNorm(config.model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttention(config)

        self.feed_forward_norm = nn.LayerNorm(config.model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(config)

    @staticmethod
    def _gate_sum(x, out, gate):
        return x + gate[:, None, :] * out

    def forward(self, x, time_embed, rope, attention_mask=None):
        self_attn_params, ff_params = torch.chunk(self.text_modulation(time_embed), 2, dim=-1)

        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        out = self.apply_scale_shift_norm(self.self_attention_norm, x, scale[:, None, :], shift[:, None, :])
        out = self.self_attention(out, rope, attention_mask)
        x = self._gate_sum(x, out, gate)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        out = self.apply_scale_shift_norm(self.feed_forward_norm, x, scale[:, None, :], shift[:, None, :])
        out = self.feed_forward(out)
        x = self._gate_sum(x, out, gate)
        return x

    def apply_scale_shift_norm(self, norm, x, scale, shift):
        return (norm(x) * (scale + 1.0) + shift)
    



class TransformerDecoderBlock(nn.Module):
    def __init__(self, model_dim, time_dim, ff_dim, head_dim, attention_engine="auto"):
        super().__init__()
        self.visual_modulation = Modulation(time_dim, model_dim, 9)

        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttentionEnc(model_dim, head_dim, attention_engine)

        self.cross_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.cross_attention = MultiheadCrossAttention(model_dim, head_dim, attention_engine)

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim)


    def forward(self, x, time_embed, text_embed, rope=None, attention_mask=None, key_padding_mask=None):
        self_attn_params, cross_attn_params, ff_params = torch.chunk(self.visual_modulation(time_embed), 3, dim=-1)

        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        scale = scale[:, None, :]  # (B, 1, D)
        shift = shift[:, None, :]  # (B, 1, D)
        out = apply_scale_shift_norm(self.self_attention_norm, x, scale, shift)
        out = self.self_attention(out, rope, attention_mask)
        x = apply_gate_sum(x, out, gate)


        shift, scale, gate = torch.chunk(cross_attn_params, 3, dim=-1)
        scale = scale[:, None, :]  # (B, 1, D)
        shift = shift[:, None, :]  # (B, 1, D)
        out = apply_scale_shift_norm(self.cross_attention_norm, x, scale, shift)
        out = self.cross_attention(out, text_embed, key_padding_mask)
        x = apply_gate_sum(x, out, gate)


        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        scale = scale[:, None, :]  # (B, 1, D)
        shift = shift[:, None, :]  # (B, 1, D)
        out = apply_scale_shift_norm(self.feed_forward_norm, x, scale, shift)
        out = self.feed_forward(out)
        x = apply_gate_sum(x, out, gate)
        return x


