import torch
from dimol.models.models import TransformerConfig, DiffusionTransformer
from dimol.diffusion.paths import GaussianConditionalProbabilityPath
from dimol.diffusion.conditionals import CosineAlpha, CosineBeta
from dimol.tokenizer.smiles_tokenizer import SmilesTokenizer
from dimol.diffusion.distributions import GaussianMixture

device = 'cuda'
model = DiffusionTransformer.from_pretrained(load_dir='checkpoints/v7/499', map_location=device)
model.eval()
tok = SmilesTokenizer.load("data/smiles_bpe.json")
vocab = tok.get_vocab()
ids_to_toks = {value:key for key,value in vocab.items()}

sm = 'COC(=O)/C=C1/CC[C@H]2[C@@H]3CCC4=CC(=O)CC[C@]4(C)[C@H]3C(=O)C[C@]12C'
real_ids, mask = tok.encode_padded(sm, add_special_tokens=True)
real_ids = torch.tensor(real_ids).unsqueeze(0).to(device)
mask = torch.tensor(mask).unsqueeze(0).to(device).bool()
real_embed = model.token_embedding(real_ids)

p_data = GaussianMixture.symmetric_2D(nmodes=5, std=1.0, scale=10.0)
path = GaussianConditionalProbabilityPath(
    p_data=p_data.to(device),
    p_simple_shape=[208, 768],
    alpha=CosineAlpha(device),
    beta=CosineBeta(device)
).to(device)


print(f"=== Tying check ===")
print(f"out_proj tied to embedding: {model.out_proj.weight.data_ptr() == model.token_embedding.weight.data_ptr()}")

print(f"\n=== Embedding stats ===")
norms = model.token_embedding.weight.norm(dim=-1)
print(f"norm mean: {norms.mean():.3f}, std: {norms.std():.3f}")
print(f"norm min: {norms.min():.3f} (token id {norms.argmin().item()}) - {ids_to_toks[norms.argmin().detach().item()]}")
print(f"norm max: {norms.max():.3f} (token id {norms.argmax().item()} - {ids_to_toks[norms.argmax().detach().item()]})")
print(f"norm of token 132 (OH): {norms[132]:.3f}")
print(f"norm of token 404 ([Br-]): {norms[404]:.3f}")

print(f"\n=== Forward output check ===")
with torch.no_grad():
    out = model(input_embeddings=real_embed, time=torch.zeros(1,1,device=device), attention_mask=mask)
    print(f"eps sum here: {out.sum()}")
    print(f"output shape: {out.shape}")
    print(f"output norm at first non-pad position: {out[0, 0].norm().item():.3f}")
    print(f"output norm at pad position: {out[0, -1].norm().item():.3f}")

    
with torch.no_grad():
    # ===== Test 1: direct decode (no diffusion) =====
    direct_logits = model.out_proj(real_embed)
    direct_decoded = direct_logits.argmax(-1)
    real_pos_mask = mask.squeeze(0)  # only check non-pad positions
    real_pos_match = (real_ids[0][real_pos_mask] == direct_decoded[0][real_pos_mask]).float().mean()
    print(f"Direct decode (real positions only): {real_pos_match.item():.3f}")
    print(f"  input:   {real_ids[0][real_pos_mask].cpu().tolist()}")
    print(f"  decoded: {direct_decoded[0][real_pos_mask].cpu().tolist()}")

    # ===== Test 2: at t=1 (data endpoint) =====
    t_one = torch.ones(1, 1, device=device)
    print(f"\nAt t=1: alpha={path.alpha(t_one[..., None]).item():.4f}, beta={path.beta(t_one[..., None]).item():.4f}")
    eps = model(input_embeddings=real_embed, time=t_one, attention_mask=mask)
    alpha = path.alpha(t_one[..., None])
    beta = path.beta(t_one[..., None])
    x0_hat = (real_embed - beta * eps) / alpha.clamp(min=1e-4)
    logits = model.out_proj(x0_hat)
    decoded = logits.argmax(-1)
    t1_match = (real_ids[0][real_pos_mask] == decoded[0][real_pos_mask]).float().mean()
    print(f"t=1 match (real positions only): {t1_match.item():.3f}")
    print(f"eps sum here: {eps.sum()}")
    print(f"  input:   {real_ids[0][real_pos_mask].cpu().tolist()}")
    print(f"  decoded: {decoded[0][real_pos_mask].cpu().tolist()}")

    # ===== Test 3: at t=0.99 (very low noise but valid training distribution) =====
    t_high = torch.full((1, 1), 0.99, device=device)
    print(f"\nAt t=0.99: alpha={path.alpha(t_high[..., None]).item():.4f}, beta={path.beta(t_high[..., None]).item():.4f}")
    
    # Add the actual diffusion noise as in training
    noise = torch.randn_like(real_embed)
    x_t = path.alpha(t_high[..., None]) * real_embed + path.beta(t_high[..., None]) * noise
    eps_pred = model(input_embeddings=x_t, time=t_high, attention_mask=mask)
    
    # check noise prediction quality
    noise_mse = ((eps_pred - noise) ** 2).mean().item()
    print(f"  noise prediction MSE (should be ~0.4 if model learned): {noise_mse:.4f}")
    print(f"eps sum here: {eps_pred.sum()} vs noise: {noise.sum()}")
    
    # decode
    x0_hat_high = (x_t - path.beta(t_high[..., None]) * eps_pred) / path.alpha(t_high[..., None]).clamp(min=1e-4)
    logits_high = model.out_proj(x0_hat_high)
    decoded_high = logits_high.argmax(-1)
    t_high_match = (real_ids[0][real_pos_mask] == decoded_high[0][real_pos_mask]).float().mean()
    print(f"  match at t=0.99: {t_high_match.item():.3f}")
    print(f"  input:   {real_ids[0][real_pos_mask].cpu().tolist()}")
    print(f"  decoded: {decoded_high[0][real_pos_mask].cpu().tolist()}")

    # ===== Test 4: at mid-noise t=0.5 =====
    t_mid = torch.full((1, 1), 0.5, device=device)
    print(f"\nAt t=0.5: alpha={path.alpha(t_mid[..., None]).item():.4f}, beta={path.beta(t_mid[..., None]).item():.4f}")
    noise = torch.randn_like(real_embed)
    x_t = path.alpha(t_mid[..., None]) * real_embed + path.beta(t_mid[..., None]) * noise
    eps_pred = model(input_embeddings=x_t, time=t_mid, attention_mask=mask)
    noise_mse = ((eps_pred - noise) ** 2).mean().item()
    print(f"  noise MSE: {noise_mse:.4f}")
    print(f"eps sum here: {eps_pred.sum()} vs noise: {noise.sum()}")
    x0_hat_mid = (x_t - path.beta(t_mid[..., None]) * eps_pred) / path.alpha(t_mid[..., None]).clamp(min=1e-4)
    logits_mid = model.out_proj(x0_hat_mid)
    decoded_mid = logits_mid.argmax(-1)
    t_mid_match = (real_ids[0][real_pos_mask] == decoded_mid[0][real_pos_mask]).float().mean()
    print(f"  match at t=0.5: {t_mid_match.item():.3f}")
    print(f"  input:   {real_ids[0][real_pos_mask].cpu().tolist()}")
    print(f"  decoded: {decoded_mid[0][real_pos_mask].cpu().tolist()}")

    # ===== Test 4: at mid-noise t=0.5 =====
    t_high = torch.full((1, 1), 0.001, device=device)
    print(f"\nAt t=0.001: alpha={path.alpha(t_high[..., None]).item():.4f}, beta={path.beta(t_high[..., None]).item():.4f}")
    noise = torch.randn_like(real_embed)
    x_t = path.alpha(t_high[..., None]) * real_embed + path.beta(t_high[..., None]) * noise
    eps_pred = model(input_embeddings=x_t, time=t_high, attention_mask=mask)
    noise_mse = ((eps_pred - noise) ** 2).mean().item()
    print(f"  noise MSE: {noise_mse:.4f}")
    print(f"eps sum here: {eps_pred.sum()} vs noise: {noise.sum()}")
    x0_hat_high = (x_t - path.beta(t_high[..., None]) * eps_pred) / path.alpha(t_high[..., None]).clamp(min=1e-4)
    logits_high = model.out_proj(x0_hat_high)
    decoded_high = logits_high.argmax(-1)
    t_high_match = (real_ids[0][real_pos_mask] == decoded_high[0][real_pos_mask]).float().mean()
    print(f"  match at t=0.001: {t_high_match.item():.3f}")
    print(f"  input:   {real_ids[0][real_pos_mask].cpu().tolist()}")
    print(f"  decoded: {decoded_high[0][real_pos_mask].cpu().tolist()}")