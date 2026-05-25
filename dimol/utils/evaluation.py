import torch
from collections import Counter
from tqdm.auto import tqdm


@torch.no_grad()
def reconstruct_autoregressive(model, x, max_len=None):
    """
    Autoregressive reconstruction using sampled latent z.
    Returns reconstructed token ids [bs, seq_len]
    """
    model.eval()
    device = x.device
    bs, seq_len = x.shape
    max_len = max_len or seq_len

    # Encode full sequence
    mu, logvar = model.encode(x)
    z = model.reparameterize(mu, logvar)

    # Apply same latent shift as training
    z = torch.roll(z, shifts=1, dims=1)
    z[:, 0, :] = 0

    # Start token = first token of input
    out = torch.full_like(x, model.pad_idx)
    out[:, 0] = x[:, 0]

    for t in tqdm(range(1, max_len)):
        logits = model.decode(out[:, :t], z[:, :t])
        next_token = logits[:, -1].argmax(dim=-1)
        out[:, t] = next_token

    return out

@torch.no_grad()
def token_accuracy(pred, target, pad_idx=0):
    mask = target != pad_idx
    correct = (pred == target) & mask
    return correct.sum().item() / mask.sum().item()

@torch.no_grad()
def sequence_accuracy(pred, target, pad_idx=0):
    mask = target != pad_idx
    exact = ((pred == target) | ~mask).all(dim=1)
    return exact.float().mean().item()

@torch.no_grad()
def common_mistakes(pred, target, pad_idx=0, top_k=20, tokenizer=None):
    errors = []
    for p, t in zip(pred.view(-1), target.view(-1)):
        if t != pad_idx and p != t:
            idx_t, idx_p = int(t), int(p)

            if tokenizer is not None:
                idx_t, idx_p = tokenizer.decode(idx_t), tokenizer.decode(idx_p)
            errors.append((idx_t, idx_p))
    return Counter(errors).most_common(top_k)

@torch.no_grad()
def positional_error_rate(pred, target, pad_idx=0):
    seq_len = target.size(1)
    errors = torch.zeros(seq_len)
    counts = torch.zeros(seq_len)

    for i in range(seq_len):
        mask = target[:, i] != pad_idx
        counts[i] = mask.sum()
        errors[i] = ((pred[:, i] != target[:, i]) & mask).sum()

    return (errors / counts.clamp_min(1)).cpu()

@torch.no_grad()
def kl_statistics(mu, logvar):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return {
        "kl_mean": kl.mean().item(),
        "kl_per_dim": kl.mean(dim=(0,1)).cpu(),
        "mu_std": mu.std().item(),
        "logvar_mean": logvar.mean().item()
    }


@torch.no_grad()
def latent_interpolation(model, x1, x2, steps=8):
    model.eval()
    mu1, logvar1 = model.encode(x1)
    mu2, logvar2 = model.encode(x2)

    z1 = model.reparameterize(mu1, logvar1)
    z2 = model.reparameterize(mu2, logvar2)

    outs = []
    for alpha in torch.linspace(0, 1, steps):
        z = (1 - alpha) * z1 + alpha * z2
        z = torch.roll(z, shifts=1, dims=1)
        z[:, 0, :] = 0

        out = reconstruct_autoregressive(model, x1)
        outs.append(out)

    return outs

def latent_sensitivity(model, x):
    mu, logvar = model.encode(x)
    z = model.reparameterize(mu, logvar)
    
    # Detach z, then create a new tensor that requires gradients
    with torch.no_grad():
        z_shifted = torch.roll(z, 1, 1)
        z_shifted[:, 0, :] = 0
    
    # Create a new tensor that requires gradients
    z_modified = z_shifted.clone().requires_grad_(True)
    
    logits = model.decode(x[:, :-1], z_modified[:, :-1])
    loss = logits.norm()
    loss.backward()
    
    return z_modified.grad.norm().item()



@torch.no_grad()
def evaluate_model(model, dataloader, pad_idx=0, device="cuda"):
    token_accs = []
    seq_accs = []
    mistake_counter = Counter()
    pos_errors = None
    kl_stats_all = []

    for x in dataloader:
        x = x.to(device)

        # Reconstruction
        pred = reconstruct_autoregressive(model, x)

        token_accs.append(token_accuracy(pred, x, pad_idx))
        seq_accs.append(sequence_accuracy(pred, x, pad_idx))

        mistake_counter.update(common_mistakes(pred, x, pad_idx))

        if pos_errors is None:
            pos_errors = positional_error_rate(pred, x, pad_idx)
        else:
            pos_errors += positional_error_rate(pred, x, pad_idx)

        # Latent stats
        mu, logvar = model.encode(x)
        kl_stats_all.append(kl_statistics(mu, logvar))

    report = {
        "token_accuracy": sum(token_accs) / len(token_accs),
        "sequence_accuracy": sum(seq_accs) / len(seq_accs),
        "common_mistakes": mistake_counter.most_common(20),
        "positional_error_rate": (pos_errors / len(dataloader)).tolist(),
        "latent_stats": kl_stats_all[0]  # representative
    }

    return report
