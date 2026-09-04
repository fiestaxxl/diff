# Review of the padding handling and of the diffusion setup

Every number below was measured on this repository with the reference config
(`configs/diffusion_chebi.yaml`, model_dim 768, 4 blocks, emb_dim 32, vocab 512,
seq_len 208) unless stated otherwise. Nothing here has been changed silently: the
items marked *config* are switchable today, the items marked *decision* need an
experiment and a new baseline.

## Summary

| # | Finding | Evidence | Status |
|---|---|---|---|
| P1 | The loss is dominated by padding | the `mse_t0` term is 89% of the total at T=208 | config: `loss.mask_padding` |
| P2 | Compute is dominated by padding | 80% of positions are pads; bucketing gives 3.1x | config: `bucket_by_length` |
| P3 | The denoiser attends over pads | 168 of 208 positions carry no molecule | config: `loss.mask_padding` |
| P4 | Length is modelled implicitly | pad latent is the zero embedding plus noise | decision |
| D1 | The latents carry almost no signal | token identity is 0.63% of the variance of x0 | decision |
| D2 | `mse_t0` is normalized per sample, not per token | the term grows linearly with T | config: `loss.mask_padding` |
| D3 | Cross-entropy is gated by t although its input does not depend on t | 59% of the batch is discarded for free | config: `loss.ce_alpha_threshold` |
| D4 | The readout is trained on x0 and used on x_T | different distributions, same scale | decision |
| D5 | The reverse SDE keeps injecting noise at the data end | 23% of the signal scale per step | config: `sampling.variance` |
| D6 | Sampling runs in fp32 on every rank | 300 forward passes per rank per round | code |

---

## Padding

**P1. The loss is mostly about padding.** With a batch of 64 molecules of length
20 to 60 stored at T=208 (80% padding), the loss splits as

| T | total | mse | ce | mse_t0 |
|---|---|---|---|---|
| 208 | 68.18 | 1.30 | 6.24 | 60.64 (89%) |
| 64 | 24.77 | 1.30 | 6.24 | 17.23 (70%) |

The denoising term the diffusion is actually about (`mse`) is 2% of the objective.
`loss.mask_padding: true` removes the padded positions from both MSE terms and
from attention; the same batch then gives a total of 7.80 instead of 68.18, and
the number stops depending on T.

**P2. Compute is mostly about padding.** The mean molecule is about 40 tokens
against a stored length of 208. Length bucketing plus per-batch trimming, on a
synthetic corpus with a realistic length profile:

| Batching | median step | mean batch width |
|---|---|---|
| fixed length 208 | 772 ms | 208 tokens |
| bucketing + trim | 247 ms | 46 tokens |

**P3. The denoiser conditions on padding.** Attention is bidirectional over all
208 positions and `attention_mask` is not passed, so every real token attends to
about 168 pad positions. Whether that hurts is an empirical question, but it does
mean the model's function changes with the amount of padding in a batch, which is
also what blocks bucketing.

**P4. Length is modelled implicitly.** The pad row of the embedding table is exactly
zero (`padding_idx=0` keeps its gradient at zero, measured norm 0.000000), so a pad
position is `0 + 0.25 * noise`: pure noise with the same variance as any other
latent. During sampling the model has to decide where the molecule ends by emitting
pad-like latents that the readout maps to `<pad>`/`<eos>`. Worth a diagnostic
metric: the share of generated sequences that contain `<eos>` at all, and the
distribution of generated lengths against the training distribution.

---

## The diffusion setup

**D1. The latents carry almost no signal.** The embedding table is initialized at
std 0.02 and `x0 = emb + 0.25 * randn`:

```
var(x0) = 0.00040 (token identity) + 0.06250 (injected noise) = 0.06290
share carrying token identity: 0.63%
```

With the variance-preserving cosine path, `SNR(t) = alpha(t)^2 * var(x0) / beta(t)^2`:

| t | 0.25 | 0.50 | 0.59 | 0.75 | 0.90 | 0.99 |
|---|---|---|---|---|---|---|
| SNR | 0.011 | 0.063 | 0.112 | 0.367 | 2.51 | 255 |

SNR crosses 1 at t = 0.844, so 84% of the sampled timesteps are noise dominated.
With unit-variance latents it would cross at t = 0.500. Two knobs move this:
`diffusion.x0_noise_std` and the scale of the embeddings themselves (normalize the
table, or scale x0 to unit variance). Both change the objective, so this is an
experiment, not a fix; but it is the first thing to look at, because it decides
how much of the schedule does any useful work.

**D2. `mse_t0` is normalized per sample, not per token.** The mask is `(B, 1)`
while the squared error is `(B, L)`, so the sum over L is divided by the number of
selected samples. The term therefore grows linearly with T: 60.64 at T=208 against
17.23 at T=64, a ratio of 3.52 against T ratio 3.25. It also has no weight of its
own, it is multiplied by `lambda_mse`. `loss.mask_padding: true` switches to the
per-token normalization that the old trainer carried as commented-out code.

**D3. Cross-entropy is gated by t although its input is not a function of t.** The
readout is fed `x0`, not `x_t`:

```python
logits = get_logits(x0)                      # x0 does not depend on t
ce_sample_mask = (alpha > 0.80)              # yet only 41% of the batch is used
```

41% of the batch reaches the cross-entropy and 67% reaches the grammar term, while
the logits are computed for the whole batch anyway. Lowering
`loss.ce_alpha_threshold` to 0 gives 2.4x more cross-entropy signal per step at no
extra compute. That is the cheapest experiment on this list.

**D4. The readout is trained on one distribution and used on another.** Training
computes `out_proj(x0)` where `x0 = emb + 0.25 * noise`; sampling computes
`out_proj(x_T)` where `x_T` is the endpoint of the reverse SDE. The scales agree by
construction (both about 0.25: measured 0.2494 for training input, and the forward
path implies 0.249 as the marginal std at t=1), but the structure does not: one is
an embedding plus isotropic noise, the other is whatever the reverse process
produced. Training the readout on `x0_hat` instead would remove the mismatch and is
the single change most likely to move validity, which is why it is listed as a
decision rather than a fix.

**D5. The reverse SDE keeps injecting noise at the data end.** With
`sampling.variance: 1.0` and 300 steps over t in [1e-4, 0.999], each step adds
`sigma * sqrt(h) = 0.058` of noise while the marginal std of the state at that
point is about 0.25, i.e. 23% of the signal scale per step, and the score term
carries a factor `1/beta^2 = 3.9e3` at t=0.99. The formulation is standard, but the
last steps are stiff: `sampling.variance: 0` turns the sampler into the
probability-flow ODE, which is the usual ablation here. Cheap to try, it only
affects generation.

**D6. Sampling is fp32 and runs on every rank.** `dimol/eval/sampling.py` performs
300 model forwards outside autocast, and every rank generates its own
`sampling.num_samples` molecules to report one averaged validity. On 8 GPUs that is
8x the work for the same number. Two fixes, both purely mechanical: run the SDE
under autocast, and either shard the samples across ranks or sample on rank 0 only.

---

## Where the compute actually goes

For the reference config at batch 512, forward pass only:

| Part | FLOPs | Share |
|---|---|---|
| transformer body (4 blocks) | ~6.3 TFLOP | 99.9% |
| readout `out_proj` (emb_dim 32 -> vocab 512) | ~3.5 GFLOP | 0.06% |

So the readout matmul is not worth optimizing; what makes it expensive is memory,
not arithmetic: the `(B, L, V)` logits are 218 MiB in fp32 (109 MiB now that
autocast covers the readout), and the grammar term takes a softmax over the same
shape. The dominant cost is the transformer body, and 80% of that is spent on
padding, which is what P2 addresses.

---

## Suggested order

1. `loss.ce_alpha_threshold: 0` — free, more signal per step, no compute change.
2. `loss.mask_padding: true` — makes the objective length-invariant and stops
   padding from dominating the loss; needed before step 3.
3. `train_loader.bucket_by_length: true` — 3.1x on the measured profile.
4. `diffusion.x0_noise_std` sweep (and/or normalized embeddings) — moves the SNR
   schedule, the most consequential of the four.
5. `sampling.variance: 0` — generation only, cheap to compare.

Steps 1 to 3 need a new baseline because they change the objective; run the same
short schedule with the current defaults first so that the comparison is fair.
