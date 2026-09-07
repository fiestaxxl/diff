# The pipeline as it stands

Every stage is one script taking one yaml. Nothing reads environment variables, nothing
takes positional arguments, and every knob below has a comment in the config saying what
it does and what it measured.

## 1. Data

    python scripts/prepare_data.py configs/zinc250k.yaml

Downloads the corpus, curates it and writes tokenized memmap shards. Streaming and
multiprocess throughout: the corpus is never held in memory, `Curator` filters in worker
processes, `TextShardWriter` and `ArrayShardWriter` append to shards, and the tokenized
output is `<split>_tokens_NNNNN.npy` plus `<split>_attn_mask_NNNNN.npy` read back through
`np.load(mmap_mode="r")`.

Curation: canonicalize with rdkit, drop what does not parse, drop duplicates by canonical
string, drop anything longer than the canvas, split train/val/test.

`meta.json` records the token statistics the training budget is derived from: for
ZINC-250k, 224,568 training molecules at 22.51 tokens each, 5.06M tokens per epoch,
64.8% of canvas positions are padding.

    python scripts/train_tokenizer.py configs/zinc250k.yaml

Byte-pair over SMILES with the chemistry protected: bracket atoms are inviolable single
tokens, two-letter elements (Cl, Br) never split, the initial alphabet comes from the
corpus. `dimol/tokenization/chem_audit.py` scores a candidate tokenizer on whether it
splits functional groups and how far apart it puts ring-digit pairs, which is what the
model has to learn to match.

## 2. Training

    bash scripts/train.sh configs/diffusion_zinc250k.yaml
    bash scripts/train.sh configs/diffusion_zinc250k.yaml run_name=x optimizer.lr=3e-4

One yaml holds the model, the data, the diffusion path, the loss, the optimizer, the
schedule, precision, parallelism, evaluation, checkpointing and sampling. Dotted overrides
on the command line go through `OmegaConf.from_cli` and are merged into a structured
config in struct mode, so a typo is an error rather than a silent no-op.

The diffusion itself is unchanged from the original work: a variance-preserving cosine
path, `x_t = alpha(t) x0 + beta(t) eps` with `alpha = sin(pi t / 2)`, epsilon
parameterization, and a loss that is the denoising MSE plus a cross-entropy on the readout
plus a reconstruction term, the last two gated to the low-noise part of each batch. Every
formula is carried over verbatim; everything below is an opt-in flag that defaults to the
original behaviour.

What the model can be told to do, with what it measured:

| flag | effect |
|---|---|
| `model.self_conditioning` | second input, the model's own previous x0 estimate. **+38% usable, -13% FCD, three seeds.** The one change that improves both axes |
| `model.length_conditioning` | the non-padding token count as an input, embedded like the timestep. Matches the corpus length and ring statistics almost exactly; costs yield on its own, best in combination |
| `model.tie_readout` | readout is the embedding table itself. No effect |
| `diffusion.time_sampler` | `logit_normal` instead of uniform t. Doubles validity and worsens FCD: a trade, not a win |
| `diffusion.x0_noise_kind` | gaussian, laplace, sphere, token_mixup. laplace and mixup break generation |
| `loss.pad_weight` | padding weight in the noise loss. 0.6 gave the best validity in the study and the worst distribution: length gaming |
| `loss.min_snr_gamma` | min-SNR weighting. Strongly harmful here, because the token-level terms are already gated to high SNR |
| `loss.gate_mode` | `topk` makes the gated batch fraction deterministic. Lowers the mean, does not narrow the seed spread |
| `loss.ce_include_pad` | cross-entropy supervises padding. Nothing alone, harmful combined |
| `ema.*` | weight averaging. 4.40% to 0.24%: harmful, and the averaged checkpoint is legitimate |

Logging is one line per step with loss, tokens per second and accuracy, plus a metrics
jsonl. DDP is one flag; for screening it is the wrong tool, because with more runs than
GPUs one run per GPU is strictly faster.

## 3. Generation

    python scripts/generate.py configs/generate_zinc250k.yaml generate.checkpoint=runs/x/ep80-ba17600

Euler-Maruyama over the reverse SDE, then the readout, then decoding. Sampler settings
that were swept: sigma 1.0 beats the deterministic limit by 1.4 points, and 100 solver
steps match 1000, so generation runs at 100. The solver time grid is a knob and uniform
won.

Decoding, in the order of how much it is worth:

| flag | effect |
|---|---|
| `generate.decode=grammar_close` | repairs brackets and ring digits while writing the string. **4.60% to 13.20% usable on the reference**, length distribution intact. The default worth having |
| `generate.decode=grammar_mixed` | trims as well as closes. Higher validity, 59% trivial molecules. Do not use |
| `generate.strict_decode` | full connectivity check instead of counting brackets: no ring closed onto its own atom, no closure duplicating a bond. Zero false rejections on all 12,443 corpus molecules |
| `generate.restrict_atoms` | forbids bracket atoms the corpus never uses, which 14-16% of accepted molecules carry |
| `generate.length_prior` | draws the length from the corpus and pins the rest of the canvas to padding. Required by a length-conditioned model, nearly useless without one |
| `generate.length_floor` | also forbids stopping early. 0% usable, 60-atom strings. Kept only to reproduce that |
| `generate.refine_rounds` | denoise-renoise cycles after the trajectory. Changes nothing: the sample is a fixed point |

## 4. Measurement

    python scripts/evaluate_samples.py configs/generate_zinc250k.yaml generate.samples=<dir>
    python scripts/analyze_distribution.py --reference data/zinc250k/corpus/val.txt <files>
    python scripts/compare_configs.py --reference r_base80 <metrics.json ...>

Validity alone is not the metric and the study proves it: across eleven configurations
spanning a 2.4x range in sample quality, final validation loss varies by 1.5% and
correlates with quality at r = +0.46, the wrong sign. Validity itself is gameable three
ways at once, by trimming in the decoder, by ending sequences early, and by inventing
atoms.

So the headline number is **usable**: distinct, valid, at least ten heavy atoms, at least
one ring, no bracket atom absent from the corpus, as a share of all attempts. Next to it,
**FCD** against the corpus, and thirteen descriptors in units of the corpus's own standard
deviation. `compare_configs.py` refuses to call an effect unless the worst seed of a group
beats the best seed of the reference, because a configuration at this size is a
distribution and not a number: two runs with the same seed landed at 23.9% and 55.5%.

**Three seeds per configuration is the unit of measurement.**

## 5. What the measurement says the model still cannot do

With length conditioning the samples match the corpus on length, weight and ring count to
within a quarter of a standard deviation. Aromatic rings stay at 0.8 against the corpus
1.85, and nothing in section 2 or 3 moves that: not the objective, not the timestep
density, not depth from 2 to 16 blocks at fixed parameters, not sampling-time refinement.
Parameter count moves it, 0.8 at 5M to 1.1 at 14.6M, and the 14.6M model was the most
undertrained of the set.

That is what the corpus was blocking. ZINC-250k is 5.06M tokens per epoch, so 14.6M
parameters at 80 tokens per parameter is 231 passes and we measured that 161 passes hurt.
ZINC-20 is a billion molecules, about 22.6B tokens, which is 0.64 passes for a 180M model
at the same budget.
