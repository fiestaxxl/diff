# dimol: diffusion models for SMILES generation

Continuous embedding-space diffusion over SMILES, in two stages: unconditional pretraining
on a large molecule corpus, then a text-conditional fine-tune where a caption describes the
molecule to generate.

A run is described by a single yaml file whose path is passed to the script. The config
conventions follow llm-foundry: `ba`/`ep` durations, dotted CLI overrides, and
`global_train_batch_size` + `device_train_microbatch_size` with grad accumulation derived
from them. The config is merged in struct mode, so a mistyped key is an error rather than a
silent no-op.

```bash
pip install -e ".[chem,data,logging,dev]"
```

**[EXPERIMENTS.md](EXPERIMENTS.md) is the backlog**: every hypothesis tested, what it
measured, and the verdict — including the failures and the retracted claims. Read it before
proposing anything; a good fraction of the obvious ideas have already been run.

---

## The two stages

### Stage 1 — unconditional pretraining

```bash
# tokenizer, then tokenize the corpus into memmap shards
python scripts/train_tokenizer.py  configs/zinc250k.yaml
python scripts/tokenize_dataset.py configs/zinc250k.yaml

# train (the launcher picks python or torchrun and tees the log)
bash scripts/train.sh configs/diffusion_zinc20.yaml --gpus 4
```

The token budget is the first-order knob and everything in this repo is sized by the same
rule: **80 real (non-padding) tokens per parameter**. `meta.json` written by the tokenizer
carries the statistics to compute it — for a 171M model at 27.22 tokens per molecule and a
global batch of 4096, that is 122,637 steps.

### Stage 2 — text-conditional fine-tune

```bash
# molecules and captions, kept aligned row by row; two stages because curation needs
# RDKit and caption encoding needs a GPU
python scripts/prepare_paired.py --stage molecules --config configs/chebi20.yaml
python scripts/prepare_paired.py --stage text      --config configs/chebi20.yaml

# extend the tokenizer and grow the pretrained checkpoint into the larger vocabulary
python scripts/extend_tokenizer.py --tokenizer <stage-1 tokenizer> --corpus <corpus> --out <out>
python scripts/grow_vocab.py --checkpoint runs/<pretrain>/ep<E>-ba<N> \
    --vocab-size 640 --max-pos 128 --out runs/<pretrain>_grown

# fine-tune from the grown checkpoint
bash scripts/train.sh configs/chebi20_finetune.yaml --gpus 4 load_path=runs/<pretrain>_grown
```

Extending rather than retraining the tokenizer is deliberate: `add_tokens` appends after the
existing vocabulary, so every trained id keeps its meaning and the embedding table can be
warm-started. `grow_vocab.py` draws new rows at the *trained* scale, because the table grows
about fiftyfold over a run and a row at initialisation scale would sit near the origin and
never be reached.

---

## Generation and scoring

Sampling needs a GPU; scoring needs RDKit. They are separate commands so they can run in
different places.

```bash
# unconditional
python scripts/generate.py configs/generate_zinc250k.yaml generate.checkpoint=runs/x/ep80-ba17600
python scripts/evaluate_samples.py configs/generate_zinc250k.yaml generate.samples=<dir>

# text-conditional: generate, then score
python scripts/eval_text.py --stage generate --config configs/chebi20_finetune.yaml \
    --checkpoint runs/x/ep2020-ba200000 --split test --steps 300 \
    --length-file data/<corpus>/lenc/test_pred_lengths.npy --out <dir>
python scripts/eval_text.py --stage score --dir <dir>

# the published-benchmark columns: BLEU, exact, Levenshtein, MACCS/RDK/Morgan, FCD, validity
python scripts/score_benchmark.py --dir <dir>
```

### Measurement rules this repo enforces

These are not style preferences; each one is a result. See
[EXPERIMENTS.md](EXPERIMENTS.md#how-to-read-any-number-here) for the evidence.

- **Never select a checkpoint by validation loss.** Across eleven configurations spanning a
  2.4x range in quality, loss varied 1.5% and correlated at r = +0.46 — the wrong sign.
- **Validity alone is gameable**, by trimming in the decoder, by stopping early, and by
  inventing atoms. The unconditional headline is `usable` (distinct, valid, ≥10 heavy atoms,
  ≥1 ring, no out-of-corpus bracket atom, over *all attempts*), reported with FCD and
  descriptor shifts in corpus sigmas.
- **Three seeds are the unit of measurement.** `scripts/compare_configs.py` will not call an
  effect unless the worst seed of a group beats the best seed of the reference.
- **For a text-conditional model, always report the shuffled-caption floor** — every molecule
  paired with someone else's caption (`--shuffle-captions`). Without it you cannot tell
  conditioning from a good prior over the corpus.

---

## Helper models

Two small models sit beside the denoiser. Both predict the molecule's token length from the
caption, which the sampler uses to pin the rest of the canvas to padding.

```bash
python scripts/train_length_head.py    --config <cfg> --out <path>.pt      # frozen states
python scripts/train_length_encoder.py --config <cfg> --out <dir>          # fine-tuned encoder
```

The head reads frozen caption states and plateaus; fine-tuning the encoder roughly halves
the error. The encoder writes one predicted length per caption per split, and the sampler
consumes the file via `--length-file`, so any predictor can be swapped in without touching
the diffusion code.

---

## What the yaml controls

| Section | Contents |
|---|---|
| `model` | denoiser architecture (`diffusion_transformer`) or AR baseline (`smiles_ar`); `self_conditioning`, `length_conditioning`, `text_dim`, `text_global` |
| `diffusion` | regime (`epsilon`/`x`), `alpha`/`beta` schedules, embedding noise, timestep sampler, `t_eps` |
| `loss` | term weights (`lambda_mse`, `lambda_ce`, `lambda_grammar`), alpha gates, padding treatment (`mask_padding`, `pad_weight`, `ce_include_pad`), `self_cond_prob`, `caption_dropout` |
| `train_loader` / `eval_loader` | dataset (`smiles_npy`, `smiles_text_npy`), workers, shuffling, length bucketing |
| `optimizer` / `scheduler` / `algorithms` | AdamW, warmup+cosine, gradient clipping |
| batching | `global_train_batch_size`, `device_train_microbatch_size` (accumulation is derived) |
| compute | `precision` (`amp_bf16`/`amp_fp16`/`fp32`), `tf32`, `compile`, `ddp_config` |
| control | `max_duration`, `eval_interval`, `save_interval`, `autoresume`, `load_path`, `load_weights_only` |
| logging | `loggers` (console / jsonl / tensorboard), `console_log_interval` |
| `sampling` | periodic generation during training, and the sampler itself: solver, steps, decoding, length prior |

Every field can be overridden from the command line, so sweeps need no code changes:

```bash
python scripts/train.py configs/diffusion_zinc20.yaml \
    run_name=s42 seed=42 model.model_dim=1024 model.num_heads=16 model.num_text_blocks=12 \
    max_duration=122637ba scheduler.t_max=122637ba
```

Two knobs that have bitten before: **`scheduler.max_lr` overrides `optimizer.lr`** — the peak
rate lives on the scheduler. And **`autoresume=true` obeys `load_weights_only`**, so a
fine-tune config that sets it will resume the weights while resetting the step counter and
restarting the cosine; pass `load_weights_only=false` to resume properly.

---

## Training log

Every step reports loss, accuracy and throughput:

```
[train ba 1200/9500 ep 3] loss 1.2346 | acc 0.8123 | mse 0.9800 | ce 0.2500 | mse_t0 0.0041 |
gram 0.0002 | lr 2.900e-04 | gnorm 0.834 | tok/s 145.2K | mol/s 698.1 | dt 210ms | mem 34.1G
```

`acc 1.000` with `ce 0.000` within the first epochs is expected, not a bug: with
`ce_input=x0` the readout only has to invert the embedding table it owns, and the loss is
carried by the MSE terms. With `self_cond_prob: 0.5`, `dt` is bimodal — half the steps run
an extra forward pass for the self-conditioning estimate.

The same numbers reach the jsonl and tensorboard logs under Composer-style names:
`loss/train/total`, `metrics/train/token_acc`, `throughput/tokens_per_sec`,
`lr-AdamW/group0`, `l2_norm/grad/global`.

## Run artifacts

```
runs/<run_name>/
  config.resolved.yaml      the config after overrides and interpolation
  env.json                  torch/cuda/gpu versions plus git sha and dirty flag
  metrics.jsonl             every metric, one json line per event
  samples/ba<N>.txt         molecules sampled during training
  ep<E>-ba<N>/              checkpoint: weights + optimizer/scaler/step (autoresume)
```

Keep `save_interval` short enough that losing a run costs less than it took to notice.

---

## Layout

```
dimol/
  registry.py               name -> class: models, tasks, schedules, loggers, datasets
  config.py                 config schema, ba/ep durations, derived grad accumulation
  builders.py               config -> objects
  data/                     datasets (.npy shards), loaders, collation, raw SMILES sources
  tokenization/             SMILES BPE tokenizer + chemistry audit
  models/                   layers (adaLN, RoPE, cross-attention), denoiser (DiT),
                            score wrapper, length head, AR baseline
  diffusion/                alpha/beta schedules, probability path, SDE, simulators
  training/                 trainer, tasks (the losses), optimizers, checkpoints, DDP
  eval/                     sampling, grammar-constrained decoding, SMILES state machine,
                            distribution metrics, reports
scripts/
  train.sh, train.py        training entrypoints
  prepare_data.py           download, curate and shard a corpus
  train_tokenizer.py        train and audit a SMILES tokenizer
  tokenize_dataset.py       corpus -> memmap token shards
  prepare_paired.py         molecules + captions, kept row-aligned
  extend_tokenizer.py       add a second corpus's tokens without moving any id
  grow_vocab.py             grow a checkpoint's vocabulary, canvas and length table
  augment_paired.py         several SMILES spellings per molecule (measured harmful; see backlog)
  train_length_head.py      caption -> length, on frozen encoder states
  train_length_encoder.py   caption -> length, fine-tuning the encoder
  generate.py               unconditional sampling
  eval_text.py              text-conditional generation and scoring
  evaluate_samples.py       validity / uniqueness / novelty / diversity
  score_benchmark.py        the published ChEBI-20 benchmark columns
  analyze_distribution.py   descriptors, FCD, scaffolds against a reference corpus
  compare_configs.py        seed-aware verdicts across runs
  bench_ladder.py           speed checklist, one item per rung
configs/                    one yaml per kind of run
tests/                      pytest: configs, formulas, losses, checkpoints, smoke training
```

### Documentation

| File | Contents |
|---|---|
| [EXPERIMENTS.md](EXPERIMENTS.md) | the backlog: every hypothesis, result and verdict |
| [docs/experiments-zinc250k.md](docs/experiments-zinc250k.md) | the running lab notes, with the full tables |
| [docs/pipeline.md](docs/pipeline.md) | module-by-module walkthrough of a training run, speed checklist |
| [docs/pipeline-current.md](docs/pipeline-current.md) | the pipeline as it stands, stage by stage |
| [docs/architecture.md](docs/architecture.md) | what moved where and what was deliberately not changed |
| [docs/diffusion-review.md](docs/diffusion-review.md) | measured findings about padding and the diffusion setup |
| [docs/tokenizer-zinc250k.md](docs/tokenizer-zinc250k.md) | how the tokenizer was chosen, with the audit |

## Tests

```bash
pytest -q                    # includes the CPU smoke training run
pytest -q -k "not smoke"     # fast subset
```

## Running on a shared machine

Nothing here assumes a particular cluster, but two habits are worth keeping anywhere the
hardware is shared or the session can drop.

- **Cap the thread pools.** Torch takes every core it can see for intra-op parallelism, and
  RDKit- or FCD-based scoring will do the same. Set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS` and `NUMEXPR_NUM_THREADS` before launching, and keep the data-prep
  worker counts explicit in the config rather than defaulting to `os.cpu_count()`.
- **Detach long jobs and judge them by their output.** Start training under `setsid nohup`
  so a dropped connection does not take it down, check progress by the log growing and the
  checkpoints appearing, and verify the GPUs are actually idle before trusting a benchmark —
  a killed `torchrun` leaves workers spinning that no process listing will show you.
