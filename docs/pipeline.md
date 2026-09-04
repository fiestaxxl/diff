# Training pipeline, module by module

One yaml describes a run; `scripts/train.py` turns it into objects and hands them
to a single loop. Nothing else reads the config.

```bash
python scripts/train.py configs/diffusion_chebi.yaml
torchrun --nproc_per_node=8 scripts/train.py configs/diffusion_chebi.yaml precision=amp_bf16
```

## 1. Startup order

`scripts/train.py::main` follows the llm-foundry order (tokenizer, model, data,
optimizer, schedule, loggers, trainer):

| # | Call | Module | Result |
|---|---|---|---|
| 1 | `load_from_argv()` | `dimol/config.py` | yaml + CLI overrides merged into the `DimolConfig` schema, struct mode on |
| 2 | `init_distributed()` | `dimol/training/distributed.py` | `DistEnv(rank, local_rank, world_size, device)`; no-op without torchrun |
| 3 | `update_batch_size_info()` | `dimol/config.py` | `device_train_batch_size` and `device_train_grad_accum` derived from the global/micro batch |
| 4 | `seed_all(seed + rank)` | `dimol/training/distributed.py` | python/numpy/torch seeded per process, optional TF32 |
| 5 | `build_tokenizer()` | `dimol/builders.py` | `SmilesTokenizer` loaded from `tokenizer.path` (skipped when unset) |
| 6 | `build_model()` | `dimol/builders.py` → `dimol/registry.py` | `model.name` selects a factory: `diffusion_transformer` or `smiles_ar` |
| 7 | `torch.compile` / `DDP` | `scripts/train.py` | compile first, then the DDP wrapper (that order matters) |
| 8 | `build_dataloader()` | `dimol/data/loaders.py` | `train_loader` / `eval_loader` plus a `DistributedSampler` under DDP |
| 9 | `build_optimizer()` | `dimol/builders.py` → `dimol/training/optim.py` | parameter groups (no weight decay on `token_embedding`) + AdamW |
| 10 | `build_scheduler()` | `dimol/training/optim.py` | `warmup_cosine`; `ba`/`ep` durations converted with `steps_per_epoch` |
| 11 | `build_loggers()` | `dimol/builders.py` → `dimol/training/logging.py` | console / jsonl / comet / tensorboard, rank 0 only |
| 12 | `build_path()` | `dimol/builders.py` → `dimol/diffusion/paths.py` | `alpha`/`beta` schedules + `p_simple` of shape `[seq_len, emb_dim]` |
| 13 | `build_task()` | `dimol/builders.py` → `dimol/training/tasks.py` | `DiffusionTask` or `ARTask`: owns the loss, its weights and the grammar tables |
| 14 | `Trainer(...).maybe_resume()` | `dimol/training/trainer.py`, `checkpoint.py` | loads `load_path`, or the latest checkpoint when `autoresume: true` |
| 15 | `Trainer.fit()` | `dimol/training/trainer.py` | the loop below |

## 2. Module graph

```
                       configs/*.yaml
                             |
                     dimol/config.py            schema, ba/ep durations, batch math
                             |
   scripts/train.py ----> dimol/builders.py ----> dimol/registry.py   (name -> class)
                             |        |    \
                             |        |     \____ dimol/models/        diffusion_transformer, gpt, layers
                             |        |                                 denoiser (eps/x -> score)
                             |        \_________ dimol/diffusion/      conditionals (alpha/beta), paths,
                             |                                          diff_eqs (SDE), simulators
                             \__________________ dimol/data/           datasets (.npy), loaders, sources
                                        |
                             dimol/training/trainer.py                  the loop
                                        |
              +-------------------------+--------------------------+
              |                         |                          |
      training/tasks.py         training/optim.py          training/checkpoint.py
      (loss + metrics)          (param groups, lr)         (weights + trainer state)
              |                         |                          |
              +----------> training/logging.py <-------------------+
                           (console every step, jsonl, comet)
                                        |
                             dimol/eval/sampling.py                 periodic generation
                             dimol/eval/report.py                   validity / uniqueness / novelty
```

## 3. What one step does

`Trainer.fit()` in `dimol/training/trainer.py`:

1. **microbatch**: move the batch to the device, set `require_backward_grad_sync`
   to `False` on all but the last microbatch (so DDP all-reduces once per step).
2. **loss**: `task.compute_loss(model, batch, step)`. For diffusion this samples
   `t`, builds `x_t = alpha*x0 + beta*eps`, runs the denoiser inside autocast and
   combines MSE, cross-entropy and the optional grammar term. Metrics come back in
   the same dict.
3. **backward**: `loss / grad_accum` then `backward()` (`scaler.scale(...)` for fp16).
4. **step boundary**, once every `device_train_grad_accum` microbatches:
   `unscale_` → `clip_grad_norm_` → write the lr from the schedule → `optimizer.step()`
   → `zero_grad(set_to_none=True)`.
5. **metrics**: `reduce_metrics` averages the accumulated dict across ranks with a
   single all-reduce, then the console line and the namespaced metrics are logged.
6. **periodic events**, checked on the step counter: `eval_interval` runs the eval
   loader, `sampling.interval` generates molecules and logs rdkit validity,
   `save_interval` writes a checkpoint. `max_duration` ends the run.

The console line (every step, `console_log_interval: 1ba`):

```
[train ba 1200/9500 ep 3] loss 1.2346 | acc 0.8123 | mse 0.9800 | ce 0.2500 |
lr 2.900e-04 | gnorm 0.834 | tok/s 145.2K | mol/s 698.1 | dt 210ms | mem 34.1G
```

## 4. Data

`scripts/tokenize_dataset.py` writes one file pair per split:

```
<out_dir>/train_tokens.npy      (N, T) uint16     <out_dir>/train_attn_mask.npy   (N, T) uint8
<out_dir>/val_tokens.npy                          <out_dir>/val_attn_mask.npy
<out_dir>/test_tokens.npy                         <out_dir>/test_attn_mask.npy
```

`SmilesDataset` (`dimol/data/datasets.py`) also accepts a sharded layout, for a
corpus that does not fit into a single file:

```
<out_dir>/train_tokens_00000.npy, train_tokens_00001.npy, ...
<out_dir>/train_attn_mask_00000.npy, train_attn_mask_00001.npy, ...
```

Shards are sorted by name and concatenated logically, so one dataset covers the
whole split. Two consequences worth being explicit about:

* **Splits are separate datasets.** `train_loader` and `eval_loader` are built
  from their own `dataset.split`, so training never touches `val`/`test`; `test`
  is only read by `scripts/generate.py` when computing novelty.
* **Nothing "runs out".** When the train loader is exhausted that is the end of
  an epoch: the loop increments the epoch counter, calls `sampler.set_epoch` (so
  DDP reshuffles differently) and iterates the same dataset again. Training stops
  only on `max_duration`. Under DDP the `DistributedSampler` gives each rank a
  disjoint slice of every epoch, so the effective epoch is the whole split
  regardless of the number of GPUs.

Arrays are memory-mapped by default (`mmap: true` on the dataset node), so RAM
holds only the rows a worker touches.

`dimol/data/sources.py` is a different thing: it reads *raw* SMILES strings (a
HuggingFace dataset or a local `.txt`/`.csv`/`.parquet` file) and is used only by
the preparation scripts, `scripts/train_tokenizer.py` and
`scripts/tokenize_dataset.py`. Training never imports it.

## 5. Run artifacts

```
checkpoints/<run_name>/
  config.resolved.yaml      the config after overrides and interpolation
  env.json                  torch/cuda/gpu versions, git sha, dirty flag
  metrics.jsonl             one json line per logged event
  samples/ba<N>.txt         molecules sampled during training
  ep<E>-ba<N>/              config.json + weights + trainer_state.pt (autoresume)
```

## 6. Speed: checklist item -> code -> measurement

Measured with `scripts/bench_step.py` (synthetic batch, forward + backward +
optimizer step) and `scripts/bench_ladder.py` (each rung adds one item on top of
the previous one, every rung in a fresh process). The numbers below are CPU
medians for micro_bs=8, seq_len=208, 44.9M parameters. TF32, bf16 and fused AdamW
are CUDA features and read n/a here; rerun the ladder on the node:

```bash
python scripts/bench_ladder.py configs/diffusion_chebi.yaml
```

| # | Item | Where it lives | Measured here |
|---|---|---|---|
| 1 | TF32 | `dimol/training/distributed.py:96` (`set_float32_matmul_precision`), called from `scripts/train.py:125`; key `tf32` | see the ladder |
| 2 | bf16 autocast | `scripts/train.py:66` `_autocast_factory`, applied in `dimol/training/tasks.py::DiffusionTask.compute_loss`; key `precision` | see the ladder |
| 2b | fp16 + GradScaler for pre-Ampere | `scripts/train.py:76` `_build_scaler`; order scale -> backward -> unscale -> clip -> step in `dimol/training/trainer.py:193-222` | see the ladder |
| 3 | torch.compile | `scripts/train.py:146`; the graph-break fix is `dimol/models/layers.py:148` `apply_rotary`; keys `compile`, `compile_backend` | 152.7 -> 22.3 ms with tf32 and bf16, **6.85x** cumulative |
| 4 | Flash attention | `dimol/models/layers.py:140` and `dimol/models/gpt.py:65` (`F.scaled_dot_product_attention`) | 20.9 -> 15.4 ms per attention fwd+bwd, **1.36x** |
| 5 | Powers of two | config: `vocab_size 512`, `model_dim 768`, `num_heads 12` (head_dim 64) | vocab 500 -> 512: 786 -> 770 ms, **1.02x** |
| 6 | Fused AdamW | `dimol/training/optim.py:73` (auto on CUDA); key `optimizer.fused` | see the ladder |
| 7 | Grad accumulation | derived in `dimol/config.py:249` `update_batch_size_info`; loss divided at `dimol/training/trainer.py:193`, boundary at `:208` | not a speed knob |
| 8 | DDP | wrap `scripts/train.py:148`, sync flag `dimol/training/trainer.py:190`, one all-reduce `dimol/training/distributed.py:108`, `set_epoch` `dimol/training/trainer.py:181` | needs >1 GPU |
| + | Gradient clipping | `dimol/training/trainer.py:292` `_clip_gradients` | - |
| + | Warmup + cosine | `dimol/training/optim.py:92` `WarmupCosine` | - |
| + | Weight decay on 2D only | `dimol/training/optim.py:26` `param_groups` | - |
| + | Honest timing | `torch.cuda.synchronize()` at `dimol/training/trainer.py:227`, tok/s in the step line | - |
| + | One host copy per step | `dimol/training/trainer.py:304` `_to_floats` (was ~16 blocking `float()` calls) | see the ladder |
| + | autocast over readout+loss | `dimol/training/tasks.py::compute_loss`; key `loss.autocast_scope` | logits 218 -> 109 MiB at batch 512 |

## 7. Sequence length: bucketing instead of truncation

At `seq_len=208` with a mean molecule of about 40 tokens, roughly 80% of every
batch is padding. Trimming the corpus would lose molecules, so the code instead
supports length bucketing: batches are drawn from pools of similar length
(`dimol/data/samplers.py:30` `BucketBatchSampler`) and each batch is cut to its own
longest sequence (`dimol/data/collate.py:15` `trim_collate`). Keys:

```yaml
train_loader:
  bucket_by_length: true
  bucket_pool_factor: 64      # pool = 64 batches; sorted inside, shuffled outside
  trim_to_multiple_of: 8
```

Measured end to end on a synthetic corpus with a realistic length profile (median
39, p95 63, 81% padding), real model and real loss:

| Batching | median step | mean batch width |
|---|---|---|
| fixed length 208 | 772 ms | 208 tokens |
| bucketing + trim | 247 ms | 46 tokens |

**Bucketing requires a decision first.** With `loss.mask_padding: false` (the
default, i.e. the original behaviour) the loss averages over padded positions, so
trimming changes the objective, not just the speed:

| | total loss at T=208 | at T=48 |
|---|---|---|
| `mask_padding: false` | 98.28 | 17.55 (-82%) |
| `mask_padding: true` | 7.80 | 7.53 (-3.5%, inside the 3.1% noise of a different draw) |

`loss.mask_padding: true` enables the masked variants that the old trainer carried
as commented-out code: pads are excluded from attention and from both MSE terms
(cross-entropy already ignored them through `ignore_index`). It makes the loss
independent of padding, which is what makes bucketing safe, and it is pinned by
`tests/test_padding_mask.py`. It also changes the objective relative to every
previous run, so the baseline has to be re-measured after switching it on.
