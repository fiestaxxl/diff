# dimol: diffusion models for SMILES generation

A run is described by a single yaml file whose path is passed to the script.
The config conventions follow llm-foundry: `ba`/`ep` durations, dotted CLI
overrides, and `global_train_batch_size` + `device_train_microbatch_size` with
grad accumulation derived from them.

```bash
pip install -e ".[chem,data,logging,dev]"

# 1. tokenizer, then tokenize the corpus
python scripts/train_tokenizer.py  configs/tokenizer_chebi.yaml
python scripts/tokenize_dataset.py configs/tokenizer_chebi.yaml

# 2. training (single GPU or DDP); the launcher picks python or torchrun and tees the log
bash scripts/train.sh configs/diffusion_chebi.yaml
bash scripts/train.sh configs/diffusion_chebi.yaml --gpus 8

# 3. generation and the quality report
python scripts/generate.py configs/generate_chebi.yaml \
    generate.checkpoint=checkpoints/v12/ep499-ba9500
```

Every config field can be overridden from the command line, so sweeps need no
code changes:

```bash
python scripts/train.py configs/diffusion_chebi.yaml \
    run_name=lg_0.01 loss.lambda_grammar=0.01 max_duration=250ep precision=amp_bf16
bash scripts/train.sh configs/diffusion_chebi.yaml --gpus 8 run_name=lg_0.01 loss.lambda_grammar=0.01
bash scripts/sweep_grammar.sh          # lambda_grammar sweep
```

## What the yaml controls

| Section | Contents |
|---|---|
| `model` | denoiser architecture (`diffusion_transformer`) or AR model (`smiles_ar`) |
| `diffusion` | regime (`epsilon`/`x`), `alpha`/`beta` schedules, embedding noise, `t_eps` |
| `loss` | term weights (`lambda_mse`, `lambda_ce`, `lambda_grammar`), alpha gates, class weights |
| `train_loader` / `eval_loader` | dataset, workers, shuffling |
| `optimizer` / `scheduler` / `algorithms` | AdamW, warmup+cosine, gradient clipping |
| batching | `global_train_batch_size`, `device_train_microbatch_size` (accum is derived) |
| compute | `precision` (`amp_bf16`/`amp_fp16`/`fp32`), `tf32`, `compile`, `ddp_config` |
| control | `max_duration`, `eval_interval`, `save_interval`, `autoresume`, `load_path` |
| logging | `loggers` (console / jsonl / comet / tensorboard), `console_log_interval` |
| `sampling` | periodic generation during training (SDE steps, number of samples) |

## Training log

Every step reports loss, accuracy and tokens-per-sec:

```
[train ba 1200/9500 ep 3] loss 1.2346 | acc 0.8123 | mse 0.9800 | ce 0.2500 | mse_t0 0.0041 |
gram 0.0002 | lr 2.900e-04 | gnorm 0.834 | tok/s 145.2K | mol/s 698.1 | dt 210ms | mem 34.1G
```

The same numbers reach comet/tensorboard/jsonl under Composer-style names:
`loss/train/total`, `metrics/train/token_acc`, `throughput/tokens_per_sec`,
`lr-AdamW/group0`, `l2_norm/grad/global`, `time_seconds/batch/batch_total`.

## Run artifacts

```
checkpoints/<run_name>/
  config.resolved.yaml      the config after overrides and interpolation
  env.json                  torch/cuda/gpu versions plus git sha and dirty flag
  metrics.jsonl             every metric, one json line per event
  samples/ba<N>.txt         molecules sampled during training
  ep<E>-ba<N>/              checkpoint: weights + optimizer/scaler/step (autoresume)
```

## Layout

```
dimol/
  registry.py               name -> class: models, tasks, schedules, loggers, datasets
  config.py                 config schema, ba/ep durations, derived grad accum
  builders.py               config -> objects
  data/                     datasets (.npy, sharded), loaders, raw SMILES sources
  tokenization/             SMILES BPE tokenizer + audit
  models/                   layers, denoiser (DiT), AR baseline, score wrapper
  diffusion/                alpha/beta, probability path, SDE, simulators
  training/                 trainer, tasks (losses), optimizers, checkpoints, loggers, DDP
  eval/                     sampling, metrics, report
scripts/                    train.sh (entrypoint), train.py, generate.py, tokenizers, benchmarks
configs/                    one yaml per kind of run
tests/                      pytest: configs, formulas, losses, checkpoints, smoke training
docs/pipeline.md            module-by-module walkthrough of a training run, speed checklist
docs/diffusion-review.md    measured findings about padding and the diffusion setup
docs/architecture.md        what moved where, what was deliberately not changed (in Russian)
```

## Tests

```bash
pytest -q                    # includes the CPU smoke training run
pytest -q -k "not smoke"     # fast subset
```
