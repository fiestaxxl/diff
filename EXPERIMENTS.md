# Experiment backlog

Every hypothesis this project has tested, what it measured, and what the verdict was —
including the ones that failed and the claims that were later retracted. The running lab
notes with the full tables live in [docs/experiments-zinc250k.md](docs/experiments-zinc250k.md);
this file is the index and the verdicts.

Two corpora and two stages run through everything below. **Stage 1** is unconditional
pretraining on ZINC (250k molecules, later 214M). **Stage 2** is a text-conditional
fine-tune on ChEBI-20, where a caption describes the molecule to generate. The model is a
continuous embedding-space diffusion (Diffusion-LM style) over a fixed-width token canvas.

---

## How to read any number here

Three rules were established the hard way and every verdict below depends on them.

**Validation loss does not rank models.** Across eleven configurations spanning a 2.4x
range in sample quality, final validation loss varied by 1.5% and correlated with quality
at **r = +0.46** — the wrong sign. The sharpest case: three ChEBI fine-tunes with losses
0.8954 / 0.8956 / 0.8976 scored MACCS 0.129 / 0.144 / 0.518, so the lowest loss was the
worst model. No conclusion in this project rests on loss, and checkpoints are never
selected by it.

**Validity is gameable three ways at once** — by trimming in the decoder, by ending
sequences early, and by inventing atoms the corpus never contains (14–16% of accepted
molecules did this at one point). The unconditional headline metric is therefore
**usable**: distinct, valid, ≥10 heavy atoms, ≥1 ring, no bracket atom outside the corpus,
as a share of *all attempts*. Next to it, FCD and thirteen descriptors measured in the
corpus's own standard deviations.

**A configuration is a distribution, not a number.** Two runs with the *same seed* once
landed at 23.9% and 55.5% usable. `scripts/compare_configs.py` refuses to call an effect
unless the worst seed of a group beats the best seed of the reference. Three seeds is the
unit of measurement. (On the final ChEBI recipe the spread collapsed to ~0.006 MACCS, but
that was a result, not an assumption.)

---

## Stage 0 — the baseline, and why its numbers were void

Three substantive bugs were found in the original training code
(`ConditionalGaussianDenoiserTrainerLite.get_loss`):

1. The cross-entropy head was trained on `out_proj(x0)`, where `x0 = emb(tokens) + noise` —
   a function of the noised *input*, never of the denoiser's output — while at sampling
   time `out_proj` is applied to the SDE result. A train/inference mismatch.
2. `attention_mask` was hardcoded to `None` and all masking was commented out. At
   `seq_len=208` roughly 80% of attention and of the MSE went into padding.
3. The config held live objects and `to_dict()` was a generator, so hyperparameters were
   never actually logged.

**Verdict:** every pre-refactor result, including "grammar loss hurts" and "continuous
diffusion does not work for SMILES", was unusable as evidence. The pipeline was rebuilt
config-driven (one yaml, dotted CLI overrides, struct mode so typos are errors) before any
new claim was made. The diffusion formulas themselves were carried over verbatim; every
feature added since is an opt-in flag whose default reproduces the original behaviour.

---

## Infrastructure

| Item | Measured | Verdict |
|---|---|---|
| TF32 + bf16 autocast + `torch.compile` | 152.7 → 22.3 ms per step, **6.85x** cumulative | kept, default |
| Flash attention (`F.scaled_dot_product_attention`) | 20.9 → 15.4 ms per attention fwd+bwd, 1.36x | kept |
| Powers of two (vocab 500 → 512) | 786 → 770 ms, 1.02x | kept, free |
| Fused AdamW, one host copy per step, autocast over readout+loss | logits 218 → 109 MiB at batch 512 | kept |
| `torch.compile` **then** DDP, in that order | reverse order breaks graphs at every all-reduce | ordering matters |

Measured with `scripts/bench_ladder.py`, each rung adding one item in a fresh process.

**Trap:** a killed `torchrun` leaves workers spinning at 100% on every GPU. They do not
appear in `ps`, they do appear in `nvidia-smi --query-compute-apps`, and until killed every
timing measurement is wrong by a factor of three to six.

---

## Data and tokenizer

The tokenizer was chosen by audit (`dimol/tokenization/chem_audit.py`), not by compression:
losslessness first, then chemical integrity, then structural bookkeeping, with efficiency
only breaking ties among survivors.

| Finding | Detail |
|---|---|
| Plain BPE splits chlorine | `CCCl` → `CCC` + `l`, 39 times in one validation split at vocab 512. A bare `l` is not a chemical symbol and the model can emit it anywhere |
| Protection is free | Protecting `Cl`/`Br` as whole tokens removes every split at identical compression (same p50, p99, chars/token) |
| Corpus alphabet | Building the initial alphabet from corpus characters cut dead vocabulary from 71 to 28 |
| **BPE shortens structural distances** | Ring-closure digits move from 11 tokens apart (atom level) to 5, p95 from 39 to 22; 18% of ring closures end up inside a single token where they cannot be mismatched |
| Structure isolation | Forbidding merges across `(` and ring digits costs compression and gains nothing |

**Chosen:** `bpe_384_alpha`, vocab 386, 22.5 tokens per molecule on ZINC-250k.

Extended (not retrained) to ChEBI-20 with `scripts/extend_tokenizer.py`: 386 → 630 tokens,
every existing id preserved so a pretrained embedding table can be warm-started.
`scripts/grow_vocab.py` then grows the checkpoint — new embedding rows drawn at the
*trained* scale (the table grows ~50x over a run, so an init-scale row would be
unreachable), new readout biases starting at the minimum of the trained ones.

---

## Stage 1 — unconditional, ZINC-250k

| Hypothesis | Result | Verdict |
|---|---|---|
| Token budget matters more than anything | Budget dominates the whole screen; 160 tokens/param takes validity from 13.55% down to 7.23% under a uniform schedule | **budget is the first-order knob** |
| Model size beats epochs at equal step count | confirmed across the ladder | size wins |
| **Self-conditioning** | **+38% usable, −13% FCD, three seeds** | **the one change that improves both axes** |
| **Length conditioning** | matches corpus length and ring statistics almost exactly; costs yield alone, best in combination | kept, in combination |
| `tie_readout` | no effect | dropped |
| `ce_input=x0_hat` (train the readout on the reconstruction) | 2.83% / 2.69% against 5.28% baseline | **refuted** — the sampler integrates to t=0.999 where the state is close to `x0`, so a readout trained on `x0` is already matched to what it sees |
| `time_sampler=logit_normal` | doubles validity, worsens FCD | a trade, not a win |
| `x0_noise_kind` laplace / token_mixup | break generation outright | dropped |
| `pad_weight=0.6` | best validity in the study, worst distribution | **length gaming**, dropped |
| `min_snr_gamma` | strongly harmful | dropped — the token terms are already gated to high SNR |
| `ce_include_pad` (stage 1) | nothing alone, harmful combined | dropped *here*; revisited in stage 2, see below |
| `gate_mode=topk` | lowers the mean, does not narrow the seed spread | dropped |
| EMA / averaged weights | 4.40% → 0.24% | **harmful**, and the averaged checkpoint is legitimate |
| Embedding scale | fixes itself over training | no action |
| Solver: 100 steps vs 1000 | identical | 100 steps (stage 1) |
| Solver: sigma 1.0 vs deterministic limit | +1.4 points | sigma 1.0 |
| Solver time grid (5 variants) | uniform wins | uniform |
| `refine_rounds` (denoise-renoise cycles) | changes nothing — the sample is a fixed point | dropped |

### Decoding

| Mode | Result | Verdict |
|---|---|---|
| `grammar_close` — close brackets and rings while writing | **4.60% → 13.20% usable**, length distribution intact | **default** |
| `grammar_mixed` — also trim | higher validity, 59% trivial molecules | **do not use** |
| `strict_decode` — full connectivity state machine | **0 false rejections on all 12,443 corpus molecules**; +8.3pp validity on the best model | **default** |
| `restrict_atoms` | nothing on a large model — it was a small-model artifact | dropped |
| `length_floor` | 0% usable, 60-atom strings of repeated `[NH-]` | **voided 21 runs** before being decoupled from `length_prior` |

**Trap:** the repair decoder originally *substituted* the next-best token when the model's
choice was illegal, which inserted atoms the model never predicted (`[P@@]`, `[P]`) and
raised the metric. Changed to *skip*: close 39.3% → 40.4%, reference 13.07% → 15.27%. No
automatic test would have caught this — the strings stayed valid.

---

## Stage 1 — the corpus was the binding constraint

The run-to-run variance that survived every explanation (seed, warmup, learning rate,
readout tying, gate determinism) turned out to be **data starvation**: 231 passes over
ZINC-250k at the target budget. Moving to ZINC-20 (214,055,665 molecules, 27.22 real tokens
each, 5.83B tokens) settled it.

Capacity series at a matched 80 real tokens per parameter, three seeds each:

| model | steps | validity by seed | usable | aromatic rings | heavy atoms |
|---|---|---|---|---|---|
| 5.06M | 3,631 | 1.26 / 1.59 / 3.28% | 2.1% | 1.42 | 33.7 |
| 14.6M | 10,491 | 14.52 / 22.84 / 23.01% | 20.6% | 0.89 | 25.3 |
| **47.9M** | 34,375 | **61.07 / 61.49 / 63.64%** | **60.0%** | **1.49** | 26.1 |
| ZINC-20 corpus | — | — | — | 1.85 | 26.8 |

Three results, all new: **aromaticity is a capacity problem** (nothing else moved it — not
the objective, not the timestep density, not depth); **the seed spread collapsed** from 2.3x
to 2.5 points; and **batch 4096 starves small models** (5M got 3,631 updates instead of
17,600 and fell apart).

Training-free wins stacked on the best model: strict decoding +8.3pp, 300 solver steps
+5.5pp, together **75.85% validity, 98.6% unique, 100% novel**.

---

## Stage 2 — text-conditional on ChEBI-20

Architecture: cross-attention in every block to frozen SciBERT caption states, output
projection zero-initialised so the graft is inert until trained. Classifier-free guidance
from the same weights with the caption masked. `caption_dropout: 0.1` trains the
unconditional branch in place.

**The control that makes every number readable:** pairing each molecule with *another*
molecule's caption gives MACCS 0.270–0.294. Without that floor, 0.5 looks like success when
half of it is just "a typical ChEBI molecule".

### Progression

| Milestone | MACCS | What was added |
|---|---|---|
| shuffled-caption floor | 0.270 | — |
| first end-to-end | 0.368 | cross-attention + CFG, 9 minutes of training |
| 60k steps | 0.518 | real training time instead of 6k |
| + length head | 0.588 | length predicted from the caption |
| 120k steps | 0.661 | time is the largest lever on this stage |
| 200k steps | 0.688 | step saturation |
| **properly annealed 200k + padding supervision** | **0.756** | see below |

### What was settled

| Hypothesis | Result | Verdict |
|---|---|---|
| Was 60k steps enough? | 40k → 240k curve: 0.517, 0.593, 0.653, 0.682, **0.688**, 0.685 | **saturates at 160–200k**; increments halve and hit zero |
| Learning rate | 1e-4 → 0.592, 2e-4 → 0.645, **3e-4 → 0.661**, 5e-4 → 0.636, 1e-3 destroys the model | 3e-4, bracketed both sides |
| Domain adaptation (15k unconditional ChEBI steps first) | +0.036 at 60k, **−0.008 at 120k** | **convergence speed, not final quality** — retracted after measuring the curve instead of the endpoint |
| Drop the length input, rely on text alone | 0.588 with it → 0.571 (canvas still pinned) → **0.511** (nothing) | **keep it**; the input is worth +0.017 and sampling-time pinning +0.060, and the two are separable |
| Classifier-free guidance at 200k | BLEU 0.676 → 0.639 → 0.580, Levenshtein 32.9 → 43.4, FCD 1.23 → 1.67 | **monotonically harmful** once conditioning is trained; closed |
| Solver steps 300 → 600 | +0.004 MACCS, −1.4pp validity | saturated at 300 |
| **Heun (2nd-order) solver** | implemented; matched-compute comparison **never completed** | **open** |
| Scale 47.9M → 171M | MACCS 0.695 vs 0.695 | **flat** — capacity is not the constraint at this data scale |
| Train on ChEBI only, no ZINC pretraining | validity **0.069** and **0.000**, strings of 400 characters | **collapses**; ZINC pretraining is required, not optional |
| SMILES augmentation (10/20 spellings per molecule) | validity 0.75 → 0.585, exact 0.045 → 0.007, BLEU 0.726 → 0.585 | **harmful**, see below |
| **Properly annealed schedule** | same checkpoint budget, cosine to the endpoint instead of a mid-schedule slice: BLEU 0.685 → 0.726, Levenshtein 30.7 → 23.2, FCD 1.14 → 0.82 | **large, free, and it invalidates earlier mid-slice comparisons** |
| **`ce_include_pad` in stage 2** | BLEU +0.017, Levenshtein −1.06, MACCS +0.009, FCD −0.06 against the best base seed | **positive but single-run**; confirmation seeds trained, never scored |

### Length prediction

Giving the sampler the true length is worth more than anything else measured on this stage,
so three predictors were built and compared:

| Length source | val MAE | exact length | BLEU | Exact | Leven | MACCS | FCD | Validity |
|---|---|---|---|---|---|---|---|---|
| head on frozen states, 12 epochs | 7.5 | 12.1% | 0.676 | 0.031 | 32.87 | 0.723 | 1.23 | 0.771 |
| head on frozen states, 80 epochs | 6.4 | 19.9% | 0.665 | 0.047 | 33.31 | 0.736 | 1.10 | 0.763 |
| **fine-tuned encoder** | **3.3** | 18.8% | 0.685 | 0.048 | 30.68 | 0.748 | 1.14 | 0.745 |
| oracle (true length) | 0.0 | 100% | 0.707 | **0.142** | 28.69 | 0.777 | 0.98 | 0.774 |

**Halving the error did not move exact match** (0.047 → 0.048). Exact match needs the length
to be *exactly* right, and the share of exactly-right lengths is 19% for both predictors
even though one is twice as accurate on average. MACCS and Levenshtein, which tolerate
approximation, did improve. A prediction of 0.07–0.08 written down beforehand was wrong
because it assumed the payoff scales with MAE; it does not.

Tighter pinning also costs validity monotonically (0.771 → 0.763 → 0.745): a fixed canvas
leaves the sampler no room to recover.

**The frozen encoder is lossy, and this is the proof.** The head on frozen SciBERT states
plateaued at MAE 6.4 while its training loss kept falling — limited by its inputs, not its
size. Fine-tuning the encoder for the same task cut the error to 3.3.

### Termination: where the problem actually lives

`ce_include_pad=false` means the cross-entropy passes `ignore_index=pad_idx`, so **the
readout is never trained to emit padding**. The hypothesis was that this is why the model
cannot decide where a molecule ends, and why an external length predictor is structurally
required.

Half right. With padding supervision the model learned the class completely (CE on padding
4.69 → 0.0004) and every metric improved slightly. But **it still cannot stop on its own**:
given no length at all, MACCS falls 0.756 → 0.599 and Levenshtein rises 21.6 → 50.3.

The MSE always covered padding positions, so the generative part never changed — only the
readout did. Naming padding from a clean latent and *producing* a padding latent at the
right position are different problems. **Termination lives in the diffusion, not the
readout**, which is where the architectural work should point.

### Why augmentation hurts

All three completed arms agree, and neither the canonical/random split nor 10 vs 20
spellings made a difference:

| | BLEU | Exact | Leven | MACCS | FCD | Validity |
|---|---|---|---|---|---|---|
| base (two seeds) | 0.726 | 0.045 | 22.9 | 0.750 | 0.82 | 0.75 |
| 10 spellings, canonical included | 0.589 | 0.007 | 37.48 | 0.686 | 1.20 | 0.585 |
| 10 spellings, all random | 0.579 | 0.007 | 38.07 | 0.686 | 1.26 | 0.607 |
| 20 spellings, canonical included | 0.584 | 0.006 | 37.88 | 0.692 | 1.27 | 0.585 |

Augmentation adds no molecular information, only **multimodality in string space**. An
autoregressive model picks one mode; a diffusion model on a continuous canvas has to spread
mass across every spelling of the same molecule, and spread mass in a latent is
interpolation rather than choice. That is a mechanism specific to continuous diffusion, and
it predicts exactly the observed collapse in validity.

---

## What ChEBI-20 actually measures

Scored on the full 3,202-molecule test split with `scripts/score_benchmark.py`, which
produces all the published columns. (Text2Mol needs an external retrieval model and is not
computed; 95 test captions describe isotopes and carry no structure at all, and stay in the
exact-match denominator as they do for every published row.)

| Relation of a test molecule to the training set | Share |
|---|---|
| appears verbatim in train | 2.1% |
| shares a Bemis-Murcko scaffold with train | 86.9% |
| nearest-neighbour Morgan Tanimoto > 0.9 | 37.5% |
| 0.7–0.9 | 28.7% |
| ≤ 0.7 | 33.8% |

The nearest-neighbour distribution has median 0.787 and **p75 = 1.000**: a quarter of the
test set has a training molecule with an identical Morgan fingerprint, differing only in
stereochemistry or tautomer.

**Every metric is a steep function of that distance:**

| Model | Bucket | Exact | MACCS | Morgan | Validity |
|---|---|---|---|---|---|
| ours, encoder length | near-duplicate | 0.088 | **0.851** | 0.655 | 0.831 |
| | related | 0.028 | 0.733 | 0.423 | 0.687 |
| | novel | 0.020 | **0.582** | 0.284 | 0.699 |
| ours, true length | near-duplicate | **0.229** | **0.880** | 0.725 | 0.852 |
| | related | 0.109 | 0.754 | 0.487 | 0.717 |
| | novel | 0.075 | 0.617 | 0.347 | 0.735 |

An aggregate on ChEBI-20 is mostly a statement about how close the test molecules are to
the training set. On the near-duplicate third our MACCS of 0.851 is level with the published
aggregate of 0.854, and with a true length 0.880 exceeds the 0.874 reported without
correction. The aggregate gap is carried by the novel third — the only part of the benchmark
where the task is what it claims to be, and where no published work reports a number.

Length improvement helps the novel bucket most (+0.021 MACCS, against +0.009 on
near-duplicates), which is the opposite of what a memorisation effect would do.

---

## Standing against the published table

Full ChEBI-20 test split, 300 solver steps, guidance 0.

| Model | BLEU | Exact | Leven↓ | MACCS | RDK | Morgan | FCD↓ | Validity |
|---|---|---|---|---|---|---|---|---|
| Transformer | 0.499 | 0.000 | 57.66 | 0.480 | 0.320 | 0.217 | 11.32 | 0.906 |
| T5-Base | 0.762 | 0.069 | 24.95 | 0.731 | 0.605 | 0.545 | 2.48 | 0.660 |
| MolT5-Base | 0.769 | 0.081 | 24.46 | 0.721 | 0.588 | 0.529 | 2.18 | 0.772 |
| TGM-DLM w/o corr | 0.828 | 0.242 | 16.90 | 0.874 | 0.771 | 0.722 | 0.89 | 0.789 |
| TGM-DLM | 0.826 | 0.242 | 17.00 | 0.854 | 0.739 | 0.688 | 0.77 | 0.871 |
| **ours (padding supervision + encoder length)** | 0.743 | 0.050 | **21.62** | **0.756** | 0.586 | 0.501 | **0.76** | 0.763 |
| ours, base recipe, seed 43 / 44 | 0.726 / 0.726 | 0.045 / 0.044 | 23.17 / 22.68 | 0.747 / 0.753 | 0.576 / 0.580 | 0.488 / 0.490 | 0.82 / 0.82 | 0.753 / 0.750 |

**FCD 0.76 is first in the table.** Levenshtein and MACCS beat MolT5-Base. The gap that
remains is concentrated in exact match and in the fine-grained fingerprints (RDK, Morgan) —
the molecule is in the right class with the wrong details — which is what points at the
frozen text encoder and the tokenizer as the next levers.

---

## Retracted claims

Kept here because each was stated with confidence before being measured properly.

| Claim | Why it fell |
|---|---|
| "Aromaticity is a depth phenomenon" | confounded — the 8-block winner had 22% more parameters. At matched parameter count (2/4/8/12/16 blocks) shape does nothing and size does everything |
| "Domain adaptation closes the distribution gap" | measured only at 60k; the advantage is gone by 120k. A warm start, not a fix |
| "Length conditioning is a shortcut the model exploits" | the 2×2 showed the right length with a wrong caption scores 0.296 against a 0.270 floor |
| "`ce_input=x0_hat` is the most underrated experiment" | it had already been run and refuted months of log earlier; proposed without checking |
| "MAE 3–4 will give exact match 0.07–0.08" | assumed the payoff scales with MAE. It does not — 0.048 |
| "Padding supervision is the first half of the termination fix" | the model learned to name padding and still cannot stop on its own |
| "171M pretraining died from an external SIGKILL" | it was alive; concluded from a single 0%-utilisation sample |
| "FCD is 2.20" | measured on 1000 captions, where the Fréchet estimate is biased upward. On the full test split, 1.23 |

---

## Open — trained but never scored

Four 200k-step runs completed and were never generated or evaluated. This is the cheapest
outstanding work in the project.

| Run | What it answers |
|---|---|
| `cepad_s43`, `cepad_s44` | whether padding supervision survives the three-seed rule |
| `cepad_tglobal` | whether pooling the caption into the adaLN vector (a global conditioning path beside per-position cross-attention) helps |
| `aug20_random` | the fourth corner of the augmentation grid |

## Open — not yet started, in priority order

1. **Unfreeze the text encoder.** The strongest evidence-backed lever left: frozen SciBERT
   states plateaued a length head at MAE 6.4 while fine-tuning the encoder reached 3.3. If
   the frozen states lose length, they lose substructure detail too — which is exactly where
   RDK and Morgan fail. Needs the caption path rebuilt: token ids plus the encoder in the
   training loop with its own learning-rate multiplier.
2. **A content channel in the latent.** The architectural answer to termination, now that it
   is localised to latent generation rather than the readout: an extra channel denoised
   alongside the embedding that says "content continues here", so length emerges from the
   model instead of being dictated to it.
3. **A tokenizer trained on ZINC and ChEBI jointly.** Current merges are ZINC-shaped, and
   fine-grained substructure accuracy is what a poorly fitted tokenizer damages. Costs a
   fresh stage-1 pretrain.
4. **Heun solver at matched compute.** Implemented and wired (`sampling.solver=heun`), never
   measured to completion.
5. **Honest test-time compute** — K length candidates scored by the model, or resampling
   invalid generations. Must be reported with `usable` and FCD alongside, or it is the same
   metric gaming the project already rejected once.
