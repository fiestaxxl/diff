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
unit of measurement. (On the final ChEBI recipe the spread collapsed — 0.0023 MACCS across
the two post-fix seeds, 0.748119 against 0.750427, and 0.0055 on the pre-fix base recipe,
0.747308 against 0.752770. The ~0.006 this rule used to quote was the subtraction of two
rounded numbers from the second pair and is retracted below. Verdicts here are read against
the wider 0.0055, which is the conservative choice. The collapse was a result, not an
assumption.)

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
| **Heun (2nd-order) solver at matched compute** | 150 Heun steps against 300 Euler steps, one run each: BLEU 0.678 vs 0.676, exact 0.034 vs 0.031, Levenshtein 32.79 vs 32.87, MACCS 0.720 vs 0.723, RDK 0.534 vs 0.537, Morgan 0.451 vs 0.453, FCD 1.28 vs 1.23, validity 0.770 vs 0.771 | **flat at single-run resolution** — every delta is at or below the seed spread measured on neighbouring configurations, and this is one run against one run, not a seeded group |
| Scale 47.9M → 171M | MACCS 0.695 vs 0.695 | **flat** — capacity is not the constraint at this data scale |
| Train on ChEBI only, no ZINC pretraining | validity **0.069** and **0.000**, strings of 400 characters | **collapses**; ZINC pretraining is required, not optional |
| SMILES augmentation (10/20 spellings per molecule) | validity 0.75 → 0.585, exact 0.044 → 0.007, BLEU 0.726 → 0.589 (best of the four arms; worst 0.579) | **harmful**, see below |
| **Properly annealed schedule** | same checkpoint budget, cosine to the endpoint instead of a mid-schedule slice: BLEU 0.685 → 0.726, Levenshtein 30.7 → 23.2, FCD 1.14 → 0.82 | **large, free, and it invalidates earlier mid-slice comparisons** |
| **`ce_include_pad` in stage 2** | three seeds against two, Δ of the group means: BLEU +0.011, exact +0.006, Levenshtein −1.24, MACCS +0.006, RDK +0.007, Morgan +0.012, validity +0.011, FCD −0.004 | **confirmed on seven metrics of eight**, FCD not established — the confirmation seeds are trained *and* scored, see the seeded table below |

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

All four arms agree, and neither the canonical/random split nor 10 vs 20 spellings made a
difference:

| | BLEU | Exact | Leven | MACCS | FCD | Validity |
|---|---|---|---|---|---|---|
| base (two seeds) | 0.726 | 0.044 | 22.9 | 0.750 | 0.82 | 0.75 |
| 10 spellings, canonical included | 0.589 | 0.007 | 37.48 | 0.686 | 1.20 | 0.585 |
| 10 spellings, all random | 0.579 | 0.007 | 38.07 | 0.686 | 1.26 | 0.607 |
| 20 spellings, canonical included | 0.584 | 0.006 | 37.88 | 0.692 | 1.27 | 0.585 |
| 20 spellings, all random | 0.580 | 0.007 | 37.96 | 0.679 | 1.31 | 0.595 |

Augmentation adds no molecular information, only **multimodality in string space**. An
autoregressive model picks one mode; a diffusion model on a continuous canvas has to spread
mass across every spelling of the same molecule, and spread mass in a latent is
interpolation rather than choice. That is a mechanism specific to continuous diffusion, and
it predicts exactly the observed collapse in validity.

---

## What ChEBI-20 actually measures

Scored on the full 3,202-molecule test split with `scripts/score_benchmark.py`, which
produces eight of the nine published columns. (Text2Mol needs an external retrieval model
and is not computed, so no ranking in this file covers it. Earlier scorings reported 95 test
captions as isotope entries carrying no structure; that was wrong. The evaluation tokenizer
was pointed at the 391-token ZINC vocabulary while the corpus used the 635-token shared one,
which decoded 196 of 3,202 references wrongly — 95 to unparseable strings and the rest to a
different molecule — and the affected rows are salts and complexes. The 95 is the
`unparseable_reference` every pre-fix full-split run records, and every post-fix run records
0. The "75 unparseable and 121 a different molecule" this file used to print is not a split
the scorer ever produced: either the total is 196 with 101 wrong molecules, or it is 216 with
121, and nothing in the repository decides it, so only the measured 95 is quoted. (The
1,000-caption subsets record 30, which is the proportional count there.) After the fix the split
has none, and exact match divides by all 3,202 captions either way. Runs scored before that
fix are marked *pre-fix* below; on the one configuration measured both ways the fix is worth
MACCS −0.006 and validity +0.015, which is the size of a real effect in this project.)

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
the training set. On the near-duplicate third our MACCS is 0.851, and 0.880 with a true
length. Those two numbers used to be set against TGM-DLM's 0.854 and 0.874; that comparison
is withdrawn. Those are full-split aggregates on the 3,300-pair split, ours is one stratum —
37.5% of our 3,202-pair split, selected for proximity to training, which this section's own
argument says drives every metric — and both of ours are pre-fix. No published work reports
a per-stratum number, so nothing here is a ranking. The aggregate gap is carried by the novel
third, the only part of the benchmark where the task is what it claims to be.

Length improvement helps the novel bucket most: against the encoder's lengths, the true
length is worth +0.035 MACCS on the novel third, +0.029 on the near-duplicate third and
+0.021 on the related third. The largest gain falls where memorisation can help least, which
is the opposite of what a memorisation effect would do.

**Provenance of the per-stratum numbers, here and in the candidate-decoding section.** The
scorer stores full-run aggregates only, so no stratified cell in this file — these, the
selection-rule buckets below, or the retrieval floor's 0.896 / 0.832 / 0.619 — has a stored
raw counterpart to check against. They are internally consistent with each other and every
quantity derived from them reproduces from the printed cells, but they have not been verified
the way the aggregate rows have, and a rerun should write them to the metrics store.

**The retrieval floor, and what it ranks.** TF-IDF over caption text, cosine to the nearest
training caption, return that molecule — a system that generates nothing — scores BLEU 0.786,
exact 0.004, Levenshtein 20.57, MACCS 0.785, RDK 0.690, Morgan 0.576, FCD 0.48 and validity
0.994 on our full split with our scorer. Ranked against a fourteen-row field — the eleven
published rows of the standing table below, the floor itself, and our two best full-split
rows, the 10-candidate worst-of-two and the single-trajectory worst-of-two — and counting only
the rows that report a column, it sits:

| column | floor's rank | what it beats |
|---|---|---|
| Validity | 3 of 14 | everything except BioT5 and BioT5+, both at 1.000 |
| FCD↓ | 5 of 13 | Transformer, T5-base, MolT5-base, T5-large, MolT5-large, TGM-DLM, both of ours |
| BLEU | 7 of 13 | Transformer, GIT-Mol, T5-base, MolT5-base, both of ours |
| MACCS / RDK / Morgan | 8 of 14 | Transformer, GIT-Mol, T5-base, MolT5-base, both of ours |
| Levenshtein↓ | 8 of 13 | Transformer, GIT-Mol, T5-base, MolT5-base, our single-trajectory row |
| Exact | 13 of 14 | the Transformer's 0.000, and nothing else |

**The floor does not beat TGM-DLM on the fingerprint metrics.** TGM-DLM reads MACCS 0.854,
RDK 0.739 and Morgan 0.688 against the floor's 0.785 / 0.690 / 0.576, and it leads on exact
match 0.242 against 0.004. Nor does it beat T5-large, MolT5-large, MolXPT, BioT5, BioT5+ or
LDMol on those columns. What the floor does beat on the fingerprints is four published
systems — Transformer, GIT-Mol, T5-base, MolT5-base — and both of our own rows. That is the
claim, and it is smaller than the one this file used to make.

It is still the useful claim, because **ordering the eight metrics by how well a lookup table
does in them is a ranking of how much each metric rewards lookup.** Validity sits at the top
and is saturable to 1.000 by generating SELFIES, which BioT5 and BioT5+ do; FCD sits next and
rewards returning real training molecules, which is exactly what retrieval returns; the
fingerprints sit in the middle, where returning a molecule of roughly the right class scores
most of what an exactly right molecule would; exact match is at the bottom, 0.004, the one
column where a retrieved molecule is almost never the right one. On the full split the floor
is above every row we produce on the fingerprints, on BLEU and on FCD. (One 1,000-caption
row, `smcsel_sharp95`, reads MACCS 0.796 and Morgan 0.579 over 873 pairs; that comparison is
on a different denominator and the file claims its gain neither way — see the particle-filter
table.)

So the aggregates are reachable by lookup, and the floor is still not a verdict on the
benchmark: TGM-DLM and LDMol both clear it on the fingerprints, and LDMol clears it on seven
of the eight columns — MACCS 0.973, RDK 0.950, Morgan 0.931, exact 0.530, FCD 0.20, BLEU
0.926, Levenshtein 6.75 — losing only validity, 0.941 against 0.994. What survives is a
statement about a set of systems, ours among them: the Transformer, GIT-Mol, T5-base,
MolT5-base and we ourselves do not clear a retrieval baseline on the fingerprint metrics.
Seven of the eleven published rows do clear it there — TGM-DLM, T5-large, MolT5-large,
MolXPT, BioT5, BioT5+ and LDMol — and they are the counterexamples that make this a claim
about a set of systems rather than a tautology; TGM-DLM and LDMol are the two diffusion rows
among them.

---

## Standing against the published table

Three kinds of row, and they are not interchangeable. **Published rows** are the numbers
their authors report, on the uncurated 3,300-pair ChEBI-20 test split, scored by their own
scorers; all eleven are now taken from one source, Table 1 of the LDMol paper. **Calibration
rows** are a published checkpoint we ran ourselves and a no-model retrieval floor, both put
through our scorer on our split. **Our rows** are produced by `scripts/score_benchmark.py` on
our curated 3,202-pair split, 300 solver steps, guidance 0. Only rows of the same kind
compare exactly.

| Model | BLEU | Exact | Leven↓ | MACCS | RDK | Morgan | FCD↓ | Validity | pairs |
|---|---|---|---|---|---|---|---|---|---|
| Transformer *(published, AR)* | 0.499 | 0.000 | 57.660 | 0.480 | 0.320 | 0.217 | 11.32 | 0.906 | not reported |
| GIT-Mol *(published, AR)* | 0.756 | 0.051 | 26.315 | 0.738 | 0.582 | 0.519 | — | 0.928 | not reported |
| T5-base *(published, AR)* | 0.765 | 0.069 | 24.950 | 0.731 | 0.605 | 0.545 | 2.48 | 0.660 | not reported |
| MolT5-base *(published, AR)* | 0.769 | 0.081 | 24.458 | 0.721 | 0.588 | 0.529 | 2.18 | 0.772 | not reported |
| T5-large *(published, AR)* | 0.854 | 0.279 | 16.721 | 0.823 | 0.731 | 0.670 | 1.22 | 0.902 | not reported |
| MolT5-large *(published, AR)* | 0.854 | 0.311 | 16.071 | 0.834 | 0.746 | 0.684 | 1.20 | 0.905 | not reported |
| MolXPT *(published, AR)* | — | 0.215 | — | 0.859 | 0.757 | 0.667 | 0.45 | 0.983 | not reported |
| BioT5 *(published, AR)* | 0.867 | 0.413 | 15.097 | 0.886 | 0.801 | 0.734 | 0.43 | **1.000** | not reported |
| BioT5+ *(published, AR)* | 0.872 | 0.522 | 12.776 | 0.907 | 0.835 | 0.779 | 0.35 | **1.000** | not reported |
| TGM-DLM *(published, diffusion)* | 0.826 | 0.242 | 17.003 | 0.854 | 0.739 | 0.688 | 0.77 | 0.871 | not reported |
| **LDMol *(published, diffusion)*** | **0.926** | **0.530** | **6.750** | **0.973** | **0.950** | **0.931** | **0.20** | 0.941 | not reported |
| MolT5-base, our run of the released checkpoint | 0.527 | 0.043 | 40.34 | 0.755 | 0.583 | 0.512 | 1.63 | 0.782 | 2503 |
| TF-IDF caption retrieval, generates nothing | 0.786 | 0.004 | 20.57 | 0.785 | 0.690 | 0.576 | 0.48 | 0.994 | 3182 |
| ours, padding supervision, mean of 3 seeds *(pre-fix)* | 0.737 | 0.050 | 21.69 | 0.756 | 0.585 | 0.501 | 0.81 | 0.763 | 2412 |
| ours, base recipe, mean of 2 seeds *(pre-fix)* | 0.726 | 0.044 | 22.93 | 0.750 | 0.578 | 0.489 | 0.82 | 0.751 | 2389 |
| ours, single trajectory, worst of 2 post-fix seeds | 0.736 | 0.052 | 21.72 | 0.748 | 0.575 | 0.492 | 0.81 | 0.775 | 2495 |
| **ours + 10-candidate decoding, worst of 2 seeds** | 0.781 | 0.143 | 18.18 | 0.771 | 0.629 | 0.545 | 0.74 | 0.993 | 3189 / 3178 |

**Where the eleven published rows come from, and this is the first table here with a source.**
All eleven are Table 1 of the LDMol paper (Chang and Ye, ICML 2025, arXiv:2405.17829v4),
supplied by the project owner. Ten of the eleven are therefore LDMol's transcription of other
papers' reported numbers, not those papers read directly; only the LDMol row is the authors'
own measurement. That is one transcription step better than what this table used to hold and
one step short of reading each source, so the citation to give for a Transformer or a BioT5+
number is still the LDMol table, not the original paper.

**What it supersedes, and the one disagreement.** The five oldest rows were transcribed from
a chat message in commit 29ad1f6 (2026-09-25) and were never checked against anything. Four
of them overlap the sourced table — Transformer, T5-base, MolT5-base and TGM-DLM — and the
sourced table replaces them. Checked cell by cell, the Transformer, MolT5-base and TGM-DLM
rows agree exactly, and so does the T5-base validity of 0.660, which is the odd-looking number
in the whole table and which two independent transcriptions now carry. **One cell disagrees:
T5-base BLEU, 0.762 in the chat transcription against 0.765 in LDMol's Table 1.** The sourced
value stands and 0.762 is withdrawn; the difference changes no comparison in this file.
The fifth old row, TGM-DLM w/o correction (BLEU 0.828, exact 0.242, Levenshtein 16.90, MACCS
0.874, RDK 0.771, Morgan 0.722, FCD 0.89, validity 0.789), has **no counterpart in the sourced
table** — LDMol reports TGM-DLM without that variant — so it is kept here rather than in the
table, still unverified. It sits above the retrieval floor on six of the eight columns, so
including it would push the floor one place lower on each of them: the three fingerprints,
8 of 14 to 9 of 15; BLEU, 7 of 13 to 8 of 14, against its 0.828; Levenshtein, 8 of 13 to
9 of 14, against its 16.90; and exact match, 13 of 14 to 14 of 15, against its 0.242, where
the floor stays last but one. Only FCD (0.89) and validity (0.789) are worse than the floor's
0.48 and 0.994 and leave its rank untouched. It would push our best row one place lower in
the same six columns.

**LDMol.** "LDMol: A Text-to-Molecule Diffusion Model with Structurally Informative Latent
Space Surpasses AR Models", Jinho Chang and Jong Chul Ye, ICML 2025, arXiv:2405.17829 — cite
v4, the ICML version; the v1 preprint of May 2024 reports weaker numbers and must not be
quoted in its place. Code at https://github.com/jinhojsk515/LDMol. The row is
caption-to-molecule on the standard 3,300-pair test split, and its MACCS, RDK and Morgan are
Tanimoto similarities, the same quantity we compute. LDMol reports no Text2Mol column.
Two columns in the table are not reported at all by their authors and are printed as dashes:
GIT-Mol's FCD, and MolXPT's BLEU and Levenshtein. Every rank in this file counts only the
rows that report the column.

**Our scorer does not reproduce published numbers.** We ran the released MolT5-base
checkpoint (`laituan245/molt5-base-caption2smiles`) ourselves, on our split, with our scorer.
Six of eight columns move: BLEU 0.527 against a published 0.769, Levenshtein 40.34 against
24.46, exact 0.043 against 0.081, MACCS 0.755 against 0.721, Morgan 0.512 against 0.529, FCD
1.63 against 2.18. Same checkpoint, same split, same scorer. The curated split does not
explain it: for our Levenshtein to fall to 24.46 the 98 pairs we drop would have to average a
distance of −494. No TGM-DLM, LDMol, T5 or Transformer output has ever passed through our
scorer. So **every cross-paper statement in this file is a comparison against the published
numbers as reported by their authors, which our scorer does not reproduce on the one system
we can calibrate.** Within our own rows the comparisons are exact; across that line they are
indicative and nothing more.

**Split.** The published rows are 3,300 pairs; ours are 3,202. `scripts/prepare_paired.py`
drops a pair whole when the reference does not parse or exceeds the 128-token canvas — 98
pairs, 3.0% — and those are the unparsable and the longest molecules, exactly the ones our
canvas cannot emit, so the curation is not neutral with respect to validity or exact match.
Counting the 98 as failures puts our worst-seed validity at 0.963 rather than 0.993 and our
exact match at 0.138 rather than 0.143.

**Pairs.** `pairs` is the denominator of MACCS, RDK, Morgan and FCD: the captions where both
the reference and the generation parse. BLEU, exact match, Levenshtein and validity are over
all 3,202 captions in every row of ours and are comparable as printed. The published rows
report no denominator at all. Eight columns, not the published nine: Text2Mol is not computed.

**One of our two worst composites carries two denominators.** A worst-of-N row takes each
column from whichever seed did worst in it, so no single seed produced the row. The
candidate-decoding row takes its fingerprints from seed 42 (3,189 pairs) and its FCD from
seed 43 (3,178) — two `pairs`-denominated cells from two seeds — so its cell is printed as
the pair. The single-trajectory row is not like that. All four of its `pairs`-denominated
cells, MACCS 0.748, RDK 0.575, Morgan 0.492 and FCD 0.81, are `fix_k0`'s over 2,495 pairs,
because a per-metric worst takes the higher FCD and `fx_cepad_s45` scores the better one,
0.76 against 0.81. What `fx_cepad_s45` supplies is exact match 0.052, Levenshtein 21.72 and
validity 0.775, and those are over all 3,202 captions by the definition above, so 2,481 is
the denominator of nothing in the row and the cell now reads 2,495 alone. This file used to
print 3,178 alone against fingerprint cells that came from the 3,189-pair seed, and
2,495 / 2,481 against a row with one `pairs` denominator.

On the right denominator the candidate-decoding row averages its fingerprints over 3,189
captions against 2,412 and 2,389 for the two pre-fix rows above it — 777 and 800 extra
captions, nearly all of them generations that used to fail to parse, which are the hard ones.
Its 0.771 / 0.629 / 0.545 are therefore not an improvement on 0.756 / 0.585 / 0.501 measured
over the same molecules, and its FCD is a distance against a different reference set.

**Where this leaves us.** LDMol leads our best row on seven of the eight columns: BLEU 0.926
against 0.781, exact 0.530 against 0.143, Levenshtein 6.75 against 18.18, MACCS 0.973 against
0.771, RDK 0.950 against 0.629, Morgan 0.931 against 0.545, FCD 0.20 against 0.74. Four of
those seven survive the only calibration this project has. On the MolT5-base checkpoint our
scorer moves BLEU by 0.242, Levenshtein by 15.9 and FCD by 0.553, and LDMol's margins over us
in those three columns are 0.145, 11.43 and 0.54 — every one of them smaller than the shift,
so they are noise at this resolution and are not evidence of anything. The margins that do
clear it are exact match 0.387, MACCS 0.202, RDK 0.321 and Morgan 0.386, against measured
shifts of 0.038, 0.033, 0.005 and 0.017. **Four of eight beyond the calibration gap, three
more inside it**, which is enough to carry the point and is the form the claim should take.
We lead on validity alone, 0.993 against 0.941, and even that is not like for like: ours is
"at least one of ten candidates is usable", theirs is one sample per caption.

Against the eleven published rows our best row beats four — Transformer, GIT-Mol, T5-base and
MolT5-base — and nothing else. Its best placings are validity 4th of 14 and FCD 6th of 13 in
the field of the floor-rank table above; on the three fingerprints it is 9th of 14, below the
retrieval floor. Two of those placings rest on margins inside the calibration gap and are
indicative only: ahead of MolT5-base on BLEU by 0.012 against a measured BLEU disagreement of
0.242, and on Levenshtein by 6.3 against a measured 15.9. Morgan is level with T5-base rather
than ahead, 0.545 against 0.545. The candidate-decoding row is a decoding procedure on top of
the same checkpoint, costing ten sampling runs per caption; the rows above it are single
trajectories and are what the architecture does on its own.

**The cost comparison is a count of network evaluations, not a compute ratio.** LDMol spends
about 200 network evaluations per caption: 100 DDIM steps, classifier-free guidance, one
sample. Ten candidates at 300 solver steps is about 3,000 backbone passes plus ten verifier
scorings — **fifteen times the network evaluations** for the worse result. It is not fifteen
times the compute, and nothing in this repo measures FLOPs or wall-clock for either system.
The passes are not the same size: our backbone is 47.9M parameters against LDMol's DiT at
roughly 160M, and their frozen text encoder (~335M) and autoregressive latent-to-SMILES
decoder (~120M) are not counted on their side at all. In FLOPs the ratio could narrow or
invert, and the word compute should not be used for it until something measures it. What the
count does support is the shape of the result: fifteen times the passes for a worse row is not
a good place to sit on a test-time scaling curve, whichever way the FLOPs come out.

**Against MolT5-base, scoped.** Levenshtein beats it on both numbers we hold, but only one of
the two is a comparison. Against our own run of the released checkpoint, 40.34, the margin is
19 points on the same split with the same scorer and it is real. Against the published 24.46
the margin is 2.8 for padding supervision (21.69) and 1.5 for the base recipe (22.93), both
far inside the 15.9-point Levenshtein disagreement our scorer shows on that same checkpoint,
so the cross-paper half is not claimed. The split cuts the same way and has only ever been
checked against MolT5: the 98 pairs we drop are the unparsable and the longest references, and
a mean distance of about 115 on those 98 would take our 21.69 to 24.46 on its own, which is
not implausible for references above the 128-token canvas. The repo has shown the 98 cannot
explain MolT5's gap; it has never shown they cannot explain ours. MACCS beats the published
0.721 but not our own run of the same
checkpoint, 0.755, against which the worst cepad seed is 0.754 — so MACCS is not claimed. The
remaining gap to the published rows is concentrated in exact match and in the fine-grained
fingerprints (RDK, Morgan) — the molecule is in the right class with the wrong details — which
is what points at the frozen text encoder and the tokenizer as the next levers.

FCD carries no cross-paper ranking any more. Our 0.81 was called level with TGM-DLM's 0.77;
LDMol reports 0.20, and our own MolT5 run lands 0.55 from its published figure (1.63 against
2.18), which is **fourteen times** the 0.04 margin those rankings rested on. This file printed
eighteen times, which no pair of numbers here produces. The TF-IDF floor scores 0.48 without
generating anything.

An earlier version of this file claimed FCD 0.76 as first in the table. That was one seed.
Across three, padding supervision gives 0.762 / 0.867 / 0.811 against the base recipe's
0.817 / 0.818, so the effect on FCD is **not established** and the ranking claim is withdrawn.
See the seeded table below. Its successor — FCD 0.73, first in the published table — is dead
for a second and independent reason: LDMol reports 0.20.

### Padding supervision, three seeds against two

The project's rule is that the group's worst seed must beat the reference's best. All five
runs here are pre-fix and share the old reference set, so the comparison between them holds;
none of these numbers is comparable with a post-fix row.

| metric | cepad (3 seeds) | base (2 seeds) | Δ mean | verdict |
|---|---|---|---|---|
| BLEU | 0.743 / 0.736 / 0.731 | 0.726 / 0.726 | +0.011 | confirmed |
| Exact | 0.050 / 0.052 / 0.048 | 0.045 / 0.044 | +0.006 | confirmed |
| Levenshtein↓ | 21.62 / 21.72 / 21.73 | 23.17 / 22.68 | −1.24 | confirmed |
| MACCS | 0.756 / 0.754 / 0.756 | 0.747 / 0.753 | +0.006 | confirmed, margin 0.001 |
| RDK | 0.586 / 0.583 / 0.585 | 0.576 / 0.580 | +0.007 | confirmed |
| Morgan | 0.501 / 0.498 / 0.503 | 0.488 / 0.490 | +0.012 | confirmed |
| FCD↓ | 0.762 / 0.867 / 0.811 | 0.817 / 0.818 | −0.004 | **not established** |
| Validity | 0.763 / 0.764 / 0.761 | 0.753 / 0.750 | +0.011 | confirmed |

Δ mean is the difference of the raw group means, not of the rounded cells printed here. FCD
is where that matters: averaging the two-decimal display values gives −0.007, which is what
this table used to print, and the raw means give −0.004. The FCD cells are printed at three
decimals so the two readings agree.

Seven metrics of eight, with a small and consistent effect. The MACCS margin is 0.001 and
passes only because the cepad seeds happen to be tight (spread 0.002); it should be read as
real but negligible. FCD has a seed spread of 0.105, which is larger than any difference
between the groups, and is the reason the single-run claim was wrong.

### Global caption conditioning: worse against its own pre-fix group

`model.text_global`, one seed, against the three-seed cepad group it is built on:

| | BLEU | Exact | Leven↓ | MACCS | RDK | Morgan | FCD↓ | Validity |
|---|---|---|---|---|---|---|---|---|
| cepad, per-metric worst of 3 seeds | 0.731 | 0.048 | 21.73 | 0.754 | 0.583 | 0.498 | 0.87 | 0.761 |
| cepad + text_global | 0.735 | 0.043 | 22.00 | 0.748 | 0.574 | 0.487 | 0.79 | 0.768 |

Below the group's per-metric worst (no single seed produced that row) on exact match,
Levenshtein, MACCS, RDK and Morgan; above it on BLEU, validity and FCD. Five columns below
and three above, which is the whole table.

**The verdict is worse, not the "no effect" this section used to carry.** The run is
`cepad_tglobal` and the group is `cepad_len` / `cepad_s43` / `cepad_s44`, all four pre-fix
(`unparseable_reference = 95`), so nothing here crosses the tokenizer fix. On the five
columns it loses it is outside the group's whole range, not merely below the worst seed:
MACCS 0.7481 against seeds at 0.7562 / 0.7540 / 0.7564, exact 0.0431 against 0.0500 / 0.0518
/ 0.0484, and the same on RDK, Morgan and Levenshtein. The earlier reading came from a
0.748 = 0.748 tie against the *post-fix* baseline, and that tie is an artefact of the fix,
which moves MACCS by about −0.006; against its own group there is no tie. PAPER.md reads the
same run against the same group and reaches the same verdict.

Pooling the caption into the adaLN vector makes the model worse, so the text pathway is not
capacity-limited in that direction, and the next text-side attempt should be the encoder
itself rather than another route into the denoiser.

---

## Candidate decoding: diffusion proposes, an autoregressive model selects

The length predictor is exactly right only 19.5% of the time, which is what caps exact
match, but the true length is inside its **top-5 for 63.2%** of captions. The information
is there and the argmax discards it. So: generate one candidate at each of the top-5
lengths and choose between them without looking at the reference.

How the choice is made decides everything.

| selection rule | BLEU | Exact | Leven↓ | MACCS | RDK | Morgan | FCD↓ | Validity | pairs |
|---|---|---|---|---|---|---|---|---|---|
| none (top-1 length) | 0.736 | 0.052 | 21.72 | 0.754 | 0.583 | 0.498 | 0.86 | 0.764 | 2409 |
| valid, then most likely length | 0.744 | 0.056 | 21.31 | 0.731 | 0.556 | 0.462 | 0.95 | **0.949** | 2975 |
| valid, then **AR log p(SMILES\|caption)**, 60k | 0.767 | **0.121** | 19.59 | 0.769 | 0.621 | 0.536 | 0.81 | **0.949** | 2975 |
| valid, then **AR log p(SMILES\|caption)**, 20k | **0.770** | 0.120 | **19.48** | **0.770** | **0.624** | **0.537** | **0.80** | **0.949** | 2975 |

All four rows are pre-fix and share the old reference set, so they compare with each other
and with nothing else. `pairs` is the fingerprint and FCD denominator; BLEU, exact match,
Levenshtein and validity are over all 3,202 captions.

Selecting on length alone buys 18.5 points of validity and gives back 0.023 of MACCS: the
rule knows whether a string parses and how likely its length was, and nothing about what
the caption asked for, so among several valid candidates it cannot prefer the right
molecule. The newly-valid molecules are the hard captions, and they drag the mean down.

Selecting on an autoregressive model trained on the same data, tokenizer and split moves
**every metric at once**, with no trade. Exact match goes from 0.052 to 0.121, and the
length-rule and AR-rule rows share a denominator of 2975 pairs, so that gain is the
criterion and not the sample.

Two controls matter. The **20k-step AR checkpoint selects as well as the 60k one** — very
slightly better on seven of the eight columns, and behind on exact match by one caption,
0.1202 against 0.1205, which is 385 of 3,202 against 386 — which is the opposite of what
memorisation would produce: the 60k model has seen 580 epochs of 25,574 molecules and has
essentially learnt them. Neither direction is a result at that size; the column best is
bolded where it falls and nothing is read into it. And the gain is not concentrated where
the benchmark leaks:

| bucket | exact | MACCS | Morgan | validity |
|---|---|---|---|---|
| near-duplicate (nn > 0.9) | 0.091 → **0.199** | 0.857 → **0.887** | 0.674 → **0.750** | 0.858 → **0.953** |
| related (0.7–0.9) | 0.034 → **0.088** | 0.741 → **0.785** | 0.436 → **0.500** | 0.724 → **0.936** |
| novel (≤ 0.7) | 0.024 → **0.060** | 0.579 → **0.621** | 0.283 → **0.324** | 0.695 → **0.957** |

Exact match improves by 2.19x, 2.59x and 2.50x across the three buckets, so the method
improves generalisation rather than retrieval. Validity lands at about 0.95 uniformly, and on
the near-duplicate third MACCS goes 0.857 → 0.887. That third is 37.5% of the split and both
numbers are pre-fix; it is not a table row, and the comparison with TGM-DLM's full-split 0.854
and 0.874 that used to stand here is withdrawn. On the same stratum, our split and our scorer,
TF-IDF retrieval scores 0.896 — above both of ours.

Against the published MolT5-base row this row leads on five of eight metrics by margins that
clear the calibration gap: exact 0.120 against 0.081, MACCS 0.770 against 0.721, RDK 0.624
against 0.588, FCD 0.80 against 2.18, validity 0.949 against 0.772. Three columns are excluded
rather than counted. BLEU: 0.7696 against 0.769 is a margin of 0.0006, smaller than the 0.012
BLEU spread across the three padding-supervision seeds, and the 60k AR selector — which the
text above calls equivalent — reverses it at 0.767; cross-paper BLEU is not a comparison in
any case, since our own run of the MolT5-base checkpoint scores 0.527 against its published
0.769. Morgan: 0.537 against 0.529 is a margin of 0.008 against a measured Morgan
disagreement of 0.017 on that same checkpoint, so it was counted on reasoning the BLEU cell was
excluded by. Levenshtein: 19.48 against 24.46 is 5.0 against a measured disagreement of 15.9.
MACCS survives at 0.049 against a measured shift of 0.033, but not by much. The row is also a
single pre-fix seed (`cand_ar20000`), while our own MolT5 run is post-fix, so the like-for-like
version of this comparison crosses reference sets in the other direction.

Validity 0.949 is not the highest figure in the table and never was. BioT5 and BioT5+ both
report 1.000, MolXPT 0.983, and the retrieval floor reaches 0.994 while generating nothing.
The claim that it led LDMol's 0.941 is retracted below: the margin is 0.008, the pre/post-fix
validity shift alone is 0.015, and counting the 98 dropped pairs as failures takes 0.949 to
0.921. LDMol leads this row on all seven other columns.

### Ten candidates

The ceiling is how often the true length is among the candidates: 19.5% at the argmax,
63.2% in the top five, **83.6% in the top ten**. Doubling the pool moves validity
0.949 → 0.988, exact match 0.120 → 0.130, BLEU 0.770 → 0.777 and Levenshtein 19.48 → 18.77,
and costs nothing but sampling. The fingerprints move less, and over a different denominator:
MACCS +0.003, RDK +0.008, Morgan +0.006, averaged over 3,073 pairs against 2,975, where the
98 extra pairs are precisely the captions the five-candidate pool failed to parse. MACCS
+0.003 is below the base recipe's two-seed MACCS spread of 0.005 and does not clear it.

| pool | BLEU | Exact | Leven↓ | MACCS | RDK | Morgan | FCD↓ | Validity | pairs |
|---|---|---|---|---|---|---|---|---|---|
| 5 candidates | 0.770 | 0.120 | 19.48 | 0.770 | 0.624 | 0.537 | 0.80 | 0.949 | 2975 |
| **10 candidates** | **0.777** | **0.130** | **18.77** | 0.773 | 0.632 | 0.543 | **0.79** | **0.988** | 3073 |

Both rows are single seeds and both are pre-fix.

7.60 of the 10 candidates parse on average, and only 37 captions of 3202 (1.2%) have no
usable candidate at all — which is where validity of 0.988 comes from. The chosen rank is
spread right across the pool (18% rank 0, 6% rank 9), so the later candidates are not
padding the average; they are being picked.

Against the published table, validity of 0.988 is **ours, not a published figure, and not the
highest one**: BioT5 and BioT5+ report 1.000, and the retrieval floor reaches 0.994 while
generating nothing. It is above MolXPT's 0.983, LDMol's 0.941, the plain Transformer's 0.906
and TGM-DLM's 0.871, from a single pre-fix seed (`cand_ar_k10`), and it is a one-of-ten
quantity against their single samples — the caveat two paragraphs below, which belongs with
the number rather than after it. The row leads MolT5-base and T5-base on the same five columns
of eight, counting only margins that clear the calibration gap: exact match, MACCS, RDK, FCD
and validity. Three are excluded. BLEU, because our own run of the MolT5-base checkpoint
scores 0.527 against its published 0.769, so an 0.008 lead over it or an 0.012 lead over
T5-base is not a result. Levenshtein, at 5.7 and 6.2 against a measured disagreement of 15.9.
And Morgan, where 0.543 is 0.014 above MolT5-base against a measured Morgan disagreement of
0.017, and behind T5-base's 0.545 outright. TGM-DLM still leads on exact match and the
fingerprints, and LDMol leads on all seven columns other than validity.

Three things have to be said with it. The pool costs ten sampling runs per caption, so this is
a test-time result and belongs beside the network evaluations it uses — about 3,000 backbone
passes of a 47.9M-parameter model plus ten verifier scorings, against roughly 200 evaluations
of a larger stack for LDMol's better row. Validity here is close to "at least one of ten
candidates parses", which is a property of the procedure, not of a single forward pass: the
same checkpoint sampled once scores 0.779. And every published validity it is being ranked
against is one sample per caption.

### Give every candidate its own noise as well as its own length

The ten candidates above differ only in length: they share one noise draw, because they were
ten separate runs at one seed. Drawing the noise independently per candidate costs nothing
and is the better pool. Full test split:

| | BLEU | Exact | Leven↓ | MACCS | RDK | Morgan | FCD↓ | Validity |
|---|---|---|---|---|---|---|---|---|
| shared noise, seed 42 *(pre-fix)* | 0.777 | 0.130 | 18.77 | 0.773 | 0.632 | 0.543 | 0.79 | 0.988 |
| shared noise, seed 43 | 0.776 | 0.139 | 18.68 | 0.767 | 0.625 | 0.540 | 0.73 | 0.987 |
| shared noise, seed 44 | 0.772 | 0.135 | 18.86 | 0.768 | 0.625 | 0.542 | 0.75 | 0.988 |
| **own noise, seed 42** | **0.781** | **0.143** | **18.18** | 0.771 | 0.629 | 0.545 | 0.73 | **0.996** |
| **own noise, seed 43** | **0.781** | **0.147** | **18.08** | 0.774 | 0.631 | 0.548 | 0.74 | **0.993** |

Seed 42 of the shared-noise group is pre-fix: it is scored against the old reference set over
3,073 pairs, against 3,160 and 3,162 for the other two, and it is the seed that supplies that
group's best BLEU, MACCS, RDK, Morgan and validity — five columns of eight; seed 43 supplies
the other three. Set it aside and this is two seeds against two,
below the three-seed unit of measurement this file sets at the top.

Counting all three shared-noise seeds, the worst own-noise seed beats the best shared-noise
seed on BLEU, exact match, Levenshtein and validity, which is this project's bar, and the
fingerprints are level. They are level only because the pre-fix seed supplies that group's
best MACCS and RDK, 0.773 and 0.632 against the worst own-noise seed's 0.771 and 0.629. It
supplies the group's best Morgan too, 0.543, and the worst own-noise seed beats it there
anyway at 0.545. Against the two post-fix shared-noise seeds the worst own-noise seed wins
seven of eight: BLEU 0.781 against 0.776, exact 0.143 against 0.139, Levenshtein 18.18 against
18.68, MACCS 0.771 against 0.768, RDK 0.629 against 0.625, Morgan 0.545 against 0.542,
validity 0.993 against 0.988. FCD is the exception either way and does not clear the bar: the
worst own-noise seed is 0.744 against the best shared-noise seed's 0.730, so own noise is not
established as an FCD improvement.

Quoted at the group's worst seed, as the rule requires, this is **validity 0.993 and FCD
0.74**. **Validity 0.993 is not the highest in the published table either**, which is what
this file said before the table had eleven rows: BioT5 and BioT5+ both report 1.000 and are
above it, MolXPT's 0.983 is below. It is fourth of fourteen in the field of the floor-rank
table, behind those two and behind the TF-IDF floor's 0.994, which generates nothing. It is
above LDMol's 0.941, with two caveats that do not go away: it is a one-of-ten quantity against
their single sample, and counting the 98 dropped pairs as failures takes it to 0.963. Validity
is also the column the floor ranks third in, which is the column least worth leading on.
**FCD 0.74 is not first in the table.** LDMol reports 0.20. The claim that it was, stated here
at the best seed as 0.73, is retracted below. A third seed is running.

### What the pool is actually buying

| control, full split | BLEU | Exact | Leven↓ | MACCS | RDK | Morgan | FCD↓ | Validity |
|---|---|---|---|---|---|---|---|---|
| one candidate, 300 steps *(pre-fix)* | 0.736 | 0.052 | 21.72 | 0.754 | 0.583 | 0.498 | 0.86 | 0.764 |
| one candidate, 300 steps | 0.736 | 0.052 | 21.40 | 0.748 | 0.575 | 0.492 | 0.81 | 0.779 |
| one candidate, 600 steps | 0.736 | 0.055 | 21.32 | 0.750 | 0.576 | 0.497 | 0.79 | 0.784 |
| one candidate, 1200 steps | 0.744 | 0.050 | 20.89 | 0.754 | 0.583 | 0.502 | 0.71 | 0.782 |
| one candidate, 3000 steps | 0.713 | 0.042 | 22.94 | 0.730 | 0.552 | 0.470 | 0.80 | 0.732 |
| 10 noise draws, one length | 0.777 | 0.098 | 18.39 | 0.762 | 0.600 | 0.514 | 0.74 | 0.987 |
| **10 lengths, own noise each** | 0.781 | **0.143** | 18.18 | 0.771 | 0.629 | 0.545 | 0.73 | 0.996 |

The 300-step control is printed twice: the pre-fix row is the one this section used to quote,
and the post-fix row is the one that belongs beside the rest of the table, every other row of
which is post-fix. On the post-fix control the pool's validity gain is 0.996 against 0.779,
not against 0.764. The last row is a single seed.

Two controls, and both matter.

**Compute is not the explanation.** Ten times the solver steps on one trajectory moves exact
match from 0.052 to 0.042 — it gets *worse*, monotonically past 1200 steps. The same number of
backbone passes spent on ten trajectories moves it to 0.143, and the backbone is the same
model either way, so this one is a compute comparison and not just a count. What the pool buys
is not integration accuracy.

**The length predictor is not incidental.** Ten noise draws at the single argmax length
already reach 0.098 and validity 0.987, so a large part of the validity gain is simply
"sample until something parses". But exact match stops at 0.098 against 0.143 when the ten
candidates also span the predictor's top-10 lengths. Both kinds of diversity pay, and the
length is worth about as much again as the noise.

### What is in the pool: the ceiling on every selection rule

Every proposal to improve candidate decoding is a proposal to improve the *selection* rule,
and none of them can be priced without knowing what the pool contains to begin with. So:
`scripts/oracle_ceiling.py`, the ten-candidate own-noise pool at seed 42, full split, choosing
the best candidate **with the reference in hand**. That is not a result; it is the upper bound
on everything selection can ever buy.

| pool | contains the answer | deployed rule finds it | MACCS deployed | MACCS oracle | RDK oracle | Morgan oracle | usable candidates |
|---|---|---|---|---|---|---|---|
| 5 candidates | 14.1% | 12.4% (88% of what is there) | 0.739 | 0.781 | — | — | — |
| **10 candidates** | **16.7%** | **14.3%** (86% of what is there) | 0.771 | 0.840 | 0.712 | 0.609 | 7.75 of 10 |

The ten-candidate row is over 3,185 captions, and a caption whose
selection does not parse scores zero rather than being dropped — the script refuses the
smaller denominator that would flatter it. **3,185 is not the scorer's count and the two
rules differ.** `scripts/oracle_ceiling.py` keeps a reference only if RDKit parses it *and*
it has at least three heavy atoms; `scripts/score_benchmark.py` applies that heavy-atom rule
to generations only and counts a reference as parsing whenever RDKit parses it, which
post-fix is all 3,202 (`unparseable_reference = 0`). The heavy-atom rule is the only
difference, so the 17 captions separating 3,202 from 3,185 are references of one or two heavy
atoms. That is why the same run reports 3,189 `pairs` in the standing table, a larger number
than 3,185 and not in conflict with it: `pairs` keeps those 17 and drops instead the 13
captions where the generation did not parse. So the deployed and oracle cells compare with each
other and with nothing else: the ten-candidate deployed MACCS of 0.771 happens to match the
standing table because at validity 0.996 almost every selection parses, while the
five-candidate 0.739 is below the 0.770 printed there because at 0.949 one selection in twenty
does not. The five-candidate pool is the earlier shared-noise pool, so the two rows are read
for the scale of the ceiling, not as a controlled delta. The oracle columns are per-metric: no
single string achieves that row, so it bounds every per-metric selection rule at once and is
the loosest bound available.

Three readings, and they point the same way.

**The selection rule is already close to the ceiling.** 14.3% found against 16.7% present is
86% of what the pool holds. **Every selection proposal in this project is bounded by +0.024
exact match**, that being what a perfect selector would add; a real rule gets some fraction of
it, because the last 14% is what no rule can tell apart without the reference.

**Doubling the pool barely moved the ceiling.** 14.1% to 16.7% for twice the sampling. The
deployed rule tracked it almost exactly (12.4% to 14.3%), which is the same statement from the
other side: the pool, not the chooser, is the binding constraint, and the pool scales badly.

**The remaining gap is in the generator.** A perfect selector over the ten-candidate pool
would reach exact match 0.167 against TGM-DLM's published 0.242 and LDMol's 0.530 — still
behind both, and behind TGM-DLM by three times the whole ceiling-to-deployed gap. Those two
are cross-paper comparisons like every other in this file, but both margins clear the 0.038
exact-match disagreement our scorer shows on the one checkpoint we can calibrate. The
per-metric MACCS oracle of 0.840 against a deployed 0.771 says the same about the
fingerprints. No amount of work on the chooser reaches those rows. That work belongs in the
generator, which is what the open items below are about.

## Can the verifier act *during* the trajectory?

Selection works after the fact. The obvious next question is whether the same model can
steer the sample while it is being produced, which would make the hybrid generative rather
than a re-ranking wrapper. Four routes were tried. None of them beats selection at the end,
and three of the four fail for one reason, which is worth more than any of them individually.
The fourth, particle filtering, fails on validity rather than on that reason, and one corner
of its sweep is unresolved — see below.

First, how much do the two models actually disagree? Per position, on greedy argmax:

| the verifier is asked about | agreement |
|---|---|
| finished diffusion samples | 80.8% |
| real ChEBI-20 molecules | 92.7% |
| MolT5-base samples | 45.3% |

So the verifier would rewrite about one position in five of our samples. That is far too
many to force, and the measurements bear it out.

| route | what it does | result |
|---|---|---|
| gradient guidance | adds `∇ log p_AR` to the reverse drift | at 98% of the score norm it changes 24% of molecules and improves nothing; below that it changes nothing. the score norm is about 482 against a gradient norm of 0.067, a ratio of 7×10³ |
| discrete clamping | snaps the x0 estimate to the verifier's token each step | destroys the sample — **and a control with an unreachable threshold proved the damage is the snapping, not the verifier**: identical numbers with and without it (0.302 / 0.028 / 38.07 / 0.582 / 0.627) |
| selective repaint | re-noises only the disputed positions and lets the diffusion redo them, holding the rest of the molecule fixed | **nothing moves** |
| particle filtering | carries several trajectories and reallocates compute by verifier score | costs validity in all eight configurations measured; exact match survives late resampling and not early. see below |

The repaint measurement is the one that explains the rest. On 64 captions the verifier
confidently disputes **294 of 2210 real tokens (13.3%), in 62 of 64 molecules**. Those
positions were re-noised and re-solved for 40 steps with the rest of the molecule pinned
at its own value at every step:

| re-noised back to | alpha | disputed tokens that changed | of those, matching the verifier |
|---|---|---|---|
| t = 0.90 | 0.988 | 0 of 294 | 0 |
| t = 0.75 | 0.924 | 0 of 294 | 0 |
| t = 0.60 | 0.809 | 0 of 294 | 0 |
| t = 0.45 | 0.691 | 1 of 294 | 0 |
| t = 0.30 | 0.454 | 7 of 294 | 2 |

**The sample is a fixed point of its own dynamics, position by position.** Given the rest
of the molecule, the model re-derives exactly the token it produced, from noise levels that
destroy most of the signal. Six configurations — zero, one and two rounds, with gradient
guidance on and off — produced byte-identical output files.

This reframes the disagreement. It is not that the diffusion is uncertain where the
verifier is confident; the two models hold stable, incompatible opinions. A position can
therefore only be *overwritten*, and overwriting is measured to destroy the molecule. What
is left is to choose among whole states, which is the one thing the verifier is measured to
do well — and selection at the end already works. The open question is whether selection
*during* the trajectory beats selection at the end, which is what the particle filter asks.

### Softening the length inside the trajectory: a clean failure

Pinning the canvas is how the length is imposed, before the first solver step. The
alternative to running one trajectory per candidate length is to pin each position by
P(length ≤ position) from the predictor's own top-10 distribution, so a single trajectory
carries the posterior and the readout picks the stop inside its support. A power on the
posterior dials between that and a hard length.

| | BLEU | Exact | Leven↓ | MACCS | RDK | Morgan | FCD↓ | Validity |
|---|---|---|---|---|---|---|---|---|
| hard length (control) *(pre-fix)* | 0.736 | 0.052 | 21.72 | 0.754 | 0.583 | 0.498 | 0.86 | 0.764 |
| hard length (control) | 0.736 | 0.052 | 21.40 | 0.748 | 0.575 | 0.492 | 0.81 | 0.779 |
| posterior | 0.319 | 0.000 | 38.28 | 0.530 | 0.312 | 0.276 | 5.42 | 0.732 |
| posterior² | 0.340 | 0.002 | 37.91 | 0.540 | 0.323 | 0.282 | 4.99 | 0.739 |
| posterior⁴ | 0.357 | 0.005 | 37.79 | 0.551 | 0.333 | 0.289 | 4.57 | 0.750 |
| posterior⁸ | 0.363 | 0.006 | 38.10 | 0.554 | 0.339 | 0.291 | 4.46 | 0.757 |

The four posterior rows are post-fix, so the second control row is the one that matches their
reference set. At this margin it makes no difference which is read.

Catastrophic, and **monotone in the direction of the hard length in seven columns of
eight**: the sharper the posterior the better it does, improving steadily back toward the
control on BLEU, exact match, MACCS, RDK, Morgan, FCD and validity. Levenshtein is the
exception and is not monotone — 38.28, 37.91, 37.79, 38.10 — turning back up at posterior⁸.
That is a 0.3-character wobble in a column the control wins by 17, and no reading is taken
from it. A partially pinned position is not a state the model was ever trained on — it is a
blend of "the molecule continues here" and "the molecule ended here" — and the denoiser's
estimate there is wrong in a way that spreads to the whole string, costing 17 points of
Levenshtein. Exact match is **effectively gone rather than literally zero**: posterior¹
returns one molecule of 3,202, 0.0003, and the sharper rows reach 0.002, 0.005 and 0.006
against the control's 0.052.

This is the same rigidity the repaint measured, seen from the other side. The trajectory
cannot be softened any more than it can be nudged: the length has to be a decision, and the
only way to hedge it is to run separate trajectories and choose afterwards.

### Particle filtering: the fourth route, and why it fails

If the verifier can only judge whole states, the natural place to use it during sampling is
on whole trajectories. Carry several particles per caption, decode each one's current x0
estimate at a few checkpoints, score it, and clone the ones the verifier likes over the ones
it does not. Nothing is overwritten, so the model's dynamics are untouched; only which
trajectories survive changes. Cloned particles diverge on the next step because the reverse
process is stochastic.

Five particles per caption, one per top-5 predicted length, 1000 captions, selected
afterwards by the same validity-then-likelihood rule as every candidate row. All eight
measured configurations, ordered by exact match:

| resampling | BLEU | Exact | Leven↓ | MACCS | RDK | Morgan | FCD↓ | Validity | pairs | no usable candidate |
|---|---|---|---|---|---|---|---|---|---|---|
| **none (control)** | 0.777 | **0.138** | 18.15 | 0.780 | 0.640 | 0.557 | 1.50 | **0.970** | 970 | 30 (3.0%) |
| `smcsel_sharp95` | 0.777 | 0.135 | 17.90 | 0.796 | 0.659 | 0.579 | 1.47 | 0.873 | 873 | 127 (12.7%) |
| `smcsel_late1` | 0.773 | 0.122 | 18.22 | 0.785 | 0.639 | 0.559 | 1.57 | 0.889 | 889 | 111 (11.1%) |
| `smcsel_late2` | 0.777 | 0.116 | 18.10 | 0.793 | 0.651 | 0.571 | 1.48 | 0.861 | 861 | 139 (13.9%) |
| `smcsel_gentle` | 0.753 | 0.064 | 19.57 | 0.756 | 0.590 | 0.505 | 1.61 | 0.868 | 868 | 132 (13.2%) |
| α 0.5/0.7/0.85, λ=5 | 0.735 | 0.039 | 20.98 | 0.741 | 0.566 | 0.481 | 1.82 | 0.800 | 800 | 200 (20.0%) |
| α 0.3…0.95, λ=20 | 0.733 | 0.034 | 21.19 | 0.744 | 0.564 | 0.484 | 1.72 | 0.784 | 784 | 216 (21.6%) |
| α 0.5/0.7/0.85, λ=20 | 0.733 | 0.031 | 21.17 | 0.738 | 0.561 | 0.476 | 1.83 | 0.787 | 787 | 213 (21.3%) |

This table used to carry the control and the three α-schedule rows only, and concluded that
resampling costs more than a third of exact match. The other four were measured in the same
sweep and were left out. They are named here by their run directories because their α
schedules were never written down anywhere in the repo, which is itself a defect; the
schedules have to be recovered before the rows are quoted elsewhere.

With all eight in view the verdict splits three ways. **Early resampling is harmful**: the
three α-schedule rows lose 0.099 to 0.107 of exact match against the control, and
`smcsel_gentle` loses 0.074. For the three α rows the mechanism is documented — the filter
commits at α = 0.5, where the decoded draft is still half noise, and the effective sample
size collapses to 1.1–1.9 of 5 at the first checkpoint. **Late and sharp resampling are
roughly neutral on exact match** — 0.135, 0.122 and 0.116 against the control's 0.138 — which
is what that mechanism predicts, since resampling late is exactly the fix for committing
early. **Their fingerprint gain is unresolved and is claimed neither way.** sharp95 reads
MACCS 0.796, RDK 0.659 and Morgan 0.579 against the control's 0.780 / 0.640 / 0.557, but it
scores 873 pairs against the control's 970: those averages are taken over 97 fewer and easier
captions, and for the whole gap to be a denominator artifact the 97 it does not score need
only average MACCS 0.642, which is an unremarkable value here. It needs a run whose
denominators match; until then it is neither a win nor noise.

Exact match and validity are the two columns every row shares a denominator on — both divide
by the 1000 captions — and every resampling configuration loses validity: 0.970 down to
between 0.889 and 0.784. The mechanism is in the last
column. **The mean number of usable candidates is unchanged — 3.91 of 5 with resampling, 3.91
without — but the number of captions with none at all goes from 30 to between 111 and 216.**
Cloning makes the particles correlated: the pool is now all-valid or all-invalid instead of
mixed, and it is the independence of the candidates, not their average quality, that the
selection rule lives on.

### What the control says instead

The no-resampling row is the best row in this table, and it is not the filter — it is five
*independent* candidates. It is not the best configuration measured in this project: the
ten-length row below it reaches exact 0.147 on the same 1000 captions, and on the full split
the own-noise runs reach 0.143 and 0.147 at validity 0.996 and 0.993. Against the
five-candidate row, which differs in that all five shared one noise draw and varied only in
length:

| 1000 captions, pool of 5 | BLEU | Exact | Leven↓ | MACCS | RDK | Morgan | FCD↓ | Validity | pairs |
|---|---|---|---|---|---|---|---|---|---|
| one candidate, no selection *(pre-fix)* | 0.734 | 0.059 | 21.09 | 0.764 | 0.592 | 0.514 | 1.72 | 0.771 | 758 |
| 5 lengths, shared noise *(pre-fix)* | 0.776 | 0.131 | 18.70 | 0.777 | 0.634 | 0.551 | 1.64 | 0.949 | 930 |
| **5 lengths, own noise each** | 0.777 | **0.138** | 18.15 | 0.780 | 0.640 | 0.557 | **1.50** | **0.970** | 970 |
| 10 lengths, shared noise *(pre-fix)* | 0.784 | 0.147 | 17.91 | 0.782 | 0.645 | 0.560 | 1.67 | 0.991 | 961 |

Three of these four rows are pre-fix and the own-noise row is not, so the table is not a
controlled comparison — the same error class as the random-stream artefact documented below.
Rescoring the identical baseline generations across that boundary, on this same 1000-caption
subset, moves FCD by −0.085 or −0.086 and validity by +0.017 with exact match unchanged.
(There are two post-fix scorings of that baseline: `clamp_plain` at 1.6376 gives −0.085 and
`rr_0` at 1.6366, the "0 rounds" row of the random-stream table below, gives −0.086. Nothing
here turns on the 0.001, but one number was being printed for two scorings.) The gap this
section used to credit to own noise — 0.007 exact match, 0.021 validity, 0.14 FCD — therefore
survives only in its exact-match part, and that is 7 molecules of 1000 on one seed. The claim
that own noise is the better pool rests on the full-split table above, where the reference
sets match: there the own-noise group wins BLEU, exact match, Levenshtein and validity, and
not FCD. The ten-length row makes the point from the other side — it is pre-fix and still the
best row here.

### The artefact that nearly became a result

The first repaint sweep, at 1000 captions, said something else: one round changed 495 of
1000 molecules and two rounds changed another 479, with a mean length difference of 8
characters — whole different molecules, not numerical drift. That cannot be squared with
294 disputed tokens all staying put. Rerunning the three sequentially on one card
reproduced the sweep exactly, so it was not a flake.

The differing lines are what gave it away. **The first 250 are byte-identical and every
difference starts at line 251**, spread evenly from there (158, 174 and 163 per 250). The
batch size is 250.

The repaint draws noise, the sampler runs in batches, and the global random stream is
shared: the first batch sees the trajectory it would have seen anyway, and **every later
batch gets different noise because of draws the intervention made after that batch's
trajectory was already finished.** Half the test set changed for a reason that has nothing
to do with the verifier, and the metrics moved by about what reshuffling noise moves them:

| | BLEU | Exact | Leven↓ | MACCS | RDK | Morgan | FCD↓ | Validity |
|---|---|---|---|---|---|---|---|---|
| 0 rounds | 0.734 | 0.059 | 20.75 | 0.759 | 0.586 | 0.508 | 1.64 | 0.788 |
| 1 round | 0.734 | 0.053 | 20.91 | 0.761 | 0.585 | 0.505 | 1.74 | 0.778 |
| 2 rounds | 0.736 | 0.060 | 20.73 | 0.760 | 0.594 | 0.516 | 1.69 | 0.791 |

The sampler now saves and restores the stream around any post-trajectory intervention, and
the particle filter's resampling draw comes from its own generator, so a batch's samples can
no longer depend on what was done to the batch before it. The 64-caption measurement — one
batch, nothing downstream to contaminate — was clean all along and is the one that stands.
A regression test pins the guard.

**The general lesson, written here because it has now cost two experiments:** an
intervention that consumes randomness is not controlled against one that does not, however
carefully everything else is matched.

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
| "FCD 0.76, first in the table" | one seed. Across three: 0.762 / 0.867 / 0.811 against a base of 0.817 / 0.818. No effect on FCD, and no ranking claim |
| "FCD 0.73 (0.74 at the worst seed) is first in the published table" | LDMol reports 0.20 on the same benchmark. The claim was also quoted at the best of two seeds, against this file's own worst-seed rule |
| "Validity 0.996 and FCD 0.73" as the headline pair | best of two seeds. The group's worst is validity 0.993 and FCD 0.74, which is what the standing table already printed |
| "Second on everything else behind TGM-DLM" | wrong twice over. Against the five older rows it was third, not second. Against the eleven-row sourced table our best row leads four published systems — Transformer, GIT-Mol, T5-base, MolT5-base — and nothing else, ranking 9th of 14 on each of the three fingerprints, below the retrieval floor. Morgan 0.545 is level with T5-base rather than ahead of it |
| "The first configuration that beats MolT5-Base on every metric" | the BLEU margin was 0.0006 and the 60k AR selector reverses it. Our own run of the MolT5-base checkpoint scores BLEU 0.527 against its published 0.769, so cross-paper BLEU is not a comparison at all |
| Every cross-paper ranking stated in this file before 2026-10-13 | no TGM-DLM, LDMol, T5 or Transformer output has ever been through our scorer, and the one checkpoint we did run moves six of eight columns. Rankings now stand only as "against the published numbers as their authors report them" |
| "ChEBI-20 aggregates measure lookup" | scoped, not dropped. On the full split TF-IDF retrieval beats four published systems — Transformer, GIT-Mol, T5-base, MolT5-base — and both of our rows on the fingerprint metrics, and six published systems on FCD. It does not beat TGM-DLM, T5-large, MolT5-large, MolXPT, BioT5, BioT5+ or LDMol on the fingerprints. It is a statement about a set of systems, not a property of the benchmark |
| "LDMol clears the floor on every metric at once" | false on validity: LDMol 0.941 against the floor's 0.994. Seven of eight, losing the one column this project's own lead lives in |
| "Validity 0.993 is the highest in the published table" — and the same claim at 0.988 and at 0.949 | the table now holds eleven published rows and two of them, BioT5 and BioT5+, report validity 1.000; MolXPT reports 0.983. 0.993 is fourth of fourteen, behind both of those and behind the TF-IDF floor's 0.994. The 0.949 claim fails twice more: the margin over LDMol was 0.008 against a pre/post-fix validity shift of 0.015, and counting the 98 dropped pairs as failures takes it to 0.921 |
| "196 references decoded wrongly — 75 unparseable and 121 a different molecule" | 75 is not a number the scorer ever produced. Every pre-fix full-split run records `unparseable_reference = 95` and every post-fix run records 0. Either the total is 196 with 101 wrong molecules or it is 216 with 121; nothing in the repo decides it, so only the measured 95 is quoted |
| "The seed spread collapsed to ~0.006 MACCS" | the subtraction of two rounded numbers. Measured: 0.0055 pre-fix (0.747308 / 0.752770) and 0.0023 post-fix (0.748119 / 0.750427) |
| "About fifteen times the diffusion compute" | a count of network evaluations (10 candidates × 300 steps = 3,000 backbone passes against LDMol's ~200), not a compute ratio. Nothing in this repo measures FLOPs or wall-clock, and the passes differ in size: our backbone is 47.9M parameters against LDMol's DiT at ~160M, plus their uncounted ~335M frozen text encoder and ~120M autoregressive decoder |
| "Our own MolT5 run lands eighteen times the margin those rankings rested on" | fourteen times. 0.55 (1.63 against 2.18) over the 0.04 margin quoted in the same sentence is 13.8x; no pair of numbers in this file gives eighteen |
| T5-base BLEU 0.762 | the chat transcription of commit 29ad1f6. LDMol's Table 1 reports 0.765, and it is the only cell of the four overlapping rows that disagrees |
| The candidate-decoding row's `pairs` cell of 3,178, and the "about 780 extra captions" drawn from it | a per-metric worst composite has no single denominator. Its fingerprints come from seed 42 at 3,189 pairs; 3,178 is seed 43's, correct only for the row's FCD and validity. On 3,189 the extra captions are 777 and 800 |
| "On the near-duplicate third we match or exceed TGM-DLM's 0.854 / 0.874 MACCS" | a 37.5% stratum of our curated split, from pre-fix runs, against full-split aggregates on another split. On that stratum TF-IDF retrieval scores 0.896, above both of our numbers |
| "95 test captions describe isotopes and carry no structure" | an evaluation-tokenizer defect decoded 196 of 3,202 references wrongly; the rows are salts and complexes, and post-fix there are none |
| "The no-resampling row is the best configuration measured in this project" | contradicted nine lines below it by the ten-length row, exact 0.147 against 0.138, and on the full split by the own-noise runs |
| "Resampling costs more than a third of exact match" | true of four of the eight configurations measured, not three: the three α-schedule rows lose 0.099 to 0.107 of the control's 0.138, which is 72–78%, and `smcsel_gentle` loses 0.074, which is 54%. Late and sharp resampling hold exact match within 0.003–0.022 of the control and pay in validity instead |
| "The Heun solver was never measured" and "the `ce_include_pad` confirmation seeds were never scored" | both runs exist and are scored; the file had not been updated |
| "Global caption conditioning has no effect" | read from a 0.748 = 0.748 MACCS tie against the post-fix baseline, which crosses the tokenizer fix worth about −0.006 MACCS. Against its own pre-fix group — `cepad_tglobal` against `cepad_len` / `cepad_s43` / `cepad_s44` — it is below every seed on exact match, Levenshtein, MACCS, RDK and Morgan, and above the group's worst on BLEU, FCD and validity. The verdict is **worse** |

---

## Closed — the last four runs

Four 200k-step runs sat trained but unscored for two weeks; all four are now measured.

| Run | What it answers |
|---|---|
| `cepad_s43`, `cepad_s44` | **done** — padding supervision confirmed on 7 metrics of 8, FCD not established |
| `cepad_tglobal` | **done** — worse against its own pre-fix group, below all three seeds on exact match, Levenshtein and the three fingerprints |
| `a20_random` | **done** — fourth corner agrees, augmentation is harmful |

## Open — not yet started, in priority order

1. **Unfreeze the text encoder.** The strongest evidence-backed lever left: frozen SciBERT
   states plateaued a length head at MAE 6.4 while fine-tuning the encoder reached 3.3. If
   the frozen states lose length, they lose substructure detail too — which is exactly where
   RDK and Morgan fail. Needs the caption path rebuilt: token ids plus the encoder in the
   training loop with its own learning-rate multiplier.
2. **A content channel in the latent.** Now the only architectural idea left standing:
   `text_global` was tried and made the model worse, so the next attempt should target latent
   generation rather than another route into the denoiser. An extra channel denoised
   alongside the embedding that says "content continues here", so length emerges from the
   model instead of being dictated to it — termination is localised to latent generation
   rather than the readout, which padding supervision already ruled out.
3. **A tokenizer trained on ZINC and ChEBI jointly.** Current merges are ZINC-shaped, and
   fine-grained substructure accuracy is what a poorly fitted tokenizer damages. Costs a
   fresh stage-1 pretrain.
4. **Honest test-time compute** — K length candidates scored by the model, or resampling
   invalid generations. Must be reported with `usable`, with FCD, and with the backbone
   passes it costs, or it is the same metric gaming the project already rejected once. LDMol
   reaches a better row at roughly a fifteenth of the network evaluations, and that is the
   bar as a count, not as compute: their evaluations are of a larger stack (a ~160M DiT, a
   frozen ~335M text encoder, a ~120M decoder) than our 47.9M backbone, and nothing on either
   side has been measured in FLOPs or wall-clock. Measuring that is itself an open item, and
   until it is done the bar is "fewer passes for a better row", which we are far from either
   way. The oracle ceiling bounds what the selection half of this can ever return at +0.024
   exact match, so the pool is where the work is.
5. **Re-score the pre-fix rows.** The base recipe has no post-fix scoring at all, and the
   candidate-decoding and particle-filter tables each mix reference sets. Until that is done,
   several comparisons in this file are approximate at the size of the fix itself — MACCS
   −0.006, validity +0.015.
