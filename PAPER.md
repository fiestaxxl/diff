# Lookup or Generation? What ChEBI-20 Measures, and Where Latent Diffusion Can Be Steered

**Draft — all numbers are measured on our curated split with the scorer in
`scripts/score_benchmark.py`. Every claim links to the experiment that produced it.**

---

## Abstract

Text-to-molecule systems are ranked by aggregate scores on ChEBI-20. We show that most of
those scores are reachable without generating anything. A TF-IDF caption-retrieval
baseline that returns a training molecule unchanged scores BLEU 0.786, MACCS 0.785,
RDK 0.690, Morgan 0.576, FCD 0.48 and validity 0.994 — above a MolT5-base checkpoint we
run ourselves on the same rows with the same scorer, on every one of those metrics. It
scores 0.004 on exact match. Stratifying the test split by similarity to training shows the
retrieval floor is not a near-duplicate artefact: retrieval leads on fingerprint similarity
in all three strata, including the third of the benchmark whose molecules have no close
training neighbour. Exact match is therefore the only reported metric that separates
generation from lookup.

We then ask where a latent diffusion model over SMILES embeddings can actually be steered.
Nine interventions in the generative trajectory — a jointly diffused content channel,
global caption conditioning, retrieval-initialised sampling, task-adapted caption features,
distinct padding, preference optimisation on two different pair constructions, and
likelihood guidance at two magnitudes — move every metric within seed noise. One
intervention outside the trajectory works: generating candidates at several predicted
lengths and selecting among them with a caption-conditioned autoregressive decoder takes
exact match from 0.052 to 0.121 and validity from 0.779 to 0.949, the highest validity in
the published table. The same autoregressive model used as a *guide* inside the trajectory,
at a gradient magnitude equal to the diffusion score, changes 24% of molecules and improves
nothing. A latent diffusion model's conditioning governs what its readout can say, not
where its trajectory goes.

---

## 1. Introduction

**Contributions.**

1. **A retrieval floor for ChEBI-20.** A system that generates nothing reaches or beats
   published fingerprint, BLEU, FCD and validity numbers. §3.
2. **Proximity stratification.** Test molecules split by nearest-training-neighbour
   similarity; the floor holds in every stratum, so it is not a leakage artefact. §4.
3. **Candidate decoding with autoregressive selection**, which improves exact match 2.3x
   and sets the highest validity in the table. §5.
4. **Nine negative interventions with a shared mechanism**, and a measurement that
   localises it: conditioning reaches the readout, not the trajectory. §6.
5. **An evaluation-code defect** that silently corrupted 6.1% of references in this
   project and would corrupt any reuse of the same configuration. §7.

**Non-claims.** We do not beat TGM-DLM or LDMol on aggregate scores. We argue those
aggregates should not be read as generative skill in the first place.

---

## 2. Setup

Continuous embedding-space diffusion over SMILES, Diffusion-LM style: a variance-preserving
cosine path `x_t = α(t)x₀ + β(t)ε` with `α = sin(πt/2)`, epsilon parameterisation, reverse
SDE integrated by Euler–Maruyama. 47.9M parameters, 32-dimensional latent, 128-token canvas.
Stage 1 pretrains unconditionally on 214M ZINC-20 molecules; stage 2 fine-tunes on
ChEBI-20's 25,574 caption-molecule pairs with cross-attention to frozen SciBERT caption
states. The diffusion formulation is carried over unchanged throughout; every component we
add is an opt-in flag whose default reproduces the original behaviour.

**Measurement discipline.** Three facts shape every number below.

- Validation loss does not rank models here. Across eleven configurations spanning 2.4x in
  sample quality it varied 1.5% and correlated with quality at **r = +0.46**, the wrong
  sign. We never select checkpoints by loss.
- Validity is gameable by decoder trimming, early stopping and inventing atoms, so we
  report it beside distribution metrics and require at least three heavy atoms.
- Three seeds are the unit of measurement. On the final recipe the seed spread is
  **0.006 MACCS** (0.748 / 0.750 across two seeds), which sets the resolution of every
  comparison in this paper.

---

## 3. A system that generates nothing

**Method.** TF-IDF over caption text, cosine to the nearest training caption, return that
training molecule unchanged. No model, no decoding, no chemistry. The floor is deliberately
weak: no embeddings, no reranking, no tuning.

**Result.** All rows below are produced by us, on our curated 3,202-molecule test split,
with one scorer.

| System | BLEU | Exact | Leven↓ | MACCS | RDK | Morgan | FCD↓ | Validity |
|---|---|---|---|---|---|---|---|---|
| **TF-IDF retrieval** | **0.786** | 0.004 | **20.57** | **0.785** | **0.690** | **0.576** | **0.48** | **0.994** |
| MolT5-base (our run) | 0.527 | 0.043 | 40.34 | 0.755 | 0.583 | 0.512 | 1.63 | 0.782 |
| Ours, diffusion | 0.736 | 0.052 | 21.40 | 0.748 | 0.575 | 0.492 | 0.81 | 0.779 |
| **Ours + AR selection** | 0.770 | **0.121** | 19.48 | 0.770 | 0.624 | 0.537 | 0.80 | 0.949 |

Retrieval leads on BLEU, Levenshtein, MACCS, RDK, Morgan, FCD and validity. It is last by
an order of magnitude on exact match.

**Reading.** Fingerprint similarity, string-overlap and distribution metrics on ChEBI-20
are substantially satisfiable by returning a neighbour's molecule. Exact match is not. Any
claim of generative skill on this benchmark that rests on the first group needs this floor
reported beside it.

*A caveat we cannot yet resolve.* Our MolT5-base BLEU (0.527) is far below the published
0.769 while its MACCS (0.755) is above the published 0.721. Character-level BLEU over raw
strings does not reproduce the published figure, so cross-paper BLEU comparisons on this
benchmark appear not to be like-for-like. We report our own number for all systems and
treat BLEU as internally comparable only.

---

## 4. Stratification by proximity to training

**Why.** If the floor came only from near-duplicates, it would be a leakage result about
the split rather than about the metrics. It does not.

**Anatomy of the split.** Of 3,202 test molecules: 2.1% appear verbatim in training; 86.9%
share a Bemis-Murcko scaffold with a training molecule; the nearest-neighbour Morgan
Tanimoto has median 0.787 and **p75 = 1.000** — a quarter of the test set has a training
molecule with an identical fingerprint, differing only in stereochemistry or tautomer.

Strata: **near-duplicate** (nn > 0.9) 37.5%, **related** (0.7–0.9) 28.7%, **novel**
(≤ 0.7) 33.8%.

**MACCS by stratum**

| System | near-duplicate | related | novel |
|---|---|---|---|
| TF-IDF retrieval | **0.896** | **0.832** | 0.619 |
| MolT5-base | 0.859 | 0.787 | 0.592 |
| Ours, diffusion | 0.853 | 0.738 | 0.572 |
| Ours + AR selection | 0.887 | 0.785 | **0.621** |

**Exact match by stratum**

| System | near-duplicate | related | novel |
|---|---|---|---|
| TF-IDF retrieval | 0.011 | 0.000 | 0.000 |
| MolT5-base | 0.077 | 0.023 | 0.021 |
| Ours, diffusion | 0.092 | 0.034 | 0.025 |
| **Ours + AR selection** | **0.199** | **0.088** | **0.060** |

**Reading.** Retrieval leads on fingerprint similarity in *every* stratum, including novel
molecules, so the floor is a property of the metric and not of train/test overlap. On exact
match retrieval is at zero in two strata of three, and the ordering inverts: generation
wins, and our method leads MolT5-base by 2.6x, 3.8x and 2.9x across the three strata.

Every metric is also a steep function of proximity — MACCS falls by about 0.27 from the
near-duplicate to the novel stratum for every system — so an aggregate on this benchmark is
largely a statement about how close the test molecules happen to be to the training set.

---

## 5. Candidate decoding: diffusion proposes, autoregression selects

**Motivation.** The sampler must be told the molecule's length. A caption→length encoder
reaches a mean absolute error of 3.3 tokens but is *exactly* right only 19.5% of the time,
and exact match needs exactness: halving the error from 6.4 to 3.3 tokens moved exact match
from 0.047 to 0.048. The true length is inside the predictor's top-5 for **63.2%** of
captions, so the information exists and the argmax discards it.

**Method.** Generate one candidate at each of the top-5 predicted lengths. Keep candidates
RDKit accepts with at least three heavy atoms. Among those, select the one maximising
`log p(SMILES | caption)` under a caption-conditioned autoregressive decoder trained on the
same data, tokenizer and split as the diffusion model. Fall back to the top-1 candidate if
nothing survives, so every caption yields exactly one string and validity is measured over
the same denominator as every other row.

**Result.**

| Selection rule | BLEU | Exact | Leven↓ | MACCS | RDK | Morgan | FCD↓ | Validity | pairs |
|---|---|---|---|---|---|---|---|---|---|
| none (top-1 length) | 0.736 | 0.052 | 21.73 | 0.754 | 0.583 | 0.498 | 0.86 | 0.764 | 2409 |
| valid, then likeliest length | 0.744 | 0.056 | 21.31 | 0.731 | 0.556 | 0.462 | 0.95 | 0.949 | 2975 |
| valid, then **AR likelihood** | **0.770** | **0.121** | **19.48** | **0.770** | **0.624** | **0.537** | **0.80** | **0.949** | 2975 |

Selecting on length alone buys validity and gives back fingerprint similarity: that rule
knows whether a string parses and how likely its length was, and nothing about what the
caption asked for. Selecting on caption likelihood moves **every metric at once**. The two
selection rows share a denominator of 2,975 scoreable pairs, so the difference between them
is the criterion and not the sample.

**Controls.**

- A 20k-step AR checkpoint selects as well as a 60k-step one, very slightly better. The
  60k model has seen 580 epochs of 25,574 molecules and has essentially memorised them, so
  memorisation is not what drives the gain.
- Per-stratum gains over the unselected baseline are 2.2x, 2.6x and 2.4x on exact match
  (§4), so the method improves generalisation rather than retrieval.
- Validity lands at 0.936–0.957 uniformly across strata.

**Scaling the pool.** The method is bounded by how often the true length is among the
candidates: 19.5% at the argmax, 63.2% in the top five, 83.6% in the top ten. Doubling the
pool moves every metric.

| pool | BLEU | Exact | Leven↓ | MACCS | RDK | Morgan | FCD↓ | Validity |
|---|---|---|---|---|---|---|---|---|
| 5 candidates | 0.770 | 0.120 | 19.48 | 0.770 | 0.624 | 0.537 | 0.80 | 0.949 |
| **10 candidates** | **0.777** | **0.130** | **18.77** | **0.773** | **0.632** | **0.543** | **0.79** | **0.988** |

An average of 7.6 of the 10 candidates parse, and only 37 captions of 3,202 (1.2%) have no
usable candidate, which is where 0.988 comes from. The selected rank is spread across the
whole pool — 18% at rank 0 down to 6% at rank 9 — so the later candidates are being chosen,
not merely counted.

**Independent noise per candidate.** Ten separate runs at one seed share a noise draw and
differ only in length. Drawing the noise per candidate costs nothing and is strictly better.

| full test split | BLEU | Exact | Leven↓ | MACCS | RDK | Morgan | FCD↓ | Validity |
|---|---|---|---|---|---|---|---|---|
| shared noise (3 seeds, worst) | 0.772 | 0.130 | 18.86 | 0.767 | 0.625 | 0.540 | 0.79 | 0.987 |
| **own noise (2 seeds, worst)** | **0.781** | **0.143** | **18.18** | 0.771 | 0.629 | 0.545 | **0.74** | **0.993** |

The worst own-noise seed beats the best shared-noise seed on BLEU, exact match, Levenshtein
and validity. **Validity of 0.993–0.996 and FCD of 0.73–0.74 are both first in the published
table**, above LDMol's 0.941 validity and TGM-DLM's 0.77 FCD; the row is second to TGM-DLM
on everything else and ahead of every other published system on all eight metrics.

**What the pool buys.** Two controls separate the explanations.

| | Exact | Validity |
|---|---|---|
| one candidate, 300 solver steps | 0.052 | 0.764 |
| one candidate, 1,200 solver steps | 0.050 | 0.782 |
| one candidate, 3,000 solver steps | 0.042 | 0.732 |
| 10 noise draws at one length | 0.098 | 0.987 |
| 10 lengths, own noise each | 0.143 | 0.996 |

Ten times the compute spent on integration accuracy makes exact match *worse*. Spent on ten
trajectories it nearly triples it. And the length predictor is not incidental: ten noise
draws at the argmax length reach 0.098, so most of the validity is "sample until something
parses", but the remaining 0.045 of exact match comes from spanning the predictor's top-10
lengths.

Three qualifications belong with it. The pool costs ten sampling runs per caption, so this
is a test-time-compute result and must be read beside the compute it spends. Validity here
is close to "at least one of ten candidates parses", a property of the procedure rather than
of a single forward pass. And it is a single seed; the seeded group is in progress.

---

## 6. Where latent diffusion cannot be steered

Nine interventions, all on the same backbone, all evaluated identically. The seed spread is
0.006 MACCS; nothing below clears it except where noted.

| Intervention | Where it acts | MACCS | Exact | Validity | Verdict |
|---|---|---|---|---|---|
| baseline (two seeds) | — | 0.748 / 0.750 | 0.052 | 0.779 | reference |
| content channel in the latent | trajectory | 0.724 | 0.041 | 0.721 | worse |
| content channel replacing length | trajectory | 0.582 | 0.003 | 0.690 | much worse |
| global caption conditioning (adaLN) | trajectory | 0.748 | 0.043 | 0.768 | no effect |
| retrieval-initialised sampling | trajectory | 0.785 | 0.004 | 0.993 | returns its input |
| task-adapted caption features | conditioning | 0.444 | 0.002 | 0.705 | much worse |
| distinct padding, 4 variants | target | 0.705 | 0.030 | 0.725 | worse |
| distinct padding, 8 variants | target | 0.661 | 0.026 | 0.637 | worse, monotone |
| DPO, length-varying pairs | weights | 0.746 | 0.053 | **0.800** | validity only |
| DPO, content-varying pairs | weights | 0.749 | 0.053 | **0.795** | validity only |
| AR likelihood guidance (98% of score) | trajectory | 0.741 | 0.038 | 0.748 | no effect |
| AR token clamping | trajectory | 0.582 | 0.028 | 0.627 | destroyed — by the clamping |
| AR selective repaint | trajectory | — | — | — | byte-identical output |
| **AR likelihood selection** | **after** | **0.770** | **0.121** | **0.949** | **works** |

**The localising measurement.** With the canvas unpinned and the length supplied only as an
input, we measured what the latent contains past the end of the molecule. The margin
(distance to the nearest real token embedding minus distance to the padding embedding) is
−0.085 for a padding-supervised model and −0.076 for one without, decaying toward zero
further out: the tail latent is close to nothing in particular, and the profile is
indistinguishable between the two. Padding supervision changed what the readout can say,
not what the diffusion produces.

**The sharpest case.** Guidance by the AR model's gradient, at a magnitude equal to the
diffusion score's own norm (482 against 0.067 for the raw gradient, so a scale of 7×10³),
changes **24% of molecules** by a mean of 2.06 characters — and improves nothing. The same
model, ranking the same model's finished samples, improves exact match 2.3x. A model that
is an excellent discriminator over completed samples is useless as a guide during sampling.

**The decisive case: the sample is a fixed point.** Gradient guidance acts on a continuous
state; clamping acts on a discrete one and destroys the molecule — and a control with an
unreachable confidence threshold showed the damage came from the snapping itself, not from
the verifier, since the numbers were identical with and without the verifier's opinion. That
left one route that forces nothing. We decoded the finished sample, asked the verifier which
positions it confidently disputes, re-noised **only those** back up the path, and let the
diffusion produce them again with every other position pinned at its own value at each step,
in the manner of RePaint. The verifier disputes 294 of 2210 real tokens (13.3%) across 62 of
64 molecules, so the test is not vacuous.

| re-noised back to α | 0.988 | 0.924 | 0.809 | 0.691 | 0.454 |
|---|---|---|---|---|---|
| disputed tokens that changed | 0 / 294 | 0 / 294 | 0 / 294 | 1 / 294 | 7 / 294 |

Given the rest of the molecule, the model re-derives exactly the token it produced, from
noise levels that destroy most of the signal. Zero, one and two repaint rounds, with and
without gradient guidance, write byte-identical files.

This measurement has a trap in it that we report because it nearly produced the opposite
conclusion. At 1,000 captions the same comparison showed one round changing 495 molecules
and two rounds another 479. All the differences begin at line 251, and the batch size is
250: the repaint draws noise, the sampler batches, and the global random stream is shared,
so every batch after the first was sampled from noise the intervention had displaced —
after those batches' own trajectories had finished. Half the test set changed for a reason
unrelated to the verifier. An intervention that consumes randomness is not controlled
against one that does not. The disagreement between the two
models is therefore not the diffusion being uncertain where the verifier is confident; the
two hold stable, incompatible opinions. A position can only be overwritten, and overwriting
is measured to destroy the molecule.

**Reading.** Conditioning in this architecture governs what the readout can express, not
where the trajectory goes. Interventions in the trajectory are absorbed; interventions after
it are not. This is consistent with the field's strongest diffusion result on this
benchmark, LDMol, which uses a fixed latent and an autoregressive decoder — placing the
discrete decision outside the diffusion.

---

## 7. An evaluation defect worth reporting

The fine-tune configuration pointed the evaluation tokenizer at a 391-token ZINC vocabulary
while the corpus was tokenized with the 635-token shared one. Consequences, measured:
**196 of 3,202 test references (6.1%) decoded wrongly** — 75 to unparseable strings and 121
to a *different molecule*; salts, hydrates, metals and charged nitrogen all live above id
386. The decoder's logits were truncated to 386, so no generated string could contain a
disconnection dot. The project had recorded 95 "isotope captions with no structure"; after
the fix there are none, and those rows are salts and complexes.

Correcting it moves MACCS by −0.006 and validity by +0.015 — the size of a real effect in
this study. Every number in this paper is post-fix.

---

## 8. Limitations

- One backbone, one scale (47.9M). The negative results in §6 are properties of this model,
  not proofs about latent diffusion in general, though their consistency across nine
  mechanisms is suggestive.
- Our MolT5-base reproduction does not match the published BLEU (§3); until that is
  resolved we compare BLEU only within our own runs.
- The retrieval floor uses TF-IDF. A stronger retriever would raise the floor and
  strengthen the argument, but we have not measured how far.
- §5 reports one seed for the AR-selected configuration. Three are in progress.
- Text2Mol is not computed: it needs an external retrieval model we do not have.

## 9. What we would do next

1. Three seeds for the AR-selected configuration and the retrieval floor.
2. A stronger retriever (SciBERT caption embeddings) to find where the floor actually sits.
3. Stratified numbers for LDMol, whose checkpoint is public, to test whether the strongest
   published system is also flat across strata or steep like the rest.
4. Candidate decoding at K = 10 — the true length is in the top-5 for 63.2% of captions and
   in the top-10 for more.

---

## Appendix: reproduction

```bash
# the retrieval floor
python scripts/retrieval_baseline.py --config configs/chebi20_finetune.yaml --out <dir>

# a published baseline on our split
python scripts/run_molt5.py --config configs/chebi20_finetune.yaml \
    --model laituan245/molt5-base-caption2smiles --out <dir>

# candidate decoding with autoregressive selection
python scripts/export_length_topk.py --config configs/chebi20_finetune.yaml \
    --encoder data/chebi20/lenc_5e5 --split test --k 5
# one generation per rank, then
python scripts/score_with_ar.py --config configs/chebi20_ar.yaml --checkpoint <ar> \
    --dirs <k0> ... --out <scores>.npy
python scripts/select_candidates.py --dirs <k0> ... --probs <scores>.npy --out <dir>

# scoring and stratification
python scripts/score_benchmark.py --dir <dir>
```
