# Lookup or Generation? What ChEBI-20 Measures, and Where Latent Diffusion Can Be Steered

**Draft — every number we measure is on our curated 3,202-pair split with the scorer in
`scripts/score_benchmark.py`, and every claim links to the experiment that produced it. The
published rows in §2 are other people's numbers on the standard 3,300-pair split, produced
by their own scorers; §9 says why every comparison against them is indicative, and what
margin is too small to be one.**

---

## Abstract

Text-to-molecule systems are ranked by aggregate scores on ChEBI-20. We show that most of
those scores are reachable without generating anything. A TF-IDF caption-retrieval
baseline that returns a training molecule unchanged scores BLEU 0.786, MACCS 0.785,
RDK 0.690, Morgan 0.576, FCD 0.48 and validity 0.994 — above a MolT5-base checkpoint we
run ourselves on the same split with the same scorer, on every one of those metrics. It
scores 0.004 on exact match. Stratifying the test split by similarity to training shows the
retrieval floor is not a near-duplicate artefact: retrieval leads on fingerprint similarity
in the near-duplicate and the related stratum, and is level in the novel stratum — the third
of the benchmark whose molecules have no close training neighbour — where one row of ours is
0.002 ahead across two reference sets and we resolve it neither way (§4).

The floor is an instrument, not a dismissal of the benchmark, and the published table says
what it does and does not reach. Of fourteen rows — the eleven published rows of §2, our two
and the floor — the floor ranks third on validity, fifth of the thirteen that report FCD,
seventh of thirteen on BLEU, eighth on MACCS, RDK, Morgan and Levenshtein, and thirteenth of
fourteen on exact match, where it beats only the Transformer's 0.000. It outranks four
published systems on the fingerprints and six on FCD; TGM-DLM, T5-large, MolT5-large, MolXPT,
BioT5, BioT5+ and LDMol all clear it on the fingerprints. LDMol clears it on seven of the
eight metrics it reports: four of them — exact match 0.530 against 0.004, MACCS 0.973 against
0.785, RDK 0.950 against 0.690, Morgan 0.931 against 0.576 — by margins larger than the
disagreement we measure between our scorer and a published column, and three — BLEU,
Levenshtein, FCD — by margins inside it; it loses only validity, where returning a catalogued
molecule is unbeatable. So the claim is not that ChEBI-20 measures lookup. It is that
ChEBI-20's aggregate metrics are reachable by lookup, that ordering the eight metrics by how
well the floor does in them orders them by how much each one rewards lookup, and that exact
match is where the floor collapses — the one column separating lookup from generation in ten
of the eleven published rows, the exception being the Transformer's 0.000 below the floor's
0.004.

We then ask where a latent diffusion model over SMILES embeddings can actually be steered.
Nine interventions in the architecture, the target and the weights, plus likelihood guidance
at two magnitudes: eleven rows, which split four ways against the seed spread. Two move every
metric within it, both of them likelihood guidance. Six make the model clearly worse: a
jointly diffused content channel, the same channel replacing the length channel, global
caption conditioning, task-adapted caption features, and distinct padding at four and again
at eight variants. One is degenerate rather than worse: retrieval-initialised sampling
returns its own input. Two buy validity and nothing else: preference optimisation on either
pair construction. None improves exact match. One intervention outside the trajectory works:
generating candidates at several predicted lengths and selecting among them with a
caption-conditioned autoregressive decoder takes
exact match from 0.052 to 0.143 and validity from 0.779 to 0.993, each column at the worse of
two seeds: seed 42 reads exact 0.143 and validity 0.996, seed 43 reads 0.147 and 0.993. The
same autoregressive model used as a *guide* inside the trajectory, at a gradient magnitude
equal to the diffusion score, changes 24% of molecules and improves nothing. A latent
diffusion model's conditioning governs what its readout can say, not where its trajectory
goes.

---

## 1. Introduction

**Contributions.**

1. **A retrieval floor for ChEBI-20.** A lookup table that generates nothing outranks four of
   the eleven published systems of §2 on the fingerprint metrics and six of them on FCD, sits
   third of fourteen rows on validity, and collapses to last-but-one on exact match. §3.
2. **Proximity stratification.** Test molecules split by nearest-training-neighbour
   similarity; the floor holds in every stratum, so it is not a leakage artefact. §4.
3. **Candidate decoding with autoregressive selection**, which improves exact match 2.7x
   (0.052 to 0.143, worst of two seeds) and reaches validity 0.993 — below BioT5 and BioT5+ at
   1.000 and below our own retrieval floor at 0.994, and a property of the ten-candidate
   procedure rather than of a single sample, which is 0.779 on the same checkpoint. §5.
4. **Nine negative interventions with a shared mechanism**, and a measurement that
   localises it: conditioning reaches the readout, not the trajectory. §6.
5. **An evaluation-code defect** that silently corrupted 6.1% of references in this
   project and would corrupt any reuse of the same configuration. §7.

**Non-claims.** We do not beat TGM-DLM or LDMol on aggregate scores, and our best row is above
only four of the eleven published rows of §2: Transformer, GIT-Mol, T5-base and MolT5-base.
LDMol's ICML 2025 row is above ours on seven of the eight metrics it reports — exact 0.530
against 0.143, MACCS 0.973 against 0.771, RDK 0.950 against 0.629 and Morgan 0.931 against
0.545, which are margins larger than our measured scorer disagreement in those columns, and
BLEU 0.926 against 0.781, Levenshtein 6.75 against 18.18 and FCD 0.20 against 0.74, which are
margins smaller than it. We lead only on validity, 0.993 against 0.941. **Every cross-paper
statement in this paper is a comparison against published numbers our scorer does not
reproduce**, and is marked as indicative where it is made; a margin below the measured shift
for its column is written as noise (§9). Only our own runs and one public MolT5-base
checkpoint have ever been scored by our scorer, and on that checkpoint our scorer disagrees
with the published column on six of eight metrics. We argue these aggregates should not be
read as generative skill without the floor beside them.

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
- Three seeds are the unit of measurement, and a group's worst seed must beat the reference
  group's best. On the final post-fix recipe the seed spread is **0.002 MACCS** (0.748119 /
  0.750427 across two seeds); on the pre-fix base recipe it is **0.005** (0.747308 /
  0.752770). The 0.006 quoted in earlier drafts was the subtraction of two rounded numbers
  from the second pair. §6 reads its verdicts against the wider 0.005, which is the
  conservative choice; where a row sits between 0.002 and 0.005 the table says so.

**The published rows we compare against.** Eleven rows, one source: Table 1 of the LDMol
paper (Chang and Ye, ICML 2025, arXiv:2405.17829v4). It is the first table in this project
with a source behind it, and it replaces the six-row table earlier drafts argued from.

| System | Family | BLEU | Exact | Leven↓ | MACCS | RDK | Morgan | FCD↓ | Validity |
|---|---|---|---|---|---|---|---|---|---|
| Transformer | AR | 0.499 | 0.000 | 57.660 | 0.480 | 0.320 | 0.217 | 11.32 | 0.906 |
| GIT-Mol | AR | 0.756 | 0.051 | 26.315 | 0.738 | 0.582 | 0.519 | — | 0.928 |
| T5-base | AR | 0.765 | 0.069 | 24.950 | 0.731 | 0.605 | 0.545 | 2.48 | 0.660 |
| MolT5-base | AR | 0.769 | 0.081 | 24.458 | 0.721 | 0.588 | 0.529 | 2.18 | 0.772 |
| T5-large | AR | 0.854 | 0.279 | 16.721 | 0.823 | 0.731 | 0.670 | 1.22 | 0.902 |
| MolT5-large | AR | 0.854 | 0.311 | 16.071 | 0.834 | 0.746 | 0.684 | 1.20 | 0.905 |
| MolXPT | AR | — | 0.215 | — | 0.859 | 0.757 | 0.667 | 0.45 | 0.983 |
| BioT5 | AR | 0.867 | 0.413 | 15.097 | 0.886 | 0.801 | 0.734 | 0.43 | **1.000** |
| BioT5+ | AR | 0.872 | 0.522 | 12.776 | 0.907 | 0.835 | 0.779 | 0.35 | **1.000** |
| TGM-DLM | DM | 0.826 | 0.242 | 17.003 | 0.854 | 0.739 | 0.688 | 0.77 | 0.871 |
| **LDMol** | DM | **0.926** | **0.530** | **6.750** | **0.973** | **0.950** | **0.931** | **0.20** | 0.941 |

Six of these systems have never appeared in this repository: GIT-Mol, T5-large, MolT5-large,
MolXPT, BioT5 and BioT5+. Two of them — BioT5 and BioT5+ — sit above TGM-DLM on all three
fingerprints and on exact match; MolXPT is above it on MACCS (0.859 against 0.854), RDK
(0.757 against 0.739), FCD (0.45 against 0.77) and validity (0.983 against 0.871), and below
it on Morgan (0.667 against 0.688) and on exact match (0.215 against 0.242). That changes how
this paper has to describe the field.
**The top of the table is one diffusion row, then a block of autoregressive ones.** LDMol
leads every column but validity, and the four rows below it on exact match — BioT5+ 0.522,
BioT5 0.413, MolT5-large 0.311, T5-large 0.279 — are all autoregressive and all above the
other diffusion row, TGM-DLM at 0.242. LDMol's margin over BioT5+ in that column is 0.008, a
fifth of the exact-match disagreement we measure between our scorer and a published column
(0.038, §9), which is the resolution that governs a comparison across papers. So the single
strongest row on this
benchmark is a diffusion model, and its lead over the best autoregressive model is inside
every resolution anyone here can measure: LDMol's title claims it surpasses AR models, and at
this resolution that is parity. What separates the field is not diffusion but the scale and
chemistry-awareness of the pretrained representation, and that ordering runs through both
families — base-sized models, ours among them, reach 0.069 to 0.143 exact match, large ones
0.279 to 0.311, and chemistry-specialised ones 0.413 to 0.530. Validity is saturable: BioT5
and BioT5+ reach 1.000 by generating SELFIES, which parse by construction.

**What the earlier table got wrong.** Its first five rows were transcribed from a chat message
into `scripts/score_benchmark.py` in commit 29ad1f6 (2026-09-25) with no source recorded for
any of those forty numbers. Three of the five survive the source unchanged — Transformer,
MolT5-base and TGM-DLM. Two do not.

- **T5-base BLEU was transcribed as 0.762; the source says 0.765.** We print the source. The
  other seven columns of that row match. Nothing in this paper turned on the difference.
- **"TGM-DLM w/o correction" is not in the source table.** LDMol's Table 1 reports TGM-DLM
  once, without the uncorrected variant. The transcribed row read BLEU 0.828, exact 0.242,
  Levenshtein 16.897, MACCS 0.874, RDK 0.771, Morgan 0.722, FCD 0.89, validity 0.789; those
  numbers are kept here and used nowhere, because nothing in this repository says where they
  came from. Every claim in this paper that rested on the table carrying TGM-DLM twice is
  withdrawn (§8).

The source, for all eleven rows: Jinho Chang and Jong Chul Ye, *LDMol: A Text-to-Molecule
Diffusion Model with Structurally Informative Latent Space Surpasses AR Models*, ICML 2025,
arXiv:2405.17829v4, code at `https://github.com/jinhojsk515/LDMol`, Table 1. Cite v4. The
May 2024 v1
reports a weaker row (validity 0.928 / BLEU 0.900 / Levenshtein 9.259 / MACCS 0.931 /
RDK 0.860 / Morgan 0.820 / exact 0.363 / FCD 0.25) which must not be quoted as LDMol's
result. Its MACCS, RDK and Morgan are Tanimoto similarities, the same quantity we compute.
LDMol reports no Text2Mol column.

Three things follow, and they bind every comparison in this paper.

- **All eleven rows are on the standard uncurated 3,300-pair test split. Ours are on a
  curated 3,202-pair split** (§3). The 98 dropped pairs are the unparsable references and the
  longest molecules, so the curation is not neutral with respect to validity or exact match.
- **None of the eleven was produced by our scorer**, and they were not produced by each
  other's either: one table collects eleven papers' scorers. On the one system we can
  calibrate — the public `laituan245/molt5-base-caption2smiles` checkpoint, run by us on our
  split with our scorer — six of eight columns move, by BLEU 0.242, Levenshtein 15.9,
  FCD 0.553, exact match 0.038, MACCS 0.033 and Morgan 0.017 (§3, §9). **Any cross-paper
  margin smaller than the shift for its own column is noise**, and this paper writes it as
  noise wherever one is quoted.
- **Eight columns, not the published nine**, and not every row reports all eight: GIT-Mol
  reports no FCD, MolXPT neither BLEU nor Levenshtein. Text2Mol needs an external retrieval
  model we do not have, and LDMol does not report it either.

We print these rows because a reader needs to see where the field sits. Every comparison this
paper makes against them is a comparison against numbers our scorer does not reproduce, and
is labelled that way at the point it is made.

---

## 3. A system that generates nothing

**Method.** TF-IDF over caption text, cosine to the nearest training caption, return that
training molecule unchanged. No model, no decoding, no chemistry. The floor is deliberately
weak: no embeddings, no reranking, no tuning.

**Result.** All rows below are produced by us, on our curated 3,202-molecule test split,
with one scorer (`scripts/score_benchmark.py`) and against one reference set — the corrected
one of §7, `unparseable_reference = 0` for every row in this table.

| System | BLEU | Exact | Leven↓ | MACCS | RDK | Morgan | FCD↓ | Validity | pairs |
|---|---|---|---|---|---|---|---|---|---|
| **TF-IDF retrieval** | **0.786** | 0.004 | 20.57 | **0.785** | **0.690** | **0.576** | **0.48** | **0.994** | 3182 |
| MolT5-base (our run) | 0.527 | 0.043 | 40.34 | 0.755 | 0.583 | 0.512 | 1.63 | 0.782 | 2503 |
| Ours, diffusion | 0.736 | 0.052 | 21.40 | 0.748 | 0.575 | 0.492 | 0.81 | 0.779 | 2495 |
| **Ours + AR selection**, 10 candidates, worst of 2 seeds | 0.781 | **0.143** | **18.18** | 0.771 | 0.629 | 0.545 | 0.74 | 0.993 | 3189 / 3178 |

The AR-selection row was the five-candidate run at 0.770 / 0.120 / 19.48 / 0.770 / 0.624 /
0.537 / 0.80 / 0.949 over 2,975 pairs in earlier drafts. That run is scored against the
*pre-fix* reference set (§7) while the three rows above it are post-fix, so it has been
moved to §5, where its pre-fix siblings are, and replaced here by the post-fix ten-candidate
group. The numbers are not adjusted; the row is.

`pairs` is the denominator for MACCS, RDK, Morgan and FCD only: those are means over the
captions whose generation parses. BLEU, Levenshtein, exact match and validity are over all
3,202 captions in every row and are comparable as printed. The last row has two denominators
because it is a per-metric worst of two seeds rather than a run: its MACCS, RDK and Morgan are
seed 42's, over 3,189 pairs, and its FCD is seed 43's, over 3,178. Earlier drafts printed
3,178 for all four. The 679-caption gap between
retrieval (3,182) and MolT5-base (2,503) is exactly their validity gap, and it runs against
us: MolT5-base's fingerprint and FCD means exclude the captions it could not answer with a
parseable molecule, retrieval's include everything.

Retrieval leads on BLEU, MACCS, RDK, Morgan, FCD and validity. It loses on exact match by an
order of magnitude, 0.004 against 0.143, and on Levenshtein, 20.57 against 18.18. Those two
are the only metrics in the table on which a system that generates nothing can be beaten by
one of ours.

**Reading.** Fingerprint similarity, string-overlap and distribution metrics on ChEBI-20
are substantially satisfiable by returning a neighbour's molecule. Exact match is not, with
one exception in the published table: the Transformer reports 0.000, below the floor's 0.004,
so the separation holds against ten of the eleven published rows of §2 rather than all of
them. Any claim of generative skill on this benchmark that rests on the first group needs this
floor reported beside it.

This is a statement about a set of systems, not a property of the benchmark, and the eleven
published rows of §2 say which systems. Of fourteen rows — those eleven, our two and the
floor — the floor ranks third on validity, fifth of the thirteen that report FCD, seventh of
thirteen on BLEU, eighth on MACCS, on RDK, on Morgan and on Levenshtein, and thirteenth of
fourteen on exact match, ahead only of the Transformer's 0.000.

On the three fingerprints the floor beats Transformer, GIT-Mol, T5-base, MolT5-base and both
of our own rows, and loses to TGM-DLM, T5-large, MolT5-large, MolXPT, BioT5, BioT5+ and LDMol.
Neither diffusion row is the nearest one above it. The nearest is T5-large on MACCS (0.823
against 0.785) and on RDK (0.731 against 0.690), and MolXPT on Morgan (0.667 against 0.576);
the rows between the floor and LDMol on MACCS are T5-large 0.823, MolT5-large 0.834, TGM-DLM
0.854, MolXPT 0.859, BioT5 0.886 and BioT5+ 0.907. TGM-DLM, the other diffusion row, is fifth
on MACCS, sixth on RDK and fourth on Morgan, with one to three rows between it and the floor
in each column. It reads MACCS 0.854 against 0.785, RDK 0.739 against 0.690, Morgan 0.688
against 0.576, and it clears the floor on six of the eight metrics — four of those six by
margins larger than our measured scorer disagreement in their column, the other two being
BLEU 0.826 against 0.786 and Levenshtein 17.003 against 20.57, which are inside it. On FCD,
which GIT-Mol does not report, the floor beats six published rows — Transformer, T5-base,
MolT5-base, T5-large, MolT5-large and TGM-DLM — plus both of ours, and loses to MolXPT,
BioT5, BioT5+ and LDMol; the margin over TGM-DLM, 0.29, is inside the 0.553 FCD shift. Ten
published rows report FCD, and six beaten plus four above accounts for all of them. On BLEU it
beats the same four published systems it beats on the fingerprints.

So the right formulation is: **ChEBI-20's aggregate metrics are reachable by lookup, and
ordering the eight metrics by how well the floor does in them orders them by how much each one
rewards lookup.** Validity comes first and is saturable to 1.000 by generating SELFIES, as
BioT5 and BioT5+ do. FCD comes second, because returning real training molecules is what FCD
was built to reward. The fingerprints and the string metrics sit in the middle. Exact match
comes last, and is the one column where the floor collapses.

Seven of the eleven published rows clear the floor on the fingerprints, so this is not a
verdict on the benchmark. LDMol clears it on seven of the eight metrics it reports (MACCS
0.973, RDK 0.950, Morgan 0.931, exact 0.530, BLEU 0.926, Levenshtein 6.75, FCD 0.20), losing
only validity, where a catalogued molecule parses by construction — but only the first four of
those seven are margins larger than our measured scorer disagreement in their columns. The
other three are 0.140 BLEU against a measured shift of 0.242, 13.82 Levenshtein against 15.9,
and 0.281 FCD against 0.553. Without the rows above it the floor would only say the benchmark
is broken; with them, the floor says which systems have shown generative skill on this
benchmark and which have only shown a number.

*A caveat that is larger than it looks.* Our MolT5-base run reproduces the published row on
only two of eight metrics. BLEU 0.527 against 0.769, Levenshtein 40.34 against 24.46 and
exact match 0.043 against 0.081 are far below it, and Morgan 0.512 against 0.529 is below it
too; MACCS 0.755 against 0.721 and FCD 1.63 against 2.18 run the other way; only RDK (0.583
against 0.588) and validity (0.782 against 0.772) land close. Same checkpoint, same split, same scorer, and six of eight columns move.
The 98 captions our curation drops cannot account for it: for our Levenshtein to fall to
24.46 those 98 would need a mean distance of −494, and for exact match to reach 0.081 they
would need 130 exact matches among 98 captions. We report our own number for every system we
run, treat BLEU, Levenshtein, exact match and FCD as internally comparable only, and read
every cross-paper statement in this paper as a comparison against numbers our scorer does not
reproduce, with any margin below the measured shift for its column written as noise (§9).

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
| Ours, diffusion (post-fix) | 0.853 | 0.738 | 0.572 |
| Ours, diffusion (pre-fix) | 0.857 | 0.741 | 0.579 |
| Ours + AR selection (pre-fix) | 0.887 | 0.785 | 0.621 |

**Exact match by stratum**

| System | near-duplicate | related | novel |
|---|---|---|---|
| TF-IDF retrieval | 0.011 | 0.000 | 0.000 |
| MolT5-base | 0.077 | 0.023 | 0.021 |
| Ours, diffusion (post-fix) | 0.092 | 0.034 | 0.025 |
| Ours, diffusion (pre-fix) | 0.091 | 0.034 | 0.024 |
| **Ours + AR selection** (pre-fix) | **0.199** | **0.088** | **0.060** |

*Provenance.* The retrieval and MolT5-base rows are post-fix. The AR-selection row is the
five-candidate run `cand_ar20000`, scored before the tokenizer fix of §7 against a reference
set with 95 unparseable references, so its own baseline is given twice: the post-fix
rescoring of the same configuration and the pre-fix one that shares its reference set. Read
the gains down the pre-fix pair. The fix moved full-split MACCS by −0.006, and per stratum
by −0.004 / −0.003 / −0.007, which is the whole of the difference between the two baseline
rows. No per-stratum number has been recomputed post-fix for the selected arm; a rescore is
needed rather than a scalar shift, because the defect decoded about a hundred references to a
*different molecule* (§7) and those fingerprint means were taken against the wrong ground
truth. One caveat belongs to both tables: the per-stratum cells come from the scorer's
stratified breakdown and have no counterpart in the stored run metrics, which keep full-run
aggregates only. They are consistent with each other, and every quantity derived from them
below reproduces from the printed cells, but none of them has been checked against a raw
source.

**Reading.** Like-for-like, within the pre-fix pair, candidate decoding gains +0.030 / +0.044
/ +0.042 MACCS and +0.108 / +0.054 / +0.036 exact match across the three strata, a ratio of
2.19x, 2.59x and 2.50x. Retrieval leads on fingerprint similarity in every stratum, including
novel molecules, with one apparent exception: the selected arm's 0.621 against retrieval's
0.619 in the novel stratum. That 0.002 is a comparison across two reference sets, and the fix
cost the baseline 0.007 in the same stratum, so we do not read it as a lead — but we do not
claim the reverse either until the selected arm is rescored. Everywhere else the floor is
ahead, so it is a property of the metric and not of train/test overlap. On exact match
retrieval is at zero in two strata of three, and the
ordering inverts: generation wins. Our method's lead over MolT5-base across the three strata
(2.6x, 3.8x and 2.9x) is a cross-reference-set comparison and should be read as a direction,
not a figure.

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

All three rows are scored against the pre-fix reference set of §7 and are comparable with
each other, not with the post-fix rows of §3.

| Selection rule (pre-fix references) | BLEU | Exact | Leven↓ | MACCS | RDK | Morgan | FCD↓ | Validity | pairs |
|---|---|---|---|---|---|---|---|---|---|
| none (top-1 length) | 0.736 | 0.052 | 21.72 | 0.754 | 0.583 | 0.498 | 0.86 | 0.764 | 2409 |
| valid, then likeliest length | 0.744 | 0.056 | 21.31 | 0.731 | 0.556 | 0.462 | 0.95 | 0.949 | 2975 |
| valid, then **AR likelihood** | **0.770** | **0.120** | **19.48** | **0.770** | **0.624** | **0.537** | **0.80** | **0.949** | 2975 |

Selecting on length alone buys validity and gives back fingerprint similarity: that rule
knows whether a string parses and how likely its length was, and nothing about what the
caption asked for. Selecting on caption likelihood moves **every metric at once**. The two
selection rows share a denominator of 2,975 scoreable pairs, so the difference between them
is the criterion and not the sample. The same top-1-length configuration rescored post-fix
reads 0.736 / 0.052 / 21.40 / 0.748 / 0.575 / 0.492 / 0.81 / 0.779 over 2,495 pairs, so the
ladder's baseline validity is 0.764 against these references and 0.779 against the corrected
ones; the 0.015 is the defect, not the method.

**Controls.**

- A 20k-step AR checkpoint selects as well as a 60k-step one: exact 0.1202 against 0.1205 —
  385 captions against 386 of 3,202, a one-caption difference — and BLEU 0.770 against 0.767.
  The 60k model has seen 580 epochs of
  25,574 molecules and has essentially memorised them, so memorisation is not what drives
  the gain. The two checkpoints are 0.0011 MACCS apart — a fifth of the 0.005 pre-fix seed
  spread, which is the applicable one, since both are pre-fix five-candidate runs — and the
  ordering should not be leaned on; what the control establishes is that more memorisation
  does not help.
- Per-stratum gains over the unselected baseline are 2.19x, 2.59x and 2.50x on exact match
  (§4), so the method improves generalisation rather than retrieval.
- Validity lands at 0.936–0.957 uniformly across strata.

**Scaling the pool.** The method is bounded by how often the true length is among the
candidates: 19.5% at the argmax, 63.2% in the top five, 83.6% in the top ten.

| pool (pre-fix references) | BLEU | Exact | Leven↓ | MACCS | RDK | Morgan | FCD↓ | Validity | pairs |
|---|---|---|---|---|---|---|---|---|---|
| 5 candidates | 0.770 | 0.120 | 19.48 | 0.770 | 0.624 | 0.537 | 0.80 | 0.949 | 2975 |
| **10 candidates** | **0.777** | **0.130** | **18.77** | 0.773 | 0.632 | 0.543 | **0.79** | **0.988** | 3073 |

Doubling the pool moves validity 0.949 → 0.988, exact match 0.120 → 0.130, BLEU 0.770 →
0.777 and Levenshtein 19.48 → 18.77. The fingerprints move less and over a different
denominator: MACCS +0.003, RDK +0.008, Morgan +0.006, against a ten-candidate seed spread of
0.003 MACCS. The ten-candidate row scores 3,073 pairs against 2,975, and the 98 extra pairs
are precisely the captions the five-candidate pool failed to parse, so the fingerprint
averages are not over the same molecules. Both rows are single seeds. The fingerprint gain
is not established; the validity, exact-match and string gains are.

An average of 7.6 of the 10 candidates parse, and only 37 captions of 3,202 (1.2%) have no
usable candidate, which is where 0.988 comes from. The selected rank is spread across the
whole pool — 18% at rank 0 down to 6% at rank 9 — so the later candidates are being chosen,
not merely counted.

**Independent noise per candidate.** Ten separate runs at one seed share a noise draw and
differ only in length. Drawing the noise per candidate costs nothing and is better on seven
of the eight metrics; FCD is not separated.

| full test split | BLEU | Exact | Leven↓ | MACCS | RDK | Morgan | FCD↓ | Validity |
|---|---|---|---|---|---|---|---|---|
| shared noise (2 post-fix seeds, worst) | 0.772 | 0.135 | 18.86 | 0.767 | 0.625 | 0.540 | 0.75 | 0.987 |
| shared noise, seed 42 (pre-fix references) | 0.777 | 0.130 | 18.77 | 0.773 | 0.632 | 0.543 | 0.79 | 0.988 |
| **own noise (2 seeds, worst)** | **0.781** | **0.143** | **18.18** | **0.771** | **0.629** | **0.545** | 0.74 | **0.993** |

The shared-noise group was reported as three seeds. Seed 42 is scored against the pre-fix
reference set and is printed separately rather than dropped; the group row is the worst of
the two post-fix seeds. Removing the pre-fix seed moves the group's exact match from 0.130 to
0.135 and its FCD from 0.79 to 0.75.

The worst own-noise seed beats the best shared-noise seed on seven of eight metrics — BLEU
0.781 against 0.776, exact match 0.143 against 0.139, Levenshtein 18.18 against 18.68, MACCS
0.771 against 0.768, RDK 0.629 against 0.625, Morgan 0.545 against 0.542 and validity 0.993
against 0.988 — which is this project's bar. **FCD is the exception and no gain is claimed
there**: the best shared-noise seed reaches 0.730, better than either own-noise seed (0.732
and 0.744). The 0.05 FCD advantage this paragraph used to report was almost exactly the
measured pre/post-fix FCD shift of −0.059, carried in by the pre-fix seed.

**Where this row sits.** Validity 0.993 at the worst of two seeds is fourth of the fourteen
rows in play. Above it are BioT5 and BioT5+ at 1.000, which generate SELFIES and parse by
construction, and our own retrieval floor at 0.994, which generates nothing. It is above
LDMol's 0.941 and TGM-DLM's 0.871, with the caveat that it is not the same quantity: ours is
the share of captions for which at least one of ten candidates is usable, one minus a
fall-back rate, where the published column is one sample per caption. We also require at least
three heavy atoms where the usual convention requires only a parse.

The same checkpoint's single-sample validity is 0.779, and we give it no rank. Two things stop
us. Like-for-like, the one published system we can run ourselves scores 0.782 on our split
with our scorer against this row's 0.779, so the only comparison here that does not pass
through a foreign scorer puts us below MolT5-base rather than above it. And under this paper's
own split adjustment — count the 98 dropped pairs as failures (§9) — 0.779 becomes 0.756,
below MolT5-base's published 0.772 as well.

Nothing else about this row is a ranking. In particular:

- **FCD 0.73–0.74 is not first.** It is sixth of the thirteen rows that report FCD. LDMol
  reports 0.20, BioT5+ 0.35, BioT5 0.43 and MolXPT 0.45. The claim that it was first was made
  against a table that had no LDMol row in it, and it is withdrawn. Our own retrieval floor,
  which generates nothing, scores 0.48, and our own 1,200-step single-trajectory run scores
  0.71; neither is first either.
- **LDMol is above this row on seven of the eight metrics it reports**: exact 0.530/0.143,
  MACCS 0.973/0.771, RDK 0.950/0.629 and Morgan 0.931/0.545 by margins larger than our
  measured scorer disagreement in those columns, and BLEU 0.926/0.781, Levenshtein 6.75/18.18
  and FCD 0.20/0.74 by margins smaller than it. Four of the seven survive the calibration;
  three sit inside it.
- **The row is above four of the eleven published rows and nothing else**: Transformer,
  GIT-Mol, T5-base and MolT5-base. Of the fourteen rows in play it is ninth on each of the
  three fingerprints, eighth on exact match, eighth of thirteen on BLEU, seventh of thirteen
  on Levenshtein, sixth of thirteen on FCD and fourth on validity. TGM-DLM is above it on
  BLEU, exact match, Levenshtein, MACCS, RDK and Morgan, and so are T5-large, MolT5-large,
  BioT5, BioT5+ and LDMol; MolXPT is above it on the four of those six it reports. Earlier
  drafts put the row third on those six metrics against a six-row table that carried TGM-DLM
  twice and held none of the five further systems that outrank it.
- **Two components of that placement sit inside the calibration gap.** Being ahead of
  MolT5-base on BLEU is a margin of 0.012 against a measured BLEU disagreement of 0.242 on
  that same checkpoint, and on Levenshtein 6.3 against a measured 15.9. Neither is a result.
- **Morgan is level with T5-base, not ahead.** The two numbers are 0.544985 and 0.545, a
  difference of 1.5×10⁻⁵ in a column whose measured cross-scorer shift is 0.017. They are the
  same number at the only resolution this paper can support.
- And none of these comparisons is like-for-like anyway: different split, different scorer
  (§2, §3, §9).

**What the pool buys.** Two controls separate the explanations. All six rows are post-fix.

| | Exact | Validity |
|---|---|---|
| one candidate, 300 solver steps | 0.052 | 0.779 |
| one candidate, 600 solver steps | 0.055 | 0.784 |
| one candidate, 1,200 solver steps | 0.050 | 0.782 |
| one candidate, 3,000 solver steps | 0.042 | 0.732 |
| 10 noise draws at one length | 0.098 | 0.987 |
| 10 lengths, own noise each (seed 42) | 0.143 | 0.996 |

Ten times the compute spent on integration accuracy makes exact match *worse*, 0.052 to
0.042 — monotonically from 600 steps on, after a single bump to 0.055 at 600. Spent on ten
trajectories it nearly triples it. And the length predictor is not incidental: ten noise
draws at the argmax length reach 0.098, so most of the validity is "sample until something
parses", but the remaining 0.045 of exact match comes from spanning the predictor's top-10
lengths. The 300-step row reads 0.052 / 0.764 against the pre-fix references; the validity
difference of 0.015 is the §7 defect, not the solver.

Four qualifications belong with it.

The pool costs ten sampling runs of 300 solver steps per caption, about 3,000 backbone passes
plus ten autoregressive scorings, where a single trajectory costs 300. **This is the wrong
side of the comparison.** LDMol spends roughly 200 network evaluations per caption — 100 DDIM
steps with classifier-free guidance, one sample — and reports exact match 0.530 against our
0.143. That is not a like-for-like comparison (§9), but it is not a close one either: we make
about fifteen times as many backbone passes for a worse result on seven of the eight metrics
LDMol reports, four of those beyond the calibration gap and three inside it.

**Fifteen is a ratio of network evaluations, not of compute, and we do not have a compute
ratio.** Nothing in this repository measures FLOPs or wall-clock for either system, and the
parameter counts run the other way: our backbone is 47.9M, where LDMol's DiT is roughly 160M
and is driven by a frozen text encoder of about 335M and an autoregressive latent-to-SMILES
decoder of about 120M, neither of which is counted in their 200 evaluations at all. A FLOP
accounting could narrow the ratio or invert it. Earlier drafts called this "about fifteen
times the diffusion compute"; that phrasing is withdrawn (§8).

Whatever is being demonstrated here, it is not that test-time sampling is how one wins this
benchmark. What the pool demonstrates is narrower and still worth stating: *within this
model*, sampling more trajectories buys what integrating one trajectory more accurately does
not, and the gain is realised by a selector that acts after the trajectory rather than inside
it.

Validity here is close to "at least one of ten candidates parses", a property of the
procedure rather than of a single forward pass.

The headline is the worse of two seeds (exact 0.143 and 0.147; validity 0.996 and 0.993), one
short of this project's three-seed unit; a third is running. The control rows above are one
run each, and the "10 lengths, own noise each" row is seed 42 alone.

The selection-rule and pool tables above are scored against the pre-fix references; the
control table here and the seeded tables are post-fix. The two sets are not comparable
cell-for-cell (§7).

---

## 6. Where latent diffusion cannot be steered

Nine interventions in the architecture, the target and the weights, plus four that use the
autoregressive model — guidance, clamping, repaint and selection. Thirteen rows against one
baseline, all on the same backbone, all scored by the same script. They are **not** evaluated
identically, and the table says how they differ. Eleven rows use the full 3,202-caption
split; likelihood guidance was swept on 500 captions, clamping on 1,000 and repaint on 64.
Four rows predate the tokenizer fix of §7. Each is read against a control at its own caption
count and reference set, named below the table. The seed spread is 0.005 MACCS on the pre-fix
recipe and 0.002 post-fix; nothing below clears the wider figure except where noted.

| Intervention | Where it acts | MACCS | Exact | Validity | n, refs | Verdict |
|---|---|---|---|---|---|---|
| baseline (two seeds) | — | 0.748 / 0.750 | 0.052 | 0.775 | 3202, post | reference |
| content channel in the latent | trajectory | 0.724 | 0.041 | 0.721 | 3202, pre | worse |
| content channel replacing length | trajectory | 0.582 | 0.003 | 0.690 | 3202, pre | much worse |
| global caption conditioning (adaLN) | trajectory | 0.748 | 0.043 | 0.768 | 3202, pre | worse |
| retrieval-initialised sampling | trajectory | 0.785 | 0.004 | 0.993 | 3202, post | returns its input |
| task-adapted caption features | conditioning | 0.444 | 0.002 | 0.705 | 3202, post | much worse |
| distinct padding, 4 variants | target | 0.705 | 0.030 | 0.725 | 3202, post | worse |
| distinct padding, 8 variants | target | 0.661 | 0.026 | 0.637 | 3202, post | worse, monotone |
| DPO, length-varying pairs | weights | 0.746 | 0.053 | **0.800** | 3202, post | validity only |
| DPO, content-varying pairs | weights | 0.749 | 0.053 | **0.795** | 3202, post | validity only |
| AR likelihood guidance (98% of score) | trajectory | 0.741 | 0.038 | 0.748 | 500, post | no effect |
| AR token clamping | trajectory | 0.582 | 0.028 | 0.627 | 1000, post | destroyed — by the clamping |
| AR selective repaint | trajectory | — | — | — | 64, post | byte-identical output |
| **AR likelihood selection** | **after** | **0.770** | **0.120** | **0.949** | 3202, pre | **works** |

**Matched controls**, since the baseline row does not serve all of these.

- *Likelihood guidance* is compared to its own zero-scale control on the same 500 captions:
  MACCS 0.744, exact 0.036, validity 0.764. Against it the swept run is −0.004 MACCS, +0.002
  exact, −0.016 validity. The "no effect" verdict is the matched one.
- *Token clamping* is compared to the unclamped run on the same 1,000 captions: MACCS 0.759,
  exact 0.059, validity 0.788. Against it clamping is −0.177 / −0.031 / −0.161. "Destroyed"
  survives at full strength.
- *adaLN* was previously read as "no effect" from a 0.748 = 0.748 tie against a post-fix
  baseline. The tie is an artefact of crossing the fix. Against its own pre-fix group of three
  seeds (MACCS 0.756 / 0.754 / 0.756, exact 0.050 / 0.052 / 0.048) it is 0.748 and 0.043,
  below the group's worst seed on both by about one seed spread, so the verdict is **worse**,
  not no effect.
- *AR likelihood selection* is pre-fix throughout, and its matched pre-fix baseline is the
  0.754 / 0.052 / 0.764 row of §5, not the post-fix 0.748 / 0.052 / 0.779 printed here.
- The DPO rows move MACCS by −0.003 and +0.001 against the baseline's worst seed. Both are
  inside the 0.005 pre-fix spread; against the 0.0023 post-fix one they part, the
  length-varying row at −0.0025 falling outside it and the content-varying row at +0.0010
  inside. Neither clears the project's worst-beats-best rule against the baseline's better
  seed at 0.750; "validity only" is the claim.

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
model, ranking the same model's finished samples, improves exact match 2.3x on the same
reference set (0.052 to 0.120), and 2.7x in the ten-candidate configuration of §5. A model
that is an excellent discriminator over completed samples is useless as a guide during
sampling.

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
it are not.

**LDMol is the independent confirmation, and it is a better argument than the ranking we
used to make here.** Earlier drafts called LDMol "the field's strongest diffusion result on
this benchmark" on the strength of one number, validity 0.941 — the only LDMol figure this
project held, and the single metric on which we were ahead of it. The real row (§2) is
BLEU 0.926, exact match 0.530, Levenshtein 6.75, MACCS 0.973, RDK 0.950, Morgan 0.931,
FCD 0.20, validity 0.941, and it is ahead of ours on seven of eight — four of those by
margins beyond the calibration gap and three inside it (§5).

What matters is not the ordering but where LDMol puts the discrete decision. Our measurement
says the decision cannot live inside the trajectory: a finished sample is a fixed point, the
model re-derives the token it already produced from noise levels that destroy most of the
signal, and the only intervention that moves anything acts after the trajectory has
finished. LDMol's design agrees. It diffuses in a contrastively-trained latent space that
carries structural information, and it converts that latent into a string with an
autoregressive decoder — the discrete step is outside the diffusion, exactly where our
negative results say it has to be. A system built on that separation produces the strongest
row of §2 — leading that table on seven of its eight columns and losing validity to BioT5 and
BioT5+ at 1.000 and to MolXPT at 0.983, which puts it fourth of eleven on that column. That
is an ordering of eleven papers' scorers against each other, none of which our own numbers
ever passed through, so we read it as the field's own ordering and not as a measurement. A
system that tries to steer the discrete content from inside the trajectory, as we did nine
times, produces nothing.

The same table bounds what that observation is worth. LDMol's exact match is 0.008 above
BioT5+'s, a tie at our resolution. BioT5 and BioT5+ are above TGM-DLM on every column both
report, and MolXPT on MACCS, RDK, FCD and validity but not on exact match or Morgan; between
LDMol and TGM-DLM the table is entirely autoregressive. So the strongest single row is a
diffusion model and diffusion is still not what puts a row at the top of this benchmark: the
rows are ordered by the scale and chemistry-awareness of the pretrained representation, not by
family. LDMol is evidence about where a diffusion model has to put its discrete step, which is
what our negative results are about. It is not evidence that diffusion is how one scores on
ChEBI-20.

That is the claim worth making, and it does not depend on who is first in any table. It also
suggests the obvious next experiment: our continuous latent is a frozen embedding space, and
the part of LDMol's recipe that our architecture lacks is a latent trained to be structurally
informative in the first place.

---

## 7. An evaluation defect worth reporting

The fine-tune configuration pointed the evaluation tokenizer at a 391-token ZINC vocabulary
while the corpus was tokenized with the 635-token shared one. Consequences, measured:
**196 of 3,202 test references (6.1%) decoded wrongly** — 95 to unparseable strings and the
rest to a *different molecule*; salts, hydrates, metals and charged nitrogen all live above
id 386. The decoder's logits were truncated to 386, so no generated string could contain a
disconnection dot. The project had recorded 95 "isotope captions with no structure"; after
the fix there are none, and those rows are salts and complexes.

The split of the 196 needs rerunning. Earlier drafts, and `configs/chebi20_finetune.yaml`
until it was corrected, reported "75 unparseable and 121 a different molecule". 75 is not a
number the scorer ever produced: every pre-fix run on the full split records `unparseable_reference = 95`, the five
pre-fix runs on the 1,000-caption subsets record the proportional 30, and every post-fix run
records 0. Either the total is 196 with 101 wrong molecules, or the total is 216 (6.7%) with
121. Nothing in the repository decides it, so we quote the measured 95 and flag the rest.

Correcting it moves MACCS by −0.006, RDK by −0.008, Morgan by −0.007 and validity by +0.015
— the size of a real effect in this study, and the reason reference sets cannot be mixed
inside a table.

**Which rows are post-fix.** Not all of them, contrary to what this section said before. The
§3 table, the §4 "Ours, diffusion (post-fix)" rows, the §5 seeded tables and control table,
and ten of the fourteen §6 rows are post-fix (`unparseable_reference = 0`). Scored against
the pre-fix reference set, and marked as such wherever they appear: the three selection-rule
rows and both pool rows of §5, the shared-noise seed-42 row, the §4 rows labelled pre-fix,
and four §6 rows — content channel, content channel replacing length, global caption
conditioning, and AR likelihood selection. No post-fix rescoring exists for those runs;
rescoring alone would not be enough, since the defect also truncated the decoder's reachable
vocabulary, so they must be regenerated. Until then they are read against pre-fix controls,
never against post-fix ones.

---

## 8. Retracted claims

Kept here because each was stated with confidence before being checked. The measurements
behind them are unchanged and still printed above; what fell is the claim made about them.

| Claim | Why it fell |
|---|---|
| "FCD 0.73–0.74 is first in the published table" | LDMol reports 0.20, BioT5+ 0.35, BioT5 0.43 and MolXPT 0.45; the row is sixth of thirteen. The table we checked against had none of them in it. Our own retrieval floor scores 0.48 and our own 1,200-step run 0.71, so it is not first among our rows either |
| "Ahead of every other published system on all eight metrics" | false on six of eight against TGM-DLM (BLEU 0.826, exact 0.242, Levenshtein 17.003, MACCS 0.854, RDK 0.739, Morgan 0.688) and against five further rows the repo did not hold — T5-large, MolT5-large, MolXPT, BioT5 and BioT5+ — and Morgan is a tie with T5-base, 0.544985 against 0.545 in a column whose cross-scorer shift is 0.017, not a lead |
| "The row is second to TGM-DLM on everything else" | not second and not third. The sourced table (§2) carries TGM-DLM once and adds six systems. The row is above Transformer, GIT-Mol, T5-base and MolT5-base and nothing else; it is ninth of fourteen on each fingerprint |
| "The floor beats TGM-DLM, MolT5, T5 and us on the fingerprint columns" | TGM-DLM is above the floor on all three (MACCS 0.854, RDK 0.739, Morgan 0.688 against 0.785, 0.690, 0.576), and so are T5-large, MolT5-large, MolXPT, BioT5, BioT5+ and LDMol. The floor is above four published rows on the fingerprints, not all but one |
| "A system that generates nothing reaches or beats every published row except LDMol" | the same error in contribution 1. Seven of the eleven published rows clear the floor on the fingerprints. What is true is the ranking: third of fourteen on validity, fifth of thirteen on FCD, eighth on the fingerprints, thirteenth of fourteen on exact match |
| "Most published systems do not clear that floor by much" | TGM-DLM clears it by +0.069 MACCS, +0.049 RDK, +0.112 Morgan and by 0.242 against 0.004 on exact match, and six further rows clear it on the fingerprints as well. The floor is a ranking of metrics by how much each rewards lookup, not a statement that the field sits near it |
| "Validity 0.993 is above every published row we hold" | BioT5 and BioT5+ report 1.000, reached by generating SELFIES, and our own retrieval floor reaches 0.994. The row is fourth of fourteen |
| "Single-sample validity 0.779 is fifth of the seven rows" | no rank is claimed now. Our own run of the MolT5-base checkpoint scores 0.782 on the same split with the same scorer, and the split adjustment takes 0.779 to 0.756, below MolT5-base's published 0.772 |
| "Beyond that we make no cross-paper ranking claim at all" | the paper made at least eight. The disclaimer is replaced by the scoped form: every cross-paper statement here is a comparison against numbers our scorer does not reproduce, and any margin below the measured shift for its column is written as noise |
| "LDMol beats our row / the floor on seven of eight" (as a flat count) | four of those seven margins are larger than the measured scorer disagreement in their column; three — BLEU, Levenshtein, FCD — are smaller. The count stands only with that split stated |
| "About fifteen times LDMol's diffusion compute" | fifteen is a ratio of backbone passes, 3,000 against about 200. Nothing here measures FLOPs or wall-clock, and our 47.9M backbone is smaller than LDMol's ~160M DiT, whose ~335M text encoder and ~120M decoder are not counted on their side at all |
| "Six published rows, five of them with no source" | replaced by the eleven rows of LDMol's Table 1 (Chang and Ye, ICML 2025). The transcription's T5-base BLEU was 0.762 against the source's 0.765, and its "TGM-DLM w/o correction" row is not in the source table |
| "Exact match separates lookup from generation in the rows we hold" | the Transformer reports 0.000, below the floor's 0.004. It separates them in ten of the eleven published rows |
| "Retrieval leads on fingerprint similarity in all three strata" | the novel stratum carries an exception the paper declines to resolve, 0.621 against 0.619 across two reference sets (§4). The abstract now carries the same hedge as the section |
| "The 20k and 60k AR checkpoints differ by exact 0.1202 against 0.1206" | the 60k figure is 0.1205, 386 captions of 3,202. The difference between the checkpoints is one caption |
| "LDMol is the field's strongest diffusion result, which we beat on validity" | the premise was right and the evidence was one number. LDMol's real row beats ours on seven of eight, four of them beyond the calibration gap; the interesting relationship is mechanistic, not ordinal (§6) |
| "ChEBI-20 measures lookup" | too strong. Its aggregate metrics are *reachable* by lookup, and how far each metric is reachable is what the floor measures: seven of the eleven published rows clear it on the fingerprints, and it is last but one on exact match. The floor is an instrument, not a verdict on the benchmark |
| "Every number in this paper is post-fix" | eight runs feeding §3, §4, §5 and §6 carry `unparseable_reference = 95`. The affected rows are now marked (§7) |
| "Validity 0.996 and FCD 0.73, first in the table" | quoted at the best seed, against this project's own worst-beats-best rule. At the worst seed the group is 0.993 and 0.74 |
| "Drawing the noise per candidate is strictly better" | seven of eight. The best shared-noise seed reaches FCD 0.730 against the own-noise group's 0.732 and 0.744 |
| "Doubling the pool moves every metric" | the fingerprint columns move by +0.003 MACCS, +0.008 RDK, +0.006 Morgan over a denominator that grew by 98 pairs, which is not a measured gain |
| "Global caption conditioning has no effect" | the 0.748 tie was a pre-fix row read against a post-fix baseline. Against its own group it is worse on MACCS and exact match |
| "A test-time-compute result" (as a competitive claim) | we make about 3,000 backbone passes plus ten verifier scorings per caption; LDMol makes about 200 and scores better on seven of eight metrics. The pool result is internal to this model, not a statement about what wins the benchmark, and the 15:1 is a count of network evaluations rather than of compute |
| "95 test captions describe isotopes and carry no structure" | the evaluation tokenizer was decoding salts and complexes wrongly. Post-fix the full split has zero unparseable references (§7) |
| "The strongest results on ChEBI-20 are not diffusion models" | self-contradictory beside the table it was read from: LDMol is a diffusion model and leads seven of the eight columns. What survives is narrower — the single strongest row is diffusion, its 0.008 exact-match lead over BioT5+ is inside every resolution we can measure, and everything between it and the other diffusion row is autoregressive, so the ordering tracks the pretrained representation rather than the family (§2, §6) |
| "MolXPT, BioT5 and BioT5+ all sit above TGM-DLM on the three fingerprints and on exact match" | true of BioT5 and BioT5+ only. MolXPT reads Morgan 0.667 against TGM-DLM's 0.688 and exact match 0.215 against 0.242; it leads on MACCS, RDK, FCD and validity (§2, §6) |
| "TGM-DLM, not LDMol, is the first row above the retrieval floor" | the nearest row above the floor is T5-large on MACCS and on RDK and MolXPT on Morgan. TGM-DLM is fifth on MACCS, sixth on RDK and fourth on Morgan, with one to three rows between it and the floor in each column (§3) |
| "On FCD the floor beats seven published rows" | six. GIT-Mol reports no FCD, so ten published rows carry the column: six beaten — Transformer, T5-base, MolT5-base, T5-large, MolT5-large, TGM-DLM — and four above it. The abstract and contribution 1 always said six; §3 and `scripts/score_benchmark.py` said seven |
| "LDMol loses validity to BioT5 and BioT5+" | and to MolXPT at 0.983. LDMol's 0.941 is fourth of the eleven published rows on that column (§6) |
| "The 20k and 60k AR checkpoints are a third of a seed spread apart" | 0.0011 MACCS, a fifth of the 0.005 pre-fix spread that applies to both rows. A third is the ratio against the 0.003 ten-candidate spread quoted for a different configuration (§5) |
| "Both DPO rows sit inside the pre-fix seed spread and outside the post-fix one" | true of the length-varying row, −0.0025 against a post-fix spread of 0.0023, and false of the content-varying one, +0.0010, which is inside both. The "validity only" verdict does not rest on it (§6) |

## 9. Limitations

- **No cross-paper ranking in this paper is sound, and every one made is flagged as
  indicative.** Earlier drafts claimed none was made; the paper makes several, and they are
  now marked rather than disclaimed away. Only our own runs and one public MolT5-base
  checkpoint have been scored by our scorer. On that checkpoint our numbers and the published
  column disagree on six of eight metrics: BLEU 0.527 against 0.769, Levenshtein 40.34 against
  24.46, exact match 0.043 against 0.081, MACCS 0.755 against 0.721, Morgan 0.512 against
  0.529, FCD 1.63 against 2.18; only RDK (0.583 against 0.588) and validity (0.782 against
  0.772) land close. Same checkpoint, and the 98 captions our curation drops cannot account
  for gaps that size (§3). No TGM-DLM, LDMol, T5, BioT5, MolXPT, GIT-Mol or Transformer output
  has ever passed through our scorer.
- **The calibration gap, as a rule.** Those disagreements are the only measurement we have of
  what a foreign scorer does to a column: BLEU 0.242, Levenshtein 15.9, FCD 0.553, exact match
  0.038, MACCS 0.033, Morgan 0.017, RDK 0.005. A cross-paper margin smaller than the shift for
  its own column is noise and is written as noise wherever this paper quotes one. It is a
  single-checkpoint calibration on one architecture, so it bounds nothing; it is the floor
  under what we are willing to call a difference.
- **Split mismatch.** Our rows are on a curated 3,202-pair split; every published row is on
  the standard uncurated 3,300-pair split. `scripts/prepare_paired.py` drops a pair whole
  when the reference SMILES does not parse under RDKit or exceeds the 128-token canvas,
  which removes 98 pairs (3.0%) — the unparsable and the longest molecules, the ones a
  128-token model cannot emit. The curation is not neutral with respect to validity or exact
  match. Counting the dropped pairs as failures would put our validity at 0.963–0.966 rather
  than 0.993–0.996.
- **Provenance of the published rows.** All eleven now come from one cited table, LDMol's
  Table 1 (§2), which is the first table in this project with a source. We have not opened the
  primary papers for the ten rows LDMol reports from other work, so each of those is a
  transcription at one remove, and a table assembled by one group collects other groups'
  scorers. The six-row
  table it replaces was transcribed from a chat message in commit 29ad1f6 with no source at
  all; two of its rows did not survive the check (§2).
- One backbone, one scale (47.9M). The negative results in §6 are properties of this model,
  not proofs about latent diffusion in general, though their consistency across nine
  mechanisms is suggestive.
- The retrieval floor uses TF-IDF. A stronger retriever would raise the floor and
  strengthen the argument, but we have not measured how far.
- §5's ten-candidate own-noise headline is two seeds, not the three this project calls a
  unit; a third is running. The shared-noise comparison group has two post-fix seeds plus one
  pre-fix run, printed separately. The K=5 AR-selection run is a single pre-fix run.
- Eight columns, not the published nine. Text2Mol needs an external retrieval model we do
  not have, and LDMol does not report it either.
- **Network evaluations, not compute.** The headline configuration makes about fifteen times
  as many backbone passes per caption as LDMol — about 3,000 against about 200 — for a worse
  result on seven of the eight metrics LDMol reports, four of them beyond the calibration gap
  (§5). That is a count of network evaluations. Nothing in this repository measures FLOPs or
  wall-clock for either system, and the parameter counts run the other way: 47.9M for our
  backbone against roughly 160M for LDMol's DiT, plus their frozen ~335M text encoder and
  ~120M autoregressive decoder, neither of which is in their 200. We do not know the compute
  ratio and do not claim one.

## 10. What we would do next

1. Score someone else's outputs with our scorer. Until a TGM-DLM or LDMol generation file
   goes through `scripts/score_benchmark.py`, the 0.24 BLEU and 16-point Levenshtein gaps we
   measure on MolT5-base make every cross-paper number in the field unusable to us. LDMol's
   code and weights are public, which makes it the one to do first.
2. Regenerate and rescore the eight pre-fix runs (§7), starting with the five-candidate
   AR-selection run that §3, §4 and §6 all lean on.
3. A third seed for the ten-candidate own-noise configuration, and seeds for the retrieval
   floor, which is still a single run.
4. A stronger retriever (SciBERT caption embeddings) to find where the floor actually sits.
5. Stratified numbers for LDMol, to test whether the system that clears the floor by the
   widest margin is also flat across proximity strata or steep like the rest. If LDMol is
   flat, the floor plus the
   stratification is a usable two-axis report for this benchmark; if it is steep, the
   benchmark's aggregate is still mostly a proximity statistic.
6. A structurally informative latent, which is the part of LDMol's recipe our architecture
   does not have (§6).

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
# the headline configuration of §5 is --k 10 with a separate noise draw per candidate;
# --k 5 with one shared draw reproduces the pre-fix selection table.
# one generation per rank, then
python scripts/score_with_ar.py --config configs/chebi20_ar.yaml --checkpoint <ar> \
    --dirs <k0> ... --out <scores>.npy
python scripts/select_candidates.py --dirs <k0> ... --probs <scores>.npy --out <dir>

# scoring and stratification
python scripts/score_benchmark.py --dir <dir>
```
