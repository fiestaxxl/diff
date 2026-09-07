# Experiment log: ZINC-250k

Everything below runs on the workhorse model (256 wide, 4 blocks, 5.04M parameters),
one H100 per run, batch 1024 molecules, bf16 + tf32 + torch.compile, comet off. The
corpus and the tokenizer are described in docs/tokenizer-zinc250k.md.

Evaluation protocol: 10,000 molecules sampled with the Euler-Maruyama sampler over 300
steps from a fixed seed, scored with RDKit. Validity carries a Wilson 95% interval;
uniqueness is computed among the valid ones, novelty against the 224,568 training
molecules.

## Budget: how many tokens per parameter

Constant learning rate after warmup, so the curve is not confounded by a cosine decay.
Validation loss against the consumed budget:

| run | parameters | 2 t/p | 5 | 10 | 20 | 40 | best | within 1% at |
|---|---|---|---|---|---|---|---|---|
| 5.0M seed 42 | 5,038,272 | 3.494 | 0.877 | 0.830 | 0.798 | 0.799 | 0.750 | 34 t/p |
| 5.0M seed 43 | 5,038,272 | 3.543 | 0.862 | 0.824 | 0.811 | 0.801 | 0.758 | 24 t/p |
| 16.6M | 16,594,496 | 0.861 | 0.770 | 0.800 | 0.783 | 0.783 | 0.748 | 10 t/p |
| 44.9M | 44,939,968 | 0.788 | 0.800 | 0.800 | 0.800 | 0.800 | 0.761 | 1.4 t/p |

Two things came out of this.

**Validation loss cannot rank model sizes here.** All four runs land on the same
0.75-0.80, and the seed-to-seed spread is about 1%. The loss has a floor it cannot go
below: the objective asks the model to predict the noise that we inject into the
embeddings ourselves (`x0_noise_std = 0.25`), and that part is unpredictable by
construction. Bigger models reach the floor with fewer tokens per parameter, they do not
reach a lower floor.

**Validity keeps improving long after the loss has flattened.** The probes above ran to
60 t/p and gave 3.7% and 4.9% validity, while the 20 t/p runs of the next section gave
around 1%. So the loss curve is not a stopping criterion, and the budget has to be set
from the generation metric instead.

## Grammar loss: does it help?

The penalty on parenthesis balance and ring parity, at the pre-refactor weight 0.001,
against the same run with the term switched off. Both at 20 t/p (4,400 steps).

| run | validity | uniqueness | diversity |
|---|---|---|---|
| grammar 0.001, seed 42 | 1.051% (0.87-1.27) | 100% | 0.919 |
| grammar 0.001, seed 43 | 0.854% (0.69-1.05) | 100% | 0.921 |
| grammar off, seed 42 | **1.504%** (1.28-1.76) | 96.0% | 0.918 |
| grammar off, seed 43 | **1.352%** (1.14-1.60) | 86.6% | 0.931 |

Switching it off gives 1.43% against 0.95% on the mean of two seeds, and the intervals
barely touch. The term hurts, exactly as the old ChEBI sweep suggested, and it also
costs a softmax over the whole `(B, L, V)` logits tensor. It is off in everything below.

## The ladder, at 20 tokens per parameter

Every row is 10,000 sampled molecules from a fixed sampling seed; the model differs only
in the switch named. Two seeds where noted. Grammar loss off everywhere except the first
two rows of the previous section.

| run | validity | 95% CI | uniqueness | diversity | failures: parens / rings / other |
|---|---|---|---|---|---|
| grammar off (reference) s42 | 1.50% | 1.28-1.76 | 96.0% | 0.918 | 68 / 26 / 6 |
| grammar off (reference) s43 | 1.35% | 1.14-1.60 | 86.6% | 0.931 | 67 / 27 / 6 |
| cross-entropy gate off s42 | 1.26% | 1.06-1.50 | 94.4% | 0.918 | 70 / 25 / 5 |
| cross-entropy gate off s43 | 1.33% | 1.12-1.58 | 88.6% | 0.930 | 68 / 26 / 6 |
| x0_noise_std 0.10 | 1.19% | 1.00-1.42 | 99.2% | 0.906 | 70 / 25 / 6 |
| x0_noise_std 0.50 | 0.78% | 0.63-0.97 | 100% | 0.915 | 69 / 25 / 5 |
| mask_padding s42 | 0.07% | 0.03-0.15 | 100% | 0.952 | 65 / 33 / 3 |
| mask_padding s43 | 0.03% | 0.01-0.09 | 100% | 0.959 | 68 / 30 / 2 |
| mask_padding + bucketing s42 | 0.11% | 0.06-0.20 | 90.9% | 0.960 | 65 / 32 / 3 |
| mask_padding + bucketing s43 | 0.08% | 0.04-0.16 | 87.5% | 0.968 | 67 / 30 / 3 |

**Masking the padding destroys generation, 20 to 50 times worse.** This is the most
useful negative result of the batch, and the mechanism is clear once seen. On a fixed
canvas of 64 positions the padding is not waste: it is the only place where the model
learns where a molecule ends. Excluding pads from attention and from the loss removes
every gradient that ever taught the model to emit a pad latent, so at sampling time the
region past the molecule is unconstrained noise, the decoder writes junk into it and the
string never parses. Length bucketing, which needs masking to be well defined, inherits
the same collapse.

So the padding waste measured earlier cannot be reclaimed by masking alone. It needs an
explicit length mechanism: predict the length, or supervise `<eos>` separately, or
diffuse a variable-length latent.

**Opening the cross-entropy gate does nothing**, 1.30% against 1.43%, inside the seed
spread. That is consistent with the diagnosis: with `ce_input=x0` the readout only has
to invert the embedding table it owns, the task saturates within the first epochs
(accuracy 1.000, cross-entropy 0.000), and adding more of the same signal changes
nothing.

**The injected noise is near its optimum at 0.25.** Lowering it to 0.10 or raising it to
0.50 both come out worse, though only one seed each.

## Budget dominates everything

Same configuration (grammar off), only the number of steps changes.

| budget | steps | validity | 95% CI | failures: parens / rings |
|---|---|---|---|---|
| 20 t/p | 4,400 | 1.43% (mean of two seeds) | | 68 / 26 |
| 40 t/p | 8,800 | 2.65% | 2.35-2.98 | 63 / 30 |
| 80 t/p | 17,600 | 5.28% | 4.86-5.73 | 51 / 41 |

Validity roughly doubles with every doubling of the budget and shows no sign of
flattening, while the validation loss stopped moving at 10-20 t/p. The share of failures
caused by unbalanced parentheses falls from 68% to 51% as the budget grows: the model
learns branches first and ring closures later.
