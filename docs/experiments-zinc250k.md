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
| 160 t/p | 35,200 | 9.69% | 9.13-10.29 | 42 / 48 |
| 320 t/p | 70,400 | 12.85% | 12.21-13.52 | 34 / 54 |

Validity roughly doubles with every doubling of the budget up to 160 tokens per
parameter, and then the curve bends: the last doubling buys only a third more. So 160 is
the knee for this model, and the validation loss, which stopped moving at 10-20, was
never a useful signal for it.

The share of failures caused by unbalanced parentheses falls from 68% to 34% as the
budget grows while ring-closure errors rise from 26% to 54%: the model learns branches
first and ring bookkeeping much later. Whatever is worked on next should target rings.

Note also that 5.0M at 320 t/p (70,400 steps, 12.85%) loses to 44.9M at 20 t/p
(39,000 steps, 21.74%): at equal or lower wall clock, parameters beat passes.

## Everything on one table

10,000 sampled molecules per row, fixed sampling seed, scored with RDKit. The last
column is the share of invalid strings caused by unbalanced parentheses, odd ring digits
and everything else.

| run | model | budget | schedule | validity | 95% CI | uniq | div | parens / rings / other |
|---|---|---|---|---|---|---|---|---|
| probe_45m | 44.9M | 20 t/p, 39,000 steps | constant, grammar on | **21.74%** | 20.94-22.56 | 99.6% | 0.884 | 35 / 50 / 15 |
| probe_17m | 16.6M | 40 t/p, 28,800 steps | constant, grammar on | **11.58%** | 10.97-12.23 | 100% | 0.887 | 27 / 60 / 13 |
| bud160 | 5.0M | 160 t/p, 35,200 steps | cosine | 9.69% | 9.13-10.29 | 99.3% | 0.892 | 42 / 48 / 10 |
| bud80 | 5.0M | 80 t/p, 17,600 | cosine | 5.28% | 4.86-5.73 | 99.4% | 0.895 | 51 / 41 / 8 |
| ce_input=x0_hat s42 | 5.0M | 80 t/p | cosine | 2.83% | 2.52-3.17 | 100% | 0.897 | 56 / 35 / 9 |
| ce_input=x0_hat s43 | 5.0M | 80 t/p | cosine | 2.69% | 2.39-3.03 | 100% | 0.897 | 64 / 28 / 7 |
| bud40 | 5.0M | 40 t/p, 8,800 | cosine | 2.65% | 2.35-2.98 | 100% | 0.903 | 63 / 30 / 6 |
| grammar off s42 | 5.0M | 20 t/p | cosine | 1.50% | 1.28-1.76 | 96.0% | 0.918 | 68 / 26 / 6 |
| grammar off s43 | 5.0M | 20 t/p | cosine | 1.35% | 1.14-1.60 | 86.6% | 0.931 | 67 / 27 / 6 |
| ce gate off s43 | 5.0M | 20 t/p | cosine | 1.33% | 1.12-1.58 | 88.6% | 0.930 | 68 / 26 / 6 |
| ce gate off s42 | 5.0M | 20 t/p | cosine | 1.26% | 1.06-1.50 | 94.4% | 0.918 | 70 / 25 / 5 |
| x0_noise 0.10 | 5.0M | 20 t/p | cosine | 1.19% | 1.00-1.42 | 99.2% | 0.906 | 70 / 25 / 6 |
| baseline s42 | 5.0M | 20 t/p | cosine, grammar on | 1.05% | 0.87-1.27 | 100% | 0.919 | 72 / 23 / 5 |
| baseline s43 | 5.0M | 20 t/p | cosine, grammar on | 0.85% | 0.69-1.05 | 100% | 0.921 | 65 / 29 / 6 |
| x0_noise 0.50 | 5.0M | 20 t/p | cosine | 0.78% | 0.63-0.97 | 100% | 0.915 | 69 / 25 / 5 |
| mask_padding + bucketing s42 | 5.0M | 20 t/p | cosine | 0.11% | 0.06-0.20 | 90.9% | 0.960 | 65 / 32 / 3 |
| mask_padding + bucketing s43 | 5.0M | 20 t/p | cosine | 0.08% | 0.04-0.16 | 87.5% | 0.968 | 67 / 30 / 3 |
| mask_padding s42 | 5.0M | 20 t/p | cosine | 0.07% | 0.03-0.15 | 100% | 0.952 | 65 / 33 / 3 |
| mask_padding s43 | 5.0M | 20 t/p | cosine | 0.03% | 0.01-0.09 | 100% | 0.959 | 68 / 30 / 2 |

Novelty is 99.9-100% everywhere, so nothing memorizes the training set.

## Model size beats epochs at equal step count

The two probes were meant only to calibrate the budget, and they turned out to be the
best runs of the whole batch. At a comparable number of optimizer steps:

| model | steps | validity |
|---|---|---|
| 5.0M | 35,200 | 9.69% |
| 16.6M | 28,800 | 11.58% |
| 44.9M | 39,000 | 21.74% |

Four times the parameters at the same step count more than doubles validity, and this is
invisible in the validation loss, which was 0.75-0.80 for all three. So the tokens per
parameter rule has to be read the other way round: with a fixed wall-clock budget, spend
it on parameters, not on extra passes over the corpus. Nothing has saturated yet in
either direction.

## What the readout tells us, and why the obvious fix backfired

With the original `ce_input=x0` the readout is fed the data latents, so it only has to
invert the embedding table it owns: cross-entropy reaches 0.000 and token accuracy 1.000
within the first epochs. That looked like the defect to fix, since at sampling time the
readout is fed the output of the reverse SDE instead.

Training it on the reconstruction (`ce_input=x0_hat`) does make the task real -
accuracy settles around 0.71-0.79 instead of 1.000 - and it makes generation clearly
worse: 2.83% and 2.69% against 5.28% at the same budget. The reason is that the sampler
integrates to t = 0.999, where the state is by construction close to x0, so a readout
trained on x0 is matched to what it will see, while x0_hat at low alpha is dominated by
the 1/alpha amplification of the denoiser error. The original design was right and the
hypothesis is refuted.

## The embedding scale fixes itself

The signal-to-noise analysis in docs/diffusion-review.md was done at initialization,
where the token identity carries 0.63% of the variance of x0. During training the mean
embedding norm grows from 0.112 to 5.4, i.e. a per-dimension standard deviation of about
0.95 against the injected noise of 0.25, so the identity ends up carrying roughly 93% of
the variance. The model repairs the schedule on its own, which is why lowering
`x0_noise_std` to 0.10 did not help and why absolute loss values cannot be compared
across a run.
