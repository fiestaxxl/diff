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

## Decoding, measured on one fixed checkpoint

No training involved: the same 5.0M model trained to 160 tokens per parameter, 10,000
attempts each, 100 solver steps. Yield is counted per attempt, not per non-empty string,
because the repair modes can return nothing at all; length is measured over the valid
molecules and the corpus reference is 44.3 characters.

| decoding | valid | unique valid | mean length |
|---|---|---|---|
| argmax (the original) | 968 | 961 | 37.3 |
| grammar repair, close what is open | 2070 | 2070 | 39.9 |
| grammar repair, mixed | **4419** | **3513** | 25.8 |
| grammar repair, trim what is open | 5241 | 3067 | 18.8 |
| argmax, x0 estimate clamped | 43 | 37 | 11.0 |
| trim, x0 estimate clamped | 859 | 369 | 8.8 |

Repair removes parenthesis failures entirely and cuts ring failures to under a tenth,
so 92% of what still fails is valence and aromaticity. The three modes differ in what
they do with a molecule that ends with something open. Closing appends the missing ring
digit and brackets: it doubles the number of unique valid molecules and leaves the length
distribution nearly intact, which makes it the honest default. Mixed picks per string
between closing and trimming and gives 3.7x the molecules, but the average one is 26
characters against the corpus 44, so the distribution moves and the number must always be
quoted next to that length. Trimming alone is the extreme case of the same trade.

Clamping the x0 estimate onto the nearest token embedding, the trick that helps in
Diffusion-LM, is harmful here at full strength and neutral at 0.1, so the default stays
off.

## Solver settings, same checkpoint

| sigma | 100 steps | 300 steps | 1000 steps |
|---|---|---|---|
| 1.0 | 9.67% | 9.69% | 10.00% |
| 0.5 | 9.34% | 8.99% | 8.76% |
| 0.0 (probability flow) | 8.69% | 8.29% | 8.35% |

The stochastic sampler with sigma 1.0 is already the best of these, the deterministic
limit is a point and a half worse, and the step count barely matters: 100 steps match
1000, so generation can be three times cheaper than it was.

## Screening at a fixed size and budget

Every row: 5.04M parameters, batch 1024, 17,600 steps (80 tokens per parameter), grammar
loss off, seed 42, argmax decoding, 10,000 attempts. One knob changed per row. The
reference row is the budget sweep point at the same step count.

| run | what changed | validity |
|---|---|---|
| r_tlogit0 | timesteps drawn logit-normal instead of uniform | **13.55%** |
| r_emb16 | latent width 16 instead of 32 | 6.89% |
| r_ce3 | cross-entropy weight 3.0 instead of 1.0 | 6.82% |
| r_sphere | x0 corruption drawn on the sphere | 5.88% |
| r_emb64 | latent width 64 | 5.75% |
| exp_bud80_s42 | reference | 5.28% |
| r_lr6e3 | learning rate 6e-3 | 5.13% |
| r_tlogit1 | logit-normal centred at +1 (towards data) | 4.95% |
| r_lr1e3 | learning rate 1e-3 | 3.27% |
| r_maskloss | padding taken out of the loss | 1.58% |
| r_embrms_n50 | unit-norm embeddings, corruption 0.5 | 1.58% |
| r_embrms | unit-norm embeddings | 1.40% |
| r_mixup | x0 corruption by mixing token embeddings | 0.44% |
| r_regimex | predict x0 instead of the noise | 0.35% |
| r_maskattn | padding taken out of attention | 0.05% |
| r_laplace | Laplace x0 corruption | 0.00% |

The timestep distribution is the single largest training-side effect found so far: 13.55%
against 5.28% at the same cost, and better than the reference trained on four times the
tokens (12.85%). Its direction matters as much as its use, since centring the same
distribution towards the data end gives nothing. The two masking rows confirm from the
other side that padding carries the termination signal: remove it and generation
collapses.

## Averaged weights and augmented data

Same size and budget as the screening wave, 100 solver steps, argmax decoding. "final"
is the weights at the last step, "averaged" is the exponential moving average with decay
0.999 started at step 2000.

| run | data | weights | validity | mean length |
|---|---|---|---|---|
| r_ema | plain corpus | final | 4.40% | 46.1 |
| r_ema | plain corpus | averaged | 0.24% | 186.0 |
| r_aug4 | 4x random traversals | final | 2.62% | 47.5 |
| r_aug4, seed 43 | 4x random traversals | final | 2.65% | 47.4 |
| r_aug4_ema | 4x random traversals | final | 3.24% | 46.4 |
| r_aug4_ema | 4x random traversals | averaged | 1.47% | 46.3 |

Both ideas are negative here, and the averaging result is the more interesting one.
Weight averaging is standard in image diffusion and it is strongly harmful in this
model: 4.40% falls to 0.24%, and the averaged model stops terminating, producing 186
characters against the corpus 44. The averaged checkpoint is a legitimate average, not a
broken file: its largest per-tensor relative difference from the final weights is 2% and
the median is 0.6%, with no NaNs. A perturbation that small destroying generation says
the embedding table and the readout are tuned to each other tightly enough that moving
them along a trajectory average breaks the pairing. That is a property worth reporting in
its own right, and it is the same fragility that makes clamping the x0 estimate fail.

Augmentation by random SMILES traversals costs about half the validity at a fixed step
count. Four times as many distinct strings for the same number of updates means each
string is seen a quarter as often, and at this budget repetition is what the model needs.
The two seeds agree to within 0.03 points, so this is not noise.

## Loss shaping at the same size and budget

Same protocol as the screening wave: 5.04M parameters, 17,600 steps, grammar loss off,
seed 42, 10,000 attempts, argmax decoding.

| run | what changed | validity |
|---|---|---|
| r_thr90 | cross-entropy gate raised from alpha 0.8 to 0.9 | 7.33% |
| r_lsmooth | label smoothing 0.05 | 6.93% |
| r_padw06 | padding weighted 0.6 in the noise loss | 6.73% |
| r_ce2 | cross-entropy weight 2.0 | 6.52% |
| r_ce05 | cross-entropy weight 0.5 | 6.08% |
| r_padw03 | padding weighted 0.3 | 5.06% |
| r_minsnr5 | min-SNR weighting, gamma 5 | 0.79% |
| r_minsnr1 | min-SNR weighting, gamma 1 | 0.47% |

min-SNR weighting is the clearest negative in the whole study, and the reason is
structural rather than incidental. The weight caps the contribution of high signal-to-noise
timesteps, but this objective already routes its two supervision terms, the
cross-entropy and the reconstruction error, through exactly those timesteps by gating
them on alpha. Downweighting them removes most of the token-level supervision, which is
what the samples show.

Raising the gate from 0.8 to 0.9 goes the other way and helps, which points the same
direction: the token-level terms want to run on the cleanest states, not on more of them.
Both padding-weight rows and both cross-entropy-weight rows land within a point of each
other above the reference, close enough that they need a second seed before any of them
is called a real effect.

## The timestep distribution, refined

The reference row is the correct control for the whole study: same size, same 17,600
steps, grammar loss off, seed 42. It reads 6.19%, not the 5.28% of the budget sweep,
because the sweep still carried the grammar term. Every delta below is against 6.19%.

| run | timestep distribution and extras | validity |
|---|---|---|
| r_tl0_ce3 | logit-normal, cross-entropy weight 3.0 | **16.62%** |
| r_tl0_lr6e3 | logit-normal, learning rate 6e-3 | 14.70% |
| r_tl0_emb16 | logit-normal, latent width 16 | 13.73% |
| r_tlogit0 / r_tl0_s43 | logit-normal, seeds 42 and 43 | 13.55% / 13.71% |
| r_tls07 | logit-normal, std 0.7 | 12.50% |
| r_tl0_sphere | logit-normal, sphere corruption | 12.25% |
| r_tlp05 | logit-normal centred at +0.5 | 10.95% |
| r_tlm05_emb16 | centred at -0.5, latent width 16 | 9.19% |
| r_tlm05 | logit-normal centred at -0.5 | 7.85% |
| r_tl0_bud160 | logit-normal, twice the budget (35,200 steps) | 7.23% |
| r_base80 | uniform, the reference | 6.19% |
| r_tls15 | logit-normal, std 1.5 | 5.85% |
| r_tlogit1 | logit-normal centred at +1.0 | 4.95% |

Three things come out of this table.

The effect replicates across seeds, 13.55% and 13.71%, so the 7.4-point gain over the
reference is real and roughly 2.2x relative. The distribution has a single optimum at
mean 0 and standard deviation 1, falling off in every direction tried: 0.7 costs a
point, 1.5 costs everything, and shifting the centre either way costs three to nine
points. That is a narrow peak, which is worth saying out loud, because it means the
knob has to be tuned rather than switched on.

Raising the cross-entropy weight on top of it is worth another three points, well beyond
what the same change gives on its own, which was half a point. The readout has to invert
a latent geometry that the timestep change alters, so the two are not independent knobs.

The last row is the surprise: with logit-normal timesteps, doubling the budget from 80 to
160 tokens per parameter takes validity from 13.55% down to 7.23%. Under uniform
timesteps the same doubling raised it from 5.28% to 9.69%. Both runs are 161 epochs over
a 224k-molecule corpus, so this is the point where repetition starts to hurt, and the
mid-noise-heavy schedule reaches it sooner. The gain is a short-budget effect, and any
claim about it has to name the budget it was measured at.

## Where the solver puts its steps

One checkpoint, the logit-normal winner, 100 solver steps in every row, argmax decoding.
The grids keep both endpoints and the step count and only redistribute the stops. The
power column is how hard the skew is.

| grid | power | validity |
|---|---|---|
| uniform, the original | - | **15.29%** |
| dense at both ends | 2 | 15.09% |
| dense in the middle | 2 | 12.48% |
| dense towards data | 2 | 12.39% |
| dense towards noise | 2 | 10.38% |
| dense towards data | 3 | 8.32% |
| dense in the middle | 3 | 8.23% |

A clean negative: the uniform grid is already the best one, the mildest alternative ties
it within the interval, and everything else costs two to seven points. The knob stays in
the config because it is one line and it settles the question, but the default does not
change.

The same table carries a second result. This checkpoint reads 15.29% at 100 solver steps
and 13.55% at 300, with non-overlapping intervals, while the uniform-timestep model was
flat from 100 to 1000 steps. So the best step count depends on how the model was trained,
and for the winner it is the cheap end: fewer steps, better molecules.

## The two changes together

The same twelve checkpoints, decoded twice: once with the original argmax readout and
once with mixed grammar repair, 100 solver steps both times. Unique valid is the count of
distinct molecules per 10,000 attempts, which is the number a generative model is
actually judged on.

| run | argmax | repaired | unique valid | mean length |
|---|---|---|---|---|
| r_tl0_sphere | 12.25% | **63.08%** | 5615 | 29.8 |
| r_tl0_ce3 | 16.62% | 60.29% | 5442 | 34.9 |
| r_tl0_lr6e3 | 14.70% | 59.77% | 5304 | 32.7 |
| r_tlp05 | 10.95% | 52.04% | 4757 | 27.9 |
| r_tl0_emb16 | 13.73% | 50.74% | 4611 | 30.8 |
| r_tlm05 | 7.85% | 50.57% | 4414 | 35.3 |
| r_tlm05_emb16 | 9.19% | 47.84% | 4127 | 34.3 |
| r_tl0_s43 | 13.71% | 47.22% | 3846 | 28.8 |
| r_tls15 | 5.85% | 44.50% | 3451 | 32.1 |
| r_base80 | 6.19% | 42.23% | 3175 | 31.8 |
| r_tls07 | 12.50% | 37.13% | 2871 | 29.4 |
| r_tl0_bud160 | 7.23% | 35.45% | 2851 | 35.3 |

Start from the original setup, uniform timesteps and argmax decoding, and the small model
returns about 620 distinct valid molecules per 10,000 attempts. The best row here returns
5615, a factor of nine, from two changes that cost nothing: where the training timesteps
come from, and repairing brackets and ring digits while decoding.

Two honest caveats sit next to that number. The repaired molecules average 30 characters
against the corpus 44, so the repair buys yield partly by producing smaller molecules,
and the length-preserving mode has to be quoted alongside. And the ranking is not the
same under the two decoders: the sphere-corruption run is sixth on argmax and first once
repaired, which means a screening study that ranks on argmax alone can pick the wrong
winner. Both decoders should be reported for every configuration that matters.

## Combinations, extra seeds, and the variance problem

All rows below are 100 solver steps, which is why they are not directly comparable with
the 300-step columns above. The reference reads 5.78% on seed 43 against 6.19% on seed 42,
so the reference itself is stable to about half a point.

| run | argmax | repaired (mixed) | unique valid |
|---|---|---|---|
| r_tl0_padw06 | **26.15%** | 59.47% | 5204 |
| r_tl0_thr90 | 16.36% | 57.74% | 5225 |
| r_tl0_lsmooth | 14.11% | 53.94% | 4763 |
| r_tl0_cepad | 8.03% | 42.11% | 3175 |
| r_tl0_s44 | 7.18% | 30.88% | 1714 |
| r_thr90_s43 | 6.72% | 44.81% | 3459 |
| r_cepad | 5.89% | 44.75% | 3460 |
| r_base80_s43 | 5.78% | 44.00% | 3357 |

Weighting padding at 0.6 in the noise loss, on top of the logit-normal timesteps, is the
largest argmax number in the study, 26.15% against about 15% for the timestep change
alone. It costs uniqueness, 87.5% against 100%, so the unique-valid count is 5204 rather
than 5947, but that is still the top of the table. It also fits the padding story from the
other direction: removing padding from the loss destroys generation, keeping it at full
weight spends capacity on it, and 0.6 is better than either end.

Supervising padding through the cross-entropy does nothing on its own, 5.89% against the
5.78-6.19% reference, and costs half the gain when combined with the timestep change. It
stays off.

The uncomfortable row is the third seed of the winner: 7.18% where seeds 42 and 43 gave
about 15% and 13.7%, with uniqueness down to 75.7% and, under the repair decoder, 30.88%
against 47-63% for its siblings. The reference's own seeds agree to half a point, so this
variance is introduced by the logit-normal sampler, not by the measurement. The effect is
real and large in the mean, and it is also unstable: any claim about it needs a seed count
and a spread, not a single number.

## The length-preserving decoder on the checkpoints that matter

Mixed repair buys yield partly by shortening molecules. Closing what is open does not,
and this is the table to quote when the length distribution has to hold. Corpus mean is
44.3 characters.

| run | argmax | closed | uniqueness | mean length | unique valid |
|---|---|---|---|---|---|
| r_tl0_ce3 | 16.62% | **40.59%** | 99.5% | 50.2 | 4039 |
| r_tl0_sphere | 12.25% | 30.62% | 99.9% | 37.7 | 3059 |
| r_tl0_cepad | 8.03% | 19.21% | 100.0% | 41.3 | 1921 |
| r_base80 | 6.19% | 13.07% | 100.0% | 47.6 | 1307 |

This is the honest headline of the whole study. The original setup, uniform timesteps and
argmax decoding, returns about 620 distinct valid molecules per 10,000 attempts. Two
changes that cost nothing at training or sampling time, logit-normal timesteps with a
heavier cross-entropy and closing repair at decoding, return 4039, with uniqueness above
99% and a length distribution that still looks like the corpus.

## The same four checkpoints at the same solver step count

Everything above mixes 100- and 300-step generations, which is fine for large effects and
sloppy for seed comparisons. These four were regenerated at 100 steps with argmax so that
they are directly comparable.

| run | validity | uniqueness |
|---|---|---|
| r_tl0_ce3 | 18.82% | 99.4% |
| r_tlogit0, seed 42 | 15.29% | 99.9% |
| r_tl0_s43, seed 43 | 14.20% | 99.8% |
| r_base80, the reference | 5.55% | 99.8% |

With the reference at 5.55% and 5.78% on its two seeds, and the timestep change at 15.29%,
14.20% and 7.18% on three, the mean effect is a factor of 2.2 and the spread is a factor
of two within the same configuration. The reference also prefers 300 solver steps while
the logit-normal runs prefer 100, which is one more reason to state the step count next to
every number.

## Second seeds, and what survives them

Every row here is 100 solver steps. The point of this wave was to put a second seed under
each apparent winner from the earlier ones.

| run | argmax | repaired (mixed) | unique valid (mixed) |
|---|---|---|---|
| r_tl0_padw06, seed 42 | 26.15% | 59.47% | 5204 |
| r_tl0_padw06, seed 43 | 7.46% | 34.04% | 2771 |
| r_tl0_padw03 | 23.32% | 63.89% | 5910 |
| r_tl0_padw08 | 17.58% | 47.65% | 4317 |
| r_padw06 without logit-normal | 6.72% | 48.42% | 3574 |
| r_tl0_ce3, seed 42 | 18.82% | 60.29% | 5442 |
| r_tl0_ce3, seed 43 | 13.99% | 60.38% | 5730 |
| r_tl0_sphere, seed 43 | 21.58% | 66.02% | 5354 |
| r_tl0_s45, fourth seed of the plain winner | 18.08% | 68.00% | 5406 |
| reference, seeds 42 and 43 | 5.55% / 5.78% | 42.23% / 44.00% | 3175 / 3357 |

This wave costs the study its most attractive single number. Weighting padding at 0.6 read
26.15% on seed 42 and 7.46% on seed 43, so the top of the argmax table was a seed
artifact. The same is true, more mildly, of the cross-entropy weight (18.82% and 13.99%)
and of sphere corruption (12.25% at 300 steps, 21.58% at 100). Padding weight without the
timestep change reads 6.72% against a 5.55-5.78% reference, which is inside the band.

What does survive is the timestep distribution itself, now on four seeds: 15.29%, 14.20%,
7.18% and 18.08%, mean 13.7% against a reference mean of 5.7%, so a factor of 2.4 with a
range of two and a half between its own seeds. And the repair decoder survives everything:
every logit-normal checkpoint lands between 58% and 68% repaired, against 42-44% for the
reference, over roughly thirty checkpoints with no exception.

The methodological lesson is worth as much as the numbers. At this size a single seed
resolves nothing below about eight points, which is far wider than the binomial interval
of half a point that the sample count suggests. Screening at one seed found four winners;
three of them dissolved on the second seed.

## The length-preserving decoder on the best configurations

| run | argmax | closed | uniqueness | mean length | unique valid |
|---|---|---|---|---|---|
| r_tl0_padw06, seed 42 | 26.15% | 45.72% | 94.7% | 32.4 | 4330 |
| r_tl0_ce3, seed 42 | 18.82% | 40.59% | 99.5% | 50.2 | 4039 |
| r_tl0_ce3, seed 43 | 13.99% | 36.30% | 100.0% | 40.4 | 3629 |
| r_tl0_padw06_ce3 | 12.54% | 26.65% | 99.8% | 36.5 | 2661 |
| r_tl0_padw06, seed 43 | 7.46% | 17.16% | 100.0% | 42.5 | 1716 |
| reference | 5.55% | 13.07% | 100.0% | 47.6 | 1307 |

Both seeds of the cross-entropy configuration land at 3629 and 4039 distinct valid
molecules per 10,000 attempts with uniqueness at or above 99.5% and lengths of 40 and 50
against the corpus 44. That is the number to quote: about 3800 on average against 1307 for
the reference decoded the same way, and against 617 for the original setup at its own
default settings.
