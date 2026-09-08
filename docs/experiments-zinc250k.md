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

## Reading the same runs with a seed-aware tool

`scripts/compare_configs.py` groups sampled runs by configuration, averages over seeds and
calls an effect only when a group's worst seed beats the reference's best. Applied to every
generation done with the repair decoder at 100 solver steps, against the uniform-timestep
reference:

| configuration | seeds | mean | range | verdict |
|---|---|---|---|---|
| logit-normal + sphere corruption | 2 | 64.55% | 63.08-66.02 | clears the reference |
| logit-normal + cross-entropy weight 3 | 2 | 60.33% | 60.29-60.38 | clears the reference |
| logit-normal alone | 3 | 48.70% | 30.88-68.00 | inside the spread |
| logit-normal + padding weight 0.6 | 2 | 46.75% | 34.04-59.47 | inside the spread |
| reference | 2 | 43.11% | 42.23-44.00 | - |

This reverses the earlier reading in a useful way. Plain logit-normal timesteps have a high
mean and a range so wide that the conservative test cannot separate them from the
reference. Adding a heavier cross-entropy on top does not just raise the mean, it collapses
the spread: 60.29% and 60.38% on two seeds is tighter than the reference's own two seeds.
Sphere corruption behaves the same way, 63.08% and 66.02%.

So the configuration to carry forward looked like the timestep change with the readout
weighted up, preferred for stability as much as for the mean.

**Corrected by the next wave, see the final aggregate below.** Two more seeds of that
configuration read 10.58% and 9.50% on argmax, which widens its range to 9.50-18.82,
exactly as wide as the plain configuration's. The apparent stability was two draws that
happened to land together. Nothing added to the timestep change separates from it.

## The final aggregate, three seeds where it matters

Every run regenerated or generated at 100 solver steps with argmax, grouped by
configuration with `scripts/compare_configs.py`, reference is the uniform-timestep run.

| configuration | seeds | mean | range | verdict |
|---|---|---|---|---|
| logit-normal + padding weight 0.6 | 2 | 16.81% | 7.46-26.15 | clears the reference |
| logit-normal + learning rate 1.5e-4 | 3 | 13.90% | 8.61-17.99 | clears the reference |
| logit-normal + cross-entropy weight 3 | 4 | 13.22% | 9.50-18.82 | clears the reference |
| logit-normal alone | 4 | 13.69% | 7.18-18.08 | clears the reference |
| logit-normal + warmup 1600 steps | 3 | 8.44% | 4.27-12.11 | inside the spread |
| reference, uniform timesteps | 2 | 5.67% | 5.55-5.78 | - |

This is the table the study actually supports, and it is shorter than every intermediate
version of it. One training-side change clears the reference: the logit-normal timestep
distribution, on every one of its four seeds, mean 13.7% against 5.67%, a factor of 2.4.
Nothing added on top of it separates from it. The cross-entropy weight looked like a
stabiliser on two seeds, 60.29% and 60.38% repaired, and on four seeds its argmax range is
9.50-18.82%, as wide as the plain configuration's. Padding weight, sphere corruption, the
raised gate and label smoothing are all single or double draws from that same wide
distribution.

Two attempts to reduce the variance failed. A four-times longer warmup lowers the mean to
8.44% and keeps the spread. A halved peak learning rate leaves both the mean and the
spread where they were. So the instability is not obviously an optimisation artefact, and
finding its source is the open question this study ends on.

## Was the validity bought by generating simpler molecules? Yes, partly

This section exists because validity on its own is gameable in three ways, and this study
hit all three. `dimol/eval/distribution.py` measures the rest: thirteen descriptors
against the corpus in units of its own standard deviation, Bemis-Murcko scaffold counts,
the share of molecules a chemist would throw out, bracket atoms the corpus never uses, and
the Frechet ChemNet Distance. The headline number becomes **usable**: distinct, valid, at
least ten heavy atoms, at least one ring, no invented atom, counted per attempt.

Reference for all of it: the ZINC-250k validation split, 23.2 heavy atoms, 2.7 rings, 330
daltons on average, and no molecule without a ring.

| set | usable | FCD | trivial | scaffolds | heavy atoms | rings | weight |
|---|---|---|---|---|---|---|---|
| reference + closing repair | 13.20% | **14.41** | **2.6%** | 224 | 20.5 (-0.6s) | **2.48** | 295 |
| logit-normal + CE 3 + closing | **21.53%** | 18.62 | 36.1% | 216 | 14.2 (-1.9s) | 0.84 | 204 |
| logit-normal + CE 3 + mixed | 20.13% | 20.22 | 59.4% | 138 | 11.7 (-2.5s) | 0.50 | 170 |
| logit-normal + padding 0.6 + closing | 20.00% | 23.41 | 54.7% | 136 | 12.6 (-2.3s) | 0.54 | 180 |
| reference + argmax | 4.60% | - | 5.4% | 145 | 19.1 (-0.9s) | 1.90 | 277 |

Three things follow, and two of them are corrections to earlier claims in this document.

The repair decoder survives the check. On the reference model, closing repair raises usable
molecules from 4.60% to 13.20% of attempts while leaving the molecules alone: 2.6% trivial,
20.5 heavy atoms against the corpus 23.2, 2.48 rings against 2.7. That is a real 2.9x, not
the 7x that raw validity suggested, and it is honest. Mixed repair is not: it trims to
fragments, 59.4% trivial, and both its FCD and its scaffold count get worse.

The timestep result is real but half of it is simplification. Usable molecules go from
13.20% to 21.53%, a 63% relative gain, so something genuine is there. But the same model
generates 14.2 heavy atoms instead of 20.5, 0.84 rings instead of 2.48, and 36% of its
output is trivial, and its FCD is 29% worse than the reference. Validity went up 2.7x and
usable molecules only 1.6x, and the difference is exactly the simplification the check was
built to find.

The padding-weight configuration was gaming length outright. It had the highest raw
validity of anything measured, 47.79%, and it is the worst row here: 54.7% trivial, the
worst FCD, the fewest scaffolds. Down-weighting padding in the loss reduces the pressure
to fill the canvas, the model terminates early, short strings are easier to make valid,
and validity rises while the molecules get worse. It is dropped.

Absolute FCD is 14 to 23 across the board, where a good model on this kind of corpus is
under 1. At 5M parameters and 80 tokens per parameter that is expected, and it is the
number to watch when the model is scaled, because it is the one that says whether the
distribution is being learned at all.

## A protocol error worth recording

Waves 18 and 19, twenty-one runs covering self-conditioning, a tied readout and trained-in
length conditioning, were generated with `generate.length_prior` set, and the sampler at
that moment also switched on a decode floor: a stop token before the drawn length was
refused and replaced by the best content token. Wave 17 had already measured that floor at
0% usable with 60-atom strings, and it was still on, because the floor had no flag of its
own and rode along with the prior.

The signature is unmistakable in the output, a tail of one repeated rare token:

    [P@]CC[S@](=O)c1cc(=O)c(C#N)n(-n2c(=O)c2)c1C(=)cc1[o+][NH-][NH-][NH-][NH-]...

So none of those twenty-one numbers say anything about the three ideas they were meant to
test, and all of them were regenerated. The floor is now `generate.length_floor`, default
false, with the measurement that condemns it written next to it in the config.

The general lesson is about coupling, not about length: a knob that silently turns on a
second behaviour will eventually be used after that second behaviour has been ruled out.
Anything measured as harmful gets its own flag and its own default.

## Self-conditioning and length conditioning, three seeds each

All rows: 5.04M parameters, 17,600 steps, closing repair, 100 solver steps, judged on
usable molecules per attempt and on the Frechet ChemNet Distance at a matched 1500
molecules. The reference is the uniform-timestep model, 13.20% usable and FCD 14.41.

| configuration | seeds | usable, mean | range | heavy atoms | rings | trivial |
|---|---|---|---|---|---|---|
| self-conditioning | 3 | **17.38%** | 14.00-19.93 | 20.3 | 2.38 | 3.0% |
| length + self-conditioning | 3 | 13.20% | 9.13-17.67 | 21.2 | 2.47 | 1.7% |
| reference, uniform timesteps | 2 | 12.90% | 12.65-13.20 | 20.6 | 2.48 | 2.5% |
| tied readout | 3 | 12.42% | 11.47-13.53 | 20.5 | 2.41 | 3.0% |
| length conditioning | 3 | 9.16% | 5.07-12.13 | 22.0 | 2.63 | 1.5% |

And the aggregate metric on one representative of each, same sample size:

| run | usable | FCD | scaffolds | heavy atoms | rings |
|---|---|---|---|---|---|
| self-conditioning | 18.20% | 12.47 | 293 | 20.9 | 2.53 |
| length + self-conditioning | 17.67% | **11.61** | 280 | 21.3 | 2.47 |
| reference | 13.20% | 14.41 | 224 | 20.5 | 2.48 |
| length conditioning | 10.27% | 15.88 | 176 | 22.5 | 2.75 |
| corpus | - | 0 | - | 23.3 | 2.79 |

**Self-conditioning is the first change that improves both axes at once.** Usable
molecules rise 38%, FCD falls 13%, distinct scaffolds rise 31%, and the molecules
themselves are unchanged: 2.2% trivial against the reference's 2.6%, 20.9 heavy atoms
against 20.5. Its worst seed, 14.00%, beats the reference's best, 13.20%, so it clears the
conservative rule. Every earlier "win" in this document failed that test or bought its
validity by shrinking the molecules; this one does neither. It is also the result that
transfers: nothing about it is specific to SMILES.

**Length conditioning does exactly what it was built to do and still loses.** Trained-in
length conditioning puts heavy atoms at 22.5 against the corpus 23.3, a tenth of a
standard deviation, and rings at 2.75 against 2.79 - by far the closest length and ring
statistics measured anywhere here. And usable molecules fall to 10.27% and FCD rises to
15.88. So the degenerate short-sequence optimum was doing real work: with it removed, the
model has to produce a full-length molecule and cannot yet get the content right. That is
a more useful failure than the collapse it replaced, because it names the real limit.

**Together they are the best configuration measured**: FCD 11.61, the lowest of any run in
this study, at 17.67% usable, with 1.0% trivial molecules. Length conditioning supplies the
distribution and self-conditioning supplies the accuracy.

**The tied readout does nothing** - 12.42% against 12.90%, ranges overlapping - and it does
not narrow the seed spread either, 2.1 points against the reference's 0.6. That rules out
one of the two suspects for the run-to-run variance: the readout chasing a moving embedding
table is not the cause.

The remaining gap is now specific. On the length-conditioned model, where lengths and ring
counts match the corpus, aromatic rings are 0.82 against 1.89, a shift of 1.09 standard
deviations with a KS statistic of 0.46, and the synthetic accessibility score is 1.9
standard deviations worse. The model builds rings of the right number and the wrong kind.
That, not validity, is the next target.

## Aromaticity is a capacity problem (this section's first reading was wrong)

**Read the correction below the tables.** The depth claim in this section came from a
comparison in which the deeper model also carried 22% more parameters, and it does not
survive a parameter-matched series.

## Aromaticity: the first reading, kept for the record

The gap left after length and self-conditioning was specific: the model produced the right
number of rings and the wrong kind, 0.8 aromatic rings against the corpus 1.85. Across the
eight training configurations measured before this, that number does not move at all.

| configuration | usable | aromatic rings |
|---|---|---|
| length conditioning | 9.16% | 0.79 |
| length + self-conditioning | 13.20% | 0.77 |
| reference | 13.20% | 0.75 |
| tied readout | 12.42% | 0.73 |
| self-conditioning | 17.38% | 0.72 |
| self-conditioning + logit-normal | 30.02% | 0.69 |
| tied readout + logit-normal | 20.49% | 0.60 |
| corpus | - | **1.85** |

Usable molecules range over a factor of three in that table and aromatic rings sit between
0.60 and 0.79 throughout, with the highest-yield configuration the worst of them. No
objective, no timestep density, no gating and no decoding rule touches it.

Depth does. Five shapes, all on the same stack (self-conditioning, trained-in length
conditioning, uniform timesteps), three seeds each, head size held at 64:

| shape | parameters | blocks | usable, mean | range | aromatic rings | FCD | SA |
|---|---|---|---|---|---|---|---|
| 384 x 2 | 5.17M | 2 | 10.22% | 7.60-14.33 | 0.75 | 12.68 | 4.81 |
| 256 x 4 | 5.06M | 4 | 14.78% | 12.93-15.73 | 0.85 | 11.09 | 4.29 |
| 256 x 4, latent 64 | 5.12M | 4 | 16.29% | 13.53-18.13 | 0.86 | - | 4.31 |
| 192 x 8 | 6.19M | 8 | **19.93%** | 17.33-23.27 | **1.05** | **9.05** | 4.09 |
| 384 x 6 | 14.62M | 6 | **24.02%** | 21.20-29.27 | **1.10** | 9.03 | 3.99 |
| corpus | - | - | - | - | 1.85 | 0 | 3.07 |

The first three rows are matched within 2% on parameters, so 2 against 4 blocks is a clean
depth comparison at fixed size: 10.22% against 14.78% usable, 0.75 against 0.85 aromatic
rings. Eight blocks carries 22% more parameters, because the adaLN modulation scales with
depth, and the capacity probe carries 2.9x, so those two rows mix depth with size. Even so
the ordering is monotone in both, and it is monotone in exactly the quantity that no
training change could move: 0.75, 0.85, 0.86, 1.05, 1.10. Synthetic accessibility follows
it down, 4.81 to 3.99 against the corpus 3.07, and FCD follows it down too, 12.68 to 9.03,
the best in this study.

Widening the latent from 32 to 64 does almost nothing, 0.85 to 0.86, which says the
bottleneck is not how much room a token's representation has. It is how many rounds of
mixing the positions get before they are decoded, which is what an aromatic ring needs:
five or six atoms and a matched ring digit that all have to agree, decoded from a latent
by a readout that sees each position on its own.

That is the argument for scale, and it is a specific one rather than a hope. Every
objective-side change in this study either moves yield at the cost of fidelity or improves
both by a modest amount; the one structural property the corpus has and the samples lack
responds only to depth and capacity.

## Iterating the same model does not create what it lacks

If the positions only need to see each other's decisions, refinement at sampling time
should be enough: re-noise the finished sample part of the way back and denoise it again,
so each pass is conditioned on the last one's output. It costs no training. Eight settings
on the best checkpoint, 6000 attempts each:

| rounds | re-noise to t | usable | aromatic rings | rings |
|---|---|---|---|---|
| 0 | - | 15.80% | 0.91 | 2.49 |
| 1 | 0.9 | 15.80% | 0.91 | 2.49 |
| 2 | 0.9 | 15.73% | 0.91 | 2.49 |
| 4 | 0.9 | 15.73% | 0.91 | 2.49 |
| 1 | 0.7 | 15.80% | 0.91 | 2.49 |
| 2 | 0.7 | 15.80% | 0.91 | 2.49 |
| 4 | 0.7 | 15.87% | 0.91 | 2.49 |
| 2 | 0.5 | 15.53% | 0.88 | 2.41 |

Nothing moves, and not because the code does nothing: at t = 0.5, where the re-noising is
substantial, the numbers do shift, slightly downward. The finished sample sits at a fixed
point of the reverse process, so putting it back through the same score field returns it
where it was. Long-range consistency is not information the sampler is discarding, which
was the hypothesis; it is information the model does not have. That is the negative that
pairs with the depth result and rules out the cheap way of getting it.

## The variance survives every explanation offered for it

Four hypotheses have now been tested against the run-to-run spread, three seeds each.

| hypothesis | test | usable, mean | spread | verdict |
|---|---|---|---|---|
| the schedule needs longer warmup | warmup x4 | 8.44% | 7.8 pts | mean lower, spread unchanged |
| the peak learning rate is too high | lr halved | 13.90% | 9.4 pts | nothing |
| the readout chases a moving embedding table | readout tied to the table | 12.42% | 2.1 pts | nothing |
| the gated batch fraction fluctuates | fixed top-k gate | 10.69% | 1.5 pts | mean lower, spread unchanged |
| reference, for comparison | - | 12.90% | 0.6 pts | - |

The deterministic gate is the most informative of the failures. Under uniform timesteps it
lowers the mean, 10.69% against 12.90%, and leaves the spread where it was; under
logit-normal timesteps its three seeds read 6.53%, 16.40% and 17.67%, an eleven-point
spread with the lowest seed showing the usual collapse signature, 33.6% trivial molecules
at 13.6 heavy atoms. So the amount of token-level supervision per step is not what makes
runs differ.

What is left is uncomfortable and worth stating plainly: two runs with the same seed and
the same configuration landed at 23.9% and 55.5% validity, so the spread does not need a
seed to appear. Numerical nondeterminism in the backward pass is enough to send a run to a
different place on the length-fidelity trade-off, and nothing tried so far narrows it. The
practical consequence stands: three seeds per configuration, and an effect is only an
effect when the worst seed of a group beats the best seed of the reference.

## The parameter-matched depth series, which corrects the section above

To match parameters at greater depth the modulation width has to shrink, because adaLN
costs `time_dim x 6 x model_dim` per block. That gives a series within 4% on parameters
and identical FLOPs per token. Per-seed numbers, because the group means hid what matters:

| shape | parameters | usable by seed | aromatic rings by seed |
|---|---|---|---|
| 384 x 2 | 5.17M | 8.7 7.6 14.3 | 0.65 0.73 0.86 |
| 256 x 4 | 5.06M | 12.9 15.7 15.7 | 0.77 0.88 0.90 |
| 256 x 4, latent 64 | 5.12M | 18.1 13.5 17.2 | 0.94 0.76 0.86 |
| 192 x 8 | 4.88M | 16.1 17.3 10.1 | 0.97 0.87 0.55 |
| 128 x 12 | 4.96M | 13.5 17.2 13.5 | 0.78 0.84 0.75 |
| 128 x 16 | 4.86M | 12.8 13.6 14.7 | 0.73 0.69 0.78 |
| 192 x 8 | 6.19M | 19.2 17.3 23.3 | **1.08 1.02 1.05** |
| 384 x 6 | 14.62M | 21.6 29.3 21.2 | **1.02 1.22 1.05** |
| corpus | - | - | 1.85 |

**Shape does not matter and parameter count does.** The six configurations between 4.86M
and 5.17M cover 2, 4, 8, 12 and 16 blocks and latent widths 32 and 64, and their aromatic
ring counts all fall in 0.55 to 0.97 with every range overlapping every other. Usable
molecules are equally flat, 13.7% to 16.3% by group mean, with only the two-block model
clearly worse. The two models above 6M separate cleanly: their worst seed, 1.02, beats the
best seed of every 5M configuration, 0.97, which is the only kind of separation this study
accepts.

So the earlier claim, that aromaticity is a depth phenomenon, was an artefact of the
comparison. The 8-block winner in that table carried 22% more parameters than the
baseline, precisely because I had left the modulation width alone; once the modulation is
narrowed to match, 8 blocks reads 0.80 against the baseline's 0.85. Depth beyond four
blocks buys nothing here, and neither does a wider latent.

One thing sharpens the capacity result rather than weakening it: every run in this table
took the same 17,600 steps, so the token budget per parameter is not matched. The 5M
models saw 80 tokens per parameter, the 6.19M model 66, and the 14.62M model **28**. The
largest model is the most undertrained of them and still has the best aromaticity, so 1.02
to 1.22 is a lower bound on what its capacity can do.

And that is where the corpus becomes the binding constraint. Training the 14.62M model to
80 tokens per parameter needs 1.17 billion tokens, which is 231 passes over ZINC-250k, and
this study already measured that 161 passes hurt. The capacity that fixes aromaticity
cannot be fed by this corpus.

## ZINC-20: what a real token budget does

The corpus was the binding constraint, and removing it changed the answer to every
question this study had left open. ZINC-20 batch one: 214,055,665 curated and tokenized
training molecules, 27.22 real tokens each, 5.83B tokens. Three sizes at a matched 80
tokens per parameter, one pass each, three seeds, four-way DDP at 1024 molecules per GPU,
self-conditioning and length conditioning on, closing repair with the length drawn from
the corpus.

| model | steps | validity by seed | usable | aromatic rings | rings | heavy atoms | QED | SA |
|---|---|---|---|---|---|---|---|---|
| 5.06M | 3,631 | 1.26 / 1.59 / 3.28% | 2.1% | 1.42 | 4.52 | 33.7 | 0.51 | 5.94 |
| 14.6M | 10,491 | 14.52 / 22.84 / 23.01% | 20.6% | 0.89 | 2.79 | 25.3 | 0.63 | 4.58 |
| 47.9M | 34,375 | **61.07 / 61.49 / 63.64%** | **60.0%** | **1.49** | 2.84 | 26.1 | 0.65 | 3.72 |
| ZINC-20 | - | - | - | 1.85 | 3.10 | 26.8 | 0.64 | 3.47 |

Four things, and three of them are new.

**Capacity is the answer to aromaticity.** The gap that no objective, no timestep density,
no decoding rule and no rearrangement of depth could move is now nearly closed: 1.49
aromatic rings against the corpus 1.85, a shift of -0.36 standard deviations, where the
best ZINC-250k model managed 1.10 against the same 1.85. Every other descriptor of the
47.9M runs sits within 0.4 sigma of the corpus, and QED matches it outright, 0.65 against
0.64. The samples are 100% unique and 100% novel.

**The variance problem was data starvation.** On ZINC-250k, three seeds of one
configuration spread over eight to nineteen points and two runs with the *same* seed
landed at 23.9% and 55.5%. Here the three 47.9M seeds read 61.07, 61.49 and 63.64%, a
spread of two and a half points. Four hypotheses about the optimizer and the loss gates
were tested and all failed; the actual cause was 231 passes over a corpus too small to
support the model.

**Large batches cost small models.** The 5.06M runs collapsed, 1.26 to 3.28% validity
against the 13 to 16% the same size reached on ZINC-250k. The token budget is identical;
what changed is that batch 4096 gives it 3,631 optimizer steps where batch 1024 gave
17,600. At a fixed token budget the update count is what a small model needs, and the
throughput gain that made the big runs cheap is exactly what starved the small ones. The
batch has to be part of the size ladder, not a constant.

**And 60% usable molecules per attempt, without any conditioning**, is the number to carry
into the comparison with published work: 61 to 64% validity, 100% uniqueness, 100%
novelty, at 47.9M parameters. The reference this study started from produced 5.5%.

## Text guidance on ChEBI-20: the first end-to-end number

The 47.9M ZINC-20 model, vocabulary grown 448 to 640 and canvas 64 to 128, with
zero-initialised cross-attention into frozen SciBERT states, fine-tuned for 60 epochs on
25,574 caption-molecule pairs with 10% caption dropout. Nine minutes of training. 500
validation captions, one molecule generated per caption, strict decoding with closing
repair.

| setting | validity | exact match | MACCS | RDK | Morgan | token F1 |
|---|---|---|---|---|---|---|
| guidance 0 | 63.4% | 0.00% | 0.331 | 0.169 | 0.122 | 0.377 |
| guidance 1 | 56.8% | 0.00% | 0.346 | 0.170 | 0.138 | 0.375 |
| guidance 3 | 63.2% | 0.00% | **0.368** | 0.190 | 0.150 | 0.373 |
| guidance 3, captions shuffled | 59.2% | 0.00% | 0.246 | 0.118 | 0.079 | 0.287 |
| MolT5-large, published | ~96% | ~31% | ~0.83 | ~0.75 | ~0.68 | - |

Two things are established and one is not.

The mechanism works. Guidance moves the fingerprint similarities monotonically, 0.331 to
0.346 to 0.368 on MACCS, which is what classifier-free guidance is supposed to do. And the
shuffled-caption control separates weak conditioning from none: pairing every molecule
with another one's caption costs a third of the MACCS similarity and nearly half the
Morgan, so the caption is being read. Without that control the numbers would be
uninterpretable, because a model that ignored the text entirely would still score around
0.25 by producing generically ChEBI-like molecules.

What is not established is any claim of competitiveness. MACCS 0.37 against a published
0.83, and exact match zero against 0.31, is a wide gap, and the reasons are not mysterious:
the backbone barely moved in nine minutes at a peak learning rate of 3e-4, the text
encoder is frozen, and only the grafted cross-attention learned anything. The two learning
rates tried were indistinguishable on loss, which says the run was too short to separate
them rather than that they are equivalent.

The cost of a run being nine minutes is the useful fact here. The next step is not a new
idea but a proper budget: longer schedules, higher peak rates, and unfreezing the encoder,
measured against the shuffled-caption floor each time.

## Text guidance given real time, and where the gap actually is

Same setup, 60,000 steps instead of 6,000, three learning rates. 500 validation captions,
one molecule per caption, strict decoding with closing repair.

| run | step | validity | exact | MACCS | RDK | Morgan |
|---|---|---|---|---|---|---|
| lr 3e-4 | 5,000 | 50.2% | 0% | 0.371 | 0.193 | 0.148 |
| lr 3e-4 | 20,000 | 64.2% | 0% | 0.474 | 0.245 | 0.186 |
| lr 3e-4 | 40,000 | 67.2% | 0% | 0.509 | 0.276 | 0.214 |
| lr 3e-4 | 60,000 | **69.2%** | 0% | **0.518** | 0.286 | 0.219 |
| lr 1e-3 | 60,000 | 7.8% | 0% | 0.144 | 0.070 | 0.052 |
| lr 3e-3 | 60,000 | 16.8% | 0% | 0.129 | 0.039 | 0.039 |

**The final training losses of those three runs were 0.8976, 0.8956 and 0.8954.** The
lowest loss belongs to the model that scores 0.129 on the task and the highest to the one
that scores 0.518. This is the sharpest form the loss-blindness result has taken anywhere
in this study: a fourfold quality difference, ranked backwards, at a loss spread of 0.2%.
Anything that selects a checkpoint or a hyperparameter by loss here selects the worst
model on offer.

At 3e-4 the curve is monotone and still rising at 60k, so the run was too short rather
than converged.

### Guidance, and the floor it should be quoted against

On the best checkpoint:

| setting | validity | exact | MACCS | RDK | Morgan | token F1 |
|---|---|---|---|---|---|---|
| guidance 0 | 69.6% | 0% | 0.509 | 0.270 | 0.219 | 0.456 |
| guidance 3 | 69.8% | 0% | 0.509 | 0.279 | 0.220 | 0.449 |
| guidance 6 | 65.6% | 0% | 0.499 | 0.271 | 0.202 | 0.442 |
| guidance 10 | 60.6% | 0% | 0.460 | 0.239 | 0.174 | 0.418 |
| captions shuffled | 69.2% | 0% | **0.270** | 0.143 | 0.083 | 0.310 |
| **reference length given** | 72.6% | **2.0%** | **0.615** | **0.416** | **0.326** | 0.657 |

Guidance has stopped helping. At nine minutes of training it was worth 11% on MACCS;
now scale 0 and scale 3 are identical and anything higher costs quality. That is what
guidance is for: it amplifies a conditioning signal that the model is underusing, and
once the conditioning is trained there is nothing left to amplify. Worth remembering
before reaching for it as a free win.

The signal against its own floor has doubled: 0.509 against a shuffled-caption 0.270,
where the nine-minute model was 0.368 against 0.246.

### The length is a fifth of the remaining gap

The last row is an oracle, not a result: each molecule was generated at its reference's
own token length instead of a length drawn from the corpus. Knowing the length alone is
worth 21% on MACCS, 49% on RDK, 48% on Morgan, and it is the difference between zero and
two percent exact match.

That is not a benchmark number, but it is a design instruction. A caption very often says
how large the molecule is, sometimes literally ("N-nonacosanoyl" is twenty-nine carbons),
and the model already takes a length as an input. A head predicting length from the frozen
caption states would convert most of that oracle gain into a real one, and it is a much
smaller job than anything else on the list.

Published MolT5-large on this benchmark is about 96% validity, 31% exact match and 0.83
MACCS, so the gap is still wide. What has changed is that it is now itemised: the run was
too short, the caption's length information is unused, and the text encoder is frozen.

## Should the length input be dropped in favour of text alone? No: predict it

The suspicion was reasonable. The length is given from ground truth during training, so
the model could learn to lean on it and never extract size from the caption, and then
obey a randomly drawn length at generation time. Crossing the two inputs settles it.

| caption | length | validity | exact | MACCS | RDK | Morgan |
|---|---|---|---|---|---|---|
| right | drawn from corpus | 69.6% | 0% | 0.509 | 0.270 | 0.219 |
| right | true (oracle) | 72.6% | 2.0% | 0.615 | 0.416 | 0.326 |
| shuffled | drawn from corpus | 69.2% | 0% | 0.270 | 0.143 | 0.083 |
| shuffled | true (oracle) | 71.6% | 0.4% | 0.296 | 0.175 | 0.101 |

The caption is worth +0.24 to +0.32 MACCS; the length is worth +0.11 with a correct
caption and +0.03 with a wrong one. A shortcut would look the opposite way round: the true
length with a wrong caption would score well, and it scores 0.296 against a 0.270 floor.
So the two inputs are complementary rather than competing, and the length carries almost
no structure of its own - it constrains the realisation of a structure the caption has
already specified.

### The head

`dimol/models/length_head.py` attention-pools the frozen caption states and classifies the
length over the canvas, with neighbouring lengths given partial credit because being one
token out is nearly right. Twelve epochs, two minutes:

| predictor | MAE on val |
|---|---|
| the corpus mean | 18.91 tokens |
| ridge on mean-pooled states | 9.03 |
| **this head** | **7.77** (43% within three tokens, 12.5% exact) |

And end to end, on the same 500 captions:

| length source | validity | exact | MACCS | RDK | Morgan | token F1 |
|---|---|---|---|---|---|---|
| drawn from the corpus | 69.6% | 0.00% | 0.509 | 0.270 | 0.219 | 0.456 |
| **predicted from the caption** | **74.8%** | **0.80%** | **0.588** | **0.389** | **0.289** | 0.601 |
| true length (oracle) | 72.6% | 2.00% | 0.615 | 0.416 | 0.326 | 0.657 |

The head recovers 75% of the oracle gain on MACCS, 82% on RDK and 65% on Morgan, and it
beats the oracle on validity, 74.8% against 72.6%, presumably because it predicts slightly
short and short molecules are easier to get right. It was still improving when training
stopped at twelve epochs.

So the design is: keep the length input, predict it from the caption, and the answer to
"rely on text alone" is that this *is* relying on text - the length now comes from the
caption too, through a two-minute head rather than through the diffusion model's own
capacity. A control run with the length input removed entirely is queued behind the
current sweep, because the reasoning above deserves a measurement of its own.
