# Where the quality is lost, and what to try

Fixed for every experiment below: the 5.04M workhorse (256 wide, 4 blocks, emb_dim 32),
global batch 1024, budget 80 tokens per parameter (17,600 steps, about four minutes on
one H100), grammar loss off, evaluation on 10,000 sampled molecules. Winners are
re-run with a second seed and then at 160 t/p. Model size and budget are deliberately
frozen: they are known to work (see docs/experiments-zinc250k.md) and would mask
everything else.

The failure budget we are attacking, measured at 9.7% validity: 42% of invalid strings
are unbalanced parentheses, 47% have an odd number of some ring digit, 11% are
everything else (valence, aromaticity, charges). So almost nine tenths of the loss is
grammar bookkeeping, not chemistry.

## 1. Decoding: the model is better than its argmax

The latents are decoded position by position, independently, by argmax. Nothing checks
that the result is a well-formed SMILES, even though the rules are known in advance and
cost nothing to enforce.

| idea | status | result |
|---|---|---|
| grammar repair, trim what is left open | done | 9.67% -> 62.91%, but mean length 25.6 against 44 in the corpus and uniqueness 58% |
| grammar repair, close what is left open | done | 9.67% -> **20.70%** at length 45.0 and uniqueness 100% |
| grammar repair, trim rings then close branches | queued | |
| snap the x0 estimate onto the nearest embedding (clamping) | done | harmful at full strength: 9.67 -> 0.43%; light strengths queued |
| valence-aware decoding: also track open valences per atom | open | the remaining failures are 91% valence, this is the next frontier |
| beam or best-of-n over the same latent, ranked by RDKit | open | cheap at generation time, changes the sampling cost only |

The split between trim and close is the honest trade-off: trimming buys validity by
producing shorter molecules, closing keeps the length distribution and still triples
validity.

## 2. The diffusion pipeline itself

The forward process corrupts a token embedding with isotropic Gaussian noise of a fixed
size and then follows a variance-preserving cosine path. Both halves are choices.

| idea | status | rationale |
|---|---|---|
| noise intensity 0.10 / 0.25 / 0.50 | done | 0.25 best, the model adapts the embedding scale anyway |
| Laplace noise (heavier tails, same variance) | queued | tails decide how often the denoiser sees a hard example |
| noise on the sphere (fixed norm per position) | queued | removes the norm degree of freedom the model has to learn to ignore |
| corruption towards another token's embedding | queued | keeps the corruption on the data manifold; a continuous relaxation of a discrete transition |
| fix the latent scale by RMS-normalizing the embeddings | queued | the raw table grows 50x during a run, so the schedule the model sees drifts |
| time sampled logit-normal instead of uniform | queued | uniform t spends most of the budget where the task is trivial or impossible |
| predict x0 instead of the noise | queued | changes what the network has to represent at low SNR |
| velocity parameterization | open | the usual third option, needs a small formula addition |
| loss weighting by signal-to-noise (min-SNR) | open | standard in image diffusion, not tried here |
| self-conditioning: feed the previous x0 estimate back in | open | the single largest reported gain in continuous text diffusion |

## 3. Padding, length and the canvas

Measured: masking the padding out of attention and out of the loss collapses generation
from 1.43% to 0.03%, because on a fixed canvas the padding is the only place the model
learns where a molecule ends.

| idea | status | rationale |
|---|---|---|
| mask attention only, keep the loss on pads | queued | separates the two halves of that collapse |
| mask the loss only, keep pads visible | queued | same |
| predict the length with a separate head | open | moves termination out of the canvas, then masking and bucketing become safe and buy 1.7x speed |
| left-pad or centre the molecule on the canvas | open | changes what "position" means for RoPE |

## 4. Data

| idea | status | rationale |
|---|---|---|
| random SMILES traversals, 4 per molecule | queued | 224,568 -> 898,271 strings; at 160 t/p the model otherwise sees the same string over a hundred times |
| more traversals (10-20) | open | if 4 helps, the curve is worth following |
| curriculum by molecule size | open | short molecules first, since ring bookkeeping is what fails |
| scaffold-balanced sampling | open | the corpus is dominated by a few scaffolds |

## 5. Training

| idea | status | rationale |
|---|---|---|
| cross-entropy weight 1 vs 3 | queued | how much the readout should shape the latents |
| cross-entropy on the reconstruction | done | worse, 2.7% against 5.3%: the sampler ends where x0 lives, so training the readout on x0 is right |
| learning rate 1e-3 / 3e-3 / 6e-3 | queued | never tuned for this model size |
| averaged weights for sampling | queued | standard variance reduction, off by default until measured |
| embedding dimension 16 / 32 / 64 | queued | 386 classes have to be separable in the diffusion latent |

## 6. Architecture

The denoiser is a DiT: adaLN-Zero conditioning on the timestep, RoPE, bidirectional
attention over the canvas, a linear projection in and out of the 32-dimensional latent.

| idea | status | rationale |
|---|---|---|
| depth against width at fixed parameters | open | 4x256 was inherited, never compared |
| learned absolute positions instead of RoPE | open | the canvas is fixed and short, relative positions may not be what matters |
| normalized readout (the unused NormalizedLinear) | open | the embedding norms grow 50x, so the logit scale drifts |
| a second readout head predicting the ring-digit parity | open | supervise the bookkeeping the model actually fails at |
