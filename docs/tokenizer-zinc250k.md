# Tokenizer choice for ZINC-250k

Corpus: `yairschiff/zinc250k`, curated with RDKit canonicalization, deduplication and a
length guard. Upstream ships train and validation only, so half of validation is routed
to a test split by a stable hash of the molecule.

| split | molecules | chars p50 | chars p95 | chars max |
|---|---|---|---|---|
| train | 224,568 | 44 | 60 | 109 |
| val | 12,470 | 44 | 60 | 103 |
| test | 12,417 | 44 | 60 | 88 |

Nothing was dropped: every molecule parsed, none exceeded the length guard, and there
were no duplicates. Element coverage of train: C 224,568, N 219,426, O 209,053,
S 78,989, F 42,117, Cl 32,729, Br 11,151, I 791, P 123.

## How the candidates were judged

A SMILES tokenizer is not judged by compression alone. The audit
(`dimol/tokenization/chem_audit.py`) checks three things, in this order.

1. **Losslessness.** `decode(encode(s)) == s` for the whole split and zero unknown
   tokens. A candidate that fails this is out.
2. **Chemical integrity.** The reference segmentation is the standard atom-level SMILES
   pattern. The audit compares token boundaries against it and counts atoms whose
   characters end up in different tokens, with two-letter elements and bracket atoms
   reported separately.
3. **Structural bookkeeping**, the part that models actually get wrong: how far apart
   the two digits of a ring closure and the two halves of a branch end up in token
   space, and how often a ring closure lands entirely inside one token, where it cannot
   be broken at all.

Efficiency (tokens per molecule, characters per token, dead vocabulary) decides between
candidates that pass the first two.

## Results, audited on the validation split

| tokenizer | vocab | atoms split | Cl/Br split | ring gap p50 | ring gap p95 | paren gap p95 | rings closed inside a token | tok p50 | tok p99 | chars/tok | dead vocab |
|---|---|---|---|---|---|---|---|---|---|---|---|
| atomwise | 67 | 0.000% | 0 | 11 | 39 | 28 | 0.0% | 38 | 59 | 1.16 | 20 |
| bpe_1024 unprotected | 1024 | 0.003% | 13 | 4 | 22 | 16 | 23.3% | 19 | 34 | 2.27 | 70 |
| bpe_256 protected | 258 | 0.000% | 0 | 5 | 24 | 17 | 13.0% | 21 | 37 | 2.05 | 71 |
| **bpe_512 protected** | **514** | **0.000%** | **0** | **5** | **22** | **16** | **19.2%** | **20** | **35** | **2.19** | **71** |
| bpe_1024 protected | 1026 | 0.000% | 0 | 4 | 22 | 16 | 23.4% | 19 | 34 | 2.27 | 71 |
| bpe_2048 protected | 2050 | 0.000% | 0 | 4 | 21 | 15 | 26.5% | 19 | 34 | 2.31 | 108 |
| bpe_512 structure-isolated | 514 | 0.000% | 0 | 7 | 30 | 22 | 0.0% | 28 | 47 | 1.55 | 166 |

Token counts on the training split, including `<bos>` and `<eos>`:

| tokenizer | p50 | p95 | p99 | p99.9 | max | above 48 |
|---|---|---|---|---|---|---|
| atomwise | 40 | 55 | 61 | 66 | 74 | 15.7% |
| bpe_256 protected | 23 | 33 | 39 | 45 | 63 | 0.029% |
| bpe_512 protected | 22 | 32 | 37 | 44 | 62 | 0.017% |
| bpe_1024 protected | 21 | 31 | 36 | 43 | 62 | 0.012% |
| bpe_2048 protected | 21 | 31 | 36 | 42 | 62 | 0.012% |
| bpe_512 structure-isolated | 30 | 43 | 50 | 56 | 69 | 1.37% |

## What the numbers said

**Plain BPE splits chlorine.** Without protection it cuts `CCCl` into `CCC` + `l`:
39 occurrences in the validation split at vocab 512, 13 at 1024, 5 at 2048. A bare `l`
is not a chemical symbol, and the model can emit it anywhere. Protecting the
two-letter elements the same way bracket atoms are protected removes every split at no
cost in compression: `bpe_1024` and `bpe_1024_prot` have the same p50, p99 and
characters per token. Protection is now the default in `SmilesTokenizer.train`.

**BPE shortens the structural distances rather than lengthening them.** This was the
open question, since the model is known to fail on brackets and digits. Merging brings
the two digits of a ring closure from 11 tokens apart (atom level) to 5, the 95th
percentile from 39 to 22, and it puts 19% of all ring closures entirely inside a single
token, where they cannot be mismatched. Branch matching shortens from 28 to 16 tokens at
the 95th percentile. Isolating parentheses and digits from every merge goes the other
way: no ring is ever closed inside a token, and the distances grow back to 7 and 30.

**Vocabulary past 512 buys almost nothing.** Going 512 -> 1024 -> 2048 saves one token
at p50 and one at p99 while doubling and then quadrupling the vocabulary. In this model
the readout has to separate the vocabulary inside a 32-dimensional diffusion latent, so
extra classes are not free.

## Decision

`bpe_512_prot`, vocabulary 514, `max_length` 64.

* zero atom splits, zero unknown tokens, lossless roundtrip on all three splits;
* 64 covers the whole corpus: the longest training molecule is 62 tokens, so nothing is
  dropped, and 64 is a multiple of 64 for the kernels;
* 22 tokens at the median means 65% of a padded batch is padding, which the length
  bucketing already implemented in the data loader turns into an average batch width of
  about 24.

Two candidates are kept as ablation axes rather than discarded: `atomwise` is the
chemically safest and longest, `bpe_512_isolated` never lets a merge touch a parenthesis
or a ring digit. Both are lossless, so a run can switch between them by changing one
path in the config.

## Reproducing

```bash
python scripts/prepare_data.py     configs/zinc250k.yaml   # corpus, 250s -> 50s with 16 procs
python scripts/train_tokenizers.py configs/zinc250k.yaml   # all candidates + the audit table
python scripts/tokenize_dataset.py configs/zinc250k.yaml   # sharded .npy, 4.3s
```
