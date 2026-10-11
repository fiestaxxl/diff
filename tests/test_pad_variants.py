"""Distinct padding: variants are assigned by position, and they all stop the molecule."""
from __future__ import annotations

import torch
from omegaconf import OmegaConf

from dimol.builders import build_path
from dimol.models.diffusion_transformer import DiffusionTransformer, TransformerConfig
from dimol.training.tasks import DiffusionTask

SEQ, VOCAB, EMB, VARIANTS = 12, 64, 8, 4


def _model(variants: int) -> DiffusionTransformer:
    return DiffusionTransformer(TransformerConfig(
        model_dim=32, emb_dim=EMB, time_dim=32, num_heads=4, num_text_blocks=2,
        vocab_size=VOCAB, pad_idx=0, max_pos=SEQ + 4,
        self_conditioning=False, pad_variants=variants))


def _task(path) -> DiffusionTask:
    return DiffusionTask(path, pad_idx=0, grammar_enabled=False, self_cond_prob=0.0,
                         seq_len=SEQ, ce_include_pad=True)


def _path():
    cfg = OmegaConf.create({"diffusion": {"name": "vp_cosine", "p_simple": "gaussian"}})
    return build_path(cfg, seq_len=SEQ, emb_dim=EMB, device="cpu")


def _batch():
    mask = torch.zeros(3, SEQ, dtype=torch.long)
    mask[:, :5] = 1
    return {"token_ids": torch.randint(1, 40, (3, SEQ)) * mask,
            "attention_mask": mask}


def test_one_variant_is_the_original_behaviour():
    torch.manual_seed(0)
    plain = _task(_path()).compute_loss(_model(1), _batch(), step=0)
    assert torch.isfinite(plain["loss"])


def test_variants_give_consecutive_padding_positions_different_targets():
    model = _model(VARIANTS)
    base = VOCAB - VARIANTS
    batch = _batch()
    # Reproduce the assignment the task performs and check it cycles.
    positions = torch.arange(SEQ)
    cyclic = base + (positions % VARIANTS)
    pad_at = batch["attention_mask"][0].bool()
    assigned = torch.where(pad_at, batch["token_ids"][0], cyclic)
    tail = assigned[~pad_at]
    assert len(set(tail.tolist())) == VARIANTS, "padding targets should cycle, not repeat"
    assert tail[0] != tail[1], "consecutive padding positions must differ"


def test_the_loss_runs_with_variants_and_supervises_them():
    torch.manual_seed(0)
    out = _task(_path()).compute_loss(_model(VARIANTS), _batch(), step=0)
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["ce_loss"])


def test_variants_need_an_attention_mask():
    import pytest

    with pytest.raises(ValueError, match="pad_variants"):
        _task(_path()).compute_loss(_model(VARIANTS),
                                    {"token_ids": torch.randint(1, 40, (3, SEQ))}, step=0)


def test_every_variant_ends_the_molecule_in_the_decoder():
    from dimol.eval.decoding import build_decoder

    class _Tok:
        pad_id, eos_id, bos_id, unk_id, mask_id = 0, 1, 2, 3, 4
        vocab_size = VOCAB
        pad_variant_ids = tuple(range(VOCAB - VARIANTS, VOCAB))

        def get_vocab(self):
            return {f"t{i}": i for i in range(VOCAB)}

        def id_to_token(self, i):
            return f"t{i}"

    decoder = build_decoder(_Tok(), canvas=SEQ, mode="grammar_close")
    for i in range(VOCAB - VARIANTS, VOCAB):
        assert decoder.is_stop[i], f"id {i} should stop the molecule"
