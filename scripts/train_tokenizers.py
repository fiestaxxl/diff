#!/usr/bin/env python3
"""Train the tokenizer candidates and audit them side by side.

    python scripts/train_tokenizers.py configs/zinc250k.yaml

Trains everything listed under prepare.tokenizer_candidates on the train corpus,
audits each one on a held-out split with dimol.tokenization.chem_audit, prints the
comparison table and writes the reports as json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dimol.config import load_from_argv  # noqa: E402
from dimol.data.sources import iter_smiles  # noqa: E402
from dimol.tokenization.atomwise import train_atomwise  # noqa: E402
from dimol.tokenization.chem_audit import audit, format_table  # noqa: E402
from dimol.tokenization.smiles_tokenizer import SmilesTokenizer  # noqa: E402


def main(cfg: DictConfig) -> None:
    prep = OmegaConf.to_container(cfg.prepare, resolve=True) or {}
    source = dict(prep.get("corpus") or prep.get("source") or {})
    candidates = prep.get("tokenizer_candidates") or []
    out_dir = Path(prep.get("tokenizer_dir") or "data/tokenizers")
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_split = str(prep.get("audit_split", "val"))
    audit_molecules = prep.get("audit_molecules")

    def train_corpus():
        return iter_smiles(source, "train", canonicalize=False)

    def audit_corpus():
        return iter_smiles(source, audit_split, canonicalize=False)

    reports = []
    for spec in candidates:
        spec = dict(spec)
        name = spec.pop("name")
        kind = spec.pop("kind", "bpe")
        path = out_dir / f"{name}.json"
        started = time.time()

        if path.exists():
            print(f"[{name}] already trained, loading {path}")
            tokenizer = SmilesTokenizer.load(path)
        else:
            print(f"\n=== training {name} ({kind}) {spec}")
            if kind == "atomwise":
                tokenizer = train_atomwise(train_corpus, show_progress=True, **spec)
            else:
                tokenizer = SmilesTokenizer.train(
                    smiles_iter=train_corpus, show_progress=True, **spec
                )
            tokenizer.save(path)
            print(f"[{name}] trained in {time.time() - started:.1f}s -> {path}")

        report = audit(tokenizer, audit_corpus(), name=name, max_molecules=audit_molecules)
        report["path"] = str(path)
        reports.append(report)
        print(f"[{name}] vocab {report['vocab_size']}, tokens p50 {report['tokens_p50']}, "
              f"atoms split {report['atoms_split_pct']:.3f}%")

    print("\n" + format_table(reports) + "\n")
    report_path = out_dir / "audit.json"
    report_path.write_text(json.dumps(reports, indent=2))
    print(f"reports written to {report_path}")

    print("Example tokenizations (first molecule of the audit split):")
    for rep in reports:
        if rep["examples"]:
            ex = rep["examples"][0]
            print(f"  {rep['name']:<22} {' '.join(ex['tokens'])}")


if __name__ == "__main__":
    main(load_from_argv())
