"""Allow TUAB runs when the raw EDF tree is absent on this machine (e.g. 2080 servers).

If metadata CSVs were built and stored in ~/.cache/pyhealth/tuab/ (typical) and
examples/conformal_eeg/cache_tuab/ (or a custom --cache-dir) already holds
samples_*/index.json from a run on a machine with the corpus, you can use the
shared NFS copy without reprocessing or rsync'ing terabytes of .edf.

Does not rebuild cache; it only relaxes the script-level ``root`` existence
check. Requires train/eval PyHealth metadata CSVs and a finished sample stream.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def tuab_sample_cache_dir(args: argparse.Namespace) -> Path:
    """Match ``_run`` cache_dir logic: .../cache_tuab or {cache_dir}_tuab."""
    cache_base = getattr(args, "cache_dir", None)
    rel = (cache_base.rstrip("/") + "_tuab") if cache_base else "examples/conformal_eeg/cache_tuab"
    p = Path(rel)
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def tuab_offline_caches_available(args: argparse.Namespace) -> bool:
    """True if we can run TUAB without the original ``--root`` EDF directory."""
    meta = Path.home() / ".cache" / "pyhealth" / "tuab"
    if not (meta / "tuab-train-pyhealth.csv").is_file():
        return False
    if not (meta / "tuab-eval-pyhealth.csv").is_file():
        return False
    cdir = tuab_sample_cache_dir(args)
    if not cdir.is_dir():
        return False
    return any(cdir.glob("samples_*.ld/index.json"))
