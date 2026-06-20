"""Compatibility helpers for legacy baseline dependencies."""

from __future__ import annotations


def patch_numpy_legacy_aliases() -> None:
    """Restore NumPy aliases expected by older fairseq snapshots."""
    import numpy as np

    aliases = {
        "bool": bool,
        "complex": complex,
        "float": float,
        "int": int,
        "object": object,
        "str": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)
