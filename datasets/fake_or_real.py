"""Convert Fake-or-Real testing folders into fold*.tsv + wav.scp files."""

from pathlib import Path

import pandas as pd

_SUBSETS = {
    None: ("for-norm", "for-norm"),
    "norm": ("for-norm", "for-norm"),
    "2sec": ("for-2sec", "for-2seconds"),
    "original": ("for-original", "for-original"),
    "rerec": ("for-rerec", "for-rerecorded"),
}


def ensure_meta(data_root: Path, meta_dir: Path, fold: int, track=None, force: bool = False) -> None:
    """Write fold{fold}_evaluation.tsv + wav.scp for a Fake-or-Real subset.

    Labels are inferred from testing/real and testing/fake. The default subset
    is for-norm because it is already normalized/16 kHz/mono in the downloaded
    tree observed on the server.
    """
    if track not in _SUBSETS:
        raise ValueError(f"Unsupported Fake-or-Real subset: {track}")

    outer, inner = _SUBSETS[track]
    test_root = Path(data_root) / outer / inner / "testing"
    meta_dir = Path(meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)

    eval_path = meta_dir / f"fold{fold}_evaluation.tsv"
    wav_scp_path = meta_dir / "wav.scp"
    if not force and eval_path.exists() and wav_scp_path.exists():
        return

    rows = []
    scp_lines = []
    for label_dir, label in (("real", "bonafide"), ("fake", "spoof")):
        for path in sorted((test_root / label_dir).glob("*.wav")):
            utt_id = f"for_{track or 'norm'}_{label_dir}_{path.stem}"
            rows.append((utt_id, label))
            scp_lines.append(f"{utt_id} {path}")

    if not rows:
        raise FileNotFoundError(f"No Fake-or-Real wav files found under {test_root}")

    pd.DataFrame(rows, columns=["utt_id", "label"]).to_csv(eval_path, sep="\t", index=False)
    with open(wav_scp_path, "w") as f:
        f.write("\n".join(scp_lines) + "\n")
