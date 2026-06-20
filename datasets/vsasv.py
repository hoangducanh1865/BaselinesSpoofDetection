"""Convert VSASV CM scenario files into fold*.tsv + wav.scp files."""

from pathlib import Path

import pandas as pd

_DEFAULT_PROTOCOL = "result_reproduce/cm/bonafide_replay_adversarial_vc.txt"
_SPOOF_HINTS = (
    "voice_conversion",
    "replay",
    "adversarial",
    "partial-spoof",
    "/spoof/",
    "_vc_",
    "_ra_",
)


def _label_from_path(path: str) -> str:
    lowered = path.lower()
    if any(hint in lowered for hint in _SPOOF_HINTS):
        return "spoof"
    return "bonafide"


def ensure_meta(data_root: Path, meta_dir: Path, fold: int, track=None, force: bool = False) -> None:
    """Write fold{fold}_evaluation.tsv + wav.scp for VSASV.

    The selected VSASV CM file stores ``audio_path score`` from a previous
    reproduction run, not explicit labels. Labels are therefore inferred from
    stable path conventions: bonafide paths are genuine speech, and replay /
    voice-conversion / adversarial paths are spoofing attacks.
    """
    data_root = Path(data_root)
    meta_dir = Path(meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)

    eval_path = meta_dir / f"fold{fold}_evaluation.tsv"
    wav_scp_path = meta_dir / "wav.scp"
    if not force and eval_path.exists() and wav_scp_path.exists():
        return

    protocol_path = data_root / _DEFAULT_PROTOCOL
    df = pd.read_csv(protocol_path, sep=r"\s+", header=None, names=["path", "score"], engine="python")
    df = df.drop_duplicates(subset=["path"]).reset_index(drop=True)
    df["utt_id"] = [f"vsasv_{idx:06d}" for idx in range(len(df))]
    df["label"] = df["path"].astype(str).map(_label_from_path)

    df[["utt_id", "label"]].to_csv(eval_path, sep="\t", index=False)
    with open(wav_scp_path, "w") as f:
        for row in df.itertuples(index=False):
            abs_path = data_root / row.path
            f.write(f"{row.utt_id} {abs_path}\n")
