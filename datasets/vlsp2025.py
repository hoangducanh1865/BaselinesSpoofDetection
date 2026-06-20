"""Convert the labeled VLSP2025 dev CM list into fold*.tsv + wav.scp files."""

from pathlib import Path

import pandas as pd

_DEFAULT_PROTOCOL = "data-train/split/original+adversarial+tts+vc/dev.txt"


def ensure_meta(data_root: Path, meta_dir: Path, fold: int, track=None, force: bool = False) -> None:
    """Write fold{fold}_evaluation.tsv + wav.scp for VLSP2025.

    The public/private VLSP2025 trial files are ASV/SASV trial pairs. For CM EER
    with AASIST, use the labeled development list containing columns:
    speaker_id audio_path label.
    """
    data_root = Path(data_root)
    meta_dir = Path(meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)

    eval_path = meta_dir / f"fold{fold}_evaluation.tsv"
    wav_scp_path = meta_dir / "wav.scp"
    if not force and eval_path.exists() and wav_scp_path.exists():
        return

    protocol_path = data_root / _DEFAULT_PROTOCOL
    df = pd.read_csv(protocol_path, sep=r"\s+", header=None, names=["speaker", "path", "label"], engine="python")
    df["utt_id"] = [f"vlsp2025_{idx:06d}" for idx in range(len(df))]
    df["label"] = df["label"].astype(str).str.lower().replace({"real": "bonafide", "fake": "spoof"})

    df[["utt_id", "label"]].to_csv(eval_path, sep="\t", index=False)
    with open(wav_scp_path, "w") as f:
        for row in df.itertuples(index=False):
            f.write(f"{row.utt_id} {row.path}\n")
