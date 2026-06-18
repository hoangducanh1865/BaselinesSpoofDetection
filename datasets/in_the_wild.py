"""Convert In-The-Wild metadata into MoLEx fold*.tsv + wav.scp files."""

from pathlib import Path

import pandas as pd


def ensure_meta(data_root: Path, meta_dir: Path, fold: int, track=None, force: bool = False) -> None:
    """Write fold{fold}_evaluation.tsv + wav.scp under meta_dir.

    Expected layout:
        data_root/
          meta.csv                 # columns: file,speaker,label
          <audio files>.wav
    """
    data_root = Path(data_root)
    meta_dir = Path(meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)

    eval_path = meta_dir / f"fold{fold}_evaluation.tsv"
    wav_scp_path = meta_dir / "wav.scp"
    if not force and eval_path.exists() and wav_scp_path.exists():
        return

    df = pd.read_csv(data_root / "meta.csv")
    df["utt_id"] = df["file"].map(lambda name: Path(str(name)).stem)
    df["label"] = df["label"].astype(str).str.lower()

    df[["utt_id", "label"]].to_csv(eval_path, sep="\t", index=False)

    with open(wav_scp_path, "w") as f:
        for row in df.itertuples(index=False):
            f.write(f"{row.utt_id} {data_root / row.file}\n")
