"""Convert In-The-Wild metadata into MoLEx fold*.tsv + wav.scp files."""

from pathlib import Path

import pandas as pd


def _metadata_is_valid(eval_path: Path, wav_scp_path: Path) -> bool:
    if not eval_path.exists() or not wav_scp_path.exists():
        return False

    eval_df = pd.read_csv(eval_path, sep="\t")
    eval_keys = set(eval_df.iloc[:, 0].astype(str))
    eval_labels = set(eval_df.iloc[:, 1].astype(str).str.lower())
    scp_keys = set()
    with open(wav_scp_path) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if parts:
                scp_keys.add(parts[0])
    return bool(eval_keys) and eval_keys <= scp_keys and eval_labels <= {"bonafide", "spoof"}


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
    if not force and _metadata_is_valid(eval_path, wav_scp_path):
        return

    df = pd.read_csv(data_root / "meta.csv")
    # In-The-Wild filenames may contain spaces. MoLEx uses Kaldi-style wav.scp
    # where the first whitespace-delimited token is the utterance ID, so generate
    # compact IDs instead of deriving IDs from filenames.
    df["utt_id"] = [f"itw_{idx:06d}" for idx in range(len(df))]
    df["label"] = (
        df["label"]
        .astype(str)
        .str.lower()
        .str.strip()
        .replace({"bona-fide": "bonafide", "bona_fide": "bonafide"})
    )

    df[["utt_id", "label"]].to_csv(eval_path, sep="\t", index=False)

    with open(wav_scp_path, "w") as f:
        for row in df.itertuples(index=False):
            f.write(f"{row.utt_id} {data_root / row.file}\n")
