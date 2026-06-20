"""Convert ASVspoof2021 LA/DF/PA CM keys into fold*.tsv + wav.scp files."""

from pathlib import Path

import pandas as pd

_TRACKS = {"LA", "DF", "PA"}


def _metadata_is_valid(eval_path: Path, wav_scp_path: Path) -> bool:
    return eval_path.exists() and wav_scp_path.exists() and eval_path.stat().st_size > 0


def _read_key(path: Path, track: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    # All ASVspoof2021 CM key files use the utterance id in column 1.
    # The label column shifts with the richer DF metadata, but it is always the
    # column containing only bonafide/spoof.
    label_col = None
    for col in df.columns:
        values = set(df[col].astype(str).str.lower().unique())
        if values and values <= {"bonafide", "spoof"}:
            label_col = col
            break
    if label_col is None:
        raise ValueError(f"Could not find bonafide/spoof label column in {path}")
    return pd.DataFrame({"utt_id": df[1].astype(str), "label": df[label_col].astype(str).str.lower()})


def _audio_dirs(data_root: Path, track: str) -> list[Path]:
    prefix = f"ASVspoof2021_{track}_eval"
    dirs = []
    for root in sorted(data_root.glob(f"{prefix}*")):
        dirs.extend(sorted(root.rglob("flac")))
    return dirs


def _build_audio_map(data_root: Path, track: str) -> dict[str, Path]:
    audio_map = {}
    for flac_dir in _audio_dirs(data_root, track):
        for path in flac_dir.glob("*.flac"):
            audio_map[path.stem] = path
    return audio_map


def ensure_meta(data_root: Path, meta_dir: Path, fold: int, track: str, force: bool = False) -> None:
    """Write fold{fold}_evaluation.tsv + wav.scp for ASVspoof2021.

    data_root must point at the extracted ASVspoof2021 directory containing
    LA-keys-full, DF-keys-full, PA-keys-full and the corresponding eval audio
    folders. Only evaluation metadata is created because ASVspoof2021 is used
    here as a cross-dataset test set.
    """
    track = track.upper()
    if track not in _TRACKS:
        raise ValueError(f"Unsupported ASVspoof2021 track: {track}")

    data_root = Path(data_root)
    meta_dir = Path(meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)

    eval_path = meta_dir / f"fold{fold}_evaluation.tsv"
    wav_scp_path = meta_dir / "wav.scp"
    if not force and _metadata_is_valid(eval_path, wav_scp_path):
        return

    key_path = data_root / f"{track}-keys-full" / "keys" / track / "CM" / "trial_metadata.txt"
    df = _read_key(key_path, track)
    audio_map = _build_audio_map(data_root, track)

    missing = [utt_id for utt_id in df["utt_id"] if utt_id not in audio_map]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} ASVspoof2021 {track} audio files; first missing: {missing[0]}"
        )

    df.to_csv(eval_path, sep="\t", index=False)
    with open(wav_scp_path, "w") as f:
        for utt_id in df["utt_id"]:
            f.write(f"{utt_id} {audio_map[utt_id]}\n")
