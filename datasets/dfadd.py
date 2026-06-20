"""Convert DFADD Hugging Face parquet test data into fold*.tsv + wav.scp files."""

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


def _normalise_label(label: str) -> str:
    label = str(label).strip().lower()
    return {
        "real": "bonafide",
        "genuine": "bonafide",
        "bonafide": "bonafide",
        "fake": "spoof",
        "spoof": "spoof",
        "spoofed": "spoof",
    }.get(label, label)


def _read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise ImportError(
            "DFADD parquet decoding requires pyarrow or fastparquet in the eval "
            "conda environment. Install one with `pip install pyarrow`."
        ) from exc


def _decode_audio(audio_obj):
    """Return (array, sampling_rate) from Hugging Face Audio parquet payloads."""
    if isinstance(audio_obj, dict):
        if audio_obj.get("bytes") is not None:
            array, sr = sf.read(BytesIO(audio_obj["bytes"]), dtype="float32")
            return array, sr
        if audio_obj.get("array") is not None:
            return np.asarray(audio_obj["array"], dtype=np.float32), int(audio_obj.get("sampling_rate", 16000))
        if audio_obj.get("path"):
            array, sr = sf.read(audio_obj["path"], dtype="float32")
            return array, sr
    if isinstance(audio_obj, (bytes, bytearray)):
        array, sr = sf.read(BytesIO(audio_obj), dtype="float32")
        return array, sr
    raise TypeError(f"Unsupported DFADD audio payload type: {type(audio_obj)!r}")


def ensure_meta(data_root: Path, meta_dir: Path, fold: int, track=None, force: bool = False) -> None:
    """Write fold{fold}_evaluation.tsv + wav.scp for the DFADD test split.

    Audio is embedded in parquet files, so this converter extracts deterministic
    16 kHz wav files into ``meta_dir/audio`` once and points wav.scp at them.
    """
    data_root = Path(data_root)
    meta_dir = Path(meta_dir)
    audio_dir = meta_dir / "audio"
    meta_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    eval_path = meta_dir / f"fold{fold}_evaluation.tsv"
    wav_scp_path = meta_dir / "wav.scp"
    if not force and eval_path.exists() and wav_scp_path.exists():
        return

    rows = []
    scp_lines = []
    idx = 0
    for parquet_path in sorted((data_root / "data").glob("test-*.parquet")):
        df = _read_parquet(parquet_path)
        for row in df.itertuples(index=False):
            utt_id = str(getattr(row, "audio_name", "") or f"dfadd_test_{idx:06d}")
            utt_id = utt_id.replace("/", "_").replace(" ", "_")
            label = _normalise_label(getattr(row, "label"))
            wav_path = audio_dir / f"{utt_id}.wav"
            if not wav_path.exists() or force:
                array, sr = _decode_audio(getattr(row, "audio"))
                if array.ndim > 1:
                    array = array.mean(axis=1)
                sf.write(wav_path, array, sr)
            rows.append((utt_id, label))
            scp_lines.append(f"{utt_id} {wav_path}")
            idx += 1

    pd.DataFrame(rows, columns=["utt_id", "label"]).to_csv(eval_path, sep="\t", index=False)
    with open(wav_scp_path, "w") as f:
        f.write("\n".join(scp_lines) + "\n")
