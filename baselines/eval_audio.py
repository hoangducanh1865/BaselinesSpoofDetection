"""Shared audio helpers for baseline evaluation adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import Tensor


def load_wav_scp(path: Path) -> dict[str, str]:
    mapping = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping


def format_error(exc: Exception) -> str:
    try:
        message = str(exc)
    except Exception:
        message = repr(exc)
    return f"{type(exc).__name__}: {message}"


def read_mono_audio(path: str, target_sr: int | None = None, resample_context: str = "eval") -> np.ndarray:
    audio, sample_rate = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.shape[0] == 0:
        raise ValueError("empty audio")
    if target_sr is not None and sample_rate != target_sr:
        try:
            import librosa
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"{resample_context} must resample non-{target_sr} Hz audio. "
                "Install librosa or convert the dataset first."
            ) from exc
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sample_rate, target_sr=target_sr)
        if audio.shape[0] == 0:
            raise ValueError("empty audio after resampling")
    return audio.astype(np.float32)


def repeat_or_trim(x: np.ndarray, max_len: int) -> np.ndarray:
    if x.shape[0] == 0:
        raise ValueError("empty audio")
    if x.shape[0] >= max_len:
        return x[:max_len]
    repeats = int(max_len / x.shape[0]) + 1
    return np.tile(x, repeats)[:max_len]


def collate_eval_fixed(batch):
    xs = []
    ys = []
    utt_ids = []
    skipped = []
    for x, y, utt_id, path, error in batch:
        if error is not None:
            skipped.append((utt_id, "bonafide" if y == 1 else "spoof", path, error))
            continue
        xs.append(x)
        ys.append(y)
        utt_ids.append(utt_id)
    if not xs:
        return None, None, None, skipped
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long), utt_ids, skipped


def collate_eval_padded(batch):
    xs = []
    ys = []
    utt_ids = []
    skipped = []
    for x, y, utt_id, path, error in batch:
        if error is not None:
            skipped.append((utt_id, "bonafide" if y == 1 else "spoof", path, error))
            continue
        xs.append(x)
        ys.append(y)
        utt_ids.append(utt_id)
    if not xs:
        return None, None, None, skipped
    max_len = max(x.numel() for x in xs)
    padded = xs[0].new_zeros((len(xs), max_len))
    for index, x in enumerate(xs):
        padded[index, : x.numel()] = x
    return padded, torch.tensor(ys, dtype=torch.long), utt_ids, skipped


def tensor_or_error(loader, utt_id: str, label: str, path: str):
    y = 1 if label == "bonafide" else 0
    try:
        audio = loader(path)
    except Exception as exc:
        return None, y, utt_id, path, format_error(exc)
    return Tensor(audio), y, utt_id, path, None


def write_skipped_audio(output_dir: Path, skipped_rows: list[tuple[str, str, str, str]]) -> None:
    if not skipped_rows:
        return
    skipped_path = output_dir / "skipped_audio.tsv"
    with open(skipped_path, "w") as f:
        f.write("utt_id\tlabel\tpath\terror\n")
        for utt_id, label, path, error in skipped_rows:
            f.write(f"{utt_id}\t{label}\t{path}\t{error}\n")
    print(f"Skipped {len(skipped_rows)} unreadable files; details written to {skipped_path}")
