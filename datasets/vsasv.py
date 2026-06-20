"""Convert VSASV CM scenario files into fold*.tsv + wav.scp files."""

from pathlib import Path

import pandas as pd

_DEFAULT_PROTOCOL = "result_reproduce/cm/bonafide_replay_adversarial_vc.txt"
_AUDIO_DIR = "dataset-16"
_SPOOF_HINTS = (
    "voice_conversion",
    "replay",
    "adversarial",
    "partial-spoof",
    "/spoof/",
    "_vc_",
    "_ra_",
)
_REQUIRED_LABELS = {"bonafide", "spoof"}


def _resolve_audio_path(raw_path: str, data_root: Path) -> Path:
    candidates = _candidate_audio_paths(raw_path, data_root)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _candidate_audio_paths(raw_path: str, data_root: Path) -> list[Path]:
    raw = Path(str(raw_path))
    parts = raw.parts
    candidates = []

    if raw.is_absolute():
        candidates.append(raw)

    for marker in ("dataset-16", "dataset"):
        if marker in parts:
            marker_idx = parts.index(marker)
            suffix = Path(*parts[marker_idx + 1:])
            candidates.append(data_root / _AUDIO_DIR / suffix)
            if len(suffix.parts) == 2:
                candidates.append(data_root / _AUDIO_DIR / suffix.parts[0] / "bonafide" / suffix.parts[1])

    if "vn-celeb" in parts:
        marker_idx = parts.index("vn-celeb")
        if len(parts) > marker_idx + 1 and parts[marker_idx + 1] == "data":
            suffix = Path(*parts[marker_idx + 2:])
            candidates.append(data_root / _AUDIO_DIR / suffix)
            if len(suffix.parts) == 2:
                candidates.append(data_root / _AUDIO_DIR / suffix.parts[0] / "bonafide" / suffix.parts[1])

    if "spoofing_data" in parts:
        marker_idx = parts.index("spoofing_data")
        suffix = Path(*parts[marker_idx + 1:])
        candidates.append(data_root / _AUDIO_DIR / suffix)

    if "adversarial_data" in parts:
        marker_idx = parts.index("adversarial_data")
        suffix = Path(*parts[marker_idx + 1:])
        candidates.append(data_root / _AUDIO_DIR / suffix)

    if not raw.is_absolute():
        candidates.append(data_root / raw)
        candidates.append(data_root / _AUDIO_DIR / raw)

    deduped = []
    seen = set()
    for candidate in candidates or [data_root / raw]:
        key = str(candidate)
        if key not in seen:
            deduped.append(candidate)
            seen.add(key)
    return deduped


def _is_under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _metadata_is_valid(eval_path: Path, wav_scp_path: Path, data_root: Path) -> bool:
    if not eval_path.exists() or not wav_scp_path.exists():
        return False

    eval_df = pd.read_csv(eval_path, sep="\t")
    if eval_df.empty:
        return False
    eval_keys = set(eval_df.iloc[:, 0].astype(str))
    eval_labels = set(eval_df.iloc[:, 1].astype(str).str.lower())
    if not eval_keys or not eval_labels <= _REQUIRED_LABELS or eval_labels != _REQUIRED_LABELS:
        return False

    scp_keys = set()
    with open(wav_scp_path) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            scp_keys.add(parts[0])
            audio_path = Path(parts[1])
            if audio_path.is_absolute() and not _is_under(audio_path, data_root):
                return False
            if not audio_path.exists():
                return False
    return eval_keys <= scp_keys


def _label_from_path(path: str) -> str:
    lowered = path.lower()
    if any(hint in lowered for hint in _SPOOF_HINTS):
        return "spoof"
    return "bonafide"


def _rows_from_protocol(df: pd.DataFrame, data_root: Path) -> tuple[list[tuple[str, str, Path]], int]:
    rows = []
    missing = 0
    for row in df.itertuples(index=False):
        abs_path = _resolve_audio_path(row.path, data_root)
        if not abs_path.exists():
            missing += 1
            continue
        utt_id = f"vsasv_{len(rows):06d}"
        rows.append((utt_id, _label_from_path(str(row.path)), abs_path))
    return rows, missing


def _rows_from_audio_tree(data_root: Path) -> list[tuple[str, str, Path]]:
    audio_root = data_root / _AUDIO_DIR
    rows = []
    if not audio_root.exists():
        return rows
    for path in sorted(audio_root.rglob("*.wav")):
        utt_id = f"vsasv_{len(rows):06d}"
        rows.append((utt_id, _label_from_path(str(path)), path))
    return rows


def _label_counts(rows: list[tuple[str, str, Path]]) -> dict[str, int]:
    counts = {}
    for _, label, _ in rows:
        counts[label] = counts.get(label, 0) + 1
    return counts


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
    if not force and _metadata_is_valid(eval_path, wav_scp_path, data_root):
        return

    protocol_path = data_root / _DEFAULT_PROTOCOL
    df = pd.read_csv(protocol_path, sep=r"\s+", header=None, names=["path", "score"], engine="python")
    df = df.drop_duplicates(subset=["path"]).reset_index(drop=True)

    rows, missing = _rows_from_protocol(df, data_root)
    source = "protocol"
    labels = set(_label_counts(rows))
    if labels != _REQUIRED_LABELS:
        scanned_rows = _rows_from_audio_tree(data_root)
        scanned_labels = set(_label_counts(scanned_rows))
        if scanned_labels == _REQUIRED_LABELS or len(scanned_rows) > len(rows):
            rows = scanned_rows
            source = "audio_tree"

    eval_df = pd.DataFrame(rows, columns=["utt_id", "label", "path"])
    eval_df[["utt_id", "label"]].to_csv(eval_path, sep="\t", index=False)
    with open(wav_scp_path, "w") as f:
        for row in eval_df.itertuples(index=False):
            f.write(f"{row.utt_id} {row.path}\n")

    counts = eval_df["label"].value_counts().to_dict() if not eval_df.empty else {}
    print(
        f"[vsasv] source={source}; {len(eval_df)}/{len(df)} files found; "
        f"skipped_missing={missing}; labels={counts}"
    )
