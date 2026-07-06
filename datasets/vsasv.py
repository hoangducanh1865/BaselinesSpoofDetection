"""Convert VSASV CM scenario files into fold*.tsv + wav.scp files."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

_DEFAULT_PROTOCOL = "result_reproduce/cm/bonafide_replay_adversarial_vc.txt"
_AUDIO_DIR = "dataset-16"
_SPOOF_HINTS = (
    "voice_conversion",
    "replay",
    "adversarial",
    "partial-spoof",
    "spoofed",
    "_vc_",
    "_ra_",
)
_REQUIRED_LABELS = {"bonafide", "spoof"}
_RESPLIT_SEED = 20260706
_RESPLIT_RATIOS = {"train": 0.7, "validation": 0.1, "evaluation": 0.2}
_AUDIO_SUFFIXES = {".wav", ".flac"}


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
    parts = {part.lower() for part in Path(path).parts}
    if "bonafide" in parts:
        return "bonafide"
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


def _stable_order(values: list[str], seed: int) -> list[str]:
    return sorted(
        values,
        key=lambda value: (
            hashlib.blake2b(
                f"{seed}:{value}".encode("utf-8"), digest_size=16
            ).hexdigest(),
            value,
        ),
    )


def _allocate_group(speakers: list[str], seed: int) -> dict[str, str]:
    """Allocate one attack-signature group while preserving rare signatures."""
    ordered = _stable_order(speakers, seed)
    count = len(ordered)
    split_order = list(_RESPLIT_RATIOS)
    if count == 1:
        counts = {"train": 1, "validation": 0, "evaluation": 0}
    elif count == 2:
        counts = {"train": 1, "validation": 0, "evaluation": 1}
    else:
        counts = {split: 1 for split in split_order}
        remaining = count - len(split_order)
        exact = {
            split: remaining * ratio
            for split, ratio in _RESPLIT_RATIOS.items()
        }
        floors = {split: int(value) for split, value in exact.items()}
        for split, value in floors.items():
            counts[split] += value
        leftover = remaining - sum(floors.values())
        fractional_order = sorted(
            split_order,
            key=lambda split: (exact[split] - floors[split], _RESPLIT_RATIOS[split]),
            reverse=True,
        )
        for split in fractional_order[:leftover]:
            counts[split] += 1

    assignment = {}
    offset = 0
    for split in split_order:
        for speaker in ordered[offset:offset + counts[split]]:
            assignment[speaker] = split
        offset += counts[split]
    return assignment


def _scan_resplit_audio(data_root: Path) -> dict[str, dict[str, list[Path]]]:
    audio_root = data_root / _AUDIO_DIR
    if not audio_root.is_dir():
        raise FileNotFoundError(f"VSASV audio directory does not exist: {audio_root}")

    inventory: dict[str, dict[str, list[Path]]] = {}
    for speaker_dir in sorted(path for path in audio_root.iterdir() if path.is_dir()):
        attacks = {}
        for attack_dir in sorted(path for path in speaker_dir.iterdir() if path.is_dir()):
            files = sorted(
                path.resolve()
                for path in attack_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in _AUDIO_SUFFIXES
            )
            if files:
                attacks[attack_dir.name] = files
        if attacks:
            inventory[speaker_dir.name] = attacks
    if not inventory:
        raise FileNotFoundError(f"No VSASV audio found under {audio_root}")
    return inventory


def prepare_resplit(
    data_root: Path,
    meta_dir: Path,
    fold: int = 1,
    force: bool = False,
    seed: int = _RESPLIT_SEED,
) -> dict:
    """Create a non-destructive speaker-disjoint 70/10/20 VSASV split."""
    data_root = Path(data_root)
    meta_dir = Path(meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    split_paths = {
        split: meta_dir / f"fold{fold}_{split}.tsv"
        for split in _RESPLIT_RATIOS
    }
    wav_scp_path = meta_dir / "wav.scp"
    manifest_path = meta_dir / "speaker_split_manifest.tsv"
    summary_path = meta_dir / "split_summary.json"
    expected = [*split_paths.values(), wav_scp_path, manifest_path, summary_path]
    if not force and all(path.exists() for path in expected):
        return json.loads(summary_path.read_text())

    inventory = _scan_resplit_audio(data_root)
    signature_groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for speaker, attacks in inventory.items():
        signature_groups[tuple(sorted(attacks))].append(speaker)

    assignment = {}
    for group_index, (signature, speakers) in enumerate(sorted(signature_groups.items())):
        group_seed = seed + group_index
        assignment.update(_allocate_group(speakers, group_seed))

    split_rows: dict[str, list[tuple[str, str, Path, str, str]]] = defaultdict(list)
    manifest_rows = []
    seen_paths = set()
    for speaker, attacks in sorted(inventory.items()):
        split = assignment[speaker]
        signature = ",".join(sorted(attacks))
        file_count = 0
        for attack, paths in sorted(attacks.items()):
            label = "bonafide" if attack.lower() == "bonafide" else "spoof"
            for path in paths:
                path_key = str(path)
                if path_key in seen_paths:
                    raise RuntimeError(f"Duplicate VSASV audio path: {path}")
                seen_paths.add(path_key)
                relative = path.relative_to((data_root / _AUDIO_DIR).resolve())
                utt_hash = hashlib.blake2b(
                    str(relative).encode("utf-8"), digest_size=10
                ).hexdigest()
                utt_id = f"vsasv_rs_{utt_hash}"
                split_rows[split].append((utt_id, label, path, speaker, attack))
                file_count += 1
        manifest_rows.append((speaker, split, signature, file_count))

    speaker_sets = {
        split: {speaker for speaker, assigned in assignment.items() if assigned == split}
        for split in _RESPLIT_RATIOS
    }
    for left_index, left in enumerate(speaker_sets):
        for right in list(speaker_sets)[left_index + 1:]:
            overlap = speaker_sets[left] & speaker_sets[right]
            if overlap:
                raise RuntimeError(f"Speaker leakage between {left} and {right}: {sorted(overlap)[:5]}")

    attack_speakers: dict[str, set[str]] = defaultdict(set)
    for speaker, attacks in inventory.items():
        for attack in attacks:
            attack_speakers[attack].add(speaker)
    for attack, speakers in attack_speakers.items():
        if len(speakers) < len(_RESPLIT_RATIOS):
            continue
        missing_splits = [
            split
            for split, split_speakers in speaker_sets.items()
            if not (speakers & split_speakers)
        ]
        if missing_splits:
            raise RuntimeError(
                f"Attack {attack!r} is absent from splits {missing_splits} "
                f"despite appearing for {len(speakers)} speakers"
            )

    scp_rows = []
    summary = {
        "dataset": "vsasv_resplit",
        "seed": seed,
        "ratios": _RESPLIT_RATIOS,
        "source_audio_root": str((data_root / _AUDIO_DIR).resolve()),
        "copies_audio": False,
        "splits": {},
    }
    for split, output_path in split_paths.items():
        rows = sorted(split_rows[split], key=lambda row: row[0])
        frame = pd.DataFrame(rows, columns=["utt_id", "label", "path", "speaker", "attack"])
        labels = set(frame["label"])
        if labels != _REQUIRED_LABELS:
            raise RuntimeError(f"Split {split} does not contain both labels: {sorted(labels)}")
        frame[["utt_id", "label"]].to_csv(output_path, sep="\t", index=False)
        scp_rows.extend((row.utt_id, row.path) for row in frame.itertuples(index=False))
        summary["splits"][split] = {
            "speakers": len(speaker_sets[split]),
            "utterances": len(frame),
            "labels": frame["label"].value_counts().sort_index().to_dict(),
            "attacks": frame["attack"].value_counts().sort_index().to_dict(),
        }

    pd.DataFrame(
        manifest_rows,
        columns=["speaker", "split", "attack_signature", "utterances"],
    ).sort_values(["split", "speaker"]).to_csv(manifest_path, sep="\t", index=False)
    with wav_scp_path.open("w") as handle:
        for utt_id, path in sorted(scp_rows):
            handle.write(f"{utt_id} {path}\n")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


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
    if track == "resplit":
        prepare_resplit(data_root, meta_dir, fold=fold, force=force)
        return

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
