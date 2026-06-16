"""Convert ASVspoof2019 LA/PA CM protocols into the fold*.tsv + wav.scp format
that baselines/molex/src/data_utils_NEW.py (gen_cyber_list) expects.

Protocol columns confirmed from the real LA files on the server
(space-separated, no header, 5 columns: SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID
KEY): col[1] = utterance id (matches the flac filename stem), col[4] = key
("bonafide" or "spoof").

Audio layout: ASVspoof2019_{LA,PA}_{train,dev,eval}/flac/<utt_id>.flac --
verified via `ls` for LA on the server. PA protocol files follow the same
historical 5-column convention but haven't been verified against real
headers -- check before relying on asvspoof2019pa.
"""

from pathlib import Path

import pandas as pd

_SPLIT_TAGS = {
    "train": "train.trn",
    "validation": "dev.trl",
    "evaluation": "eval.trl",
}
_AUDIO_DIR_TAGS = {"train": "train", "validation": "dev", "evaluation": "eval"}


def _read_protocol(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    return pd.DataFrame({"utt_id": df[1], "label": df[4]})


def ensure_meta(data_root: Path, meta_dir: Path, fold: int, track: str, force: bool = False) -> None:
    """Write fold{fold}_{train,validation,evaluation}.tsv + wav.scp under meta_dir.

    data_root must point at .../asvspoof2019/<track>/<track> (e.g. .../LA/LA),
    matching AGENT_TASK.md's documented layout. track is "LA" or "PA".
    """
    track = track.upper()
    data_root = Path(data_root)
    meta_dir = Path(meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)

    expected = [meta_dir / f"fold{fold}_{split}.tsv" for split in _SPLIT_TAGS] + [meta_dir / "wav.scp"]
    if not force and all(p.exists() for p in expected):
        return

    scp_lines = []
    for split, tag in _SPLIT_TAGS.items():
        protocol_path = (
            data_root / f"ASVspoof2019_{track}_cm_protocols"
            / f"ASVspoof2019.{track}.cm.{tag}.txt"
        )
        df = _read_protocol(protocol_path)
        df.to_csv(meta_dir / f"fold{fold}_{split}.tsv", sep="\t", index=False)

        flac_dir = data_root / f"ASVspoof2019_{track}_{_AUDIO_DIR_TAGS[split]}" / "flac"
        for utt_id in df["utt_id"]:
            scp_lines.append(f"{utt_id} {flac_dir / (utt_id + '.flac')}")

    with open(meta_dir / "wav.scp", "w") as f:
        f.write("\n".join(scp_lines) + "\n")
