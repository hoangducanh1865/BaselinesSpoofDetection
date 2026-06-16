"""Convert ASVspoof5 Track 1 protocols into the fold*.tsv + wav.scp format
that baselines/molex/src/data_utils_NEW.py (gen_cyber_list) expects.

Protocol columns confirmed from the real files on the server (space-separated,
no header, 10 columns): col[1] = utterance id (matches the flac filename
stem), col[8] = key ("bonafide" or "spoof" -- present consistently in that
position for both classes, even though col[7] holds either an attack tag
like "A11" or, for genuine speech, the literal string "bonafide"). The other
columns (gender, codec, VC source utterance, etc.) aren't needed for CM-only
training.

Audio layout: flac_T/<utt_id>.flac (train), flac_D/<utt_id>.flac (dev),
flac_E_eval/<utt_id>.flac (eval) -- verified via `ls` on the server.
"""

from pathlib import Path

import pandas as pd

_PROTOCOL_FILES = {
    "train": "ASVspoof5.train.tsv",
    "validation": "ASVspoof5.dev.track_1.tsv",
    "evaluation": "ASVspoof5.eval.track_1.tsv",
}
_FLAC_DIRS = {
    "train": "flac_T",
    "validation": "flac_D",
    "evaluation": "flac_E_eval",
}


def _read_protocol(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    return pd.DataFrame({"utt_id": df[1], "label": df[8]})


def ensure_meta(data_root: Path, meta_dir: Path, fold: int, track=None, force: bool = False) -> None:
    """Write fold{fold}_{train,validation,evaluation}.tsv + wav.scp under meta_dir.

    track is accepted (and ignored) for interface parity with
    datasets.asvspoof2019.ensure_meta -- ASVspoof5 has a single track here.
    """
    data_root = Path(data_root)
    meta_dir = Path(meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)

    expected = [meta_dir / f"fold{fold}_{split}.tsv" for split in _PROTOCOL_FILES] + [meta_dir / "wav.scp"]
    if not force and all(p.exists() for p in expected):
        return

    scp_lines = []
    for split, protocol_name in _PROTOCOL_FILES.items():
        df = _read_protocol(data_root / "protocols" / protocol_name)
        df.to_csv(meta_dir / f"fold{fold}_{split}.tsv", sep="\t", index=False)

        flac_dir = data_root / _FLAC_DIRS[split]
        for utt_id in df["utt_id"]:
            scp_lines.append(f"{utt_id} {flac_dir / (utt_id + '.flac')}")

    with open(meta_dir / "wav.scp", "w") as f:
        f.write("\n".join(scp_lines) + "\n")
