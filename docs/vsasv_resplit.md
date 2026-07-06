# VSASV speaker-disjoint resplit

`vsasv_resplit` is a deterministic, non-destructive alternative to the legacy
`vsasv` evaluation-only metadata.

- Audio remains under the original `dataset-16` tree.
- No waveform is copied, moved, renamed, or modified.
- Speakers are grouped by their available attack categories.
- Each attack-signature group is deterministically assigned 70/10/20 to
  train/validation/evaluation using seed `20260706`.
- The three splits are speaker-disjoint.
- `speaker_split_manifest.tsv` records the assignment and attack signature.
- `split_summary.json` records speaker, label, attack, and utterance counts.

Prepare and audit:

```bash
python tools/prepare_vsasv_resplit.py --force
```

Generated metadata:

```text
outputs/vsasv_resplit/meta/vsasv_resplit/
  fold1_train.tsv
  fold1_validation.tsv
  fold1_evaluation.tsv
  wav.scp
  speaker_split_manifest.tsv
  split_summary.json
```

Any baseline using the unified CLI can evaluate the new held-out split with:

```bash
python main.py --baseline <baseline> --mode eval \
  --dataset vsasv_resplit --ckpt <checkpoint>
```

The legacy `vsasv` dataset ID and its existing results are unchanged.
