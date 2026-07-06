# M2 adaptation to VSASV

This experiment adapts the best ASVspoof5-trained SEE-MoLEx M2 checkpoint to
the speaker-disjoint `vsasv_resplit` protocol.

It follows the domain-adaptation setting in MoLEx Table VII:

- retain the 12 source experts in every MoE layer;
- append 2 new experts per layer (14 total);
- initialize all matching weights and the first 12 router rows from the source
  checkpoint;
- freeze the source experts, WavLM backbone, attentive merger, and classifier;
- train only the 2 new experts and the expanded entropy routers;
- select the checkpoint using VSASV validation EER;
- evaluate the held-out VSASV split only after checkpoint selection.

The paper found that two added experts produced the main adaptation gain, while
four added experts gave only marginal improvement. It also found that replaying
10% of source-domain data reduced forgetting. This first configuration does not
mix source data, so ASVspoof5 should be re-evaluated after adaptation and
reported as a source-retention result.

## New run

```bash
CKPT=outputs/see_molex/M2/2026_06_23_17_51_31/weights/epoch_3_0.864.pth

CUDA_VISIBLE_DEVICES=0 MOLEX_NUM_GPU=1 python -u main.py \
  --baseline see_molex \
  --ablation M2 \
  --mode train \
  --dataset vsasv_resplit \
  --config configs/see_molex_m2_vsasv_adapt.yaml \
  --init-ckpt "$CKPT" \
  --seed 1234
```

Adaptation runs are isolated under:

```text
outputs/see_molex/M2/adaptations/vsasv_resplit/<timestamp>/
```

## Resume

```bash
CUDA_VISIBLE_DEVICES=0 MOLEX_NUM_GPU=1 python -u main.py \
  --baseline see_molex \
  --ablation M2 \
  --mode train \
  --dataset vsasv_resplit \
  --config configs/see_molex_m2_vsasv_adapt.yaml \
  --resume <timestamp> \
  --seed 1234
```

Do not combine `--resume` and `--init-ckpt`: initialization starts a new
optimizer and schedule, while resume restores an interrupted adaptation run.
