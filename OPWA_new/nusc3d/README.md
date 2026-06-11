# nusc3d — nuScenes 3D Conformal Prediction

## Structure

```
nusc3d/
├── cp_core/                # CP utility code (ported from exp0)
│   ├── calibration.py      # q̂ = np.quantile(scores, 0.90)
│   ├── evaluation.py       # bin stats + coverage/gap computation
│   ├── config.py           # CP parameters only (ALPHA, BIN_EDGES)
│   └── decision_tree.py    # collapse判定阈值逻辑
├── exp0_outputs/           # 2D Cityscapes results (fallback reference)
│   ├── exp0_results.json
│   ├── exp0_acdc_results.json
│   ├── miou_table.json
│   └── ... (all figures + JSON from exp0)
├── exp1_outputs/           # 2D PS-CRC / in-domain results (fallback)
│   ├── calib_bin_stats.json
│   ├── diag_A_sky_stats.json
│   └── ... (all diagnosis outputs)
└── README.md
```

## Porting Guide

| Source (OPWA_v3/exp0) | Destination | Changes |
|---|---|---|
| `calibration.py` | `cp_core/calibration.py` | Stripped SegFormer, Cityscapes, PIL. `compute_q_hat()` is pure numpy. |
| `evaluation.py` | `cp_core/evaluation.py` | Parametric bin names; no hardcoded `BIN_NAMES`/`TARGET_COVERAGE` import. |
| `config.py` | `cp_core/config.py` | Kept only CP + bin params. Removed CITYSCAPES_ROOT, FOGGY_ROOT, DAV2 paths, label mapping. |
| `decision_tree.py` | `cp_core/decision_tree.py` | Ported to class-based; accepts dict (not file path). No file I/O. |

## Not ported (2D-specific, rewrite for 3D)

- `model_loader.py` → use nuScenes LiDAR model
- `data_utils.py` → use nuScenes SDK file matching
- `gt_utils.py` → use nuScenes lidarseg labels
- `transmittance.py` → 3D uses physical distance, not DCP
- `inference.py` → rewrite for point cloud inference

## Fallback assets

`exp0_outputs/` and `exp1_outputs/` contain 2D Cityscapes experiment outputs.
These serve as:
1. **Reference baselines** — what phenomenon looks like in 2D
2. **Paper fallback** — if 3D results are insufficient, 2D Cityscapes + ACDC results can still form a paper
