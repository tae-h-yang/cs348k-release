# KIMODO Humanoid100 Repair Results

Date: 2026-05-31

## Goal

Apply the same deterministic reference repair used for the MotionBricks/Humanoid100
experiments to the KIMODO-generated G1 clips, then check whether physical-screen
metrics and SONIC tracking improve.

This is a test-time reference repair baseline. It does not fine-tune KIMODO or
change SONIC. The current repair family is retiming plus smoothing, selected by
the physical-awareness critic.

## Commands

```bash
python scripts/repair_kimodo_humanoid100.py \
  --out_dir results/kimodo_humanoid100_repaired_retimed \
  --data_dir data/kimodo_humanoid100_repaired_retimed \
  --export_sonic_refs
```

```bash
python scripts/evaluate_sonic_policy_mujoco.py \
  --reference_dir results/kimodo_humanoid100_repaired_retimed/sonic_references \
  --out_csv results/kimodo_humanoid100_repaired_retimed/sonic_tracking_4s_cuda.csv \
  --summary_csv results/kimodo_humanoid100_repaired_retimed/sonic_summary_4s_cuda.csv \
  --provider cuda \
  --max_seconds 4 \
  --init_reference_pose
```

CUDA provider loading failed in this shell because `libcublasLt.so.12` was not
available to ONNX Runtime, so SONIC evaluation fell back to CPU. The rollout
results completed successfully.

## Aggregate Results

| Measure | Original KIMODO | Repaired KIMODO | Delta |
| --- | ---: | ---: | ---: |
| Physical-pass clips | 48 / 100 | 53 / 100 | +5 |
| Critic `accept` actions | 47 / 100 | 54 / 100 | +7 |
| Critic `reject/regenerate` actions | 18 / 100 | 16 / 100 | -2 |
| Mean physical risk | 41.548 | 37.916 | -3.632 |
| Mean p95 torque-limit ratio | 22.727 | 22.117 | -0.610 |
| Mean contact artifact score | 0.264 | 0.247 | -0.016 |
| SONIC 4s no-fall clips | 53 / 100 | 56 / 100 | +3 |
| SONIC mean track seconds | 2.855 | 3.007 | +0.152 |
| SONIC mean tracking RMSE | 0.156 | 0.142 | -0.014 |

The improvement is real but not a solved-result. Retiming/smoothing helps a
small set of clips, but it also regresses a few clips and does not make KIMODO
references reliably executable.

## SONIC Rescue Cases

These are the seven clips where original KIMODO fell before the 4-second horizon
and repaired KIMODO completed the full 4 seconds in the SONIC MuJoCo harness.

| Clip | Prompt | Selected variant | SONIC seconds | RMSE |
| --- | --- | --- | ---: | ---: |
| `hrb_001_forward_walk` | Walk forward at a comfortable indoor pace with symmetric arm swing. | `retime_2p0x_gaussian` | 3.46 -> 4.00 | 0.163 -> 0.130 |
| `hrb_056_open_door` | Step forward, reach with the right hand as if opening a door, then pass through. | `retime_1p5x` | 2.62 -> 4.00 | 0.276 -> 0.241 |
| `hrb_068_stumble_left` | Recover from a small leftward stumble with a side step. | `retime_2p0x` | 1.92 -> 4.00 | 0.213 -> 0.201 |
| `hrb_069_stumble_right` | Recover from a small rightward stumble with a side step. | `retime_2p0x` | 2.72 -> 4.00 | 0.259 -> 0.271 |
| `hrb_070_backward_recovery` | Recover from a slight backward lean by stepping back. | `retime_2p0x` | 3.38 -> 4.00 | 0.154 -> 0.097 |
| `hrb_077_wave` | Stand still and wave the right hand at shoulder height. | `retime_2p0x_gaussian` | 3.86 -> 4.00 | 0.181 -> 0.122 |
| `hrb_097_split_squat_jump` | Perform a small split-squat jump switching which foot is forward. | `retime_2p0x_gaussian` | 3.42 -> 4.00 | 0.192 -> 0.146 |

Four clips regressed from no-fall to fall under the same 4-second screen:
`hrb_011_skip_forward`, `hrb_023_zombie_walk`, `hrb_067_stumble_forward`, and
`hrb_088_duck_under_bar`.

## Artifacts

- Main repair output: `results/kimodo_humanoid100_repaired_retimed/`
- Selected repair metrics: `results/kimodo_humanoid100_repaired_retimed/repair_summary.csv`
- Candidate metrics: `results/kimodo_humanoid100_repaired_retimed/candidate_metrics.csv`
- Aggregate plot: `results/kimodo_humanoid100_repaired_retimed/kimodo_repair_summary.png`
- Original-vs-repaired SONIC join: `results/kimodo_humanoid100_repaired_retimed/sonic_original_vs_repaired_4s_joined.csv`
- Rescue review videos:
  - Original failures: `results/kimodo_humanoid100_repaired_retimed/rescue_review/original_failed_videos/`
  - Repaired successes: `results/kimodo_humanoid100_repaired_retimed/rescue_review/repaired_videos/`

## Interpretation

This supports a narrow presentation claim: simple test-time repair improves
some KIMODO references under both dynamics-derived screening and SONIC tracking.
It does not support a strong claim that the current method produces generally
physical humanoid references. The next stronger method should use a controller-
or critic-aware optimization loop that directly penalizes SONIC fall risk and
tracking error, not only inverse-dynamics and contact heuristics.
