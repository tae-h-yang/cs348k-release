# Metrics Snapshot

Source: tracked release assets under `slides/assets/`.

## First-Pass KIMODO Audit

| Set | Physical Pass | SONIC No-Fall | Both Gates | Mean SONIC Sec. | Mean RMSE |
|---|---:|---:|---:|---:|---:|
| all KIMODO references | 48/100 | 53/100 | 29/100 | 2.855 | 0.156 |
| physical-pass subset | 48/48 | 29/48 | 29/48 | 3.293 | 0.170 |
| flagged subset | 0/52 | 24/52 | 0/52 | 2.450 | 0.143 |

## Non-Exclusive Failure Flags

| Flag | Count | Mean SONIC Sec. |
|---|---:|---:|
| physical screen fail | 52 | 2.450 |
| SONIC fall | 47 | 1.586 |
| torque limit >1x | 66 | 2.570 |
| high root force >5kN | 28 | 1.978 |
| self-contact >8% | 34 | 2.804 |
| non-foot floor contact | 11 | 0.473 |
| low foot support | 7 | 1.877 |
| contact artifact >0.45 | 15 | 2.120 |
| floor penetration >8cm | 10 | 0.390 |

## Deterministic Repair Snapshot

| Metric | Original | Repaired |
|---|---:|---:|
| physical pass | 48/100 | 53/100 |
| critic accept | 47/100 | 54/100 |
| reject / regenerate | 18/100 | 16/100 |
| SONIC 4s no-fall | 53/100 | 56/100 |
| mean SONIC seconds | 2.855 | 3.007 |
| mean RMSE | 0.156 | 0.142 |
| mean risk | 41.548 | 37.916 |
| contact artifact | 0.264 | 0.247 |

Repair evidence comes from `docs/kimodo_repair_results_2026-05-31.md`: seven SONIC rescues and four regressions.
