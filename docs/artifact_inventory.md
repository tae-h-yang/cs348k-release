# Artifact Inventory

Date: 2026-05-07

This page is the working map for generated artifacts. The `results/` directory
is intentionally git-ignored, so this document is the tracked pointer to what
exists locally and what should or should not be presented.

## Final SONIC Videos To Review

Use these first:

```text
results/sonic_actual_sim_examples_pose_aligned/
```

This folder contains six official SONIC examples rendered from actual MuJoCo
simulator `mj_data.qpos`, with each robot root re-centered per frame so the
reference and policy robot stay visible for pose comparison.

Key files:

```text
results/sonic_actual_sim_examples_pose_aligned/contact_sheet_pose_aligned.jpg
results/sonic_actual_sim_examples_pose_aligned/actual_qpos_summary.csv
results/sonic_actual_sim_examples_pose_aligned/forward_lunge_R_001__A359_M_actual_sim_qpos.mp4
results/sonic_actual_sim_examples_pose_aligned/walking_quip_360_R_002__A428_actual_sim_qpos.mp4
results/sonic_actual_sim_examples_pose_aligned/squat_001__A359_actual_sim_qpos.mp4
results/sonic_actual_sim_examples_pose_aligned/neutral_kick_R_001__A543_actual_sim_qpos.mp4
results/sonic_actual_sim_examples_pose_aligned/dance_in_da_party_001__A464_actual_sim_qpos.mp4
results/sonic_actual_sim_examples_pose_aligned/macarena_001__A545_M_actual_sim_qpos.mp4
```

Use this companion folder when root drift/travel matters:

```text
results/sonic_actual_sim_examples_released/
```

These videos also come from actual MuJoCo simulator `qpos`, with the elastic
support released before playback, but preserve root drift in the render. They
are harder to inspect when the robot travels out of frame, so use them as
physics/root-motion evidence rather than as the primary human-facing video set.

## Archived SONIC Outputs

These folders were moved out of the main result view but kept intact:

```text
results/archive/2026-05-07_sonic_debug_legacy/
```

Contains old native `g1_debug` videos. Do not present these as tracking or
fall evidence because SONIC's C++ debug publisher hard-codes measured base
translation.

```text
results/archive/2026-05-07_sonic_supported_probe/
```

Contains an early actual-qpos probe where the support band stayed enabled.
Useful only for debugging the qpos recorder.

```text
results/archive/2026-05-07_motionbricks_sonic_approx/
```

Contains older MotionBricks-to-SONIC videos from the approximate Python/native
conversion path. Do not present these as final SONIC evidence until the
MotionBricks exporter is upgraded to a full official-style reference package
and validated with the actual-qpos native path.

## Tracked Code And Docs

The reproducible pieces to push are code, tests, configs, paper, and docs. The
large generated `results/` artifacts are local outputs unless we decide to use
Git LFS, an external artifact store, or a release bundle.

Relevant SONIC scripts:

```text
scripts/run_sonic_sim_autodrop.py
scripts/render_sonic_actual_sim_examples.py
scripts/record_sonic_realtime_debug.py
scripts/render_sonic_native_examples.py
```

Relevant SONIC docs:

```text
docs/sonic_native_check.md
docs/artifact_inventory.md
claude.md
```
