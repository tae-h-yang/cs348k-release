# Artifact Inventory

Date: 2026-05-30

This page is the working map for generated artifacts. The `results/` directory
is intentionally git-ignored, so this document is the tracked pointer to what
exists locally and what should or should not be presented.

## Final Native SONIC Videos To Review

Use these first for the current Humanoid100 physics-execution evidence:

```text
results/ralphloop/20260529_191342/humanoid100_final_eval_k256/all100_native_sonic_release/
results/ralphloop/20260529_191342/humanoid100_final_eval_k256/all100_native_sonic_release/humanoid100_native_analysis.md
results/ralphloop/20260529_191342/humanoid100_final_eval_k256/all100_native_sonic_release/strict_presentation_pass_contact_sheet.jpg
results/ralphloop/20260529_191342/humanoid100_final_eval_k256/all100_native_sonic_release/fail_contact_sheet.jpg
```

Historical fixed-height native result: 76/100 no-fall and 66/100 strict pass
for the `sonic_verified_best` selected references. Use the reference-aware
selection table below for current claims.

Then use the failed-prompt variant sweep:

```text
results/ralphloop/20260529_191342/humanoid100_final_eval_k256/failed_prompt_native_variant_sweep/native_variant_rescue_analysis.md
results/ralphloop/20260529_191342/humanoid100_final_eval_k256/failed_prompt_native_variant_sweep/strict_presentation_pass_contact_sheet.jpg
results/ralphloop/20260529_191342/humanoid100_final_eval_k256/failed_prompt_native_variant_sweep/fail_contact_sheet.jpg
```

Historical projected result with fixed-height native verifier selection/retry
over K1/K8/K9 variants: 84/100 no-fall and 74/100 strict pass. This is kept
for provenance; use the corrected reference-aware table below for current
talk/paper numbers.

Convenience pointers:

```text
results/current_validated/native_all100_selected_release/
results/current_validated/native_failed_prompt_variant_rescue/
results/current_validated/humanoid100_native_analysis.md
results/current_validated/native_variant_rescue_analysis.md
```

Native rendering convention: white/left is the reference robot, red/right is
the SONIC-controlled robot under MuJoCo physics.

## Approximate SONIC Videos

Use these as diagnostic/legacy approximate videos, not as the first physics
result. They were generated with corrected approximate SONIC initialization:
the simulated robot starts from the reference pose with zero initial velocity.
In these videos the red translucent robot is the selected reference motion and
the solid robot is the approximate MuJoCo/SONIC rollout.

```text
results/humanoid100_final_eval/before_after_overlay_videos/*.mp4
results/humanoid100_final_eval/before_after_overlay_contact_sheet.jpg
results/humanoid100_final_eval/before_after_overlay_videos.csv
```

This folder has 100 side-by-side videos, one per Humanoid100 prompt. The left
panel is the K=1 MotionBricks baseline and the right panel is the selected
candidate from the verifier/repair pipeline.

For final-only inspection use:

```text
results/humanoid100_final_eval/final_100_selected_overlay_videos/*.mp4
results/humanoid100_final_eval/final_100_selected_overlay_contact_sheet.jpg
results/humanoid100_final_eval/final_100_selected_overlay_videos.csv
```

For baseline-only inspection use:

```text
results/humanoid100_final_eval/k1_baseline_overlay_videos/*.mp4
results/humanoid100_final_eval/k1_baseline_overlay_videos.csv
```

The index CSVs include initialization audit columns. The latest validation
found 100/100 videos in each set, max initial qpos error `0.0`, and max initial
qvel norm `0.0`.

Older representative-only endpoint:

```text
results/humanoid100_final_eval/final_selector/
```

Key files:

```text
results/humanoid100_final_eval/final_selector/README.md
results/humanoid100_final_eval/final_selector/final_selector_summary.png
results/humanoid100_final_eval/final_selector/representative_contact_sheet.jpg
results/humanoid100_final_eval/final_selector/representative_videos/*.mp4
results/humanoid100_final_eval/final_selector/selected_methods.csv
results/humanoid100_final_eval/final_selector/selector_summary.csv
```

This is no longer the first video folder to review because it only contains 27
representative videos. It remains useful for selector tables and plots. Keep
the claim bounded: it is an inference-time verification/repair result over
MotionBricks proxy references, not a trained generator and not semantic success
for the unsupported 78/100 prompts.

Older official SONIC sanity-check videos remain useful for proving that the
SONIC install itself can run:

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

## Current 100-Prompt Native SONIC Artifacts

Use these for the repeat-conservative MotionBricks verifier-selection result:

```text
results/ralphloop/20260529_191342/humanoid100_final_eval_k256/final_100_native_selection_ref_aware.csv
results/ralphloop/20260529_191342/humanoid100_final_eval_k256/final_100_native_selection_ref_aware.md
results/ralphloop/20260529_191342/humanoid100_final_eval_k256/final_100_native_selection_ref_aware.png
results/ralphloop/20260529_191342/humanoid100_final_eval_k256/all100_native_sonic_release/batch_summary_ref_aware.csv
results/ralphloop/20260529_191342/humanoid100_final_eval_k256/failed_prompt_native_variant_sweep/batch_summary_ref_aware.csv
results/ralphloop/20260529_191342/humanoid100_final_eval_k256/failed_prompt_native_variant_sweep/native_variant_rescue_analysis.md
```

Current repeat-conservative headline: native verifier selection over original
plus K1/K8/K9 variants reaches 100/100 reference-aware no-fall and 74/100
strict tracking. A deep retry batch contains one additional strict pass in its
first rollout, but the same clip did not repeat as strict in the diagnostic
contact render, so keep it exploratory rather than headline.
The 100/100 number is a survival/screening claim, not a perfect prompt or pose
accuracy claim.

Latest targeted K1024 native SONIC evidence:

```text
results/ralphloop/20260529_191342/humanoid100_final_eval_k256/final_100_native_selection_ref_aware_k1024_targeted.csv
results/ralphloop/20260529_191342/humanoid100_final_eval_k256/final_100_native_selection_ref_aware_k1024_targeted.md
results/ralphloop/20260529_191342/humanoid100_final_eval_k256/final_100_native_selection_ref_aware_k1024_targeted.png
results/ralphloop/20260530_003531/humanoid100_final_eval_k1024/nonstrict_k1024_native_sonic/batch_summary.csv
results/ralphloop/20260530_003531/humanoid100_final_eval_k1024/nonstrict_k1024_native_sonic/targeted_native_rescue.md
```

Targeted headline, with repeat-stability caveat: K1024 sampling plus native
selection rescues 10 of the 26 non-strict prompts and projects 84/100 strict
tracking while preserving 100/100 reference-aware no-fall. Treat this as the
best current result, but keep the 74/100 table as the repeat-conservative
claim because the K1024 rescues have not all been rerun multiple times.

Negative follow-up probes for the remaining 16 hard prompts:

```text
results/ralphloop/20260530_003531/humanoid100_final_eval_k1024/retimed_nonstrict_native_sonic/targeted_native_rescue.md
results/ralphloop/20260530_003531/humanoid100_final_eval_k1024/angvel_corrected_native_sonic/targeted_native_rescue.md
results/ralphloop/20260530_003531/humanoid100_final_eval_k1024/angvel_corrected_native_sonic/contact_sheet.jpg
results/ralphloop/20260530_003531/humanoid100_final_eval_k1024/upright_safe_native_sonic/contact_sheet_partial.jpg
```

Retiming/smoothing added 0/16 strict rescues. Recomputing root angular velocity
from body quaternions also added 0/16 strict rescues. A partial upright-safe
projection run added 0/8 strict rescues before it was stopped. The remaining
failures are concentrated in floor/low-posture and acrobatic-stress prompts;
they need a contact/state-conditioned retargeter or a tracker/generator trained
for those contact modes, not more blind K-sampling.

Controller-manifold projection baseline:

```text
results/ralphloop/20260530_003531/humanoid100_final_eval_k1024/sonic_projected_references/
results/ralphloop/20260530_003531/humanoid100_final_eval_k1024/sonic_projected_native_sonic/batch_summary.csv
results/ralphloop/20260530_003531/humanoid100_final_eval_k1024/sonic_projected_native_sonic/targeted_native_rescue.md
results/ralphloop/20260530_003531/humanoid100_final_eval_k1024/sonic_projected_native_sonic/contact_sheet.jpg
results/ralphloop/20260530_003531/humanoid100_final_eval_k1024/sonic_projected_iter2_native_sonic/
results/ralphloop/20260530_003531/humanoid100_final_eval_k1024/sonic_projected_iter3_native_sonic/
results/ralphloop/20260530_003531/humanoid100_final_eval_k1024/sonic_projected_iter4_native_sonic/
results/ralphloop/20260529_191342/humanoid100_final_eval_k256/final_100_native_selection_ref_aware_k1024_projected_iter4.csv
results/ralphloop/20260529_191342/humanoid100_final_eval_k256/final_100_native_selection_ref_aware_k1024_projected_iter4.md
results/ralphloop/20260529_191342/humanoid100_final_eval_k256/final_100_native_selection_ref_aware_k1024_projected_iter4.png
```

This baseline extracts the actual native SONIC MuJoCo qpos from the first
failed rollout, re-exports it as a new reference, and reruns the native tracker.
Projection iteration 1 rescues 8/16 remaining hard prompts, iteration 2 rescues
burpee, iteration 3 rescues pushup, and iteration 4 adds no new strict rescues.
The integrated experimental table is 94/100 strict with 100/100
reference-aware no-fall. This is the strongest execution result, but it is not
a prompt-preserving MotionBricks generation result: semantic/style fidelity must
be audited because the reference has been projected onto what the controller
already executed.
