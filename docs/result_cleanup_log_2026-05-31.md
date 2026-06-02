# Result Cleanup Log 2026-05-31

Purpose: keep the final result tree easier to inspect without deleting source evidence needed for reproducibility.

## Removed

- `results/presentation_redghost_representative/`
  - Reason: obsolete representative subset from before the final all-100 prompt-named red-ghost render.
  - Replacement: `results/presentation_redghost_all100/`.

## Kept

- `results/presentation_redghost_all100/`
  - Current user-facing final presentation video root.
- `results/presentation_redghost_all100/failure_success_pairs/`
  - Prompt-matched failure-to-success examples.
- `results/ralphloop/`
  - Large but kept because it contains source experiment artifacts and native SONIC MotionBricks evaluation data.
- `results/kimodo_humanoid100_full_kimodo100_full_20260530_200838/`
  - Kept because it contains Kimodo qpos, metrics, and source evaluation tables.

## Current Warning

Do not use `_motionbricks_all_redghost/` directly for presentation. It is an internal cache with selected-candidate filenames, not prompt-facing filenames.
