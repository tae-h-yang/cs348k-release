# Rendering Failure Notes 2026-05-31

Purpose: prevent repeated mistakes when making final SONIC presentation videos.

## What Went Wrong

- I treated a new "clean overlay" renderer as a cosmetic change, but it changed the coordinate/model assumptions.
- I replayed native SONIC simulator qpos through a simplified 29-DoF two-robot render model. Some native logs include qpos from a different SONIC sim model/layout, so the rendered G1 mesh can appear to float or hang.
- I also briefly reversed the visual convention: red was rendered on the physics robot rather than the reference ghost.
- I tried to force same-origin overlay while the duplicated robot XML still had hidden body offsets, creating misleading spatial separation.
- I checked only that videos opened/contact sheets existed, not that the root trajectory, foot height, and red/white identity were correct.

## Do Not Repeat

- Do not make final claims from newly styled render videos until checking actual frames.
- Do not trust a render if the physical robot's feet are visibly off the ground while native SONIC logs or original videos looked grounded.
- Do not mix qpos from one SONIC/MuJoCo model with another render XML unless the qpos layout and root-height convention have been verified.
- Do not use the broken archived/experimental clean-overlay folders as presentation artifacts.
- Do not present raw candidate-cache filenames like `_motionbricks_all_redghost/hrb_074_heel_rock_K9.mp4` as prompt-labeled outputs. Those names identify selected candidates, not final prompt labels.
- Do not label a video as success/failure from MP4 length after `--disable_early_termination`; use the original metric CSVs and copied prompt-named folders.

## Correct Source Of Truth

- For MotionBricks, use native SONIC deploy outputs from:
  `results/ralphloop/20260529_191342/humanoid100_final_eval_k256/all100_native_sonic_release/`
- For Kimodo, use the Kimodo SONIC evaluation outputs and metric tables from:
  `results/kimodo_humanoid100_full_kimodo100_full_20260530_200838/eval/`
- When making curated folders, copy or hardlink the known-good evaluated videos first. Only re-render if the render is validated frame-by-frame against the source logs/videos.
- Current prompt-named presentation root:
  `results/presentation_redghost_all100/`
- Current prompt-matched failure/success pair root:
  `results/presentation_redghost_all100/failure_success_pairs/`

## Minimum Visual QA Before Calling A Video Final

- Open the contact sheet and at least one MP4 per folder.
- Confirm red/ghost/reference identity and physical robot identity.
- Confirm the physical robot mesh is not floating.
- Confirm contact markers are near the floor/feet when contact is expected.
- Confirm failure filenames match metric evidence in `index.csv` or the source metric table.
- Confirm contact sheets for the destination folders before telling the user they are final.
