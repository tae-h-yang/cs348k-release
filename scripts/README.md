# Scripts

This folder keeps the final helper scripts needed to rebuild slides, refresh
slide metrics, and reproduce the KIMODO/SONIC audit pipeline at a high level.

Primary entry points:

- `slides/build_slides.sh` rebuilds the HTML/PDF/PPTX deck.
- `update_slide_metrics.py` refreshes slide metric snippets from tracked CSVs.
- `evaluate_kimodo_humanoid100.py` and `repair_kimodo_humanoid100.py` run the
  KIMODO reference audit and prompt-repair helpers.
- `build_humanoid_robotics_prompt_suite.py`, `build_sports_acrobatics_prompt_suite.py`,
  and `evaluate_humanoid_100_prompts.py` define the tracked prompt suites and
  prompt-support audit used by the tests.
- `evaluate_sonic_policy_mujoco.py` and `run_sonic_native_release_batch.py`
  support SONIC/MuJoCo rollout checks.
- `render_selected_overlay_videos.py` and `make_humanoid100_video_contact_sheet.py`
  produce presentation review media.

Large raw experiment outputs are intentionally not stored here.
