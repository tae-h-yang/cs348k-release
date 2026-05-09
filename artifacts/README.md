# Artifacts

This directory contains the checkpoint-facing visual examples and optional
diagnostic outputs. The README uses the lightweight snapshots directly; large
videos and tarball payloads are documented but not committed.

## Example Snapshots

The `example_snapshots/` directory contains lightweight still frames embedded
directly in the README:

- `example_snapshots/motionbricks/`: generated MotionBricks reference snapshots.
- `example_snapshots/sonic/`: SONIC tracking-policy audit snapshots.

These are included to make the project concrete for readers who have not seen
MotionBricks or the controller harness before.

## Optional Diagnostic Plots

The `plots/` directory contains diagnostic outputs from local exploratory runs.
They are useful for checking the pipeline, but the Week 6 checkpoint does not
depend on any final result number.

## Optional Videos

Large videos were generated locally but are omitted from this lightweight
GitHub push:

- `videos/risk_explainer/`: videos with risk timelines and component bars.
- `videos/comparison/`: paired kinematic reference videos for quick inspection.
- `videos/sonic_policy/`: learned-policy tracking examples.
- `videos/sonic_policy_multik/`: K-sweep learned-policy tracking examples.

Videos are qualitative inspection aids. They do not prove physical execution.

## Video Placeholders

The `video_placeholders/` directory contains still frames from generated videos
so the checkpoint page has a visual preview even without the large MP4 files:

- `video_placeholders/walk_seed0_risk_explainer_preview.png`: preview of the
  risk-explainer layout for a walking clip.
- `video_placeholders/hand_crawling_risk_preview.png`: preview of a harder
  low-posture crawling example.

## CSV Summaries

`results/` contains compact CSV files from local exploratory runs. The full
per-run result tree exists in the local release bundles.

## Full Bundles

The full bundles exist locally in the development repo under `release_assets/`.
They are not committed in this lightweight GitHub push. If the tarballs are
available locally, verify/extract the curated artifact bundle with:

```bash
cd ..
shasum -a 256 -c release_assets/cs348k_artifacts_2026-05-08.tar.zst.sha256
tar --zstd -xf release_assets/cs348k_artifacts_2026-05-08.tar.zst
```

Full local results plus MotionBricks data:

```bash
cd ..
shasum -a 256 -c release_assets/cs348k_full_results_motionbricks_2026-05-08.tar.zst.sha256
tar --zstd -xf release_assets/cs348k_full_results_motionbricks_2026-05-08.tar.zst
```

Use GNU `sha256sum -c` instead of `shasum -a 256 -c` on Linux if preferred.
