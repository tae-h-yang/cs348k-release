# Release Assets

Large generated artifacts were stored here in the local release repo via Git
LFS. This lightweight GitHub checkpoint push omits the tarball payloads because
the LFS upload was unreliable. The checksum files remain as identifiers for the
local bundles.

Local curated bundle:

```text
release_assets/cs348k_artifacts_2026-05-08.tar.zst
release_assets/cs348k_artifacts_2026-05-08.tar.zst.sha256
```

Local full-results bundle:

```text
release_assets/cs348k_full_results_motionbricks_2026-05-08.tar.zst
release_assets/cs348k_full_results_motionbricks_2026-05-08.tar.zst.sha256
```

Contents include:

- final SONIC actual-qpos videos from `results/sonic_actual_sim_examples_pose_aligned/`
- released-root SONIC videos from `results/sonic_actual_sim_examples_released/`
- generated plots and CSV summaries from `results/`
- supplementary, comparison, risk-explainer, and SONIC policy videos from `results/videos/`
- neural critic checkpoints/logs/plots
- paper PDF and key documentation

If the tarballs are available locally, restore with:

```bash
git lfs pull
tar --zstd -xf release_assets/cs348k_artifacts_2026-05-08.tar.zst
sha256sum -c release_assets/cs348k_artifacts_2026-05-08.tar.zst.sha256
```

The extracted top-level folder is `artifact_bundle/`.

Restore the full local results and MotionBricks data:

```bash
git lfs pull
tar --zstd -xf release_assets/cs348k_full_results_motionbricks_2026-05-08.tar.zst
sha256sum -c release_assets/cs348k_full_results_motionbricks_2026-05-08.tar.zst.sha256
```

The full bundle extracts `results/` and `data/motionbricks/`.
