# SONIC Native Sim2Sim Check

Date: 2026-05-05

## What Was Verified

The official SONIC MuJoCo simulator environment was missing initially. Running:

```bash
cd /home/rewardai/repos/GR00T-WholeBodyControl
bash install_scripts/install_mujoco_sim.sh
```

created `.venv_sim` and installed the official sim dependencies (`mujoco`, `tyro`, `pin`, `unitree_sdk2py`, `msgpack-numpy`, etc.).

The native C++ deployment binary then successfully initialized against the official reference set:

```bash
cd /home/rewardai/repos/GR00T-WholeBodyControl/gear_sonic_deploy
LD_LIBRARY_PATH=/home/rewardai/anaconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/home/rewardai/anaconda3/lib/python3.12/site-packages/tensorrt_libs:/home/rewardai/repos/GR00T-WholeBodyControl/gear_sonic_deploy/thirdparty/unitree_sdk2/lib/x86_64:/home/rewardai/repos/GR00T-WholeBodyControl/gear_sonic_deploy/thirdparty/unitree_sdk2/thirdparty/lib/x86_64:$LD_LIBRARY_PATH \
  ./target/release/g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example/ \
  --obs-config policy/release/observation_config.yaml \
  --encoder-file policy/release/model_encoder.onnx \
  --planner-file planner/target_vel/V2/planner_sonic.onnx \
  --input-type keyboard \
  --output-type zmq \
  --disable-crc-check \
  --enable-csv-logs
```

Observed evidence:

- 13 official reference motions loaded from `gear_sonic_deploy/reference/example/`.
- Decoder, encoder, and planner TensorRT engines initialized successfully.
- Policy observation dimension matched: `994`.
- Encoder observation dimension matched: `1762`.
- Native deploy received simulator `LowState` with millisecond-scale age after control start.
- The first official reference, `forward_lunge_R_001__A359_M`, was played to completion in the native deploy process.

## New Local Tools

Project-side helpers added for native SONIC checks:

- `scripts/run_sonic_sim_autodrop.py`: starts the official MuJoCo sim in headless mode, can release the elastic support by timer or trigger file, and can log the simulator's real `mj_data.qpos/qvel`.
- `scripts/record_sonic_realtime_debug.py`: records native `g1_debug` ZMQ output as a quick target-vs-measured MP4. This is now marked as diagnostic-only because measured root translation in the C++ debug payload is hard-coded.
- `scripts/render_sonic_actual_sim_examples.py`: launches one official reference at a time, logs actual simulator `qpos`, optionally releases the support band immediately before playback, and renders reference-vs-actual videos offline from the recorded MuJoCo state.

## Artifacts

Generated native-debug video (diagnostic only; see limitation below):

```text
results/sonic_native_check/run2_forward_lunge_tracking.mp4
```

Additional official SONIC examples rendered from the same native debug path:

```text
results/sonic_native_examples/00_forward_lunge_R_001__A359_M.mp4
results/sonic_native_examples/01_walking_quip_360_R_002__A428.mp4
results/sonic_native_examples/02_squat_001__A359.mp4
results/sonic_native_examples/04_neutral_kick_R_001__A543.mp4
results/sonic_native_examples/07_dance_in_da_party_001__A464.mp4
results/sonic_native_examples/08_macarena_001__A545_M.mp4
results/sonic_native_examples/contact_sheet.jpg
```

Batch command:

```bash
python scripts/render_sonic_native_examples.py --out_dir results/sonic_native_examples
```

Corrected actual-simulator-qpos videos, with elastic support released before playback:

```text
results/sonic_actual_sim_examples_released/forward_lunge_R_001__A359_M_actual_sim_qpos.mp4
results/sonic_actual_sim_examples_released/walking_quip_360_R_002__A428_actual_sim_qpos.mp4
results/sonic_actual_sim_examples_released/squat_001__A359_actual_sim_qpos.mp4
results/sonic_actual_sim_examples_released/neutral_kick_R_001__A543_actual_sim_qpos.mp4
results/sonic_actual_sim_examples_released/dance_in_da_party_001__A464_actual_sim_qpos.mp4
results/sonic_actual_sim_examples_released/macarena_001__A545_M_actual_sim_qpos.mp4
results/sonic_actual_sim_examples_released/contact_sheet_actual_qpos.jpg
results/sonic_actual_sim_examples_released/actual_qpos_summary.csv
```

Pose-aligned review videos, also rendered from actual simulator `qpos`, keep both robots visible even when the actual root drifts or travels:

```text
results/sonic_actual_sim_examples_pose_aligned/forward_lunge_R_001__A359_M_actual_sim_qpos.mp4
results/sonic_actual_sim_examples_pose_aligned/walking_quip_360_R_002__A428_actual_sim_qpos.mp4
results/sonic_actual_sim_examples_pose_aligned/squat_001__A359_actual_sim_qpos.mp4
results/sonic_actual_sim_examples_pose_aligned/neutral_kick_R_001__A543_actual_sim_qpos.mp4
results/sonic_actual_sim_examples_pose_aligned/dance_in_da_party_001__A464_actual_sim_qpos.mp4
results/sonic_actual_sim_examples_pose_aligned/macarena_001__A545_M_actual_sim_qpos.mp4
results/sonic_actual_sim_examples_pose_aligned/contact_sheet_pose_aligned.jpg
results/sonic_actual_sim_examples_pose_aligned/actual_qpos_summary.csv
```

Corrected batch commands:

```bash
python scripts/render_sonic_actual_sim_examples.py \
  --out_dir results/sonic_actual_sim_examples_released \
  --release_before_play \
  --release_settle 1.0

python scripts/render_sonic_actual_sim_examples.py \
  --out_dir results/sonic_actual_sim_examples_pose_aligned \
  --release_before_play \
  --release_settle 1.0 \
  --align_mode per_frame
```

Generated native logs:

```text
results/sonic_native_check/run2_logs/
results/sonic_native_check/run2_target_motion.csv
results/sonic_native_check/run2_policy_input.csv
```

## Important Limitation

The previous Python-only `evaluate_sonic_policy_mujoco.py` harness is not a trustworthy SONIC evaluation path: it fails even on SONIC's official example references. Use it only as a rough diagnostic until it is reconciled with the official simulator/deploy stack.

The native `g1_debug` stream also has a critical visualization limitation. In `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/output_interface/output_interface.hpp`, `base_trans_measured` is fixed to `{0.0, -1.0, 0.793}`. Therefore any MP4 rendered from `g1_debug` can show a measured/red robot with a fake fixed/floating root. Those videos are useful for checking that messages are published and joints are moving, but they are not valid evidence for root tracking, falling, or whole-body physical execution.

The corrected `sonic_actual_sim_examples_*` videos render actual `mj_data.qpos` from the official simulator. The `released` set preserves root drift/travel from the simulator. The `pose_aligned` set independently recenters each robot every frame for easier local pose comparison. Use both together: pose-aligned videos are easier to inspect visually, while `actual_qpos_summary.csv` reports root height and root XY travel.

For MotionBricks-to-SONIC conversion, the current exporter writes the 29 joint positions and root body pose in the format needed by the `g1` encoder observations, but it is still not a full official-style reference package: official SONIC example folders include 14 tracked bodies (`body_pos.csv` width 42, `body_quat.csv` width 56), while `scripts/export_sonic_references.py` currently writes root-only body data. Treat previous MotionBricks SONIC videos/metrics as approximate until the exporter is upgraded to produce full 14-body kinematics and validated against the actual-qpos native path above.
