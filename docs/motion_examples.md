# MotionBricks Prompt Examples and Video Map

The current public MotionBricks G1 preview used in this project is a
mode/control interface, not a full arbitrary text-to-motion endpoint. For the
checkpoint, each "prompt" below is therefore a readable description attached to
one exposed MotionBricks mode plus seed/control settings. The complete 105-row
suite is in `configs/prompt_suite_105.csv`.

## Why These Motions

The suite was chosen to stress different failure modes for generated humanoid
references:

- Static idle: should already be easy; tests whether resampling can regress a
  stable reference.
- Locomotion: core robot use case; tests forward travel, speed, and balance.
- Directional locomotion: tests whether the screen preserves requested travel
  direction.
- Expressive locomotion: tests upper-body style while walking, where visual
  plausibility can conflict with torque/contact demand.
- Whole-body low motion: deliberately hard crawling cases with non-foot
  contacts and low support, useful for exposing limitations.

## Representative Prompts

| Category | MotionBricks mode | Full prompt text | Why include it |
| --- | --- | --- | --- |
| Static | `idle` | Stand upright in a relaxed idle pose without traveling. | Low-risk control case; best-of-K should usually avoid unnecessary changes. |
| Locomotion | `walk` | Walk forward at a natural pace with balanced arm swing. | Basic forward walking reference for the main risk-reduction claim. |
| Locomotion | `slow_walk` | Walk forward slowly and carefully. | Tests whether slower references are actually lower demand. |
| Locomotion | `injured_walk` | Walk forward with an injured uneven gait while staying upright. | Asymmetric gait stress case for tracking and contact. |
| Directional | `walk_left` | Travel to the robot's left while maintaining a walking gait. | Tests preservation of requested lateral direction. |
| Directional | `walk_right` | Travel to the robot's right while maintaining a walking gait. | Companion lateral-direction case. |
| Expressive | `walk_boxing` | Walk forward while throwing boxing-style arm motions. | Upper-body motion can increase torque and self-contact risk. |
| Expressive | `walk_happy_dance` | Walk forward with a happy dance style and energetic arms. | High visible style variation; good qualitative example. |
| Expressive | `walk_gun` | Walk forward while holding the arms in a pretend aiming pose. | Tests constrained arm posture during locomotion. |
| Expressive | `walk_scared` | Walk forward in a scared style with guarded upper-body motion. | Tests style preservation under risk screening. |
| Whole-body | `hand_crawling` | Crawl forward on hands with a very low whole-body posture. | Hard low-posture contact case; often remains high risk. |
| Whole-body | `elbow_crawling` | Crawl forward on elbows with a very low whole-body posture. | Hard non-foot-contact case for limitation analysis. |

## MotionBricks Reference Videos

These videos were generated locally for the visual result of K=1 vs K=8
inverse-dynamics screening. The large MP4 files are omitted from this
lightweight GitHub push, but the intended artifact names are:

- `artifacts/videos/risk_explainer/walk_seed0_K1_vs_K8_risk_explainer.mp4`
- `artifacts/videos/risk_explainer/walk_happy_dance_seed0_K1_vs_K8_risk_explainer.mp4`
- `artifacts/videos/risk_explainer/hand_crawling_seed0_K1_vs_K8_risk_explainer.mp4`
- `artifacts/videos/comparison/walk_happy_dance_seed0_K1_vs_K8.mp4`
- `artifacts/videos/comparison/walk_boxing_seed0_K1_vs_K8.mp4`

The risk-explainer videos are the clearest checkpoint examples because they show
the motion, risk timeline, risk components, and torque-limited joints together.
For the GitHub checkpoint page, see
`artifacts/video_placeholders/walk_seed0_risk_explainer_preview.png` and
`artifacts/video_placeholders/hand_crawling_risk_preview.png` as lightweight
still-frame placeholders for those videos.

## SONIC Tracking Videos

These videos were generated locally for the controller-validation audit. The
large MP4 files are omitted from this lightweight GitHub push, but the intended
artifact names are:

- `artifacts/videos/sonic_policy/injured_walk_seed2_K1.mp4`
- `artifacts/videos/sonic_policy/injured_walk_seed2_K8.mp4`
- `artifacts/videos/sonic_policy/walk_boxing_seed3_K1.mp4`
- `artifacts/videos/sonic_policy/walk_boxing_seed3_K8.mp4`
- `artifacts/videos/sonic_policy_multik/slow_walk_seed1_K1.mp4`
- `artifacts/videos/sonic_policy_multik/slow_walk_seed1_K8.mp4`
- `artifacts/videos/sonic_policy_multik/walk_boxing_seed0_K1.mp4`
- `artifacts/videos/sonic_policy_multik/walk_boxing_seed0_K16.mp4`

These videos should be described carefully: they test whether a learned
Motion-targeting policy can track the generated references. They are not a
safety certificate. The current result is that inverse-dynamics screening is a
useful diagnostic, while controller-in-the-loop selection is the stronger next
step.
