# Presentation Notes: MotionBricks Physical-Awareness Study

## One-Slide Thesis

MotionBricks is a fast, high-quality kinematic humanoid motion generator, but
its own limitation section points to physical awareness as future work. This
project adds a no-generator-retraining test-time screen: generate K candidate
Unitree G1 qpos clips, score each with a MuJoCo inverse-dynamics heuristic
critic, and select the lowest-risk candidate.

Use this exact framing:

> This is a heuristic inverse-dynamics screening and ranking layer before
> tracking. It is not a learned controller, not a safety guarantee, and not yet
> validated on hardware.

## Commands

```bash
# Main 39-identity K sweep
python scripts/run_ablation.py
python scripts/plot_guided_ablation.py

# Extended 105-identity K=1/K=8 run
python scripts/run_extended_ablation.py --seeds 7 --k_values 1 8
python scripts/summarize_extended_ablation.py
python scripts/plot_extended_ablation.py

# Audits and negative controls
python scripts/build_prompt_suite.py
python scripts/evaluate_prompt_alignment.py
python scripts/evaluate_contact_quality.py
python scripts/semantic_preservation.py
python scripts/evaluate_computed_torque.py
python scripts/full_candidate_audit.py --seeds 2 --K 8

# Learned-policy closed-loop audit
python scripts/export_sonic_references.py \
  --modes elbow_crawling hand_crawling idle injured_walk slow_walk stealth_walk walk walk_boxing walk_gun walk_happy_dance walk_left walk_right walk_scared walk_stealth walk_zombie \
  --seeds 7 \
  --out_dir results/sonic_references_210_fixed
python scripts/evaluate_sonic_policy_mujoco.py \
  --provider cpu \
  --reference_dir results/sonic_references_210_fixed \
  --max_seconds 5.0 \
  --out_csv results/sonic_policy_mujoco_tracking_210_fixed.csv \
  --summary_csv results/sonic_policy_mujoco_summary_210_fixed.csv
python scripts/analyze_sonic_policy_results.py \
  --tracking_csv results/sonic_policy_mujoco_tracking_210_fixed.csv \
  --summary_csv results/sonic_policy_selector_summary_fixed.csv \
  --by_clip_csv results/sonic_policy_selector_by_clip_fixed.csv \
  --plot results/sonic_policy_tracking_k1_k8_selector_fixed.png

# Videos
python scripts/render_comparison.py
python scripts/render_supplementary.py --section all
python scripts/render_risk_explainer.py --default_set
python scripts/render_sonic_policy_rollouts.py results/sonic_policy_rollouts/*.npz --out_dir results/videos/sonic_policy

# Tests
pytest -q
```

## Dataset

The main K sweep uses 13 styles x 3 seeds = 39 motion identities and evaluates
K=[1,4,8,16]. The robustness extension uses all 15 exposed G1 demo modes x 7
seeds = 105 paired K=1/K=8 mode-seed identities. This is 15 unique local mode
prompts, not 100 different behaviors.

Extended run composition:

| Motion type | Identities | Examples |
| --- | ---: | --- |
| static | 7 | `idle_seed*` |
| locomotion | 56 | `walk`, `slow_walk`, `walk_left`, `walk_right`, etc. |
| expressive | 28 | `walk_boxing`, `walk_gun`, `walk_scared`, etc. |
| whole_body | 14 | `hand_crawling`, `elbow_crawling` |

## Main Quantitative Result

Lead with:

- `results/guided_extended_k1_vs_k8.png`
- `results/guided_extended_action_counts.png`
- `results/guided_risk_vs_K.png`

Extended 105-identity result:

- Mean risk: 35.32 -> 13.58.
- Aggregate reduction: 61.55%.
- Improved pairs: 81/105.
- Critic-accepted clips: 25/105 -> 78/105.
- Reject labels: 13/105 -> 9/105.

Per type:

- Locomotion: 86.88% reduction; 49/56 critic-accepted at K=8.
- Expressive: 93.26% reduction; 28/28 critic-accepted at K=8.
- Whole-body crawling: 40.52% reduction, but 9/14 still critic-rejected.
- Static idle: -14.38% regression; resampling should be gated off when K=1 is
  already low risk.

Say plainly:

> Best-of-K works well as a test-time screen for upright and expressive
> MotionBricks clips. It does not solve whole-body crawling, and it can hurt
> already-stable idle motions.

## Audits To Mention

Prompt/task proxy preservation:

- `configs/prompt_suite_105.csv`
- `results/prompt_alignment_by_category.png`
- K=1 mean proxy alignment: 0.583.
- K=8 mean proxy alignment: 0.568.
- 56/105 pairs are within -0.05 alignment; 72/105 keep displacement ratio in
  [0.5, 1.5].
- Say explicitly: this is a mode/control prompt suite, not a full
  HumanML3D-style text-to-motion retrieval evaluation.

Contact-quality diagnostics:

- `results/contact_quality_by_category.png`
- `results/risk_vs_contact_artifacts.png`
- Mean contact artifact score: 0.274 -> 0.248.
- Whole-body crawling: 0.654 -> 0.489, but non-foot floor contact remains high
  at 34.7% for K=8.
- Use this when someone asks how to see risk in the simulation/video.

Semantic preservation:

- `results/semantic_preservation.png`
- Median displacement/path/speed ratios are near 1.0.
- 15/39 benchmark pairs change root displacement outside a 0.5x-1.5x band.

Candidate audit:

- `results/candidate_audit_summary.csv`
- 10/10 representative K=8 batches had the same segment-selected and full-score
  selected winner.
- Present this as a sanity check, not a proof for all prompts.

Controller negative controls:

- `results/computed_torque_tracking.png`
- PD tracking correlation is weak (rho=-0.09).
- Computed-torque tracking also fails uniformly at frame 15 and does not improve
  K=8 RMSE.
- `results/sonic_policy_tracking_k1_k8_selector_fixed.png`
- Corrected SONIC learned-policy MuJoCo harness over 105 K=1/K=8 pairs:
  K=1 averages 2.005 s / 0.339 rad RMSE, K=8 averages 2.054 s / 0.334 rad
  RMSE, both with 98/105 falls and 0% torque saturation. K=8 deltas are small
  and non-significant.
- A policy-aware selector over the two available candidates improves survival
  to 2.299 s and RMSE to 0.310 rad.
- `results/sonic_policy_multik_selector_fixed.png`
- Corrected 39-identity K=1/4/8/16 SONIC audit: a policy-aware selector over
  stored variants improves survival from 2.134 s to 2.833 s (p=1.89e-06) and
  RMSE from 0.328 to 0.286 rad (p=0.019), while fall count remains 36/39.
  This is the best evidence for the pivot to controller-in-the-loop screening.

Say:

> The negative controls are why I call this a screening metric, not a controller
> validation.

Updated thesis after the SONIC audit:

> Inverse-dynamics screening is a useful diagnostic, but it is not sufficient
> physical awareness. A stronger final method must use controller-in-the-loop
> screening or planner-policy co-training.

## Videos

Show risk-explainer videos first when explaining what "risk" means:

- `results/videos/risk_explainer/walk_seed0_K1_vs_K8_risk_explainer.mp4`
- `results/videos/risk_explainer/walk_happy_dance_seed0_K1_vs_K8_risk_explainer.mp4`
- `results/videos/risk_explainer/hand_crawling_seed0_K1_vs_K8_risk_explainer.mp4`

These show per-frame risk, component bars, timeline spikes, and the most
torque-limited joints. They are the best answer to "where is the risk in the
motion?"

Then show paired kinematic reference videos, not old PD rollouts:

- `results/videos/comparison/walk_happy_dance_seed0_K1_vs_K8.mp4`
- `results/videos/comparison/walk_boxing_seed0_K1_vs_K8.mp4`
- `results/videos/comparison/walk_seed1_K1_vs_K8.mp4`
- `results/videos/comparison/hand_crawling_seed0_K1_vs_K8.mp4`

Use supplementary stitched videos after regeneration:

- `results/videos/supplementary/S0_main.mp4`
- `results/videos/supplementary/S1_locomotion.mp4`
- `results/videos/supplementary/S2_expressive.mp4`
- `results/videos/supplementary/S3_wholebody_crawling.mp4`
- `results/videos/supplementary/S4_k_scaling.mp4`
- `results/videos/supplementary/S5_wc_vs_ps.mp4`

Do not show `results/videos/idle_seed0.mp4` as a result. That is an old weak-PD
rollout artifact.

For learned-policy failure/screening evidence, show:

- `results/videos/sonic_policy/injured_walk_seed2_K1.mp4`
- `results/videos/sonic_policy/injured_walk_seed2_K8.mp4`
- `results/videos/sonic_policy/walk_boxing_seed3_K1.mp4`
- `results/videos/sonic_policy/walk_boxing_seed3_K8.mp4`
- `results/videos/sonic_policy_multik/walk_boxing_seed0_K1.mp4`
- `results/videos/sonic_policy_multik/walk_boxing_seed0_K16.mp4`
- `results/videos/sonic_policy_multik/slow_walk_seed1_K1.mp4`
- `results/videos/sonic_policy_multik/slow_walk_seed1_K8.mp4`
- `results/videos/sonic_policy_multik/injured_walk_seed0_K1.mp4`
- `results/videos/sonic_policy_multik/injured_walk_seed0_K4.mp4`

## Talk Flow

1. State the MotionBricks limitation: kinematic plans need physical awareness.
2. Define the narrow contribution: inverse-dynamics critic + best-of-K sampling.
3. Show the risk formula and action thresholds.
4. Show the 39-identity K sweep: risk decreases with K.
5. Show the 105-identity extension: same effect at larger scale.
6. Show videos as qualitative inspection only.
7. Show prompt proxy, contact-quality, semantic preservation, and controller
   negative-control audits.
8. Show the SONIC learned-policy audit as the honest stress test.
9. Close with the real next step: controller-in-the-loop selection and
   planner-policy co-training.

## Expected Q&A

**Is this a safety guarantee?**  
No. It is a heuristic feasibility score based on inverse dynamics and derivative
signals.

**Did you train a motion retargeting or tracking policy?**  
No. I imported the released SONIC G1 tracking-policy weights and ran them in an
approximate MuJoCo harness. After fixing the harness, K=8 is roughly neutral
under SONIC, while a policy-aware selector is clearly better. The next step is
policy-in-the-loop generation or co-training, not more heuristic-only scoring.

**Why not just use a controller as the metric?**  
That is the right validation target. The current repo did not include a reliable
learned G1 tracking policy, and the available PD/computed-torque baselines are
too weak to discriminate clip quality.

**Does K=8 preserve the original motion?**  
Often at the mode-label level, but not always geometrically. The semantic audit
shows 15/39 benchmark pairs change root displacement substantially.

**What is the publishable next step?**  
Co-train or preference-tune the kinematic planner with tracking-policy outcomes:
the generator should learn to produce references that a policy can execute,
instead of relying only on test-time rejection sampling.
