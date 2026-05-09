# Inverse-Dynamics-Critic-Guided Inference-Time Scaling for Humanoid Motion Generation

**CS 348K — Visual Computing Systems, Stanford University, Spring 2026**

---

## Abstract

Neural kinematic planners for humanoid robots such as MotionBricks can generate
expressive motion at interactive rates, but their own authors identify a
critical gap: generated plans may be physically implausible or exceed hardware
torque limits. This work measures one narrow test-time screening response to
that gap. We treat the generator as a stochastic sampler: by enabling Gumbel
pose-token sampling and varying the VQ-VAE conditioning seed, we draw K
candidate clips per task/request and select a low-risk one using a MuJoCo
inverse-dynamics heuristic critic. This is best-of-K inference scaling applied
to kinematic humanoid motion screening: the same generator is used, but extra
test-time samples are ranked by an explicit dynamics-derived score.

On a 13-style x 3-seed x 4-K ablation over the Unitree G1 (156 selected clips),
best-of-K reduces mean inverse-dynamics risk from 38.90 (K=1) to 15.93 (K=8),
a 59% reduction (Wilcoxon p<1e-6, Cohen's d=1.09), and changes the heuristic
action labels from 5 heuristic-accepted clips at K=1 to 26 at K=8. A larger K=1/K=8
extension over 15 styles x 7 seeds = 105 motion identities reproduces the
effect: mean risk falls from 35.32 to 13.58, heuristic-accepted clips increase
from 25 to 78, and 81/105 paired identities improve. A 105-row mode-seed audit
over 15 unique local mode prompts shows K=8 broadly preserves trajectory/style
proxies (mean 0.568 vs 0.583 for K=1), while a contact-quality audit improves
artifact score from 0.274 to 0.248. Locomotion achieves 86-87% aggregate risk reduction
and expressive styles 93-94% at K=8. Gaussian
smoothing and temporal retiming post-processes do not match this improvement
(best-of-smoothing mean risk 41.02; best-of-retiming 23.74 vs WC-K8's 15.93).
A diversity source ablation reveals that Gumbel stochastic sampling is the
primary diversity driver (21.15 mean risk alone vs seed diversity at 23.75),
with synergy for expressive motions when combined (3.65 vs 7.87 Gumbel-only).
We also report negative validation results: neither a weak PD controller nor a
stronger torque-limited computed-torque controller shows per-clip tracking
improvement from lower critic scores. We additionally built and exercised the
released SONIC G1 tracking-policy stack and ran its encoder/decoder weights in
a MuJoCo closed-loop harness. After fixing three implementation bugs in the
harness (wrong PD update rate, wrong robot initialization, wrong joint
permutation in export), the corrected full 105-pair evaluation achieves 0%
torque saturation. K=1 averages 2.005 s before fall with 0.339 rad RMSE, while
K=8 averages 2.054 s with 0.334 rad RMSE; 98/105 clips fall in both groups.
The K=8 deltas are small and non-significant despite a large inverse-dynamics
risk reduction. Therefore, our claims are about heuristic inverse-dynamics
screening and ranking plus evidence for controller-in-the-loop selection, not
real-robot tracking feasibility.

A follow-up SONIC audit over the 39-identity K=1/4/8/16 sweep tests a limited
controller-aware selector. Choosing the variant with the longest SONIC survival
improves mean survival from 2.134s at K=1 to 2.833s (Wilcoxon p=1.89e-06) and
lowers RMSE from 0.328 to 0.286 rad (p=0.019); fall count remains 36/39. This
is the main pivot: inverse-dynamics screening is useful as a diagnostic, but a
serious physical-awareness method must put the tracking policy in the selection
or training loop.

---

## 1. Introduction

Whole-body humanoid motion generation has advanced dramatically. MotionBricks
[1] demonstrates a real-time, composable kinematic planner with smart
locomotion and object-interaction primitives. The public preview used in this
repo exposes a Unitree G1 mode/control interface rather than the full arbitrary
text authoring stack. The MotionBricks limitation section explicitly names
physical-awareness integration as future work: kinematic planners may produce
plans that exceed hardware torque limits or require physically impossible
ground reactions.

This project makes the identified gap concrete and measurable. We instrument
the MotionBricks generation pipeline with a MuJoCo inverse-dynamics heuristic
critic that scores each candidate trajectory using required joint torques, root
wrench, and joint-space derivatives. We then exploit a design choice of the
generator to create diversity: MotionBricks uses a VQ-VAE codebook of reference
pose tokens [1]; changing the conditioning seed selects a different reference,
and enabling Gumbel sampling (vs. default argmax) varies the token sequence
within a reference. Together, these provide two sources of kinematic candidate
diversity per task/request.

**Best-of-K inference scaling.** The main finding is that spending K times the inference compute—generating K candidates and keeping the lowest-risk one—produces large, consistent reductions under the inverse-dynamics heuristic with no model retraining. This is directly analogous to best-of-N (BoN) sampling for language model reasoning [6,7], where a verifier or reward model selects the best candidate from a fixed generator. Cobbe et al. [6] first systematically characterized BoN for LLM math reasoning; we adapt this principle to motion generation, replacing the reward model with an explicit dynamics-derived score and the token sequence with a kinematic trajectory. We characterize this screening behavior across K values and motion categories.

**Per-segment steering.** A natural extension is to apply critic guidance at each
generation step rather than over complete clips. We implement per-segment online
steering: at every MotionBricks generation decision point, we save the agent
state, sample K segment candidates, score each, and commit the best. During this
implementation we observe that MotionBricks GPU inference is not bitwise
deterministic across calls, even with the same random seed. We do not isolate a
single kernel-level cause; we treat this as a practical reproducibility finding
and compare steering methods at matched compute.

**Contributions:**
1. Empirical characterization of best-of-K inference scaling for
   inverse-dynamics-ranked humanoid motion generation with no generator or
   controller retraining in the final selector.
2. Finding that K=4-8 captures most of the available heuristic risk reduction
   on the tested upright motions, while crawling remains high-demand; verified
   on both a 39-identity K-sweep and a 105-identity K=1/K=8 extension.
3. Per-segment online steering implementation and compute-matched comparison
   against whole-clip best-of-K.
4. Characterization of MotionBricks non-bitwise-deterministic GPU inference in
   this generation pipeline.
5. Side-by-side comparison videos and semantic-preservation metrics for K=1
   vs. K=8 trajectories.

---

## 2. Related Work

**Inference-time scaling.** Best-of-N sampling [6] is now a standard technique for language models: draw N completions and select the one ranked highest by a reward model. Constitutional AI and process reward models [7] apply this idea step-by-step through chain-of-thought. Our work applies the same principle to motion generation, replacing the reward model with a physics simulator and the token sequence with a kinematic trajectory.

**Neural kinematic motion generation.** MotionBricks [1] uses a VQ-VAE over motion bricks and an autoregressive transformer to generate humanoid whole-body motions in real time. PHC [2] and other physics-based character controllers demonstrate that kinematic reference tracking is feasible but sensitive to reference quality—physically implausible references lead to tracking failure. Recent Unitree G1 systems make this gap explicit: CLAW [10] composes kinematic primitives and then uses a low-level controller in MuJoCo to obtain physically grounded trajectories, while RoboForge [11] couples motion generation with physical plausibility optimization. Our final selection method is narrower: a no-generator-retraining pre-screening/ranking layer for an existing kinematic generator.

**Inverse dynamics for feasibility screening.** Inverse dynamics has been used in biomechanics and robotics to estimate joint torques from motion capture [4]. Using MuJoCo's `mj_inverse` as a pre-filter for learned-policy rollouts extends this to neural motion planners: compute implied force demand without forward simulation, cheaply flag infeasible plans.

**Test-time compute optimization.** Test-time compute scaling has emerged as a distinct research direction [7,8]: for a fixed-parameter model, spending more inference compute can match or exceed the quality of a larger trained model. Snell et al. [7] show that optimal inference-time compute allocation outperforms simply scaling model size for many reasoning tasks. We apply this perspective to MotionBricks-style humanoid reference screening.

---

## 3. Method

### 3.1 Robot and Motion Representation

We work with the Unitree G1 humanoid as modeled in the GR00T-WholeBodyControl repository [1]. The MuJoCo model has 29 actuated hinge joints plus a 6-DOF free root, giving 36-dimensional qpos. All clips are generated at 30 fps with a range of 150–200 frames (5–7 seconds).

Motion styles cover four categories: **static** (idle, 1 style), **locomotion** (6 walking variants), **expressive** (4 upper-body emotive styles), and **whole-body** (2 crawling styles).

Scope clarification: the final method is candidate reranking, not MotionBricks
fine-tuning and not motion retargeting. Neural surrogate critics are trained as
side experiments to imitate the heuristic score, but they do not update the
MotionBricks generator and are not used as the final selector because
clip-level neural selection underperforms the heuristic.

### 3.2 Physics Critic

The critic uses MuJoCo's exact-replay inverse dynamics (`mj_inverse`) to compute the joint torques required to execute each frame-to-frame transition with zero tracking error. Six signals are computed per clip:

- p95 torque-to-limit ratio (joint actuator saturation)
- p95 unactuated root force (ground reaction proxy)
- p95 unactuated root torque
- p95 joint velocity
- p95 joint acceleration
- mean joint jerk

These are aggregated into a scalar heuristic risk score:

```
risk = 35·excess(p95 τ/limit, 1.0)
     + 20·excess(p95 F_root, 10000 N)
     + 15·excess(p95 M_root, 2000 Nm)
     + 10·excess(p95 ω, 10 rad/s)
     + 10·excess(p95 α, 120 rad/s²)
     + 10·excess(mean jerk, 1200 rad/s³)
```

where `excess(v, t) = log1p(max(v-t, 0) / t)`. Clips receive heuristic action
labels from the implementation:

- `critic_accept`: risk <= 25 and p95 torque/limit <= 1.25,
- `critic_review`: risk <= 80 and p95 torque/limit <= 5.0, but not critic-accepted,
- `critic_reject`: risk > 80 or p95 torque/limit > 5.0.

These labels are critic labels only; they are not hardware safety labels.

A fast **online segment critic** (`src/physics_eval/online_critic.py`) uses the same formula on short windows (~32 frames) for generation-time scoring.

### 3.3 Diversity Sources in MotionBricks

MotionBricks generates trajectories autoregressively in segments. At each generation step the model selects pose tokens, which by default uses argmax sampling (`pose_token_sampling_use_argmax: True`). Two intervention points create diverse candidates:

1. **Seed diversity:** the `random_seed` field in control signals indexes into the VQ-VAE codebook, selecting a different reference pose frame. Different seeds produce structurally diverse candidates within the same motion style.

2. **Stochastic sampling:** setting `pose_token_sampling_use_argmax: False` enables Gumbel sampling, producing variation within a single seed's target reference. We implement this via a monkey-patch of `inferencer.predict` that injects the config key before each call.

### 3.4 Whole-Clip Best-of-K (WC-K)

For each (style, seed) pair, we generate K candidate clips: candidate 0 is
deterministic (k=0, fixed seed, argmax), candidates 1..K-1 use the same base
seed but with incremental offsets (seed + k*137) and stochastic Gumbel sampling.
The implementation selects the candidate with the lowest fast segment-critic
score, then reports the full-clip heuristic risk of the selected output. This is
important: the main ablation is **segment-critic-guided selection evaluated by
the full critic**, not an oracle that saves and full-scores every candidate. We
add a candidate-level audit to measure segment-vs-full winner agreement.

This is computationally equivalent to running the generator K times. For a 200-frame clip with ~12 generation steps, WC-K uses K×12 inference calls.

**Baseline design note.** Our K=1 baseline is deterministic (argmax, fixed seed).
Comparing K>1 stochastic sampling against K=1 deterministic sampling conflates
sample count with sampling mode. We therefore include diversity-source ablations
and describe K=1 as the original deterministic baseline, not as a pure
sample-count control.

### 3.5 Per-Segment Online Steering (PS-K)

Rather than generating K full clips and selecting one, per-segment steering applies the critic at each individual generation step. Implementation:

1. Before each generation step, save agent state: `{mujoco_qpos buffer, model_features, frame_index, mode}`.
2. Try K candidates (k=0 deterministic, k=1..K−1 stochastic with offset seeds).
3. Score each candidate using the transition-aware segment critic: prepend the last 8 committed frames to the new segment before scoring, capturing inter-segment jerk.
4. Commit the lowest-risk candidate **only if** it beats k=0 by ≥25% margin (conservative steering—prevents noisy small-margin switches that can compound into worse full trajectories).
5. Restore the winner's agent state; the next step conditions on the committed segment.

This costs K×n_segments inference calls—the same budget as WC-K for the same K and clip length, enabling a fair compute-matched comparison.

### 3.6 MotionBricks GPU Non-Determinism (New Finding)

During per-segment steering development, we discovered that MotionBricks GPU inference is **not deterministic across calls**, even with identical `torch.manual_seed(seed)` initialization. Two consecutive `generate_clip(seed=0)` calls produce different qpos outputs starting from the first generation step (frame 16), with a maximum difference of ~0.85 rad.

Likely cause: GPU libraries often use non-bitwise-deterministic kernels for
parallel reductions and related operations. We do not isolate the exact
operation inside MotionBricks. The practical conclusion is that repeated
generation calls are not exactly reproducible under the default setup.

**Methodological consequence:** Per-segment steering (PS-K) cannot be fairly compared against a "true K=1 deterministic baseline"—the baseline itself differs across runs. The correct comparison is **compute-matched**: PS-K4 vs WC-K4, both using ~48 inference calls for a 200-frame clip.

---

## 4. Results

### 4.1 Whole-Clip Best-of-K Ablation

The primary experiment runs K ∈ {1, 4, 8, 16} on 13 motion styles × 3 seeds = 39 clips per K value (156 clips total). Results are in `results/guided_ablation_full.csv`.

**Overall results (39 clips, bootstrap 95% CI, Wilcoxon signed-rank vs K=1):**

| K | Mean risk | 95% CI | Critic-accept/39 | vs K=1 | Wilcoxon p | Cohen's d |
|---|---:|---:|---:|---:|---:|---:|
| 1 (deterministic baseline) | 38.90 | [27.6, 51.6] | 5 | — | — | — |
| 4 | 23.24 | [11.3, 37.0] | 20 | −40% | p<0.001 (32/39 improved) | 0.72 |
| 8 | 15.93 | [5.3, 29.3] | 26 | −59% | p<0.001 (34/39 improved) | **1.09** |
| 16 | 13.61 | [3.5, 26.5] | 31 | −65% | p<0.001 (34/39 improved) | 1.14 |

All comparisons against K=1 are highly statistically significant (p<10⁻⁶, one-sided Wilcoxon signed-rank). Cohen's d=1.09 for K=8 indicates a large effect size by conventional standards (d>0.8). K=4 captures most of the gain. K=8 provides a further 31% relative improvement over K=4 before diminishing returns become apparent at K=16.

**By motion type at K=8 (Wilcoxon signed-rank within type, bootstrap 95% CI):**

| Type | K=1 mean risk | K=8 mean risk [95% CI] | Reduction | Critic-accept K=8 | Wilcoxon p | n |
|---|---:|---:|---:|---:|---:|---:|
| Locomotion | 21.9 | 3.0 [1.2, 5.2] | **86%** | 15/18 | p<0.001 | 18 |
| Expressive | 24.5 | 1.4 [0.0, 3.1] | **94%** | 11/12 | p=0.001 | 12 |
| Whole-body crawling | 135.1 | 87.3 [40.6, 129.8] | 35% | 0/6 | p=0.016 | 6 |
| Static | 6.1 | 8.6 [8.6, 8.6] | −41% | 3/3 | n.s. (p=1.0) | 3 |

**Whole-body:** Crawling remains high-risk under the critic at all K values.
Even though the absolute reduction is significant (p=0.016), mean risk=87.3 is
far above the critic-accept range. We interpret this narrowly: under the current MuJoCo
model, critic weights, and tested sampling budget, crawling has much higher
inverse-dynamics demand than upright motion. We do not claim a proof of
fundamental hardware impossibility.

**Static regression:** The −41% apparent regression (K=1=6.1 vs K=8=8.6) is a real, not trivially dismissed, finding. Static clips (idle pose, n=3) are already near-zero risk at K=1 (all three individual K=1 values: 1.0, 1.4, 15.9). Gumbel stochastic sampling at K>1 injects variance into an otherwise stable motion, occasionally producing worse clips that score higher. The Wilcoxon test is non-significant (p=1.0, n=3 clips), so we cannot conclude K=8 is worse than K=1 for static motions, but neither can we claim improvement. This is a genuine boundary condition: **best-of-K with stochastic candidates should not be applied to already-stable, near-zero-risk motions**. A production system should first check K=1 risk; if risk < threshold, skip resampling.

**Figures:**
- `results/guided_risk_vs_K.png` — mean risk vs K overall and by type (main result figure)
- `results/guided_action_counts_vs_K.png` — critic-action bar chart vs K
- `results/guided_k1_vs_k8_scatter.png` — per-clip risk K=1 vs K=8 scatter, colored by type

**105-identity extension.** To address sample-size concerns, we also evaluate
K=1 and K=8 on all 15 exposed G1 demo modes with 7 seeds per mode
(`results/guided_ablation_extended.csv`). This yields 105 paired motion
identities and 210 selected clips.

| Setting | n | Mean risk | Median risk | Critic-accept | Critic-review | Critic-reject |
|---|---:|---:|---:|---:|---:|---:|
| K=1 | 105 | 35.32 | 21.28 | 25 | 67 | 13 |
| K=8 | 105 | 13.58 | 0.00 | 78 | 18 | 9 |

Paired aggregate reduction is 61.55%, with 81/105 identities improved. Per-type
aggregate reductions are 86.88% for locomotion (49/56 heuristic-accepted at K=8), 93.26%
for expressive motions (28/28 heuristic-accepted), 40.52% for whole-body crawling
(still 9/14 critic-rejected), and -14.38% for static idle clips. This larger run
confirms the main conclusion and sharpens the boundary condition: best-of-K is
useful for upright/generated action clips, should be gated off for already
stable idle clips, and is not enough to solve crawling under the current critic.

Additional figures:
- `results/guided_extended_k1_vs_k8.png` — 105-identity scatter and type bars
- `results/guided_extended_action_counts.png` — action-label shifts by type

**105-row mode-seed prompt/control preservation.** The current public MotionBricks G1
preview used in this repo exposes discrete demo modes and steering controls,
not a fully arbitrary text-to-motion endpoint. To avoid overstating the
interface, we define a 105-row mode-seed suite (`configs/prompt_suite_105.csv`):
15 unique exposed G1 mode prompts x 7 seeds. This is not 100 different
behaviors. Each row has a natural-language task description, category, target
speed, target direction, and expected tags. We evaluate
prompt/task preservation with trajectory and style proxies
(`results/prompt_alignment.csv`), not learned text-motion retrieval.

| Setting | n | Mean alignment | Median alignment | Mean speed error | Mean displacement |
|---|---:|---:|---:|---:|---:|
| K=1 | 105 | 0.583 | 0.544 | 0.266 m/s | 2.416 m |
| K=8 | 105 | 0.568 | 0.562 | 0.245 m/s | 2.356 m |

The paired mean alignment delta is -0.015; 56/105 pairs stay within -0.05 of
the K=1 proxy score, and 72/105 keep displacement ratio in [0.5, 1.5]. This
supports a narrow claim: K=8 lowers inverse-dynamics risk while broadly
preserving local task proxies. It does **not** support a general claim that
screening improves language alignment. Expressive and low-posture tasks are the
weakest categories under these simple proxies, so they should be highlighted as
open evaluation work.

Additional figures:
- `results/prompt_alignment_by_category.png`
- `results/risk_vs_prompt_alignment.png`

**Contact-quality audit.** We add contact and support diagnostics on the same
210 K=1/K=8 clips (`results/contact_quality.csv`). The metrics include
self-contact frames after excluding adjacent links, max self-contact
penetration, non-foot floor contact, foot speed while contact persists, double
support, and a simple projected-COM support proxy.

| Setting | n | Contact artifact score | Self-contact frames | Non-foot floor contact | p95 foot-contact speed |
|---|---:|---:|---:|---:|---:|
| K=1 | 105 | 0.274 | 2.64% | 6.27% | 1.08 m/s |
| K=8 | 105 | 0.248 | 1.53% | 4.62% | 0.90 m/s |

Whole-body crawling improves from 0.654 to 0.489 contact artifact score, but
remains clearly problematic: non-foot floor contact is still 34.7% at K=8. This
is valuable for the presentation because it makes "risk" visually inspectable
in videos and figures, but it is still a diagnostic rather than a controller
success metric.

Additional figures:
- `results/contact_quality_by_category.png`
- `results/risk_vs_contact_artifacts.png`

**Candidate-level audit.** Because the main implementation selects candidates
with the fast online segment critic before evaluating the chosen clip with the
full critic, we reran a representative K=8 audit and saved every candidate
(`results/candidate_audit.csv`). The audit covers 10 style/seed batches across
locomotion, expressive, and whole-body categories. In this subset, the
segment-critic winner and the full-critic winner agree on 10/10 batches with
zero full-risk gap (`results/candidate_audit_summary.csv`). This does not prove
agreement for all prompts, but it addresses the immediate concern that the main
ablation might be selecting a different candidate than the full score would
have selected.

### 4.2 Qualitative Video Comparison

Side-by-side kinematic comparison videos (K=1 left, K=8 right) are in `results/videos/comparison/`. Representative results:

| Clip | K=1 risk | K=8 risk | Improvement |
|---|---:|---:|---:|
| `walk_happy_dance_seed0` | 35.9 | 0.0 | −100% |
| `walk_boxing_seed0` | 22.2 | 0.0 | −100% |
| `walk_seed1` | 14.5 | 0.0 | −100% |
| `injured_walk_seed0` | 10.3 | 0.0 | −100% |
| `walk_seed0` | 43.3 | 14.4 | −67% |
| `hand_crawling_seed0` | 161.0 | 140.7 | −13% (still rejected) |

The videos are useful for inspecting qualitative behavior, but they do not by
themselves prove semantic preservation. We therefore compute root displacement,
root path length, average speed, and joint-trajectory RMSE between K=1 and K=8
(`results/semantic_preservation.csv`). Median displacement, path-length, and
speed ratios are near 1.0 overall, but 15/39 pairs change root displacement by
more than a 0.5x-1.5x band. The safe claim is that K=8 often preserves the broad
mode label while sometimes changing motion scale substantially.

For presentation, we also render **risk-explainer videos** in
`results/videos/risk_explainer/`. These videos make the heuristic visible as
per-frame timeline spikes, risk-component bars, and the current top
torque-limited joints. They are intended for interpretability of the metric,
not as additional validation.

### 4.3 Compute-Matched Comparison: WC-K4 vs PS-K4

We compare two strategies at equal compute cost (~48 inference calls for a 200-frame clip with ~12 generation steps):

- **WC-K4:** generate 4 full clips independently, keep the one with the lowest online segment-critic score, then evaluate the selected clip with the full critic.
- **PS-K4:** at each generation step, try 4 segment candidates, score with a transition-aware window (last 8 frames + new segment), commit the best if it beats k=0 by ≥25% margin.

Results on 13 styles × 3 seeds = 39 clips (`results/steered_vs_wc_ablation.csv`):

| Strategy | Mean risk | 95% CI | Critic-accept/39 | Critic-reject/39 | Wilcoxon vs WC |
|---|---:|---:|---:|---:|---:|
| WC-K4 | **19.61** | [8.9, 32.7] | **22** | 4 | — |
| PS-K4 | 34.35 | [19.4, 51.4] | 18 | 6 | p=0.0003, d=0.61 |

**WC-K4 strictly lower risk on 25/39 clips; PS-K4 strictly lower on 14/39.** The Wilcoxon signed-rank test (p=0.0003) confirms WC-K4 produces significantly lower risk at equal compute, with a medium effect size (Cohen's d=0.61). The binomial test (p=0.054) is marginal at one-sided α=0.05, reflecting that while the direction is clear, the sample size is modest. The Wilcoxon test—which accounts for the magnitude of differences, not just direction—is the appropriate test here.

By motion type:

| Type | WC-K4 risk | WC-K4 critic-accepts | PS-K4 risk | PS-K4 critic-accepts |
|---|---:|---:|---:|---:|
| Static | 6.1 | 1/3 | 31.6 | 0/3 |
| Locomotion | **6.7** | 12/18 | 8.3 | **14/18** |
| Expressive | **4.1** | **9/12** | 18.9 | 4/12 |
| Whole-body | 96.1 | 0/6 | 145.0 | 0/6 |

WC-K4 clearly dominates except on locomotion critic-accept count, where PS-K4 achieves 14/18 vs WC-K4's 12/18. This is a meaningful exception: locomotion is highly repetitive and each segment is more independent, so greedy local decisions are less harmful. For expressive motions that require globally coordinated upper-body movement, whole-clip selection is dramatically better (WC: 4.1 vs PS: 18.9 mean risk).

**Interpretation.** The whole-clip strategy wins because: (1) the full-clip critic is better calibrated than the segment critic (which scores 20–60 for clips the full critic marks critic-accept); (2) whole-clip selection has a global view while segment steering makes irrevocable local decisions; (3) CUDA non-determinism (Section 3.6) means the k=0 "natural" candidate in per-segment steering is already a different trajectory than the standalone K=1 baseline, so switching to stochastic candidates is noisier.

For locomotion specifically, segment-level steering may eventually match or beat whole-clip given a better-calibrated segment critic—motivating the neural surrogate critic as Week 2 work.

### 4.4 Post-Hoc Baselines: Smoothing and Retiming

We compare WC-K8 against two classes of post-hoc baselines that do not generate new clips.

**Gaussian and Savitzky-Golay velocity smoothing** (`results/smoothing_baseline.csv`):

| Variant | Mean risk | Critic-accept/39 |
|---|---:|---:|
| K=1, no post-process | 38.90 | 5 |
| Best per-clip smoothing (Gaussian σ=1.5, SavGol-11, SavGol-21) | 41.02 | 8 |
| **WC-K8 (this work)** | **15.93** | **26** |

Smoothing makes risk *worse* on average (41.02 vs 38.90). Gaussian and Savitzky-Golay filters remove high-frequency components of the trajectory, which can reduce jerk-related penalties, but they also alter the intended motion kinematics—changing velocities and accelerations in ways that may increase torque demand at other joints. Best-of-K (15.93, 26 critic-accepts) is 63% better than the best smoothing baseline.

**Retiming and seed reranking** (from earlier experiments):

| Variant | Mean risk | Critic-accept/39 |
|---|---:|---:|
| Original clips (K=1) | 35.43 | 8 |
| Best of {original, slow-1.5×, slow-2×, smooth} | 23.74 | 17 |
| Seed reranking (best of 3 seeds) | 23.00 | — |
| **WC-K8 (this work)** | **15.93** | **26** |

Best-of-K outperforms retiming by 33% and almost doubles critic-accept count. Retiming helps by slowing down motions (reducing required torques through lower velocities) but changes the artistic intent of the motion. Best-of-K achieves better physical quality at the original timing, which is strictly preferable.

### 4.5 Critic Validation and Tracking Negative Controls

To partially validate the heuristic critic, we ran individual-level Spearman correlation against a weak PD forward-simulation controller:

| Signal pair | Spearman ρ | p-value |
|---|---:|---:|
| Risk score vs. PD time-to-fall | −0.09 | 0.59 |
| Risk score vs. PD tracking RMSE | −0.09 | 0.58 |

Individual-level correlation is not significant—the weak PD controller fails quickly for all clip types (31–119 frames) and cannot discriminate quality at the individual level.

However, group-level ordering is preserved:

| Type | Critic mean risk | PD mean time-to-fall |
|---|---:|---:|
| Static | 5.8 | 75 frames |
| Locomotion | 7.8 | 59 frames |
| Expressive | 9.2 | 55 frames |
| Whole-body | 109.7 | 48 frames |

The critic assigns whole-body crawling the highest demand category. Per-clip
validation requires a stronger tracking controller.

We also test a stronger torque-limited computed-torque baseline that uses
inverse-dynamics feedforward plus PD feedback (`scripts/evaluate_computed_torque.py`).
This negative control does **not** validate the critic: all 78 K=1/K=8 rollouts
fall at frame 15 (0.5 s), K=8 does not improve tracking RMSE, and the K=8
Spearman correlation between critic risk and tracking RMSE is rho=-0.078
(p=0.637). This confirms that the available controllers in this repository are
not adequate retargeting-policy evaluations. The strongest future validation is
a reliable native learned-policy deployment or real robot rollout.

We built the public SONIC G1 deployment stack and ran the released SONIC
encoder/decoder weights in a MuJoCo closed-loop harness
(`scripts/evaluate_sonic_policy_mujoco.py`). The native C++ binary waits for
live Unitree `LowState` DDS messages; the Python harness supplies the same
observation tensors from simulated G1 state, matching the observation layout
confirmed against `g1_deploy_onnx_ref.cpp`.

**Harness correctness note (2026-05-05).** Three implementation bugs in an
earlier version of the harness caused all clips to fall immediately (0.3–0.7 s,
30–40% torque saturation) regardless of clip quality, making it impossible to
distinguish K=1 from K=8. These bugs were identified and fixed:

1. *PD update rate*: torque was recomputed only once per 50 Hz policy step
   rather than at each physics substep. In training and on hardware, PD runs at
   the physics timestep (≈200 Hz). Fixing this alone dropped torque saturation
   from ~35% to 0% and increased mean track time ~5×.

2. *Robot initialization*: the robot was placed at the reference motion's
   frame-0 pose (knees nearly fully extended, body_q ≈ −0.67 from default).
   The real deployment initializes the robot at the DEFAULT standing pose before
   tracking begins. MotionBricks frame 0 is a mid-stride configuration outside
   the policy's training-time starting distribution.

3. *Export permutation*: `export_sonic_references.py` used `ISAACLAB_TO_MUJOCO`
   instead of `MUJOCO_TO_ISAACLAB` to reorder joints, scrambling all columns.
   All references were re-exported after the fix.

**Corrected small smoke test (20 clips, 5 modes × 2 seeds × K=1/K=8, max 5 s):**

| Group | n | Falls | Mean track time | Mean joint RMSE | Torque sat. |
|---|---:|---:|---:|---:|---:|
| K=1 | 10 | 10 | 2.07 s | 0.317 rad | 0% |
| K=8 | 10 | 10 | 1.81 s | 0.401 rad | 0% |
| K=8 − K=1 (paired) | 10 | — | −0.26 s | +0.084 rad | — |

Best individual clip: `walk_happy_dance_seed0_K1` tracked for 4.78 s. Side-by-side
videos (policy simulation vs reference replay) are in `results/sonic_videos/`.

All clips still fall within the 5-second window. K=8 clips have slightly
shorter tracking time and higher RMSE than K=1. The direction is interpretable:
K=8 selects more dynamic motions with larger joint excursions, which are harder
to track from a default-standing initialization. Importantly, this is now a
genuine negative result from a correctly functioning harness rather than a
measurement artifact. It remains the strongest evidence that inverse-dynamics-
only screening is not sufficient; the next step must incorporate the tracking
policy in the selection or training loop.

**Corrected full 105-pair audit (15 modes × 7 seeds × K=1/K=8, max 5 s):**

| Group | n | Falls | Mean track time | Mean joint RMSE | Torque sat. |
|---|---:|---:|---:|---:|---:|
| K=1 | 105 | 98 | 2.005 s | 0.339 rad | 0% |
| K=8 | 105 | 98 | 2.054 s | 0.334 rad | 0% |
| K=8 − K=1 (paired) | 105 | — | +0.049 s | −0.0056 rad | — |
| SONIC selector over K=1/K=8 | 105 | 98 | 2.299 s | 0.310 rad | 0% |

K=8 still reduces inverse-dynamics heuristic risk strongly
(`mean_full_risk_delta=-21.74`), but the learned-policy deltas are small and
non-significant (`p=0.159` for greater survival, `p=0.457` for lower RMSE).
The policy-aware selector over the two available candidates improves survival
and RMSE (`p=1.75e-10` and `p=0.0169`), which supports controller-in-the-loop
selection as the serious next method.

**Corrected 39-identity K=1/4/8/16 audit:**

| Group | n | Falls | Mean track time | Mean joint RMSE | Torque sat. |
|---|---:|---:|---:|---:|---:|
| K=1 | 39 | 36 | 2.134 s | 0.328 rad | 0% |
| K=4 | 39 | 36 | 2.132 s | 0.347 rad | 0% |
| K=8 | 39 | 36 | 2.193 s | 0.329 rad | 0% |
| K=16 | 39 | 36 | 2.129 s | 0.315 rad | 0% |
| SONIC selector over K=1/4/8/16 | 39 | 36 | 2.833 s | 0.286 rad | 0% |

Direct inverse-dynamics K values are not monotonic under SONIC, but the
controller-aware oracle over stored variants improves survival by +0.699 s
(`p=1.89e-06`) and RMSE by −0.042 rad (`p=0.019`). This is not a deployable
generator; it is an upper-bound audit showing that policy outcomes contain
useful selection information not captured by inverse dynamics alone.

### 4.6 Neural Surrogate Critic

To enable real-time K=8 generation, we train a lightweight 1D CNN surrogate that predicts the heuristic risk score from a qpos window, replacing the computationally expensive `mj_inverse` scoring.

**Architecture:** 4-layer causal 1D CNN mapping (36, 32) qpos windows → scalar log-risk. 162k parameters. Input shape: (batch, 36 joints, 32 frames).

**Training data:** 8,910 sliding windows (stride=4) extracted from all 234 labeled clips (156 WC ablation + 78 PS ablation). Labels are heuristic risk scores computed offline. 85%/15% train/val split. Labels are log-transformed (log1p) before training to address right-skew.

**Important caveat — circularity:** The neural critic is trained on heuristic labels and evaluated on heuristic labels. Spearman ρ=0.919 means the CNN accurately mimics the heuristic—not that the heuristic is physically correct. The neural critic inherits all limitations of the heuristic critic, including the unvalidated individual-level discrimination noted in Section 4.5. Its value is speed, not additional physical validity.

**Results after 80 epochs on RTX 4060:**

| Metric | Value | Target |
|---|---:|---:|
| Spearman ρ (val set, neural vs heuristic) | **0.919** | ≥ 0.85 ✓ |
| Inference time (GPU, single window) | **0.174 ms** | < 0.5 ms ✓ |
| Parameters | 162,241 | < 200k ✓ |
| Val Huber loss | 0.108 | — |

**Timing comparison (K=8 generation pipeline):**
- Heuristic critic, full clip (200 frames): ~80ms per candidate × 8 = **640ms per (style, seed)**
- Neural critic, full clip (overlapping 32-frame windows, stride=16, ~12 windows): 12 × 0.174ms ≈ **2.1ms per candidate** × 8 = **17ms per (style, seed)**
- Speedup for K=8 scoring: **~38×** (from 640ms to 17ms)

This brings K=8 **scoring** from 640ms to 17ms. This is not a full real-time
generation result: MotionBricks generation cost still dominates end-to-end
latency, so the neural critic only removes the scoring bottleneck.

**Selection agreement:** We further validate the neural critic by running neural-guided WC-K8 alongside heuristic-guided WC-K8 on the same 39 clips and comparing which candidate each selects. Results in Section 4.7.

Training artifacts: `results/neural_critic/model.pt`, `validation.png`, `train_log.csv`, `config.json`.

### 4.7 Neural-Guided WC-K8 vs Heuristic-Guided WC-K8

To validate the neural critic's usefulness as a selector (not just a speed benchmark), we run WC-K8 using neural critic selection on the same 39 clips and compare against heuristic-guided WC-K8. This directly tests whether ρ=0.919 surrogate fidelity translates to comparable winner selection.

Results from `results/neural_guided_ablation.csv` (39 clips, K=8):

| Method | Mean full-clip risk | Critic-accept/39 | Selection agreement |
|---|---:|---:|---:|
| Heuristic-guided WC-K8 | **15.50** | **29** | — |
| Neural-guided WC-K8 | 31.04 | 16 | 12/39 (31%) |

**Negative result — neural clip selection does not match heuristic quality.** Despite ρ=0.919 on window-level validation, neural-guided selection agrees with heuristic selection only 31% of the time, resulting in dramatically worse full-clip risk (31.04 vs 15.50, nearly 2×).

**Root cause:** The neural critic was trained on 32-frame segment labels and applied to 200-frame clips by taking the maximum neural score across overlapping windows. This max-window aggregation is a poor proxy for full-clip heuristic scoring for two reasons: (1) the neural critic sees local windows and may rank globally worse candidates higher if their worst window is marginally better; (2) the training distribution (dense windows from all clips) does not match the selection distribution (ranking K complete clips that differ in holistic physical quality).

**Key lesson:** High window-level ρ does not guarantee clip-level selection quality. A neural clip selector would need to be trained on full-clip labels, not window-level labels. The neural critic's validated use case is **per-segment steering** (PS-K), where it scores short windows — exactly matching its training distribution. For WC-K, the heuristic critic remains necessary.

**Speedup note:** Despite not matching heuristic selection quality, the neural
critic is faster per candidate (0.55ms vs 9.39ms for the segment critic), but
this only validates it as an engineering optimization target. We do not use it
for final WC-K selection.

**Clip-level neural critic (follow-up):** To address the root cause of the
max-window aggregation failure, we also trained a clip-level neural model that
takes full variable-length clips and predicts full-clip risk directly. The
original 39-clip K=1 training set achieved only rho=0.536. After the
105-identity extension, we retrained the same 270k-parameter model on all 210
extended K=1/K=8 clips (`results/neural_critic_clip_extended/`). Validation
rho improves to 0.747, but remains below the 0.85 target. We then trained a
larger 3.38M-parameter residual temporal CNN with attention pooling for 800
epochs on 497 de-duplicated full clips assembled from the main K sweep,
extended run, candidate audit, and steering ablation
(`results/neural_critic_clip_v2/`). This improves validation rho to 0.800, but
still misses the target. More data/model capacity helps, yet full-clip neural
selection still is not reliable enough to replace the heuristic critic.

### 4.8 Diversity Source Ablation

To answer whether seed diversity and Gumbel stochastic sampling each contribute independently (Reviewer R4), we run K=4 with three isolated strategies on all 39 clips:

- **Seed-only:** argmax (deterministic), 4 different seeds (base, +137, +274, +411)
- **Gumbel-only:** stochastic Gumbel, same base seed × 4 draws (pure sampling variance)
- **Combined:** k=0 argmax + k=1..3 stochastic with offset seeds (current method)

Results from `results/diversity_ablation.csv` (K=4, 39 clips):

| Strategy | Mean risk | Critic-accept/39 | Notes |
|---|---:|---:|---|
| K=1 baseline (deterministic) | 38.90 | 5 | Reference |
| Seed-only K=4 | 23.75 | 21 | Argmax, 4 different seeds |
| Gumbel-only K=4 | 21.15 | 23 | Stochastic Gumbel, 1 seed × 4 draws |
| Combined K=4 (current) | 20.98 | 22 | k=0 argmax + k=1..3 stochastic offset seeds |

**Both diversity sources contribute, but Gumbel stochastic sampling is the primary driver.** Gumbel-only (21.15) outperforms seed-only (23.75) by 11%, while combining both provides only marginal additional improvement (20.98, −1% over Gumbel-only).

**By motion type:**

| Type | Seed-only | Gumbel-only | Combined |
|---|---:|---:|---:|
| Static | 6.11 | 8.65 | 8.65 |
| Locomotion | 4.78 | 3.80 | 4.42 |
| Expressive | 12.21 | 7.87 | **3.65** |
| Whole-body | 112.54 | 106.02 | 111.47 |

For **expressive motions**, the combined strategy is dramatically better (3.65 vs 7.87 for Gumbel-only and 12.21 for seed-only), suggesting genuine synergy: expressive motions require both a good reference pose (seed diversity) and variation within that reference (Gumbel). For **locomotion**, Gumbel-only alone is sufficient (3.80). For **static** motions, both stochastic strategies produce the same regression as observed in Section 4.1.

**Interpretation:** Gumbel stochastic sampling explores variation in how the model achieves a target pose (different token sequences, same reference). Seed diversity explores different target poses altogether. Both matter most when the motion requires globally coordinated, context-sensitive execution (expressive styles), while Gumbel alone suffices for repetitive motions (locomotion).

---

## 5. Discussion

**Why best-of-K appears to work under this critic.** MotionBricks uses a VQ-VAE
codebook of reference poses. With a fixed conditioning seed, the model generates
a deterministic (via argmax) or low-entropy trajectory toward that reference.
Different seeds index different codebook entries, and stochastic token sampling
changes the generated trajectory within a mode. Some candidates have lower peak
velocities, smoother joint trajectories, or lower inverse-dynamics torque
demand. The critic selects these automatically, without any knowledge of the
codebook structure. This is a form of test-time search over the generator's
candidate space.

**Why K=4-8 often suffices for upright motions.** The improvement from K=8 to
K=16 is small (15.93 to 13.61) on the 39-identity benchmark, suggesting
diminishing returns after the first few samples. This is an empirical result on
the tested mode/seed set, not a universal sampling law.

**Why crawling remains high risk.** Whole-body crawling clips require much larger
root wrench and torque-limit demand under exact replay than upright clips. The
tested sampling budget reduces the score but does not reliably move crawling
into the critic-accepted action label. The safe interpretation is triage: these clips
should be routed to regeneration, stronger control, or a different planning
method rather than treated as solved by best-of-K sampling.

**Segment-level vs. clip-level steering.** The non-determinism finding changes the interpretation of per-segment steering results. Without determinism, the "greedy best-of-K" at each step cannot guarantee monotone improvement—a segment that scores well in isolation may not contribute to a globally better trajectory. Our 25% margin threshold partially addresses this but at the cost of making fewer actual switches. The whole-clip approach avoids this issue by evaluating complete trajectories with the full critic.

---

## 6. Limitations

- **Critic validity:** The physics critic is entirely heuristic—no model was trained and no safety guarantee is provided. Individual-level correlation with PD tracking is not significant (ρ=−0.09), limiting per-clip validation claims.
- **No real robot validation.** We now have an approximate SONIC learned-policy
  MuJoCo harness, but not a native offline deployment or G1 hardware rollout.
  The native C++ SONIC stack loads and initializes, then waits for live Unitree
  `LowState` DDS messages.
- **Evaluation coverage:** The full K-sweep uses 39 identities; the larger
  K=1/K=8 extension uses 105 identities across all currently exposed G1 demo
  modes. The prompt suite is mode/control based, not a true arbitrary
  text-to-motion benchmark. Claims about new prompts, objects, and unseen
  styles still extrapolate beyond this distribution.
- **Semantic preservation is not guaranteed.** K=8 often lowers risk by changing
  motion scale; 15/39 original benchmark pairs change root displacement by more
  than 0.5x-1.5x.
- **Prompt alignment is proxy-only.** HumanML3D-style evaluation would use a
  learned text-motion embedding to report R-Precision, matching score, FID,
  diversity, and multimodality. We do not have that evaluator for Unitree G1
  qpos prompts in the local release, so we report speed/direction/style proxies.
- **GPU non-determinism** in MotionBricks makes per-segment comparisons methodologically subtle and prevents exact reproduction of individual clip results.
- **Best-of-K requires K× inference compute.** Real-time deployment at K=8 requires the neural surrogate critic (Section 4.6).
- **Stochastic sampling can hurt near-zero-risk motions** (static regression, Section 4.1). A production system should gate resampling on K=1 risk.
- **Contact metrics are diagnostic only.** We now report self-contact, non-foot
  floor contact, foot skate, and support proxies, but these terms are not yet
  integrated into candidate selection and do not replace controller validation.

---

## 7. Next Steps

1. **Native/hardware tracking validation:** Run paired K=1/K=8 references
   through a native learned G1 tracking setup or G1 hardware and measure success
   rate, tracking error, fall time, and contact artifacts.
2. **Controller-in-the-loop selection:** Use short-horizon SONIC rollouts as
   the selector/reward rather than only inverse dynamics.
3. **Physical-aware planner training:** Use critic/controller labels as a curriculum or
   preference signal so the generator's distribution shifts toward candidates
   that a controller can execute, rather than relying only on rejection sampling.
4. **Richer critic terms:** Integrate self-collision, foot/contact consistency,
   center of pressure/support polygon diagnostics, and actuator thermal/current
   models into candidate selection instead of post-hoc reporting only.
4. **Clip-level neural selector:** Train a neural selector on hundreds or
   thousands of full-clip labels so it matches WC-K selection, not only
   window-level heuristic scoring.

---

## 8. Reproducibility

All experiments run in the `base` conda environment with MotionBricks installed editably.

```bash
# Full ablation (3hrs, GPU required)
conda run -n base python scripts/run_ablation.py

# Plots from existing CSV (fast)
conda run -n base python scripts/plot_guided_ablation.py

# Comparison videos from existing .npy files (fast)
conda run -n base python scripts/render_comparison.py

# Compute-matched PS vs WC ablation
conda run -n base python scripts/run_steered_ablation.py

# Diversity source ablation (2hrs)
conda run -n base python scripts/run_diversity_ablation.py

# Neural critic training (~5min on RTX 4060)
conda run -n base python scripts/train_neural_critic.py

# Neural-guided K=8 ablation
conda run -n base python scripts/run_neural_guided_ablation.py

# 105-identity K=1/K=8 extension
conda run -n base python scripts/run_extended_ablation.py --seeds 7 --k_values 1 8
conda run -n base python scripts/summarize_extended_ablation.py
conda run -n base python scripts/plot_extended_ablation.py

# Prompt/task, contact, semantic preservation, and controller negative controls
conda run -n base python scripts/build_prompt_suite.py
conda run -n base python scripts/evaluate_prompt_alignment.py
conda run -n base python scripts/evaluate_contact_quality.py
conda run -n base python scripts/semantic_preservation.py
conda run -n base python scripts/evaluate_computed_torque.py
conda run -n base python scripts/full_candidate_audit.py --seeds 2 --K 8

# Supplementary videos
conda run -n base python scripts/render_supplementary.py --section all
conda run -n base python scripts/render_risk_explainer.py --default_set

# Statistical tests and CIs
conda run -n base python scripts/compute_statistics.py
```

Key data files:
- `results/guided_ablation_full.csv` — 156-row main ablation table
- `results/steered_vs_wc_ablation.csv` — 39-row WC vs PS comparison
- `results/diversity_ablation.csv` — diversity source ablation
- `results/neural_guided_ablation.csv` — neural vs heuristic K=8 selection
- `results/neural_critic_clip_extended/` — extended 270k-parameter clip critic
  follow-up (rho=0.747, still below target)
- `results/neural_critic_clip_v2/` — larger 3.38M-parameter clip critic trained
  on 497 unique labeled clips for 800 epochs (rho=0.800, still below target)
- `results/guided_ablation_extended.csv` — 105-identity K=1/K=8 extension
- `configs/prompt_suite_105.csv` — 105 local mode-control prompt suite
- `results/prompt_alignment.csv` — prompt/task proxy preservation metrics
- `results/contact_quality.csv` — contact artifact and support proxy metrics
- `results/semantic_preservation.csv` — K=1/K=8 semantic-change diagnostics
- `results/computed_torque_tracking.csv` — stronger tracking negative control
- `results/candidate_audit.csv` — segment-vs-full winner agreement audit
- `results/statistical_tests.json` — Wilcoxon p-values and bootstrap CIs
- `results/neural_critic/model.pt` — trained surrogate critic weights
- `data/guided_ablation/` — 156 best .npy clips (13 styles × 3 seeds × 4 K values)
- `results/videos/comparison/` — K=1 vs K=8 side-by-side .mp4s

---

## References

[1] He et al., "MotionBricks: A Modular, Language-Conditioned Kinematic Planner for Whole-Body Humanoid Motion," NVlabs/GR00T-WholeBodyControl, 2025.

[2] Luo et al., "Perpetual Humanoid Control for Real-time Simulated Avatars," ICCV 2023.

[3] Peng et al., "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control," ACM SIGGRAPH 2021.

[4] Winter, D.A., "Biomechanics and Motor Control of Human Movement," 4th ed., Wiley, 2009.

[5] Peng et al., "DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills," ACM SIGGRAPH 2018.

[6] Cobbe et al., "Training Verifiers to Solve Math Word Problems," arXiv 2110.14168, 2021.

[7] Snell et al., "Scaling LLM Test-Time Compute Optimally Improves Diverse Reasoning Tasks," arXiv 2408.03314, 2024.

[8] Lightman et al., "Let's Verify Step by Step," arXiv 2305.20050, 2023.

[9] Todorov et al., "MuJoCo: A physics engine for model-based control," IROS 2012.

[10] Cao et al., "CLAW: Composable Language-Annotated Whole-body Motion Generation," arXiv 2604.11251, 2026.

[11] Yuan et al., "RoboForge: Physically Optimized Text-guided Whole-Body Locomotion for Humanoids," arXiv 2603.17927, 2026.
