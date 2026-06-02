---
marp: true
theme: cs348k
paginate: true
size: 16:9
style: |
  section { font-size: 23px; }
---

<!-- _class: title -->

# Physical Awareness for Generated Humanoid Motion

A test-time physical audit and repair loop for humanoid reference motions

CS348K Final Project  
Tae Hoon Yang

---

## Major Advances in Humanoid Motion Tracking

<div class="cols">

<div>

Modern humanoid trackers can follow high-quality motion clips in physics.

SONIC trains on over 100M frames / 700 hours of high-quality human motion.

The paper describes these motions as retargeted to the humanoid before tracking.

<div class="metric">

Tracking is getting strong. Where do these reference motions come from?

</div>

</div>

<div>

<img class="paper-figure" src="assets/figures/sonic_motion_dataset_samples.jpg" alt="SONIC motion dataset samples" />

<div class="hero-video small-video">

<video src="https://nvlabs.github.io/GEAR-SONIC/static/videos/teaser_title.mp4" poster="assets/video_posters/sonic_official.jpg" autoplay muted loop playsinline preload="metadata"></video>

</div>

</div>

</div>

<div class="refs"><p>Luo et al., "SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control"</p></div>

---

## KIMODO Makes References Easy To Generate

<div class="cols">

<div class="tight">

KIMODO generates G1 reference motions from text and kinematic constraints.

It is trained on 700 hours of production-quality optical mocap.

This gives us a broad 100-prompt humanoid reference-motion suite.

But kinematic generation does not by itself guarantee:

- feasible torques,
- valid support contacts,
- prompt correctness,
- trackability by a learned controller.

<div class="metric">

We insert a physical-awareness screen and repair loop after KIMODO and before tracking.

</div>

</div>

<div class="hero-video">

<video src="assets/videos/kimodo_official/kimodo_g1_text_loco.mp4" poster="assets/video_posters/kimodo_official/kimodo_g1_text_loco.jpg" autoplay muted loop playsinline preload="metadata"></video>

</div>

</div>

<div class="refs"><p>Rempe et al., "Kimodo: Scaling Controllable Human Motion Generation"</p></div>

---

## What Exactly Are We Refining?

<div class="cols">

<div>

We refine generated references using failures observed in the G1 trajectory.

Each clip contains root pose and joint angles over time.

From each generated clip, we derive:

- velocity and acceleration,
- contact diagnostics,
- SONIC tracking diagnostics.

A prompt can produce a plausible replay and still need a safer reference variant.

</div>

<div class="visual-card">

<video src="assets/videos/kimodo_failures_selected/trackability__hrb_001_forward_walk__sonic_fell3p46s_torque4p29x_rootforce6593N.mp4" poster="assets/video_posters/kimodo_failures_selected/trackability__hrb_001_forward_walk__sonic_fell3p46s_torque4p29x_rootforce6593N.jpg" autoplay muted loop playsinline preload="metadata" style="width:100%; max-height:330px; background:#101820; border:1px solid #cfd8df;"></video>

<div class="caption">Local G1 example: the reference replay looks plausible, but the physics rollout becomes unstable. This failure tag tells the repair loop what kind of variant to try next.</div>

</div>

</div>

---

## Evaluation Questions

Success has to be measured twice: first on the reference, then in the controller.

We ask four questions:

- How many first-pass KIMODO references pass physical rules?
- Which failure tags should drive repair attempts?
- Do refined or lower-risk references survive longer under SONIC?
- Which clips pass both physical rules and controller rollout?

Metric definitions:

- **physical pass:** torque/root-wrench, contact, and support thresholds,
- **SONIC no-fall:** full-horizon controller rollout,
- **RMSE:** joint tracking error,
- **failure tag:** structured context for reference repair.

<div class="refs"><p>Araujo et al., "Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking," arXiv:2510.02252.</p></div>

---

## What We Built

<div class="cols tight">

<div>

### Baseline

- one generated clip per prompt,
- KIMODO G1 reference trajectory,
- no test-time refinement loop.

</div>

<div>

### Our Layer: Screen + Repair

- generate a KIMODO reference,
- check dynamics/contact/support rules,
- run SONIC for rollout evidence,
- try retiming/smoothing repair variants,
- accept, flag, or reject the reference.

</div>

</div>

---

## Method: Physical-Awareness Loop

<div class="pipeline">
  <div class="step"><strong>1. Generate</strong><span>KIMODO reference from the current text prompt</span></div>
  <div class="step"><strong>2. Evaluate</strong><span>rules on the reference plus SONIC rollout checks</span></div>
  <div class="step"><strong>3. Repair</strong><span>retime and smooth candidate reference variants</span></div>
  <div class="step"><strong>4. Gate</strong><span>accept, flag, or reject after rescoring</span></div>
</div>

<small>No generator or controller fine-tuning here: physics rules and SONIC decide whether a reference variant passes.</small>

---

<!-- _class: compact -->

## Repair: Metric-Guided Variants

When a candidate fails, the failed metric tells us what kind of reference variant to try.

Then we rescore the variants with the same physical rules and SONIC gate.

| Failure tag | Repair pressure before rescoring |
|---|---|
| high torque / root wrench | slower, smoother limb motion; avoid abrupt arm swings or snaps |
| self-contact | keep arms away from torso; avoid crossing legs or shoulders |
| non-foot floor contact | use feet-only support unless the task explicitly says crawl or kneel |
| support proxy | wider stance; planted support foot; smaller center-of-mass shift |
| contact artifact | clean foot placement; avoid dragging or sliding contacts |
| trackability failure | reduce speed and amplitude; make transitions gradual |

<div class="metric">

The repair stays interpretable: every new candidate is tied to the metric that failed.

</div>

<div class="refs"><p>Araujo et al., "Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking," arXiv:2510.02252.</p></div>

---

<!-- _class: compact -->

## 100-Prompt Humanoid Suite

KIMODO can attempt the full prompt suite directly on the G1 skeleton.

We use concrete text prompts, not just motion labels.

<div class="warn"><strong>Benchmark caveat:</strong> some prompts are beyond current tracker capability, so failure is useful evidence rather than just a bad generation.</div>

| Family | Count | Example text prompts |
|---|---:|---|
| locomotion + recovery | 26 | "Walk forward at a comfortable indoor pace." |
| manipulation + loco-manipulation | 26 | "Step forward, reach with the right hand as if opening a door." |
| floor / low posture | 12 | "Crawl forward using hands and feet." |
| dance / expressive | 12 | "Do an energetic happy dance with bouncing knees." |
| athletic + terrain stress | 16 | "Perform a small split-squat jump." |
| communication / safety | 8 | "Stand still and wave the right hand at shoulder height." |

---

## Generated Reference Examples

<div class="video-grid six">

<div class="video-card">
<video src="assets/videos/kimodo_reference_family_examples/hrb_001_forward_walk_forward_walk_reference.mp4" poster="assets/video_posters/kimodo_reference_family_examples/hrb_001_forward_walk_forward_walk_reference.jpg" autoplay muted loop playsinline preload="metadata"></video>
<p><strong>locomotion + recovery</strong><br/>forward_walk</p>
</div>

<div class="video-card">
<video src="assets/videos/kimodo_reference_family_examples/hrb_056_open_door_open_door_reference.mp4" poster="assets/video_posters/kimodo_reference_family_examples/hrb_056_open_door_open_door_reference.jpg" autoplay muted loop playsinline preload="metadata"></video>
<p><strong>manipulation + loco-manipulation</strong><br/>open_door</p>
</div>

<div class="video-card">
<video src="assets/videos/kimodo_reference_family_examples/hrb_027_hand_crawl_hand_crawl_reference.mp4" poster="assets/video_posters/kimodo_reference_family_examples/hrb_027_hand_crawl_hand_crawl_reference.jpg" autoplay muted loop playsinline preload="metadata"></video>
<p><strong>floor / low posture</strong><br/>hand_crawl</p>
</div>

<div class="video-card">
<video src="assets/videos/kimodo_reference_family_examples/hrb_018_happy_dance_happy_dance_reference.mp4" poster="assets/video_posters/kimodo_reference_family_examples/hrb_018_happy_dance_happy_dance_reference.jpg" autoplay muted loop playsinline preload="metadata"></video>
<p><strong>dance / expressive</strong><br/>happy_dance</p>
</div>

<div class="video-card">
<video src="assets/videos/kimodo_reference_family_examples/hrb_097_split_squat_jump_split_squat_jump_reference.mp4" poster="assets/video_posters/kimodo_reference_family_examples/hrb_097_split_squat_jump_split_squat_jump_reference.jpg" autoplay muted loop playsinline preload="metadata"></video>
<p><strong>athletic + terrain stress</strong><br/>split_squat_jump</p>
</div>

<div class="video-card">
<video src="assets/videos/kimodo_reference_family_examples/hrb_077_wave_wave_reference.mp4" poster="assets/video_posters/kimodo_reference_family_examples/hrb_077_wave_wave_reference.jpg" autoplay muted loop playsinline preload="metadata"></video>
<p><strong>communication / safety</strong><br/>wave</p>
</div>

</div>

<div class="caption">Reference-only KIMODO outputs rendered on the G1 mesh. No physics rollout is shown on this slide.</div>

---

## First-Pass References Can Already Track

<div class="video-grid six">

<div class="video-card">
<video src="assets/videos/kimodo_success_selected/success__hrb_008_broad_jump__physical_pass_sonic_no_fall.mp4" poster="assets/video_posters/kimodo_success_selected/success__hrb_008_broad_jump__physical_pass_sonic_no_fall.jpg" autoplay muted loop playsinline preload="metadata"></video>
<p><strong>broad_jump</strong><br/>physical pass + SONIC no-fall</p>
</div>

<div class="video-card">
<video src="assets/videos/kimodo_success_selected/success__hrb_030_crab_walk__physical_pass_sonic_no_fall.mp4" poster="assets/video_posters/kimodo_success_selected/success__hrb_030_crab_walk__physical_pass_sonic_no_fall.jpg" autoplay muted loop playsinline preload="metadata"></video>
<p><strong>crab_walk</strong><br/>physical pass + SONIC no-fall</p>
</div>

<div class="video-card">
<video src="assets/videos/kimodo_success_selected/success__hrb_043_wipe_table__physical_pass_sonic_no_fall.mp4" poster="assets/video_posters/kimodo_success_selected/success__hrb_043_wipe_table__physical_pass_sonic_no_fall.jpg" autoplay muted loop playsinline preload="metadata"></video>
<p><strong>wipe_table</strong><br/>physical pass + SONIC no-fall</p>
</div>

<div class="video-card">
<video src="assets/videos/kimodo_success_selected/success__hrb_080_point_right__physical_pass_sonic_no_fall.mp4" poster="assets/video_posters/kimodo_success_selected/success__hrb_080_point_right__physical_pass_sonic_no_fall.jpg" autoplay muted loop playsinline preload="metadata"></video>
<p><strong>point_right</strong><br/>physical pass + SONIC no-fall</p>
</div>

<div class="video-card">
<video src="assets/videos/kimodo_success_candidates_review/success__hrb_041_reach_overhead__physical_pass_no_fall.mp4" poster="assets/video_posters/kimodo_success_candidates_review/success__hrb_041_reach_overhead__physical_pass_no_fall.jpg" autoplay muted loop playsinline preload="metadata"></video>
<p><strong>reach_overhead</strong><br/>physical pass + SONIC no-fall</p>
</div>

<div class="video-card">
<video src="assets/videos/kimodo_success_selected/success__hrb_065_single_leg_balance_right__physical_pass_sonic_no_fall.mp4" poster="assets/video_posters/kimodo_success_selected/success__hrb_065_single_leg_balance_right__physical_pass_sonic_no_fall.jpg" autoplay muted loop playsinline preload="metadata"></video>
<p><strong>single_leg_balance</strong><br/>physical pass + SONIC no-fall</p>
</div>

</div>

<div class="caption">Six generated references pass the physical rules and complete SONIC immediately. Red ghost = reference; solid G1 = controller rollout.</div>

---

<!-- _class: compact -->

## First-Pass KIMODO Audit

<div class="result-grid">

<div>

![kimodo audit summary](assets/figures/kimodo_audit_summary.png)

</div>

<div>

| Set | Physical Pass | SONIC No-Fall | Mean SONIC Sec. | Mean RMSE |
|---|---:|---:|---:|---:|
| all KIMODO refs | 48/100 | 53/100 | 2.855 | 0.156 |
| physical-pass subset | 48/48 | 29/48 | 3.293 | 0.170 |
| flagged subset | 0/52 | 24/52 | 2.450 | 0.143 |

<div class="metric">Controller check: 29/48 physical-pass references also complete SONIC; flagged references complete less often and track for less time.</div>

</div>

</div>

<div class="caption">Full 100-prompt KIMODO run. Failure tags identify references that need repair or rejection before deployment.</div>

---

## Generated Failure Modes

<div class="video-grid six">

<div class="video-card">
<video src="assets/videos/appendix_metric_failures/torque_demand__hrb_094_forward_roll.mp4" poster="assets/video_posters/presentation_failure_modes/torque_demand__hrb_094_forward_roll__late.jpg" autoplay muted loop playsinline preload="metadata"></video>
<p><strong>Torque/root wrench</strong><br/>forward_roll, 16.7x</p>
</div>

<div class="video-card">
<video src="assets/videos/kimodo_failures_selected/self_contact__hrb_005_turn_in_place__selfcontact75pct_torque18p86x.mp4" poster="assets/video_posters/presentation_failure_modes/self_contact__hrb_005_turn_in_place__selfcontact75pct_torque18p86x__late.jpg" autoplay muted loop playsinline preload="metadata"></video>
<p><strong>Self-contact</strong><br/>turn_in_place</p>
</div>

<div class="video-card">
<video src="assets/videos/appendix_metric_failures/nonfoot_floor__hrb_098_knee_slide.mp4" poster="assets/video_posters/presentation_failure_modes/nonfoot_floor__hrb_098_knee_slide__late.jpg" autoplay muted loop playsinline preload="metadata"></video>
<p><strong>Non-foot floor</strong><br/>knee_slide</p>
</div>

<div class="video-card">
<video src="assets/videos/kimodo_failure_candidates_review/support_proxy__hrb_085_step_over_right__support14pct.mp4" poster="assets/video_posters/presentation_failure_modes/support_proxy__hrb_085_step_over_right__support14pct__late.jpg" autoplay muted loop playsinline preload="metadata"></video>
<p><strong>Support proxy</strong><br/>step_over_right</p>
</div>

<div class="video-card">
<video src="assets/videos/kimodo_failure_candidates_review/trackability__hrb_051_carry_box__fell2p78s.mp4" poster="assets/video_posters/presentation_failure_modes/trackability__hrb_051_carry_box__fell2p78s__late.jpg" autoplay muted loop playsinline preload="metadata"></video>
<p><strong>Trackability</strong><br/>carry_box</p>
</div>

<div class="video-card">
<video src="assets/videos/appendix_metric_failures/contact_support__hrb_018_happy_dance.mp4" poster="assets/video_posters/presentation_failure_modes/contact_support__hrb_018_happy_dance__late.jpg" autoplay muted loop playsinline preload="metadata"></video>
<p><strong>Contact artifact</strong><br/>happy_dance</p>
</div>

</div>

<div class="caption">KIMODO references replayed through SONIC. Red ghost = reference; solid G1 = rollout. Failure labels come from metrics.</div>

---

<!-- _class: compact failure-stats-slide -->

## Failure Stats Across 100 KIMODO Clips

<div class="failure-stats-layout">

<div class="failure-summary">
  <div class="failure-stat">
    <strong>52</strong>
    <span>physical-rule fail</span>
  </div>
  <div class="failure-stat">
    <strong>47</strong>
    <span>SONIC fall</span>
  </div>
  <div class="failure-stat">
    <strong>66</strong>
    <span>torque limit &gt;1x</span>
  </div>
</div>

<p>Failure flags are non-exclusive: one clip can violate actuation, contact, and controller tracking.</p>

<div class="failure-bars">
  <div class="failure-bar" style="--value:52"><span>physical-rule fail</span><b>52</b></div>
  <div class="failure-bar" style="--value:47"><span>SONIC fall</span><b>47</b></div>
  <div class="failure-bar" style="--value:66"><span>torque limit &gt;1x</span><b>66</b></div>
  <div class="failure-bar" style="--value:28"><span>high root force &gt;5kN</span><b>28</b></div>
  <div class="failure-bar" style="--value:34"><span>self-contact &gt;8%</span><b>34</b></div>
  <div class="failure-bar" style="--value:15"><span>contact artifact &gt;0.45</span><b>15</b></div>
  <div class="failure-bar" style="--value:11"><span>non-foot floor contact</span><b>11</b></div>
  <div class="failure-bar" style="--value:10"><span>floor penetration &gt;8cm</span><b>10</b></div>
  <div class="failure-bar" style="--value:7"><span>low foot support</span><b>7</b></div>
</div>

</div>

---

<!-- _class: compact repair-improvement-slide -->

## Deterministic Repair Helps Some Clips

<div class="repair-improvement">

<div class="repair-hero">
  <span>SONIC no-fall</span>
  <strong>53 &rarr; 56</strong>
  <em>7 rescues, 4 regressions in the repair pass</em>
</div>

<div class="repair-rows">
  <div class="repair-row" style="--before:48; --after:53">
    <span class="repair-label">physical pass</span>
    <span class="repair-count">48 &rarr; 53</span>
    <span class="repair-delta">+5</span>
    <span class="repair-iters">100 clips</span>
  </div>
  <div class="repair-row" style="--before:47; --after:54">
    <span class="repair-label">critic accept</span>
    <span class="repair-count">47 &rarr; 54</span>
    <span class="repair-delta">+7</span>
    <span class="repair-iters">100 clips</span>
  </div>
  <div class="repair-row" style="--before:18; --after:16">
    <span class="repair-label">reject / regenerate</span>
    <span class="repair-count">18 &rarr; 16</span>
    <span class="repair-delta">-2</span>
    <span class="repair-iters">100 clips</span>
  </div>
  <div class="repair-row" style="--before:42; --after:38">
    <span class="repair-label">mean risk</span>
    <span class="repair-count">41.5 &rarr; 37.9</span>
    <span class="repair-delta">-3.6</span>
    <span class="repair-iters">lower is better</span>
  </div>
  <div class="repair-row" style="--before:16; --after:14">
    <span class="repair-label">mean RMSE</span>
    <span class="repair-count">0.156 &rarr; 0.142</span>
    <span class="repair-delta">-0.014</span>
    <span class="repair-iters">lower is better</span>
  </div>
</div>

</div>

<div class="caption">Repair baseline: deterministic retiming/smoothing variants selected by the same physical-awareness critic and SONIC gate.</div>

---

<!-- _class: compact -->

## Real Repair Rescues

<div class="video-grid three">

<div class="video-card">
<video src="assets/videos/kimodo_repair_rescues_presentation/rescue_preview__dynamic_locomotion__hrb_001_forward_walk__trim_end1s.mp4" poster="assets/video_posters/kimodo_repair_rescues_presentation/rescue_preview__dynamic_locomotion__hrb_001_forward_walk__trim_end1s.jpg" autoplay muted loop playsinline preload="metadata"></video>
<p><strong>forward_walk</strong><br/>caught: trackability<br/>prompt: slower, smoother steps</p>
</div>

<div class="video-card">
<video src="assets/videos/kimodo_repair_rescues/rescue__communication_safety__hrb_077_wave__3p86s_to_4p00s.mp4" poster="assets/video_posters/kimodo_repair_rescues/rescue__communication_safety__hrb_077_wave__3p86s_to_4p00s.jpg" autoplay muted loop playsinline preload="metadata"></video>
<p><strong>wave</strong><br/>caught: torque / tracking<br/>prompt: planted feet, smooth arm</p>
</div>

<div class="video-card">
<video src="assets/videos/kimodo_repair_rescues_presentation/rescue_preview__athletic_stress__hrb_097_split_squat_jump__trim_end1s.mp4" poster="assets/video_posters/kimodo_repair_rescues_presentation/rescue_preview__athletic_stress__hrb_097_split_squat_jump__trim_end1s.jpg" autoplay muted loop playsinline preload="metadata"></video>
<p><strong>split_squat_jump</strong><br/>caught: support / torque<br/>prompt: smaller jump, soft landing</p>
</div>

</div>
<div class="caption">Three representative repair previews. Left side is original KIMODO, right side is repaired. Within each panel: red ghost = reference; solid G1 = physics rollout.</div>

---

<!-- _class: compact -->

## Additional Repair Evidence

<div class="video-grid three">

<div class="video-card">
<video src="assets/videos/kimodo_repair_rescues/pushup_pose_height_offset_demo.mp4" poster="assets/video_posters/kimodo_repair_rescues/pushup_pose_height_offset_demo.jpg" autoplay muted loop playsinline preload="metadata"></video>
<p><strong>pushup_pose</strong><br/>height-clearance demo<br/>repair pressure: lift torso</p>
</div>

<div class="video-card">
<video src="assets/videos/appendix_repair_by_category/dance_expressive/repair_delta__hrb_018_happy_dance__no_to_no.mp4" poster="assets/video_posters/appendix_repair_by_category/dance_expressive/repair_delta__hrb_018_happy_dance__no_to_no.jpg" autoplay muted loop playsinline preload="metadata"></video>
<p><strong>happy_dance</strong><br/>risk reduction, still flagged<br/>repair pressure: clean foot placement</p>
</div>

<div class="video-card">
<video src="assets/videos/kimodo_repair_rescues/rescue__balance_recovery__hrb_070_backward_recovery__3p38s_to_4p00s.mp4" poster="assets/video_posters/kimodo_repair_rescues/rescue__balance_recovery__hrb_070_backward_recovery__3p38s_to_4p00s.jpg" autoplay muted loop playsinline preload="metadata"></video>
<p><strong>backward_recovery</strong><br/>support-proxy rescue<br/>repair pressure: smaller lean</p>
</div>

</div>
<div class="caption">Supporting examples, not all SONIC rescues. Red ghost = reference; solid G1 = rollout or visual repair proxy.</div>

---

<!-- _class: compact -->

## Boundary: What Still Fails

<div class="video-grid three">

<div class="video-card">
<video src="assets/videos/appendix_metric_failures/torque_demand__hrb_094_forward_roll.mp4" poster="assets/video_posters/appendix_metric_failures/torque_demand__hrb_094_forward_roll.jpg" autoplay muted loop playsinline preload="metadata"></video>
<p><strong>forward_roll</strong><br/>torque/root wrench</p>
</div>

<div class="video-card">
<video src="assets/videos/appendix_metric_failures/sonic_trackability__hrb_051_carry_box.mp4" poster="assets/video_posters/appendix_metric_failures/sonic_trackability__hrb_051_carry_box.jpg" autoplay muted loop playsinline preload="metadata"></video>
<p><strong>carry_box</strong><br/>controller trackability</p>
</div>

<div class="video-card">
<video src="assets/videos/boundary_still_fails/cartwheel_attempt_repaired_right.mp4" poster="assets/video_posters/boundary_still_fails/cartwheel_attempt_repaired_right.jpg" autoplay muted loop playsinline preload="metadata"></video>
<p><strong>cartwheel_attempt</strong><br/>torque/root wrench</p>
</div>

</div>

<div class="warn">Still unsolved: arbitrary text-to-robot motion; robust acrobatics and low postures; automatic repair for every invalid reference.</div>

---

<!-- _class: compact limitations-slide -->

## Limitations

<div class="limit-grid">

<div class="limit-card">
<strong>Reference repair is useful, not enough</strong>
<span>Retiming and smoothing can rescue some clips. Stronger text-to-motion stability likely needs fine-tuning or a learned motion-level refiner.</span>
</div>

<div class="limit-card">
<strong>Tracker-as-foundation is an assumption</strong>
<span>We treat SONIC as a foundation motion tracker, but uncovered motions remain a weak point. Dataset coverage still matters.</span>
</div>

<div class="limit-card">
<strong>Trackability is still hard to certify</strong>
<span>If a generated reference fails, deciding whether RL could learn to track it is a harder problem than our test-time metric screen.</span>
</div>

<div class="limit-card">
<strong>Iteration has deployment cost</strong>
<span>Repair and audit take extra test-time iterations, but catching infeasible motion before controller execution is still important for real-world deployment.</span>
</div>

</div>

<div class="metric">Main point: metric-guided reference repair can reject or improve generated references, but it does not replace better generators, broader tracker training, or deployment-time safety checks.</div>
