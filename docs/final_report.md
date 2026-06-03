# Test-Time Physical Awareness for Generated Humanoid Reference Motions

Tae Hoon Yang  
CS 348K: Visual Computing Systems, Stanford University, Spring 2026

## Abstract

Modern humanoid motion trackers such as SONIC can follow rich whole-body
reference trajectories in physics, but those trackers still need high-quality
target motions. KIMODO makes reference authoring much easier by generating
Unitree G1 kinematic motions from text prompts and constraints. This project
studies the interface between those systems: given a generated G1 reference,
can we detect physical risks, attempt a simple test-time repair, and then gate
the result with a controller rollout?

I built a physical-awareness loop for generated reference motions. It evaluates
each trajectory with MuJoCo inverse-dynamics/contact diagnostics and a
pretrained SONIC controller audit. The evaluator emits structured failure
records such as joint-level torque exceedance, root-wrench demand,
self-contact pairs, non-foot floor contacts, weak support, contact artifacts,
and controller trackability failure. Those records can drive either an LLM
prompt-refinement pass or a direct reference-repair pass. The measured repair
baseline in this release uses simple retiming/smoothing variants, and the same
physical and SONIC gates decide whether a candidate is accepted, flagged, or
rejected.

On the 100-prompt KIMODO benchmark, first-pass references pass the physical
screen in 48/100 cases, complete the nominal 4-second SONIC rollout in 53/100
cases, and pass both gates in 29/100 cases. The deterministic repair snapshot
improves physical pass from 48/100 to 53/100, SONIC no-fall from 53/100 to
56/100, mean SONIC tracking time from 2.855 s to 3.007 s, and mean RMSE from
0.156 to 0.142. It also has regressions. The contribution is not a solved
text-to-robot motion generator; it is a concrete evaluation, repair, and
acceptance-gate artifact that exposes where generated references are
physically brittle.

## Project Goal

The project asks:

> Can a physical-awareness layer identify risky KIMODO-generated humanoid
> reference motions, and can those failure tags guide useful test-time repair
> before controller execution?

Success is measured at two levels:

- reference-level physical checks: torque/root wrench, contact, support, and
  contact-artifact diagnostics,
- controller-level checks: SONIC no-fall, survival time, and tracking RMSE.

The claim boundary is important. This project does not fine-tune KIMODO, does
not train a new controller, and does not certify hardware safety. It tests
whether metric-derived failure tags can make generated references easier to
screen, repair, reject, and explain.

The project sits between two related lines of work. Physics-based imitation and
control methods such as DeepMimic, AMP, Perpetual Humanoid Control, ASE, CALM,
SONIC, and GMR show how reference motions, retargeting quality, and tracking
controllers interact under simulation. Text-to-motion systems and datasets such
as HumanML3D, T2M-GPT, MDM, MotionDiffuse, MoMask, and KIMODO make it easier to
author diverse kinematic references. This report focuses on the interface
between those lines: what happens when generated references are handed to a
physics tracker and screened with test-time diagnostics.

## System

The final system is a test-time reference screening and repair loop:

| Stage | Input | Contribution | Output |
|---|---|---|---|
| Generate | text prompt | KIMODO G1 reference generation | root + joint trajectory |
| Evaluate | generated reference | MuJoCo dynamics/contact checks + SONIC rollout | risk flags and rollout metrics |
| Repair | failed candidate | retiming/smoothing or prompt refinement | candidate repaired references |
| Repeat / gate | new candidate | rescore until no flag or budget ends | accepted, flagged, or rejected clip |

Failure tags are intentionally interpretable. For example, a high torque/root
wrench tag indicates abrupt motion; a self-contact tag indicates limb/body
intersection risk; a weak support tag indicates a likely balance problem; a
trackability tag indicates that the pretrained controller could not follow the
reference for the rollout horizon.

SONIC is used as a foundation-style tracking evaluator and acceptance gate. It
is not the optimizer that changes KIMODO.

### Structured Failure Records

The evaluator does not only return a scalar risk. It returns local evidence
that can explain the failure and condition the next repair attempt:

| Failure mode | Example record fields | Repair signal |
|---|---|---|
| torque/root wrench | joint name, peak torque ratio, frame, root force | slow or smooth the offending limb |
| self-contact | body pair, contact fraction, first frame | separate limbs or torso |
| non-foot floor | body name, contact frames, task contact allowance | lift or justify non-foot support |
| support proxy | support margin, low-support frame fraction | widen stance or reduce COM shift |
| contact artifact | foot slip, penetration, impulse spike | clean foot placement |
| SONIC trackability | fall frame, track seconds, RMSE, root error | lower speed or amplitude |

An illustrative torque/root-wrench record is:

```text
failure=torque_root_wrench;
joint=right_shoulder_pitch;
peak_torque_limit_ratio=16.7;
frame=84;
root_force_newton=6900;
sonic_track_seconds=2.78.
```

For prompt refinement, the record is inserted into a constrained LLM rewrite
task:

```text
Rewrite the humanoid motion prompt so it preserves task intent but reduces the
listed physical failure. Keep the motion on a Unitree G1 humanoid, avoid adding
objects not in the original task, and produce one concise revised prompt.
```

Inputs are the original prompt, the failure tag, the structured metric record,
and a small repair vocabulary. A typical output is: "walk forward with slower,
smoother arm swing and shorter steps while keeping both feet under the body."

The aggregate repair numbers below come from the deterministic
retiming/smoothing baseline. Prompt regeneration should be counted separately
once a full regeneration run is available.

## Benchmark

The benchmark contains 100 humanoid robotics prompts. They are deliberately
broader than walking-only examples:

| Family | Count | Example prompt |
|---|---:|---|
| locomotion + recovery | 26 | "Walk forward at a comfortable indoor pace." |
| manipulation + loco-manipulation | 26 | "Step forward, reach with the right hand as if opening a door." |
| floor / low posture | 12 | "Crawl forward using hands and feet." |
| dance / expressive | 12 | "Do an energetic happy dance with bouncing knees." |
| athletic + terrain stress | 16 | "Perform a small split-squat jump." |
| communication / safety | 8 | "Stand still and wave the right hand at shoulder height." |

The suite includes walking, shuffling, crawling, carrying, panel pressing,
gestures, obstacle-style steps, forward rolls, cartwheel attempts, jumping, and
low-posture motions. Some prompts are beyond the current generator/controller
coverage; those failures are useful stress tests rather than merely bad demos.

## Evaluation

For each generated G1 trajectory, the evaluator computes:

- torque demand relative to actuator limits,
- unactuated root force and root torque,
- velocity, acceleration, and jerk,
- self-contact,
- non-foot floor contact,
- foot-contact quality and contact artifacts,
- support-region proxy violations,
- SONIC no-fall, survival time, and joint RMSE.

The screen is not a single magic score. It is a collection of failure signals
that can disagree. A physical-pass reference can still fail SONIC; a physically
flagged reference can sometimes still be tracked. That disagreement is one of
the useful results.

## Results

### First-Pass KIMODO Audit

| Set | Physical pass | SONIC no-fall | Both gates | Mean SONIC sec. | Mean RMSE |
|---|---:|---:|---:|---:|---:|
| all KIMODO references | 48/100 | 53/100 | 29/100 | 2.855 | 0.156 |
| physical-pass subset | 48/48 | 29/48 | 29/48 | 3.293 | 0.170 |
| flagged subset | 0/52 | 24/52 | 0/52 | 2.450 | 0.143 |

![KIMODO audit summary](../slides/assets/figures/kimodo_audit_summary.png)

Figure: first-pass KIMODO audit over 100 prompts. The physical screen and the
SONIC controller rollout are related but not equivalent.

![Success video strips](../slides/assets/figures/paper_success_video_strips.jpg)

Figure: six passing references sampled across time. Red ghost is the reference
target; solid G1 is the MuJoCo/SONIC rollout.

Only 29 of 48 physical-pass references also complete SONIC. Meanwhile, 24 of
52 flagged references still complete SONIC. This means the physical screen is a
useful risk signal, but not a complete controller-success predictor.

### First-Pass Failure Tags

The main failure tags are non-exclusive:

| Failure tag | Count |
|---|---:|
| physical screen fail | 52 |
| SONIC fall | 47 |
| torque limit > 1x | 66 |
| high root force > 5 kN | 28 |
| self-contact > 8% | 34 |
| non-foot floor contact | 11 |
| low foot support | 7 |
| contact artifact > 0.45 | 15 |
| floor penetration > 8 cm | 10 |

The representative failure videos in the deck are:

| Failure family | Example |
|---|---|
| torque/root wrench | `hrb_094_forward_roll` |
| self-contact | `hrb_005_turn_in_place` |
| non-foot floor contact | `hrb_098_knee_slide` |
| support proxy | `hrb_085_step_over_right` |
| controller trackability | `hrb_051_carry_box` |
| contact artifact | `hrb_018_happy_dance` |

![Failure video strips](../slides/assets/figures/paper_failure_video_strips.jpg)

Figure: representative generated-reference failures used in the slide deck.
Rows correspond to torque/root-wrench demand, self-contact, non-foot floor
contact, support proxy, controller trackability, and contact artifact.

### Deterministic Repair Snapshot

The measured repair baseline applies deterministic retiming/smoothing variants
and selects candidates with the same physical-awareness critic and SONIC gate.

| Metric | Original | Repaired |
|---|---:|---:|
| physical pass | 48/100 | 53/100 |
| critic accept | 47/100 | 54/100 |
| critic reject status | 18/100 | 16/100 |
| SONIC 4 s no-fall | 53/100 | 56/100 |
| mean SONIC sec. | 2.855 | 3.007 |
| mean RMSE | 0.156 | 0.142 |
| mean risk | 41.548 | 37.916 |
| contact artifact | 0.264 | 0.247 |

Here, "critic reject status" means the candidate is not accepted by the current
gate and should trigger another attempt. It is not counted as evidence that a
full KIMODO prompt-regeneration run was performed.

The repair pass produced seven SONIC rescues and four regressions. The main
repair videos show representative improvements:

| Prompt | Caught issue | Repair pressure shown in deck |
|---|---|---|
| `forward_walk` | trackability | slower, smoother steps |
| `wave` | torque / tracking | planted feet, smooth arm |
| `split_squat_jump` | support / torque | smaller jump, soft landing |
| `happy_dance` | contact artifact | clean foot placement |
| `backward_recovery` | support proxy | smaller lean, step back |

The deck also includes an illustrative `pushup_pose` height-clearance demo.
That demo is useful for visual intuition, but the measured aggregate above is
the evidence used for the report.

The boundary slide shows failed runs that remain hard: `forward_roll` for
torque/root wrench, `carry_box` for controller trackability, and
`cartwheel_attempt` for acrobatic torque/root-wrench stress.

![Repair video strips](../slides/assets/figures/paper_repair_video_strips.jpg)

Figure: before/after repair previews sampled across time. Each panel is a
paired video: original KIMODO on the left and selected repaired candidate on
the right.

![Boundary video strips](../slides/assets/figures/paper_boundary_video_strips.jpg)

Figure: remaining hard cases after the current screen/repair loop.

## Qualitative Artifacts

The presentation-critical artifacts are:

- [Slide PDF](../slides/build/deck.pdf)
- [Slide PPTX](../slides/build/deck.pptx)
- `slides/assets/videos/kimodo_repair_rescues/`
- `slides/assets/videos/kimodo_repair_rescues_presentation/`
- `slides/assets/videos/appendix_metric_failures/`
- `slides/assets/videos/boundary_still_fails/`
- `slides/assets/figures/kimodo_audit_summary.png`
- `slides/assets/figures/kimodo_failure_flag_counts.png`

Experiment videos use the same convention unless noted otherwise: red
translucent G1 is the reference target, and solid G1 is the MuJoCo/SONIC
physics rollout.

## Interpretation

The results support the project hypothesis in a bounded way:

- generated references have measurable physical failure modes,
- failure tags are useful for curation and simple repair,
- SONIC is stricter than kinematic replay,
- human visual review is a sanity check for presentation examples, not the
  source of metric labels.

The current method still fails on arbitrary text-to-robot motion, robust
acrobatic tracking, low-posture tracking, and automatic repair for every
invalid reference. Stronger results likely require fine-tuning or training a
motion-level refiner, not only retiming and smoothing.

## Limitations

- Deterministic retiming/smoothing repair helps some clips but is not a
  general motion generator.
- The physical screen is heuristic and simulator-dependent.
- SONIC rollout is a simulation diagnostic, not real hardware validation.
- Some motions need semantic context: floor contact can be correct for crawling
  and wrong for walking.
- Prompt fidelity is checked qualitatively; there is no full human-study or
  vision-language benchmark.

## Future Work

The next step is to train a risk-conditioned refiner that learns from the
failure tags and SONIC rollouts. A stronger version would change the motion
trajectory itself, not only retime it. Another direction is to use short
controller rollouts or a learned surrogate during candidate selection so that
the generator is steered by trackability earlier.

## References

- Davis Rempe, Mathis Petrovich, Ye Yuan, Haotian Zhang, Xue Bin Peng, Yifeng
  Jiang, Tingwu Wang, Umar Iqbal, David Minor, Michael de Ruyter, Jiefeng Li,
  Chen Tessler, Edy Lim, Eugene Jeong, Sam Wu, Ehsan Hassani, Michael Huang,
  Jin-Bey Yu, Chaeyeon Chung, Lina Song, Olivier Dionne, Jan Kautz, Simon Yuen,
  and Sanja Fidler. **Kimodo: Scaling Controllable Human Motion Generation.**
  NVIDIA Research / arXiv, 2026.
- Zhengyi Luo, Ye Yuan, Tingwu Wang, Chenran Li, Sirui Chen, Fernando
  Castaneda, Zi-Ang Cao, Jiefeng Li, David Minor, Qingwei Ben, Xingye Da, Runyu
  Ding, Cyrus Hogg, Lina Song, Edy Lim, Eugene Jeong, Tairan He, Haoru Xue,
  Wenli Xiao, Zi Wang, Simon Yuen, Jan Kautz, Yan Chang, Umar Iqbal, Linxi Fan,
  and Yuke Zhu. **SONIC: Supersizing Motion Tracking for Natural Humanoid
  Whole-Body Control.** NVIDIA Research / arXiv, 2025.
- Joao Pedro Araujo, Yanjie Ze, Pei Xu, Jiajun Wu, and C. Karen Liu.
  **Retargeting Matters: General Motion Retargeting for Humanoid Motion
  Tracking.** arXiv:2510.02252, 2025.
- Xue Bin Peng, Pieter Abbeel, Sergey Levine, and Michiel van de Panne.
  **DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based
  Character Skills.** ACM Transactions on Graphics, 2018.
- Xue Bin Peng, Ze Ma, Pieter Abbeel, Sergey Levine, and Angjoo Kanazawa.
  **AMP: Adversarial Motion Priors for Stylized Physics-Based Character
  Control.** ACM Transactions on Graphics, 2021.
- Zhengyi Luo, Jinkun Cao, Alexander Winkler, Kris Kitani, and Weipeng Xu.
  **Perpetual Humanoid Control for Real-time Simulated Avatars.** ICCV, 2023.
- Xue Bin Peng, Yunrong Guo, Lina Halper, Sergey Levine, and Sanja Fidler.
  **ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically
  Simulated Characters.** ACM Transactions on Graphics, 2022.
- Chen Tessler, Yoni Kasten, Yunrong Guo, Shie Mannor, Gal Chechik, and Xue Bin
  Peng. **CALM: Conditional Adversarial Latent Models for Directable Virtual
  Characters.** ACM SIGGRAPH Conference Proceedings, 2023.
- Chuan Guo, Shihao Zou, Xinxin Zuo, Sen Wang, Wei Ji, Xingyu Li, and Li Cheng.
  **Generating Diverse and Natural 3D Human Motions from Text.** CVPR, 2022.
- Jianrong Zhang, Yangsong Zhang, Xiaodong Cun, Yong Zhang, Hongwei Zhao,
  Hongtao Lu, Xi Shen, and Ying Shan. **Generating Human Motion From Textual
  Descriptions With Discrete Representations.** CVPR, 2023.
- Guy Tevet, Sigal Raab, Brian Gordon, Yonatan Shafir, Daniel Cohen-Or, and
  Amit H. Bermano. **Human Motion Diffusion Model.** ICLR, 2023.
- Mingyuan Zhang, Zhongang Cai, Liang Pan, Fangzhou Hong, Xinying Guo, Lei Yang,
  and Ziwei Liu. **MotionDiffuse: Text-Driven Human Motion Generation With
  Diffusion Model.** IEEE Transactions on Pattern Analysis and Machine
  Intelligence, 2024.
- Chuan Guo, Yuxuan Mu, Muhammad Gohar Javed, Sen Wang, and Li Cheng.
  **MoMask: Generative Masked Modeling of 3D Human Motions.** CVPR, 2024.
- Todorov, Erez, and Tassa. **MuJoCo: A Physics Engine for Model-Based
  Control.** IROS, 2012.
