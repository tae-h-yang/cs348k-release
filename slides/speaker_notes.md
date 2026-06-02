# Speaker Notes

Target length: complete by the 8-minute timer inside a 10-minute CS348K slot.
The talk should focus on the project goal, technical challenge, definition of
success, evaluation questions, experiments, and what the results say.

## 0. Title

This project is about the object between a motion generator and a humanoid
controller: the reference trajectory. I am not claiming solved text-to-robot
motion. I am claiming a physical audit and repair loop for generated
references.

## 1. Major Advances in Humanoid Motion Tracking

Start optimistic: humanoid tracking policies have become impressive. SONIC is
the example in this deck. Explain it at the input level: a policy tracks
root-and-joint reference trajectories built from high-quality human motion.
Transition line: tracking is strong, but where do the reference motions come
from?

## 2. KIMODO Makes References Easy To Generate

KIMODO gives broad kinematic G1 reference generation from text and constraints,
which makes a 100-prompt benchmark plausible. The caveat is the thesis of the
project: kinematic generation is not enough to ensure feasible torques, valid
contacts, prompt correctness, or controller trackability.

## 3. What Exactly Are We Refining?

Define the object: the generated root-and-joint trajectory. The controller is
fixed, and the generator is not fine-tuned. We use failures observed in the
reference and controller rollout to decide whether to accept, repair, or reject
the candidate.

## 4. Evaluation Questions

State success explicitly: better physical-risk and tracking evidence than the
one-sample baseline, not guaranteed execution. The key questions are how many
references pass the physical rules, how often SONIC tracks them, what failure
tags explain bad cases, and whether repair variants improve the evidence.

## 5. What We Built

The baseline is one generated reference per prompt. The added layer generates a
reference, checks physical rules and SONIC rollout behavior, tries simple
retiming/smoothing repair variants, then accepts, flags, or rejects the clip.

## 6. Method: Physical-Awareness Loop

Explain the loop simply: generate, evaluate the reference and SONIC rollout,
try repair variants from the failed metric, rescore, and gate. The point is
test-time reference repair, not training a new generator or controller.

## 7. Repair: Metric-Guided Variants

Repair is not a black box. Each candidate is tied to a structured failure tag
and a constraint vocabulary, such as smoother limbs for torque spikes or wider
stance for support failures. Every repair attempt is traceable to the metric
that failed.

## 8. 100-Prompt Humanoid Suite

The benchmark has 100 humanoid robotics prompts across locomotion,
manipulation, floor, dance, athletic, and communication categories. Some are
outside current generator/controller coverage; those are useful stress tests.

## 9. Generated Reference Examples

Show what raw generated references look like before physics rollout. This slide
is reference-only, so do not describe it as SONIC or controller execution.

## 10. First-Pass References Can Already Track

Before showing failures, show that the pipeline is not only negative. Six
KIMODO references pass the physical rules and the pretrained SONIC rollout
immediately, including broad jump, crab walk, wipe table, point right, reach
overhead, and single-leg balance.

## 11. First-Pass KIMODO Audit

The audit snapshot shows many usable references and many failures. Read this
honestly: 48 of 100 pass the physical screen, 53 of 100 do not fall in this
SONIC rollout, and only 29 pass both. Offline physical pass correlates with
longer SONIC tracking, but it is not identical to controller success.

## 12. Generated Failure Modes

Point to the six failure families: torque/root-wrench demand, self-contact,
non-foot floor contact, support proxy, controller trackability, and contact
artifact. Each video corresponds to one evaluator check, so the failure tags
are not just abstract numbers.

## 13. Failure Stats Across 100 KIMODO Clips

Use this slide to show that failures are non-exclusive. A clip can violate
actuation, contact, and controller tracking at the same time. The important
headline is the map of where generated references become risky.

## 14. Deterministic Repair Helps Some Clips

This is the quantitative repair slide. Retiming and smoothing improve some
metrics and rescue some clips, but the improvement is modest and there are
regressions. Keep the claim bounded.

## 15. Real Repair Rescues

Show representative before/after videos. Left side is original KIMODO; right
side is repaired. Within each panel, red ghost is reference and solid G1 is the
physics rollout. These are qualitative examples of the measured repair pass.

## 16. Additional Repair Evidence

Use this only briefly. The pushup example is a visual height-clearance demo;
happy dance is a risk-reduction example that remains flagged, and backward
recovery is a support-proxy rescue. The main claim still comes from the
aggregate table and logged rollout records.

## 17. Boundary: What Still Fails

Show failed-run videos directly. The purpose is honesty: forward roll,
carry-box tracking, and cartwheel-style acrobatics remain hard for this simple
repair layer.

## 18. Limitations

Close with the useful lesson: generated humanoid motion needs a physical
awareness layer, but deterministic repair is not enough. The next version
should learn a risk-conditioned motion refiner from controller-labeled SONIC
rollouts.
