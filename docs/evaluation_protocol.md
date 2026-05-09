# Evaluation Protocol

## Scope

This project evaluates a **no-generator-retraining inference-time screening
layer** on top of the current public MotionBricks G1 preview. The local interface exposes
discrete motion modes and steering/control signals, not a full arbitrary
text-to-motion endpoint. We therefore evaluate a 105-row mode-seed audit over
15 unique mode-control prompts rather than claiming 100 different behaviors or
general natural-language coverage.

## Prompt/Task Following

Primary artifacts:

- `configs/prompt_suite_105.csv`
- `results/prompt_alignment.csv`
- `results/prompt_alignment_summary.csv`
- `results/prompt_alignment_by_category.png`
- `results/risk_vs_prompt_alignment.png`

The suite contains 15 exposed G1 demo modes x 7 seeds. It has 105 rows but only
15 unique local behavior prompts. Each row has a natural language prompt,
category, expected tags, target speed, target direction, and frame count.

Proxy metrics:

- Direction/progress score: root displacement along requested direction.
- Speed score: progress speed relative to nominal target speed.
- Static score: low displacement and low joint motion.
- Low-posture score: low mean root height for crawling tasks.
- Expressive score: upper-body motion relative to leg motion.
- Paired preservation: K=8 vs K=1 alignment delta and displacement ratio.

Current result:

- K=1 mean proxy alignment: 0.583.
- K=8 mean proxy alignment: 0.568.
- K=8 is within -0.05 alignment of K=1 on 56/105 paired mode-seed rows.
- Displacement ratio remains in [0.5, 1.5] on 72/105 mode-seed rows.

Interpretation: screening mostly preserves task proxies, but it is not a
semantic guarantee. Expressive and low-posture categories are the weakest under
these simple proxies.

## Physical Plausibility

Primary artifacts:

- `results/guided_ablation_extended.csv`
- `results/contact_quality.csv`
- `results/contact_quality_summary.csv`
- `results/contact_quality_by_category.png`
- `results/risk_vs_contact_artifacts.png`

Existing inverse-dynamics metrics:

- required joint torques relative to actuator limits,
- unactuated root force and torque,
- joint velocity, acceleration, and jerk.

New contact-quality metrics:

- self-contact frames after excluding adjacent kinematic links,
- max self-contact penetration,
- non-foot floor contact frames,
- foot-contact frames and double support,
- foot speed while contact persists,
- projected COM outside a simple two-foot support bounding box.

Current result:

- K=1 mean contact artifact score: 0.274.
- K=8 mean contact artifact score: 0.248.
- Whole-body contact artifact score drops from 0.654 to 0.489.
- Whole-body non-foot floor contact remains high: 47.0% at K=1 and 34.7% at K=8.

Interpretation: best-of-K improves several contact proxies, especially crawling,
but crawling remains problematic. The contact score is a diagnostic, not a
hardware safety certificate.

## Controller Validation

Current controller evidence is negative:

- Weak PD tracking does not correlate strongly with heuristic risk.
- Torque-limited computed-torque tracking falls at frame 15 for all paired K=1
  and K=8 clips in the current setup.

This means the present paper should not claim physically executable robot
motion. A stronger future validation is to run the same paired K=1/K=8
references through a learned G1 tracking policy such as a PHC/SONIC-style
controller and measure success rate, time to fall, tracking error, contact
artifacts, and torque usage.

## Commands

```bash
python scripts/build_prompt_suite.py
python scripts/evaluate_prompt_alignment.py
python scripts/evaluate_contact_quality.py
python scripts/run_extended_ablation.py --seeds 7 --k_values 1 8
python scripts/render_risk_explainer.py --default_set
pytest -q
```
