# CS348K Final Release

Final project: **Physical Awareness for Generated Humanoid Motion**.

This release contains the final slides, report, and selected video/figure
artifacts for a test-time physical-audit and reference-repair loop for
KIMODO-generated Unitree G1 reference motions. The system uses MuJoCo physical
checks and SONIC rollout evidence to produce structured failure records. Those
records support LLM prompt refinement or direct repair; the measured baseline
in this release tests simple retiming/smoothing repairs before accepting,
flagging, or rejecting references.

## Start Here

- [Final slides PDF](slides/build/deck.pdf)
- [Final slides PPTX](slides/build/deck.pptx)
- [Final report](docs/final_report.md)
- [Compact report PDF](paper/main.pdf)
- [All 100 KIMODO reference + SONIC rollout videos](videos/kimodo100/ref_plus_sonic/)
- [KIMODO100 SONIC metrics](metrics/kimodo100_sonic_tracking_cuda.csv)

## Final Report

Read the full report as [Markdown](docs/final_report.md) or as the compact
[PDF](paper/main.pdf).

This project studies a test-time physical-awareness layer for
KIMODO-generated Unitree G1 reference motions. The system evaluates each
generated trajectory with MuJoCo inverse-dynamics/contact diagnostics and a
pretrained SONIC controller rollout, then emits structured failure records for
torque/root-wrench demand, self-contact, non-foot floor contacts, weak support,
contact artifacts, and controller trackability failure.

On the 100-prompt KIMODO benchmark, first-pass references pass the physical
screen in 48/100 cases, complete the nominal 4-second SONIC rollout in 53/100
cases, and pass both gates in 29/100 cases. The deterministic repair snapshot
improves physical pass from 48/100 to 53/100, SONIC no-fall from 53/100 to
56/100, mean SONIC tracking time from 2.855 s to 3.007 s, and mean RMSE from
0.156 to 0.142, while still showing regressions and clear failure boundaries.

## One-Line Result

First-pass KIMODO references are often plausible but brittle: 48/100 pass the
physical screen, 53/100 complete the nominal 4-second SONIC rollout, and 29/100
pass both gates. Deterministic test-time repair improves a small set of clips,
but does not solve arbitrary text-to-robot motion.

## Viewing

This repo uses Git LFS for embedded slide videos. On a new machine, install
Git LFS once and pull media objects:

```bash
git lfs install
git lfs pull
```

```bash
open slides/build/deck.pdf
open slides/build/deck.pptx
```

On Linux:

```bash
xdg-open slides/build/deck.html
```

To rebuild the deck:

```bash
python -m pip install -r requirements.txt
bash slides/build_slides.sh
```

Selected videos and posters used by the slides live under `slides/assets/`.
Final helper scripts are in `scripts/`; raw experiment directories remain in
the dev repo and are intentionally not part of this clean release.
