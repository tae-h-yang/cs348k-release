# CS348K Slide Guidelines

Use this as the project-local source of truth when preparing checkpoint or final
presentation slides.

## Official Timing

Current official project guidance says:

- all final presentation slots are **10 minutes**,
- the prepared talk should complete in **8 minutes**,
- course staff will set the timer to expire at 8 minutes and politely cut off
  after that point to keep the schedule on track.

Latest source: Ed post #76, "Sign up for final presentation slots -- final
tips/reminders" by Kayvon Fatahalian. Earlier source:
<https://github.com/stanford-cs348k/project>, Final Report section.

Do **not** describe the final presentation as a 5-minute talk unless a newer
course announcement explicitly changes the requirement.

## Latest Instructor Emphasis

The May 31 Ed post says short presentations should communicate only the most
important parts:

- What project was being solved, defined in terms of goals and constraints.
- What specific technical challenge had to be overcome.
- What success means in the project's own terms.
- What questions were asked to determine whether success was achieved.
- What experiments were performed to answer those questions.
- What the results suggest about the extent to which success was achieved.

If the team has multiple students, all students should play a part in the
communication of the presentation.

## What The Course Emphasizes

The project guidelines prioritize evaluation and results. A good deck should
make these points legible:

- what problem is being solved,
- what the inputs and outputs are,
- what question or falsifiable hypothesis the project tests,
- what baseline or starting code/model was used,
- what was built beyond the baseline,
- what metrics, demos, plots, tables, or images define success,
- what the results show,
- what failed or remains unanswered,
- what the next step is.

## Recommended 8-Minute Deck Shape

Aim for roughly 12-15 content slides plus title. Prefer evidence-heavy slides
over prose-heavy slides. A slide should either set up the required project
question or answer it with evidence.

1. Title and one-sentence claim boundary.
2. Why now: humanoid tracking progress makes reference quality the bottleneck.
3. Baseline opportunity: KIMODO creates broad G1 reference motions from prompts.
4. Problem and technical challenge: kinematic references can fail physics.
5. Success criteria and evaluation questions.
6. System: baseline vs curation layer.
7. Benchmark: 100 prompts and semantic-support caveat.
8. Method: validation, risk screening, SONIC verification.
9. Main quantitative result table or plot.
10. Failure-mode/category plot.
11. SONIC tracking audit.
12. Visual evidence/contact sheet or one representative video.
13. Interpretation: what improved and what did not.
14. Next step: controller-labeled learning/refinement.
15. Takeaway.

## Claim Discipline For This Project

Safe claims:

- The project builds a physical-awareness and controller-in-the-loop evaluation
  layer for generated humanoid motion.
- KIMODO can generate G1 reference motions for the full 100-prompt suite.
- The project screens KIMODO references with dynamics/contact metrics before
  controller rollout.
- The physical screen separates safer references from references that need
  repair or regeneration.
- SONIC tracking is a stricter test than inverse-dynamics risk alone.
- All-100 videos and contact sheets are necessary because scalar metrics can
  miss visually obvious failures.

Unsafe claims unless new evidence is generated:

- "All 100 motions are physically valid."
- "KIMODO was fine-tuned."
- "The method repairs every invalid KIMODO reference."
- "The method guarantees executable robot motion."
- "The prompt semantics are solved for unsupported actions."
- "SONIC tracks every selected reference to completion."

## Deck Maintenance Checklist

Before presenting:

- Rebuild the deck with `bash slides/build_slides.sh`.
- If RalphLoop finished, update the metrics with
  `python scripts/update_slide_metrics.py` before rebuilding.
- Open `slides/build/deck.pdf` and skim every slide for clipped text.
- Check exported slide text for awkward line breaks, especially splits between
  phrases that should read together, such as "reference motions" or
  "target trajectories." Prefer shorter slide sentences over long wrapped
  paragraphs.
- Avoid document-style paragraphs in columns. Use short clauses or bullets when
  text sits next to an image/video/table; otherwise Marp will wrap narrow
  columns into awkward mid-phrase line breaks.
- Render a contact sheet from the PDF during final passes and scan it for
  visual rhythm: no clipped captions, no lonely one-word lines, no crowded
  footnotes, and no line breaks that change how the sentence is read.
- After adding a backup slide, verify both `pdfinfo slides/build/deck.pdf` and
  a rendered image of the new final page. A misplaced `---` can land inside an
  earlier two-column layout and make the slide count look plausible while the
  content appears in the wrong place.
- Check that every plotted number in the deck has a matching CSV under
  `results/`.
- Verify that video paths mentioned in the deck exist.
- Give every embedded video a `poster` image. PDF/PPTX exports often show only
  the poster frame before playback, so black loading frames are a slide bug.
- Do not use native `controls` on slide videos. Marp/Chromium captures the
  control bar and loading spinner into static PDF/PPTX exports; use autoplaying,
  muted loop videos with clean posters instead.
- For all experiment videos, keep the convention stable: red translucent G1 is
  the reference target and solid G1 is the physics rollout. Official background
  videos may use their original project visuals, but experiment-result slides
  should not mix conventions.
- For floor, crawling, kneeling, rolling, or acrobatic qualitative evidence,
  use no-early-termination renders. Do not let a low root-height termination
  visually imply failure when the reference itself intentionally goes low.
- Keep the title-slide timing at **8 minutes speaking / 10-minute slot**.
