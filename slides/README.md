# Slides

This folder contains the presentation deck for the KIMODO/SONIC Humanoid100
project.

Use `docs/cs348k_slide_guidelines.md` as the source of truth for course timing,
slide structure, and claim boundaries. Current official guidance is an
approximately 8-minute talk in a 10-minute presentation slot.

## Fast Preview

Open either of these directly after cloning:

```text
slides/build/deck.pdf
slides/build/deck.html
```

On macOS:

```bash
open slides/build/deck.pdf
open slides/build/deck.html
```

The PDF/PPTX/HTML are committed so viewing does not require rebuilding. To play
embedded videos, open `slides/build/deck.html` or `slides/build/deck.marp.html`.
The background videos use official KIMODO and GEAR-SONIC project media, so HTML
playback may need internet for the streamed SONIC clip. PDF export cannot play
MP4, and PPTX video support from Marp is not reliable across machines.

## Rebuild

```bash
bash slides/build_slides.sh
xdg-open slides/build/deck.html
```

The build script always produces `slides/build/deck.html` using the local Python
fallback. If Marp is installed, it also exports PDF and PPTX.

Minimal Python dependency for the fallback builder:

```bash
python -m pip install Markdown
```

## Marp Export Tool

This machine is now set up with Node and Marp in the active conda environment.
To recreate the setup elsewhere:

```bash
conda install -y -c conda-forge nodejs
npm install -g @marp-team/marp-cli
```

Then run:

```bash
bash slides/build_slides.sh
```

Expected extra outputs:

```text
slides/build/deck.pdf
slides/build/deck.pptx
```

Current generated artifacts:

```text
slides/build/deck.html
slides/build/deck.marp.html
slides/build/deck.pdf
slides/build/deck.pptx
```

## Live Video Clips

Small MP4s for optional live playback are in:

```text
slides/assets/videos/
```

These are intentionally separate from the PDF/PPTX because embedded video can
be brittle across machines. Open them directly during rehearsal/presentation.

## Metrics Source

The committed deck currently uses the latest completed K1024 Humanoid100
snapshot copied from the main working repo:

```text
results/ralphloop/20260530_003531/humanoid100_final_eval_k1024/
```

If newer results are copied into `slides/assets/selector_summary.csv`, rebuild
the metrics snapshot:

```bash
python scripts/update_slide_metrics.py
bash slides/build_slides.sh
```

The deck intentionally keeps the claim boundary explicit: current evidence
supports controller-in-the-loop curation and abstention, not solved arbitrary
text-to-physical humanoid motion.
