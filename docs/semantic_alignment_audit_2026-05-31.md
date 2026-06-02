# Semantic Alignment Audit 2026-05-31

Purpose: record the correction after visually noticing that some prompt-matched MotionBricks/Kimodo pairs had different or weak semantic behavior.

## What Was Checked

- Verified that final red-ghost video filenames, prompt IDs, selected references, manifests, and video paths are mechanically consistent.
- Checked `motionbricks_redghost_index.csv`, `kimodo_redghost_index.csv`, `failure_success_pair_manifest.csv`, and `selected_talk_pairs_12_manifest.csv`.
- No prompt-ID or copy-path mismatch was found.

## Actual Problem

The issue is semantic, not bookkeeping:

- The cross-generator pair folders compare different generated motions for the same prompt.
- A MotionBricks clip can pass a physical/SONIC gate while weakly following the prompt.
- Therefore those pairs do not prove that the method repaired the failed motion or preserved prompt semantics.

## Fix Applied

- Moved `failure_success_pairs/selected_talk_pairs_12/` into `_quarantine_cross_generator_not_method_claim/`.
- Updated pair README files to warn that cross-generator pairs are diagnostic, not method-improvement evidence.
- Created/kept `method_before_after_pairs/` for same-generator K1-vs-selected comparisons.

## Current Honest Result

- Same-generator true metric rescue: 1 case (`hrb_028_elbow_crawl`), still requiring semantic visual review.
- Same-generator partial improvements: 16 cases with longer SONIC tracking, but still not presentation-pass successes.
- Cross-generator diagnostic pairs: many, but not valid method-rescue evidence unless separately semantically reviewed.

## Rule Going Forward

Do not claim a video is a final success unless it passes both:

1. Physical/SONIC gate.
2. Human or VLM semantic prompt-alignment review.
