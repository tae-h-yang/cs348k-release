"""
SIGGRAPH-quality supplementary video generator.

Produces:
  results/videos/supplementary/
    S1_locomotion.mp4         — K=1 vs K=8, all locomotion clips (6 styles × 3 seeds)
    S2_expressive.mp4         — K=1 vs K=8, all expressive clips (4 styles × 3 seeds)
    S3_wholebody.mp4          — crawling: K=1/K=4/K=8/K=16 all still rejected
    S4_k_scaling.mp4          — K=1 → K=4 → K=8 → K=16 progression (4-panel)
    S5_wc_vs_ps.mp4           — WC-K4 vs PS-K4 comparison
    S0_main.mp4               — curated highlights stitched together with title cards

Each panel has:
  - Risk score overlay (color-coded by the same heuristic action thresholds as the report)
  - Accept/Reject label
  - Frame counter + timestamp
  - Motion style label
"""
import sys, csv, argparse
from pathlib import Path

import numpy as np
import mujoco
import imageio

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: cv2 not found, text overlays will be minimal")

REPO = Path(__file__).parent.parent
DATA_GUIDED = REPO / "data" / "guided_ablation"
DATA_STEERED = REPO / "data" / "steered_ablation"
OUT_DIR = REPO / "results" / "videos" / "supplementary"
MODEL_XML = REPO / "assets" / "g1" / "scene_29dof.xml"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Video settings
W2 = 640   # panel width for 2-panel videos
W4 = 320   # panel width for 4-panel videos (fits same total width)
H = 480    # panel height (shared)
FPS = 30
DIVIDER_PX = 4  # divider between panels
# Total widths
TOTAL_W2 = 2 * W2 + DIVIDER_PX           # 1284
TOTAL_W4 = 4 * W4 + 3 * DIVIDER_PX       # 1292
# Use consistent canvas width for stitching = TOTAL_W2
CANVAS_W = TOTAL_W2

# ─── Risk table ───────────────────────────────────────────────────────────────
def load_risks():
    risks = {}
    csv_path = REPO / "results" / "guided_ablation_full.csv"
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            risks[(row["clip"], int(row["K"]))] = float(row["full_risk"])
    return risks

def load_steered_risks():
    risks_wc, risks_ps = {}, {}
    csv_path = REPO / "results" / "steered_vs_wc_ablation.csv"
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            c = row["clip"]
            try:
                risks_wc[c] = float(row["risk_wc"])
                risks_ps[c] = float(row["risk_ps"])
            except (ValueError, KeyError):
                pass
    return risks_wc, risks_ps

# ─── Rendering utilities ──────────────────────────────────────────────────────

def risk_color(risk: float):
    """BGR color for risk label."""
    if risk <= 25:
        return (50, 200, 50)    # green
    elif risk <= 80:
        return (30, 130, 220)   # orange
    else:
        return (40, 40, 220)    # red

def risk_label(risk: float) -> str:
    if risk <= 25:
        return f"ACCEPT  risk={risk:.1f}"
    elif risk <= 80:
        return f"REVIEW  risk={risk:.1f}"
    else:
        return f"REJECT  risk={risk:.1f}"

def draw_overlay(frame: np.ndarray, top_text: str, risk: float,
                 style_name: str = "", t: int = 0, fps: int = FPS) -> np.ndarray:
    frame = frame.copy()
    if not HAS_CV2:
        return frame

    h, w = frame.shape[:2]
    rlabel = risk_label(risk)
    rcolor = risk_color(risk)

    # Semi-transparent risk bar at top
    bar_h = 32
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Risk text
    cv2.putText(frame, rlabel, (8, 22), cv2.FONT_HERSHEY_DUPLEX, 0.62, rcolor, 1, cv2.LINE_AA)

    # Risk bar graphic. Omit it in compact 4-panel renders; otherwise it
    # overlaps the text label at W4=320.
    if w >= 420:
        bar_x = w - 140
        bar_w = 120
        bar_inner_h = 8
        bar_y = 12
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_inner_h), (60, 60, 60), -1)
        fill_w = int(bar_w * min(risk / 100, 1.0))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_inner_h), rcolor, -1)

    # Top text (method label)
    cv2.putText(frame, top_text, (8, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    # Style name bottom
    cv2.putText(frame, style_name, (8, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Timestamp
    ts = f"{t/fps:.1f}s"
    cv2.putText(frame, ts, (w - 55, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    return frame

def title_card(text: str, subtext: str = "", n_frames: int = 60,
               w: int = CANVAS_W, h: int = H) -> list:
    """Generate a black title card."""
    frames = []
    for _ in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        if HAS_CV2:
            cx, cy = w // 2, h // 2
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
            cv2.putText(frame, text, (cx - tw // 2, cy), cv2.FONT_HERSHEY_DUPLEX, 1.0,
                        (220, 220, 220), 2, cv2.LINE_AA)
            if subtext:
                (sw, sh), _ = cv2.getTextSize(subtext, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
                cv2.putText(frame, subtext, (cx - sw // 2, cy + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (140, 140, 140), 1, cv2.LINE_AA)
        frames.append(frame)
    return frames

def divider_bar(h: int = H) -> np.ndarray:
    return np.full((h, DIVIDER_PX, 3), 30, dtype=np.uint8)

# ─── Core render ──────────────────────────────────────────────────────────────

class Renderer:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self._r2 = mujoco.Renderer(model, height=H, width=W2)
        self._r4 = mujoco.Renderer(model, height=H, width=W4)

    def render_frame(self, qpos: np.ndarray, azimuth: float = 100.0,
                     small: bool = False) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat = np.array([self.data.qpos[0], self.data.qpos[1], 0.85])
        cam.azimuth = azimuth
        cam.elevation = -18.0
        cam.distance = 3.2
        r = self._r4 if small else self._r2
        r.update_scene(self.data, camera=cam)
        return r.render().copy()

    def close(self):
        self._r2.close()
        self._r4.close()


def render_two_panel(renderer, q1, q8, risk1, risk8, style_name, label1, label2):
    """Render one frame of a two-panel comparison."""
    T = min(len(q1), len(q8))
    frames = []
    div = divider_bar()
    for t in range(T):
        f1 = renderer.render_frame(q1[t])
        f8 = renderer.render_frame(q8[t])
        f1 = draw_overlay(f1, label1, risk1, style_name, t)
        f8 = draw_overlay(f8, label2, risk8, style_name, t)
        frames.append(np.concatenate([f1, div, f8], axis=1))
    return frames

def render_four_panel(renderer, clips_dict, risks_dict, style_name):
    """
    clips_dict: {label: qpos_array}
    risks_dict: {label: float}
    Returns frames at CANVAS_W width (uses small=True renderers for W4-wide panels)
    """
    labels = list(clips_dict.keys())
    arrays = [clips_dict[l] for l in labels]
    T = min(len(a) for a in arrays)
    div = divider_bar()
    frames = []
    for t in range(T):
        panels = []
        for i, (label, arr) in enumerate(zip(labels, arrays)):
            f = renderer.render_frame(arr[t], small=True)  # W4-wide
            f = draw_overlay(f, label, risks_dict[label], style_name, t, fps=FPS)
            panels.append(f)
            if i < len(labels) - 1:
                panels.append(div)
        combined = np.concatenate(panels, axis=1)  # (H, ~TOTAL_W4, 3)
        # Pad/crop to CANVAS_W so it matches 2-panel frames
        cw = combined.shape[1]
        if cw < CANVAS_W:
            pad = np.zeros((H, CANVAS_W - cw, 3), dtype=np.uint8)
            combined = np.concatenate([combined, pad], axis=1)
        elif cw > CANVAS_W:
            combined = combined[:, :CANVAS_W]
        frames.append(combined)
    return frames


def save_video(frames, path: Path, fps: int = FPS):
    if not frames:
        return
    imageio.mimwrite(str(path), frames, fps=fps, macro_block_size=1,
                     quality=8, output_params=["-vf", "format=yuv420p"])
    size_mb = path.stat().st_size / 1e6
    print(f"  -> {path.name}  ({len(frames)} frames, {size_mb:.1f} MB)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", default="all",
                        help="Which section to render: s1|s2|s3|s4|s5|s0|all")
    args = parser.parse_args()

    risks = load_risks()
    risks_wc, risks_ps = load_steered_risks()

    print(f"Loading model: {MODEL_XML}")
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    renderer = Renderer(model)
    print("Model loaded.\n")

    TYPE_CONFIGS = {
        "locomotion": [
            "walk_seed0", "walk_seed1", "walk_seed2",
            "slow_walk_seed0", "slow_walk_seed1",
            "stealth_walk_seed0", "stealth_walk_seed1",
            "injured_walk_seed0", "injured_walk_seed1",
            "walk_zombie_seed0", "walk_zombie_seed1",
            "walk_stealth_seed0", "walk_stealth_seed1",
        ],
        "expressive": [
            "walk_boxing_seed0", "walk_boxing_seed1",
            "walk_happy_dance_seed0", "walk_happy_dance_seed1",
            "walk_gun_seed0", "walk_gun_seed1",
            "walk_scared_seed0", "walk_scared_seed1",
        ],
        "whole_body": [
            "hand_crawling_seed0", "hand_crawling_seed1",
            "elbow_crawling_seed0", "elbow_crawling_seed1",
        ],
    }

    # ── S1: Locomotion ────────────────────────────────────────────────────────
    if args.section in ("s1", "all"):
        print("=== S1: Locomotion K=1 vs K=8 ===")
        all_frames = []
        all_frames += title_card("S1: Locomotion",
                                 "K=1 (left) vs K=8 (right) | Green=Accept, Orange=Review, Red=Reject")
        for clip in TYPE_CONFIGS["locomotion"]:
            p1 = DATA_GUIDED / f"{clip}_K1.npy"
            p8 = DATA_GUIDED / f"{clip}_K8.npy"
            if not p1.exists() or not p8.exists():
                continue
            q1 = np.load(p1)
            q8 = np.load(p8)
            r1 = risks.get((clip, 1), float("nan"))
            r8 = risks.get((clip, 8), float("nan"))
            print(f"  {clip}: K1={r1:.1f} → K8={r8:.1f}")
            all_frames += title_card(clip.replace("_", " ").title(),
                                     f"K=1 risk={r1:.1f}  →  K=8 risk={r8:.1f}", n_frames=25)
            all_frames += render_two_panel(renderer, q1, q8, r1, r8,
                                           clip, "K=1 (baseline)", "K=8 (best-of-8)")
        save_video(all_frames, OUT_DIR / "S1_locomotion.mp4")

    # ── S2: Expressive ────────────────────────────────────────────────────────
    if args.section in ("s2", "all"):
        print("=== S2: Expressive K=1 vs K=8 ===")
        all_frames = []
        all_frames += title_card("S2: Expressive Motions",
                                 "K=1 (left) vs K=8 (right) | heuristic risk reduction")
        for clip in TYPE_CONFIGS["expressive"]:
            p1 = DATA_GUIDED / f"{clip}_K1.npy"
            p8 = DATA_GUIDED / f"{clip}_K8.npy"
            if not p1.exists() or not p8.exists():
                continue
            q1 = np.load(p1)
            q8 = np.load(p8)
            r1 = risks.get((clip, 1), float("nan"))
            r8 = risks.get((clip, 8), float("nan"))
            print(f"  {clip}: K1={r1:.1f} → K8={r8:.1f}")
            all_frames += title_card(clip.replace("_", " ").title(),
                                     f"K=1 risk={r1:.1f}  →  K=8 risk={r8:.1f}", n_frames=25)
            all_frames += render_two_panel(renderer, q1, q8, r1, r8,
                                           clip, "K=1 (baseline)", "K=8 (best-of-8)")
        save_video(all_frames, OUT_DIR / "S2_expressive.mp4")

    # ── S3: Whole-body (crawling — all K rejected) ────────────────────────────
    if args.section in ("s3", "all"):
        print("=== S3: Whole-body — K scaling does not fix crawling ===")
        all_frames = []
        all_frames += title_card("S3: Whole-body Crawling",
                                 "K=1 vs K=8 — still high demand under this critic")
        for clip in TYPE_CONFIGS["whole_body"][:2]:
            for K_pair in [(1, 8), (4, 16)]:
                k1, k2 = K_pair
                p1 = DATA_GUIDED / f"{clip}_K{k1}.npy"
                p2 = DATA_GUIDED / f"{clip}_K{k2}.npy"
                if not p1.exists() or not p2.exists():
                    continue
                q1 = np.load(p1)
                q2 = np.load(p2)
                r1 = risks.get((clip, k1), float("nan"))
                r2 = risks.get((clip, k2), float("nan"))
                print(f"  {clip}: K{k1}={r1:.1f}, K{k2}={r2:.1f}")
                all_frames += title_card(clip.replace("_", " ").title(),
                                         f"K={k1} risk={r1:.1f}  vs  K={k2} risk={r2:.1f}", n_frames=20)
                all_frames += render_two_panel(renderer, q1, q2, r1, r2, clip,
                                               f"K={k1}", f"K={k2}")
        save_video(all_frames, OUT_DIR / "S3_wholebody_crawling.mp4")

    # ── S4: K-scaling progression (4-panel: K=1, K=4, K=8, K=16) ─────────────
    if args.section in ("s4", "all"):
        print("=== S4: K-scaling progression (4-panel) ===")
        highlight_clips = [
            "walk_happy_dance_seed0",  # expressive, dramatic improvement
            "walk_boxing_seed0",        # expressive
            "walk_seed0",               # locomotion, partial improvement
            "walk_zombie_seed0",        # locomotion
        ]
        all_frames = []
        all_frames += title_card("S4: K-Scaling Progression",
                                 "K=1 | K=4 | K=8 | K=16  (left to right)")
        for clip in highlight_clips:
            k_clips = {}
            k_risks = {}
            for K in [1, 4, 8, 16]:
                p = DATA_GUIDED / f"{clip}_K{K}.npy"
                if not p.exists():
                    continue
                k_clips[f"K={K}"] = np.load(p)
                k_risks[f"K={K}"] = risks.get((clip, K), float("nan"))
            if len(k_clips) < 4:
                continue
            print(f"  {clip}: " + "  ".join(f"K{k}={k_risks[f'K={k}']:.1f}" for k in [1,4,8,16]))
            risk_str = "  ".join(f"K={k}:{k_risks[f'K={k}']:.1f}" for k in [1,4,8,16])
            all_frames += title_card(clip.replace("_", " ").title(), risk_str, n_frames=30)
            all_frames += render_four_panel(renderer, k_clips, k_risks, clip)
        save_video(all_frames, OUT_DIR / "S4_k_scaling.mp4")

    # ── S5: WC-K4 vs PS-K4 comparison ─────────────────────────────────────────
    if args.section in ("s5", "all"):
        print("=== S5: WC-K4 vs PS-K4 ===")
        highlight_clips = [
            "walk_boxing_seed0",
            "walk_happy_dance_seed0",
            "walk_seed0",
            "slow_walk_seed0",
        ]
        all_frames = []
        all_frames += title_card("S5: WC-K4 vs PS-K4",
                                 "Whole-Clip (left) vs Per-Segment Steering (right) | Equal compute budget")
        for clip in highlight_clips:
            p_wc = DATA_STEERED / f"{clip}_WC4.npy"
            p_ps = DATA_STEERED / f"{clip}_PS4.npy"
            if not p_wc.exists() or not p_ps.exists():
                print(f"  skip {clip}")
                continue
            q_wc = np.load(p_wc)
            q_ps = np.load(p_ps)
            r_wc = risks_wc.get(clip, float("nan"))
            r_ps = risks_ps.get(clip, float("nan"))
            print(f"  {clip}: WC={r_wc:.1f}  PS={r_ps:.1f}")
            all_frames += title_card(clip.replace("_", " ").title(),
                                     f"WC-K4 risk={r_wc:.1f}  vs  PS-K4 risk={r_ps:.1f}", n_frames=25)
            all_frames += render_two_panel(renderer, q_wc, q_ps, r_wc, r_ps, clip,
                                           "WC-K4 (whole-clip)", "PS-K4 (per-segment)")
        save_video(all_frames, OUT_DIR / "S5_wc_vs_ps.mp4")

    # ── S0: Main curated highlights ───────────────────────────────────────────
    if args.section in ("s0", "all"):
        print("=== S0: Main supplementary video (curated highlights) ===")
        # This stitches key clips together with proper section title cards
        main_frames = []

        # 1. Title
        main_frames += title_card(
            "Physics-Critic-Guided Inference-Time Scaling",
            "for Humanoid Motion Generation | CS 348K Stanford 2026",
            n_frames=90
        )

        # 2. Problem: one clip showing K=1 bad, K=8 good
        main_frames += title_card("The Problem", "K=1 (single generation) often produces physically infeasible motion", n_frames=45)
        clip = "walk_boxing_seed0"
        p1 = DATA_GUIDED / f"{clip}_K1.npy"
        p8 = DATA_GUIDED / f"{clip}_K8.npy"
        if p1.exists() and p8.exists():
            q1, q8 = np.load(p1), np.load(p8)
            r1 = risks.get((clip, 1), 22.2)
            r8 = risks.get((clip, 8), 0.0)
            main_frames += render_two_panel(renderer, q1[:90], q8[:90], r1, r8, clip,
                                            "K=1 (baseline)", "K=8 (best-of-8)")

        # 3. K-scaling: walk_happy_dance
        main_frames += title_card("Best-of-K Scaling", "More candidates → lower physics risk", n_frames=45)
        clip = "walk_happy_dance_seed0"
        k_clips = {}
        k_risks = {}
        for K in [1, 4, 8, 16]:
            p = DATA_GUIDED / f"{clip}_K{K}.npy"
            if p.exists():
                k_clips[f"K={K}"] = np.load(p)[:120]
                k_risks[f"K={K}"] = risks.get((clip, K), float("nan"))
        if len(k_clips) == 4:
            main_frames += render_four_panel(renderer, k_clips, k_risks, "walk happy dance")

        # 4. Locomotion highlights (3 clips, K=1 vs K=8)
        main_frames += title_card("Locomotion: Large K=8 Risk Reduction", "", n_frames=45)
        for clip in ["walk_seed1", "injured_walk_seed0", "walk_zombie_seed0"]:
            p1 = DATA_GUIDED / f"{clip}_K1.npy"
            p8 = DATA_GUIDED / f"{clip}_K8.npy"
            if not p1.exists():
                continue
            q1, q8 = np.load(p1), np.load(p8)
            r1 = risks.get((clip, 1), float("nan"))
            r8 = risks.get((clip, 8), float("nan"))
            main_frames += title_card(clip.replace("_", " ").title(),
                                      f"K=1: {r1:.1f} → K=8: {r8:.1f}", n_frames=20)
            main_frames += render_two_panel(renderer, q1[:90], q8[:90], r1, r8,
                                            clip, "K=1", "K=8")

        # 5. Expressive highlights
        main_frames += title_card("Expressive Motions: Large K=8 Risk Reduction", "", n_frames=45)
        for clip in ["walk_happy_dance_seed0", "walk_scared_seed0"]:
            p1 = DATA_GUIDED / f"{clip}_K1.npy"
            p8 = DATA_GUIDED / f"{clip}_K8.npy"
            if not p1.exists():
                continue
            q1, q8 = np.load(p1), np.load(p8)
            r1 = risks.get((clip, 1), float("nan"))
            r8 = risks.get((clip, 8), float("nan"))
            main_frames += title_card(clip.replace("_", " ").title(),
                                      f"K=1: {r1:.1f} → K=8: {r8:.1f}", n_frames=20)
            main_frames += render_two_panel(renderer, q1[:90], q8[:90], r1, r8,
                                            clip, "K=1", "K=8")

        # 6. Whole-body: high demand remains under the current critic
        main_frames += title_card(
            "Whole-body Crawling: Still High Demand",
            "Under this model and critic, crawling remains high demand after resampling",
            n_frames=60
        )
        clip = "hand_crawling_seed0"
        p1 = DATA_GUIDED / f"{clip}_K1.npy"
        p16 = DATA_GUIDED / f"{clip}_K16.npy"
        if p1.exists() and p16.exists():
            q1, q16 = np.load(p1), np.load(p16)
            r1 = risks.get((clip, 1), float("nan"))
            r16 = risks.get((clip, 16), float("nan"))
            main_frames += render_two_panel(renderer, q1[:90], q16[:90], r1, r16,
                                            clip, "K=1 (rejected)", "K=16 (still rejected)")

        # 7. WC-K4 vs PS-K4
        main_frames += title_card(
            "WC-K4 vs PS-K4: Whole-Clip Selection Wins",
            "Equal compute budget (~48 inference calls per clip)",
            n_frames=45
        )
        clip = "walk_boxing_seed0"
        p_wc = DATA_STEERED / f"{clip}_WC4.npy"
        p_ps = DATA_STEERED / f"{clip}_PS4.npy"
        if p_wc.exists() and p_ps.exists():
            q_wc, q_ps = np.load(p_wc), np.load(p_ps)
            r_wc = risks_wc.get(clip, float("nan"))
            r_ps = risks_ps.get(clip, float("nan"))
            main_frames += render_two_panel(renderer, q_wc[:90], q_ps[:90], r_wc, r_ps,
                                            clip, "WC-K4 (whole-clip)", "PS-K4 (per-segment)")

        save_video(main_frames, OUT_DIR / "S0_main.mp4")

    renderer.close()
    print(f"\nAll videos saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
