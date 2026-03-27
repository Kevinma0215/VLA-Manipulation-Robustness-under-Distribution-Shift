"""GIF grid assembler. Canonical location: scripts/make_grid_gif.py"""

import glob
import os
import subprocess
import sys

VIDEOS_DIR = "experiments/media/videos"
OUT_GIF    = "experiments/media/results_grid.gif"
CELL_W, CELL_H = 256, 128       # each cell (half of original 512×256)
FPS        = 12

CONDITIONS = ["nominal", "mild", "medium", "strong"]

# ── SEED SELECTION ────────────────────────────────────────────────────────────
# Each condition maps to a list of exactly 3 seed strings ("00"–"19").
SEED_SELECTION = {
    "nominal": ["00", "01", "02"],
    "mild":    ["00", "01", "14"],
    "medium":  ["02", "01", "03"],
    "strong":  ["00", "02", "19"],
}
# ─────────────────────────────────────────────────────────────────────────────

N_ROWS = len(CONDITIONS)   # 4
N_COLS = 3


def find_video(condition: str, seed: str) -> str:
    pattern = os.path.join(VIDEOS_DIR, f"{condition}_seed{seed}_*.mp4")
    matches = glob.glob(pattern)
    if not matches:
        sys.exit(f"[error] no video found for {condition} seed{seed}")
    return matches[0]


def outcome_label(path: str) -> str:
    name = os.path.basename(path)
    if "success" in name:
        return "ok"
    for ft in ("no_grasp", "drop", "wrong_place"):
        if ft in name:
            return ft.replace("_", " ")
    return ""


def build_filter(videos: list[str]) -> str:
    """Build ffmpeg filtergraph for a 4×3 grid with per-cell labels."""
    parts = []

    for i, path in enumerate(videos):
        row  = i // N_COLS
        col  = i % N_COLS
        cond = CONDITIONS[row]
        seed = SEED_SELECTION[cond][col]
        label = f"{cond} s{seed} {outcome_label(path)}"
        parts.append(
            f"[{i}:v]scale={CELL_W}:{CELL_H},"
            f"drawtext=text='{label}':x=4:y=4:"
            f"fontsize=11:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=2"
            f"[v{i}]"
        )

    for row in range(N_ROWS):
        cells = "".join(f"[v{row * N_COLS + col}]" for col in range(N_COLS))
        parts.append(f"{cells}hstack={N_COLS}[r{row}]")

    row_refs = "".join(f"[r{r}]" for r in range(N_ROWS))
    parts.append(f"{row_refs}vstack={N_ROWS}[grid]")

    return ";".join(parts)


def main():
    # Validate seed selection
    for cond in CONDITIONS:
        seeds = SEED_SELECTION.get(cond, [])
        if len(seeds) != N_COLS:
            sys.exit(f"[error] SEED_SELECTION['{cond}'] must have exactly {N_ROWS} seeds, got {seeds}")

    os.makedirs(os.path.dirname(OUT_GIF), exist_ok=True)

    # row-major order: row=condition, col=seed
    videos = [
        find_video(CONDITIONS[row], SEED_SELECTION[CONDITIONS[row]][col])
        for row in range(N_ROWS)
        for col in range(N_COLS)
    ]

    print("Videos selected (row=condition × col=seed):")
    for row in range(N_ROWS):
        for col in range(N_COLS):
            v = videos[row * N_COLS + col]
            print(f"  [{CONDITIONS[row]}, s{SEED_SELECTION[CONDITIONS[row]][col]}] {os.path.basename(v)}")

    filter_complex = build_filter(videos)
    tmp_mp4 = OUT_GIF.replace(".gif", "_tmp.mp4")

    # ── step 1: render grid to MP4 ────────────────────────────────────────────
    cmd_mp4 = (
        ["ffmpeg", "-y"]
        + [arg for v in videos for arg in ["-i", v]]
        + [
            "-filter_complex", filter_complex,
            "-map", "[grid]",
            "-r", str(FPS),
            "-c:v", "libx264", "-crf", "23", "-preset", "fast",
            tmp_mp4,
        ]
    )
    print(f"\nRendering grid MP4 → {tmp_mp4}")
    subprocess.run(cmd_mp4, check=True, stderr=subprocess.DEVNULL)

    # ── step 2: MP4 → GIF via palettegen ─────────────────────────────────────
    palette = OUT_GIF.replace(".gif", "_palette.png")

    subprocess.run([
        "ffmpeg", "-y", "-i", tmp_mp4,
        "-vf", f"fps={FPS},palettegen=max_colors=128:stats_mode=diff",
        palette,
    ], check=True, stderr=subprocess.DEVNULL)

    subprocess.run([
        "ffmpeg", "-y", "-i", tmp_mp4, "-i", palette,
        "-lavfi", f"fps={FPS}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
        OUT_GIF,
    ], check=True, stderr=subprocess.DEVNULL)

    os.remove(palette)
    os.remove(tmp_mp4)

    size_mb = os.path.getsize(OUT_GIF) / 1024 / 1024
    print(f"\nDone → {OUT_GIF}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
