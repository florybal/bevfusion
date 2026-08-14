#!/usr/bin/env python3
"""Assemble per-frame PNGs from infer_frames.py into a side-by-side evaluation video.

Layout (3 panels, left→right):
  camera  |  BEV detection  |  BEV map segmentation

Usage:
    python tools/make_eval_video.py \\
      --frames-dir output/qualitative_vis/eval_v1/frames \\
      --output     output/qualitative_vis/eval_v1/eval_video.mp4 \\
      --fps 5

Arguments:
    --frames-dir  Directory with {ts}_camera.png, {ts}_det.png, {ts}_map.png
    --output      Output MP4 path
    --fps         Frames per second (default: 5)
    --panel-size  Each panel square size in pixels (default: 640)
    --include     Which panels to include: camera det map (space-separated, default: all)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


PANEL_NAMES = ("camera", "det", "det_gt", "map", "map_gt")
PANEL_TITLES = {
    "camera":  "Câmera ZED-left + detecções projetadas",
    "det":     "Saída 2: BEV — Pred (sólido) | GT (tracejado)",
    "det_gt":  "Saída 2: BEV — Ground Truth",
    "map":     "Saída 1: Mapa + Detecções projetadas (sólido) | GT (tracejado)",
    "map_gt":  "Saída 1: Mapa + GT segmentação (verde) + GT boxes",
}


def parse_args():
    p = argparse.ArgumentParser(description="Assemble evaluation video from per-frame PNGs")
    p.add_argument("--frames-dir", required=True, help="Directory with {ts}_camera/det/map.png")
    p.add_argument("--output", required=True, help="Output MP4 file path")
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--panel-size", type=int, default=640, help="Square size for each panel")
    p.add_argument("--include", nargs="+", default=["camera", "det", "map"],
                   choices=list(PANEL_NAMES),
                   help="Painéis a incluir em sequência (padrão: camera det map). "
                        "Use 'det_gt' para GT-only det, 'map_gt' para GT mapa.")
    p.add_argument("--title-bar", type=int, default=36, help="Height of title bar in pixels (0 to disable)")
    p.add_argument("--font-scale", type=float, default=0.55)
    return p.parse_args()


def _collect_frames(frames_dir: Path, panels: List[str]) -> List[str]:
    """Return sorted list of timestamps that have at least one requested panel."""
    timestamps = set()
    for panel in panels:
        for f in frames_dir.glob(f"*_{panel}.png"):
            timestamps.add(f.stem.replace(f"_{panel}", ""))
    return sorted(timestamps)


def _load_panel(frames_dir: Path, ts: str, panel: str, size: int) -> np.ndarray:
    path = frames_dir / f"{ts}_{panel}.png"
    if path.exists():
        img = cv2.imread(str(path))
        if img is not None:
            return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    # placeholder if panel is missing
    placeholder = np.full((size, size, 3), 30, dtype=np.uint8)
    cv2.putText(placeholder, f"[{panel} N/A]", (size // 2 - 60, size // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 140, 140), 1, cv2.LINE_AA)
    return placeholder


def _make_title_bar(text: str, width: int, height: int, font_scale: float) -> np.ndarray:
    bar = np.full((height, width, 3), 20, dtype=np.uint8)
    cv2.putText(bar, text, (12, height - 8), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (220, 220, 220), 1, cv2.LINE_AA)
    return bar


def _compose_frame(frames_dir: Path, ts: str, panels: List[str],
                   panel_size: int, title_height: int, font_scale: float) -> np.ndarray:
    strips = []
    for panel in panels:
        img = _load_panel(frames_dir, ts, panel, panel_size)
        if title_height > 0:
            title_bar = _make_title_bar(PANEL_TITLES[panel], panel_size, title_height, font_scale)
            img = np.vstack([title_bar, img])
        strips.append(img)

    frame = np.hstack(strips)

    # bottom info bar
    if title_height > 0:
        info = _make_title_bar(f"frame: {ts}", frame.shape[1], title_height, font_scale)
        frame = np.vstack([frame, info])

    return frame


def _write_with_opencv(frames_dir: Path, timestamps: List[str], panels: List[str],
                        output: Path, fps: float, panel_size: int,
                        title_height: int, font_scale: float) -> bool:
    test_frame = _compose_frame(frames_dir, timestamps[0], panels,
                                panel_size, title_height, font_scale)
    h, w = test_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (w, h))
    if not writer.isOpened():
        return False

    for i, ts in enumerate(timestamps):
        frame = _compose_frame(frames_dir, ts, panels, panel_size, title_height, font_scale)
        if frame.shape[0] != h or frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(timestamps)}] {ts}")

    writer.release()
    return True


def _write_with_ffmpeg(frames_dir: Path, timestamps: List[str], panels: List[str],
                        output: Path, fps: float, panel_size: int,
                        title_height: int, font_scale: float, tmp_dir: Path):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for i, ts in enumerate(timestamps):
        frame = _compose_frame(frames_dir, ts, panels, panel_size, title_height, font_scale)
        cv2.imwrite(str(tmp_dir / f"{i:06d}.png"), frame)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(timestamps)}] {ts}")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(tmp_dir / "%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output),
    ]
    subprocess.run(cmd, check=True)
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    args = parse_args()
    frames_dir = Path(args.frames_dir).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    panels = [p for p in PANEL_NAMES if p in args.include]
    if not panels:
        print("Nenhum painel selecionado.")
        sys.exit(1)

    timestamps = _collect_frames(frames_dir, panels)
    if not timestamps:
        print(f"Nenhum frame encontrado em {frames_dir}")
        sys.exit(1)

    print(f"Frames encontrados: {len(timestamps)}")
    print(f"Painéis: {' | '.join(panels)}")
    print(f"FPS: {args.fps}  |  Painel: {args.panel_size}px")
    print(f"Saída: {output}")

    ok = _write_with_opencv(frames_dir, timestamps, panels,
                             output, args.fps, args.panel_size,
                             args.title_bar, args.font_scale)
    if not ok:
        print("OpenCV VideoWriter falhou, tentando ffmpeg...")
        tmp_dir = output.parent / "_tmp_frames"
        _write_with_ffmpeg(frames_dir, timestamps, panels,
                            output, args.fps, args.panel_size,
                            args.title_bar, args.font_scale, tmp_dir)

    print(f"\nVídeo salvo em: {output}")


if __name__ == "__main__":
    main()
