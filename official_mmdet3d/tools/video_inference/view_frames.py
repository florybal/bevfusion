#!/usr/bin/env python3
"""Visualizador interativo de frames gerados por infer_frames.py.

Navega pelos PNGs de {frames-dir} com setas ← → ou botões.
Também suporta modo batch (--save-all) para exportar composites.

Uso:
    python tools/view_frames.py \\
      --frames-dir output/qualitative_vis/fix_v3/frames

    python tools/view_frames.py \\
      --frames-dir output/qualitative_vis/fix_v3/frames \\
      --panels camera map map_gt --start 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np


PANEL_NAMES = ("camera", "det", "det_gt", "map", "map_gt")
PANEL_TITLES = {
    "camera":  "Câmera ZED-left",
    "det":     "Saída 2 — BEV: Pred + GT",
    "det_gt":  "Saída 2 — BEV: apenas GT",
    "map":     "Saída 1 — Mapa + Pred BEV",
    "map_gt":  "Saída 1 — Mapa + GT BEV",
}

BG      = "#1c1c1c"
PANEL_BG = "#111111"
TEXT    = "#d0d0d0"
BTN_C   = "#2e2e2e"
BTN_H   = "#484848"


def parse_args():
    p = argparse.ArgumentParser(description="Visualizador interativo de inferências")
    p.add_argument("--frames-dir", required=True,
                   help="Dir com {ts}_{panel}.png gerados por infer_frames.py")
    p.add_argument("--panels", nargs="+", default=["camera", "det", "map"],
                   choices=list(PANEL_NAMES),
                   help="Painéis a exibir (padrão: camera det map)")
    p.add_argument("--start", type=int, default=0, help="Índice inicial (1-based)")
    p.add_argument("--save-all", default="",
                   help="Exporta composite de todos os frames para esse dir e sai")
    p.add_argument("--dpi", type=int, default=110)
    return p.parse_args()


def collect_timestamps(frames_dir: Path, panels: List[str]) -> List[str]:
    ts_set: set = set()
    for panel in panels:
        for f in frames_dir.glob(f"*_{panel}.png"):
            ts_set.add(f.stem.replace(f"_{panel}", ""))
    return sorted(ts_set)


def load_panel_img(frames_dir: Path, ts: str, panel: str) -> Optional[np.ndarray]:
    path = frames_dir / f"{ts}_{panel}.png"
    if not path.exists():
        return None
    try:
        import cv2
        img = cv2.imread(str(path))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    except ImportError:
        from PIL import Image as _PIL
        return np.asarray(_PIL.open(path).convert("RGB"))


def main():
    args = parse_args()
    frames_dir = Path(args.frames_dir).resolve()

    panels = [p for p in PANEL_NAMES if p in args.panels]
    if not panels:
        print("Nenhum painel selecionado."); sys.exit(1)

    timestamps = collect_timestamps(frames_dir, panels)
    if not timestamps:
        print(f"Nenhum frame encontrado em {frames_dir}"); sys.exit(1)

    total = len(timestamps)
    print(f"Frames: {total}  |  Painéis: {' | '.join(panels)}")
    print("Controles: ← → para navegar  ·  Q / Esc para sair  ·  campo 'Ir:' para saltar")

    # ── matplotlib setup ──────────────────────────────────────────────────────
    import matplotlib
    if not args.save_all:
        for backend in ("TkAgg", "Qt5Agg", "GTK3Agg", "MacOSX"):
            try:
                matplotlib.use(backend); break
            except Exception:
                continue
    else:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, TextBox

    n = len(panels)
    fig_w = max(11, 4.6 * n)
    fig = plt.figure(figsize=(fig_w, 6.0), facecolor=BG)
    try:
        fig.canvas.manager.set_window_title("Visualizador de Inferências — UniTR")
    except Exception:
        pass

    # ── panel axes ────────────────────────────────────────────────────────────
    pad   = 0.008
    ax_y  = 0.18
    ax_h  = 0.78
    axes  = []
    for i, panel in enumerate(panels):
        x = i / n + pad
        w = 1 / n - 2 * pad
        ax = fig.add_axes([x, ax_y, w, ax_h], facecolor=PANEL_BG)
        ax.axis("off")
        ax.set_title(PANEL_TITLES.get(panel, panel), fontsize=8,
                     color=TEXT, pad=3)
        axes.append(ax)

    # ── info bar ──────────────────────────────────────────────────────────────
    info_ax = fig.add_axes([0.0, 0.960, 1.0, 0.038], facecolor=BG)
    info_ax.axis("off")
    info_txt = info_ax.text(0.5, 0.5, "", ha="center", va="center",
                             fontsize=9, color=TEXT)

    # ── bottom controls ───────────────────────────────────────────────────────
    bstyle = dict(color=BTN_C, hovercolor=BTN_H)

    btn_prev = Button(fig.add_axes([0.28, 0.03, 0.11, 0.08]), "◀  Anterior", **bstyle)
    btn_next = Button(fig.add_axes([0.61, 0.03, 0.11, 0.08]), "Próximo  ▶", **bstyle)

    cnt_ax = fig.add_axes([0.40, 0.03, 0.20, 0.08], facecolor="#222")
    cnt_ax.axis("off")
    cnt_txt = cnt_ax.text(0.5, 0.5, "", ha="center", va="center",
                           fontsize=11, color="white", fontweight="bold")

    jump_ax = fig.add_axes([0.77, 0.03, 0.08, 0.08])
    go_ax   = fig.add_axes([0.86, 0.03, 0.06, 0.08])
    jump_box = TextBox(jump_ax, "Ir: ", initial="", color="#2e2e2e", hovercolor="#484848")
    btn_go   = Button(go_ax, "OK", **bstyle)

    for btn in (btn_prev, btn_next, btn_go):
        btn.label.set_color(TEXT)
        btn.label.set_fontsize(9)

    # ── state ─────────────────────────────────────────────────────────────────
    cur = [max(0, min(args.start - 1, total - 1))]

    def render(idx: int):
        ts = timestamps[idx]
        for ax, panel in zip(axes, panels):
            ax.cla(); ax.axis("off")
            ax.set_facecolor(PANEL_BG)
            ax.set_title(PANEL_TITLES.get(panel, panel), fontsize=8, color=TEXT, pad=3)
            img = load_panel_img(frames_dir, ts, panel)
            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, f"[{panel}\nN/A]", ha="center", va="center",
                        transform=ax.transAxes, color="#555", fontsize=9)
        info_txt.set_text(f"Frame  {idx + 1} / {total}    ·    ts = {ts}")
        cnt_txt.set_text(f"{idx + 1} / {total}")
        fig.canvas.draw_idle()

    def go_prev(_):
        cur[0] = max(0, cur[0] - 1); render(cur[0])

    def go_next(_):
        cur[0] = min(total - 1, cur[0] + 1); render(cur[0])

    def go_jump(_=None):
        try:
            val = int(jump_box.text.strip()) - 1
            if 0 <= val < total:
                cur[0] = val; render(cur[0])
        except ValueError:
            pass

    def on_key(event):
        if   event.key in ("right", "d", "n"): go_next(None)
        elif event.key in ("left",  "a", "p"): go_prev(None)
        elif event.key in ("q", "escape"):      plt.close("all")

    btn_prev.on_clicked(go_prev)
    btn_next.on_clicked(go_next)
    btn_go.on_clicked(go_jump)
    jump_box.on_submit(lambda _: go_jump())
    fig.canvas.mpl_connect("key_press_event", on_key)

    # ── batch export mode ─────────────────────────────────────────────────────
    if args.save_all:
        save_dir = Path(args.save_all)
        save_dir.mkdir(parents=True, exist_ok=True)
        for i in range(total):
            render(i)
            out = save_dir / f"{timestamps[i]}_composite.png"
            fig.savefig(str(out), dpi=args.dpi, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"  [{i + 1}/{total}] {out.name}")
        print(f"\nComposites salvos em: {save_dir}")
        return

    render(cur[0])
    plt.show()


if __name__ == "__main__":
    main()
