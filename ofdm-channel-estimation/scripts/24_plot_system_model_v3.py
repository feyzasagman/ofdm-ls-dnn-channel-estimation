"""Figure 2.1 v3 — hand-placed SVG block diagram (fixed coordinates)."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "data" / "results" / "revision" / "final_figures"
STEM = "fig_system_model_v3"

W, H = 1600, 900
FONT = "Arial, Helvetica, sans-serif"
FONT_MAIN = 18
FONT_SUB = 16
FONT_SMALL = 14
STROKE = 2.0
R = 10

# ── fixed layout (pixels) ───────────────────────────────────────────────────
STD_W, STD_H = 150, 90
GAP = 24
MARGIN_L = 48

Y_TX = 215
Y_RX = 565

TX: list[tuple] = [
    (0, ["Binary Data"]),
    (1, ["QPSK", "Modulation"]),
    (2, ["Pilot", "Insertion"]),
    (3, ["IFFT"]),
    (4, ["CP", "Addition"]),
]

CH_X = MARGIN_L + 5 * (STD_W + GAP)
CH_W, CH_H = 304, 182
CH_Y = Y_TX - CH_H // 2

RX: list[tuple] = [
    (0, ["CP", "Removal"]),
    (1, ["FFT"]),
    (2, ["Pilot", "Extraction"]),
]

EST_X = MARGIN_L + 3 * (STD_W + GAP)
EST_W, EST_H = 440, 220
EST_Y = Y_RX - EST_H // 2

EQ_X = EST_X + EST_W + GAP
DEM_X = EQ_X + STD_W + GAP
BITS_X = DEM_X + STD_W + GAP

TAIL: list[tuple] = [
    (EQ_X, ["Equalization"]),
    (DEM_X, ["QPSK", "Demodulation"]),
    (BITS_X, ["Recovered", "Bits"]),
]

CAPTION = (
    "Figure 2.1. Block diagram of the SISO-OFDM transmission system and the "
    "channel estimation approaches evaluated in this study."
)


def _x_slot(i: int) -> int:
    return MARGIN_L + i * (STD_W + GAP)


def _box_xy(i: int, y_center: int) -> tuple[int, int]:
    x = _x_slot(i)
    return x, y_center - STD_H // 2


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _svg_header() -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}">',
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>',
        "<defs>",
        f'<style><![CDATA[',
        f"  .main {{ font-family: {FONT}; font-size: {FONT_MAIN}px; fill: #111111; }}",
        f"  .sub {{ font-family: {FONT}; font-size: {FONT_SUB}px; fill: #333333; }}",
        f"  .small {{ font-family: {FONT}; font-size: {FONT_SMALL}px; fill: #444444; }}",
        f"  .section {{ font-family: {FONT}; font-size: {FONT_SUB}px; fill: #555555; "
        f"font-weight: 600; }}",
        "]]></style>",
        "</defs>",
    ]


def _rounded_rect(x: int, y: int, w: int, h: int, fill: str = "#ffffff") -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{R}" ry="{R}" '
        f'fill="{fill}" stroke="#222222" stroke-width="{STROKE}"/>'
    )


def _text(cx: int, cy: int, label: str, cls: str = "main", lines: list[str] | None = None) -> str:
    if lines is None:
        lines = label.split("\n")
    if len(lines) == 1:
        return (
            f'<text x="{cx}" y="{cy}" class="{cls}" text-anchor="middle" '
            f'dominant-baseline="middle">{_esc(lines[0])}</text>'
        )
    start_y = cy - (len(lines) - 1) * 10
    parts = [f'<text x="{cx}" y="{start_y}" class="{cls}" text-anchor="middle">']
    for i, line in enumerate(lines):
        dy = 20 if i > 0 else 0
        parts.append(f'<tspan x="{cx}" dy="{dy}">{_esc(line)}</tspan>')
    parts.append("</text>")
    return "\n".join(parts)


def _arrow_h(x1: int, y: int, x2: int) -> str:
    return (
        f'<line x1="{x1}" y1="{y}" x2="{x2 - 10}" y2="{y}" stroke="#222222" '
        f'stroke-width="{STROKE}" marker-end="url(#arrowhead)"/>'
    )


def _arrow_v(x: int, y1: int, y2: int) -> str:
    return (
        f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2 - 10}" stroke="#222222" '
        f'stroke-width="{STROKE}" marker-end="url(#arrowhead)"/>'
    )


def _arrow_l(x1: int, y1: int, x2: int, y2: int) -> str:
    """Orthogonal L-shaped connector (vertical then horizontal or vice versa)."""
    mid_y = y2
    return "\n".join(
        [
            f'<polyline points="{x1},{y1} {x1},{mid_y} {x2},{y2}" fill="none" '
            f'stroke="#222222" stroke-width="{STROKE}" marker-end="url(#arrowhead)"/>'
        ]
    )


def build_svg() -> str:
    lines = _svg_header()
    lines.append(
        '<marker id="arrowhead" markerWidth="10" markerHeight="8" refX="9" refY="4" '
        'orient="auto"><polygon points="0,0 10,4 0,8" fill="#222222"/></marker>'
    )

    lines.append(_text(130, 72, "Section A — Transmitter + Channel", "section"))
    lines.append(_text(130, 470, "Section B — Receiver", "section"))

    # Transmitter boxes
    for i, line_list in TX:
        x, y = _box_xy(i, Y_TX)
        lines.append(_rounded_rect(x, y, STD_W, STD_H))
        lines.append(_text(x + STD_W // 2, Y_TX, "", lines=line_list))

    for i in range(4):
        x1 = _x_slot(i) + STD_W + 4
        x2 = _x_slot(i + 1) - 4
        lines.append(_arrow_h(x1, Y_TX, x2))

    # Channel box
    lines.append(_rounded_rect(CH_X, CH_Y, CH_W, CH_H, fill="#f4f4f4"))
    ch_cx = CH_X + CH_W // 2
    lines.append(_text(ch_cx, CH_Y + 40, "Channel", "main"))
    lines.append(_text(ch_cx, CH_Y + 74, "AWGN", "sub"))
    lines.append(_text(ch_cx, CH_Y + 100, "Rayleigh Flat Fading", "sub"))
    lines.append(_text(ch_cx, CH_Y + 126, "Rician Flat Fading", "sub"))
    lines.append(_text(ch_cx, CH_Y + CH_H - 20, "Single-tap channel models", "small"))

    # CP Addition → Channel
    cp_add_r = _x_slot(4) + STD_W + 4
    lines.append(_arrow_h(cp_add_r, Y_TX, CH_X - 4))

    # Channel → CP Removal (single orthogonal connector)
    ch_bottom = CH_Y + CH_H
    cp_x, cp_y = _box_xy(0, Y_RX)
    cp_cx = cp_x + STD_W // 2
    cp_top = cp_y
    ch_cx = CH_X + CH_W // 2
    mid_y = 370
    lines.append(_arrow_v(ch_cx, ch_bottom + 2, mid_y))
    lines.append(
        f'<line x1="{ch_cx}" y1="{mid_y}" x2="{cp_cx + 10}" y2="{mid_y}" '
        f'stroke="#222222" stroke-width="{STROKE}"/>'
    )
    lines.append(_arrow_v(cp_cx, mid_y, cp_top - 2))

    # Receiver front-end
    for i, line_list in RX:
        x, y = _box_xy(i, Y_RX)
        lines.append(_rounded_rect(x, y, STD_W, STD_H))
        lines.append(_text(x + STD_W // 2, Y_RX, "", lines=line_list))

    for i in range(2):
        x1 = _x_slot(i) + STD_W + 4
        x2 = _x_slot(i + 1) - 4
        lines.append(_arrow_h(x1, Y_RX, x2))

    # Pilot Extraction → Channel Estimation
    pilot_r = _x_slot(2) + STD_W + 4
    lines.append(_arrow_h(pilot_r, Y_RX, EST_X - 4))

    # Channel Estimation container
    lines.append(_rounded_rect(EST_X, EST_Y, EST_W, EST_H, fill="#f4f4f4"))
    est_cx = EST_X + EST_W // 2
    lines.append(_text(est_cx, EST_Y + 28, "Channel Estimation", "main"))

    pad = 26
    cell_w = (EST_W - 3 * pad) // 2
    cell_h = 58
    gx0 = EST_X + pad
    gx1 = gx0 + cell_w + pad
    gy0 = EST_Y + 54
    gy1 = gy0 + cell_h + pad

    grid = [
        (["LS"], gx0, gy0),
        (["Simplified-", "MMSE"], gx1, gy0),
        (["LMMSE-flat"], gx0, gy1),
        (["LS+DNN"], gx1, gy1),
    ]
    for line_list, gx, gy in grid:
        lines.append(_rounded_rect(gx, gy, cell_w, cell_h, fill="#ffffff"))
        lines.append(_text(gx + cell_w // 2, gy + cell_h // 2, "", "sub", lines=line_list))

    dnn_cx = gx1 + cell_w // 2
    lines.append(
        _text(
            dnn_cx,
            gy1 + cell_h + 24,
            "LS estimate + residual DNN correction",
            "small",
        )
    )

    # Channel Estimation → tail
    est_r = EST_X + EST_W + 4
    lines.append(_arrow_h(est_r, Y_RX, EQ_X - 4))

    for x, line_list in TAIL:
        y = Y_RX - STD_H // 2
        lines.append(_rounded_rect(x, y, STD_W, STD_H))
        lines.append(_text(x + STD_W // 2, Y_RX, "", lines=line_list))

    lines.append(_arrow_h(EQ_X + STD_W + 4, Y_RX, DEM_X - 4))
    lines.append(_arrow_h(DEM_X + STD_W + 4, Y_RX, BITS_X - 4))

    lines.append("</svg>")
    return "\n".join(lines)


def _verify_layout() -> None:
    """Basic overlap / bounds checks before export."""
    checks = []

    def rect(x, y, w, h, name):
        if x < 0 or y < 0 or x + w > W or y + h > H:
            checks.append(f"OUT OF BOUNDS: {name}")
        return (x, y, x + w, y + h)

    boxes = []
    names = []
    for i, _ in TX:
        x, y = _box_xy(i, Y_TX)
        boxes.append(rect(x, y, STD_W, STD_H, f"TX-{i}"))
        names.append(f"TX-{i}")
    boxes.append(rect(CH_X, CH_Y, CH_W, CH_H, "Channel"))
    names.append("Channel")
    for i, _ in RX:
        x, y = _box_xy(i, Y_RX)
        boxes.append(rect(x, y, STD_W, STD_H, f"RX-{i}"))
        names.append(f"RX-{i}")
    boxes.append(rect(EST_X, EST_Y, EST_W, EST_H, "Channel Estimation"))
    names.append("Channel Estimation")
    for x, _ in TAIL:
        boxes.append(rect(x, Y_RX - STD_H // 2, STD_W, STD_H, "tail"))
        names.append("tail")

    for i, a in enumerate(boxes):
        for j, b in enumerate(boxes[i + 1 :], start=i + 1):
            if a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]:
                checks.append(f"OVERLAP: {names[i]} vs {names[j]}")

    if BITS_X + STD_W > W - 20:
        checks.append("Recovered Bits too close to right margin")

    print("Layout verification:")
    if checks:
        for c in checks:
            print(f"  [WARN] {c}")
    else:
        print("  [OK] All boxes within canvas, no overlaps detected")


def export_raster_from_svg(svg_path: Path, png_path: Path, pdf_path: Path) -> None:
    """Export PNG/PDF via cairosvg, fallback to matplotlib."""
    try:
        import cairosvg

        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=W, output_height=H)
        cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
        print("Exported PNG/PDF via cairosvg")
        return
    except Exception as exc:
        print(f"cairosvg unavailable ({exc}), using matplotlib fallback")

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def box(x, y, w, h, fill="#ffffff"):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle=f"round,pad=0.02,rounding_size={R}",
                linewidth=STROKE / 1.5, edgecolor="#222", facecolor=fill,
            )
        )

    def txt(cx, cy, line_list, fs=FONT_MAIN):
        if len(line_list) == 1:
            ax.text(cx, cy, line_list[0], ha="center", va="center", fontsize=fs, family="sans-serif")
        else:
            ax.text(cx, cy, "\n".join(line_list), ha="center", va="center", fontsize=fs, family="sans-serif")

    def arr_h(x1, y, x2):
        ax.add_patch(
            FancyArrowPatch((x1, y), (x2 - 8, y), arrowstyle="-|>", mutation_scale=12,
                            linewidth=STROKE / 1.5, color="#222")
        )

    for i, line_list in TX:
        x, y = _box_xy(i, Y_TX)
        box(x, y, STD_W, STD_H)
        txt(x + STD_W // 2, Y_TX, line_list)
    for i in range(4):
        arr_h(_x_slot(i) + STD_W + 4, Y_TX, _x_slot(i + 1) - 4)

    box(CH_X, CH_Y, CH_W, CH_H, "#f4f4f4")
    ch_cx = CH_X + CH_W // 2
    txt(ch_cx, CH_Y + 40, ["Channel"])
    txt(ch_cx, CH_Y + 74, ["AWGN"], FONT_SUB)
    txt(ch_cx, CH_Y + 100, ["Rayleigh Flat Fading"], FONT_SUB)
    txt(ch_cx, CH_Y + 126, ["Rician Flat Fading"], FONT_SUB)
    txt(ch_cx, CH_Y + CH_H - 20, ["Single-tap channel models"], FONT_SMALL)
    arr_h(_x_slot(4) + STD_W + 4, Y_TX, CH_X - 4)

    cp_x, cp_y = _box_xy(0, Y_RX)
    cp_cx = cp_x + STD_W // 2
    mid_y = 370
    ax.plot([ch_cx, ch_cx], [CH_Y + CH_H + 2, mid_y], color="#222", lw=STROKE / 1.5)
    ax.plot([ch_cx, cp_cx], [mid_y, mid_y], color="#222", lw=STROKE / 1.5)
    ax.add_patch(
        FancyArrowPatch((cp_cx, mid_y), (cp_cx, cp_y - 2), arrowstyle="-|>", mutation_scale=12,
                        linewidth=STROKE / 1.5, color="#222")
    )

    for i, line_list in RX:
        x, y = _box_xy(i, Y_RX)
        box(x, y, STD_W, STD_H)
        txt(x + STD_W // 2, Y_RX, line_list)
    for i in range(2):
        arr_h(_x_slot(i) + STD_W + 4, Y_RX, _x_slot(i + 1) - 4)
    arr_h(_x_slot(2) + STD_W + 4, Y_RX, EST_X - 4)

    box(EST_X, EST_Y, EST_W, EST_H, "#f4f4f4")
    txt(EST_X + EST_W // 2, EST_Y + 28, ["Channel Estimation"])
    pad = 26
    cell_w = (EST_W - 3 * pad) // 2
    cell_h = 58
    gx0, gx1 = EST_X + pad, EST_X + pad + cell_w + pad
    gy0, gy1 = EST_Y + 54, EST_Y + 54 + cell_h + pad
    for line_list, gx, gy in [
        (["LS"], gx0, gy0),
        (["Simplified-", "MMSE"], gx1, gy0),
        (["LMMSE-flat"], gx0, gy1),
        (["LS+DNN"], gx1, gy1),
    ]:
        box(gx, gy, cell_w, cell_h)
        txt(gx + cell_w // 2, gy + cell_h // 2, line_list, FONT_SUB)
    txt(gx1 + cell_w // 2, gy1 + cell_h + 24, ["LS estimate + residual DNN correction"], FONT_SMALL)

    arr_h(EST_X + EST_W + 4, Y_RX, EQ_X - 4)
    for x, line_list in TAIL:
        box(x, Y_RX - STD_H // 2, STD_W, STD_H)
        txt(x + STD_W // 2, Y_RX, line_list)
    arr_h(EQ_X + STD_W + 4, Y_RX, DEM_X - 4)
    arr_h(DEM_X + STD_W + 4, Y_RX, BITS_X - 4)

    fig.savefig(png_path, dpi=100, bbox_inches="tight", pad_inches=0, facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)
    print("Exported PNG/PDF via matplotlib fallback")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _verify_layout()

    svg_content = build_svg()
    svg_path = OUTPUT_DIR / f"{STEM}.svg"
    svg_path.write_text(svg_content, encoding="utf-8")
    print(f"Saved: {svg_path}")

    # Validate SVG parses
    ET.fromstring(svg_content)

    png_path = OUTPUT_DIR / f"{STEM}.png"
    pdf_path = OUTPUT_DIR / f"{STEM}.pdf"
    export_raster_from_svg(svg_path, png_path, pdf_path)

    meta = {
        "figure_id": "Figure 2.1 (v3)",
        "canvas_px": [W, H],
        "method": "fixed-coordinate manual SVG",
        "caption": CAPTION,
        "notes": [
            "Residual DNN detail deferred to Figure 2.2",
            "No automatic graph layout engines used",
        ],
    }
    (OUTPUT_DIR / f"{STEM}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (OUTPUT_DIR / f"{STEM}_caption.txt").write_text(CAPTION + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
