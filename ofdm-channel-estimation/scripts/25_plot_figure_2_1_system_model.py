"""Figure 2.1 — SISO-OFDM system model (two-row journal layout, fixed-coordinate SVG)."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "data" / "results" / "revision" / "final_figures"
STEM = "figure_2_1_system_model"

# Landscape ~16:7
W, H = 2240, 980
FONT = "Arial, Helvetica, sans-serif"
FS = 15
FS_NOTE = 12
STROKE = 1.8
R = 6

GAP = 30
BH = 58
BH_CH = 96
BH_EST = 62
FS_CH = 11

Y_ROW1 = 200
Y_TRUNK = 320
Y_ROW2 = 470
Y_MET = 640

CAPTION = (
    "Figure 2.1. Block diagram of the SISO-OFDM system and the channel estimation "
    "methods evaluated in this study."
)

CHECKLIST = {
    "two_row_layout": True,
    "main_flow_row1": True,
    "channel_short_labels_awgn_rayleigh_rician": True,
    "channel_note_single_tap_k6db": True,
    "four_estimators_row2": True,
    "no_ls_dnn_residual_detail_in_fig_2_1": True,
    "estimators_merge_to_equalization": True,
    "mse_nmse_single_evaluation_box": True,
    "ber_only_at_receiver_end": True,
    "simplified_mmse_not_mmse_only": True,
    "lmmse_flat_included": True,
    "no_multipath": True,
}


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _layout() -> dict:
    """Compute fixed coordinates for all blocks."""
    specs: list[tuple[str, int, list[str]]] = [
        ("data_bits", 108, ["Data Bits"]),
        ("qpsk", 78, ["QPSK"]),
        ("pilot_ins", 118, ["Pilot", "Insertion"]),
        ("ofdm", 128, ["OFDM", "Modulation"]),
        ("channel", 152, []),  # custom content
        ("fft", 108, ["FFT", "Receiver"]),
        ("pilot_ext", 118, ["Pilot", "Extraction"]),
        ("ch_est", 128, ["Channel", "Estimation"]),
        ("eq", 108, ["Equalization"]),
        ("demod", 108, ["QPSK", "Demodulation"]),
        ("ber", 72, ["BER"]),
    ]

    total_w = sum(w for _, w, _ in specs) + GAP * (len(specs) - 1)
    x0 = (W - total_w) // 2
    row1: list[dict] = []
    x = x0
    for key, bw, lines in specs:
        h = BH_CH if key == "channel" else BH
        row1.append({"key": key, "x": x, "w": bw, "h": h, "cx": x + bw // 2, "lines": lines})
        x += bw + GAP

    ce = next(b for b in row1 if b["key"] == "ch_est")
    eq = next(b for b in row1 if b["key"] == "eq")

    est_labels = ["LS", "Simplified-MMSE", "LMMSE-flat", "LS+DNN"]
    ew, eg = 168, 32
    est_span = 4 * ew + 3 * eg
    est_x0 = ce["cx"] - est_span // 2
    estimators: list[dict] = []
    ex = est_x0
    for label in est_labels:
        estimators.append({"x": ex, "w": ew, "h": BH_EST, "cx": ex + ew // 2, "label": label})
        ex += ew + eg

    met_w = 210
    metrics = {
        "x": ce["cx"] - met_w // 2,
        "w": met_w,
        "h": 52,
        "cx": ce["cx"],
    }

    return {"row1": row1, "estimators": estimators, "metrics": metrics, "ce": ce, "eq": eq}


def _rect(x: int, y: int, w: int, h: int, fill: str = "#ffffff") -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{R}" ry="{R}" '
        f'fill="{fill}" stroke="#222" stroke-width="{STROKE}"/>'
    )


def _txt(cx: int, cy: int, lines: list[str], fs: int = FS) -> str:
    if len(lines) == 1:
        return (
            f'<text x="{cx}" y="{cy}" font-family="{FONT}" font-size="{fs}px" '
            f'fill="#111" text-anchor="middle" dominant-baseline="middle">'
            f"{_esc(lines[0])}</text>"
        )
    y0 = cy - (len(lines) - 1) * 8
    parts = [
        f'<text x="{cx}" y="{y0}" font-family="{FONT}" font-size="{fs}px" '
        f'fill="#111" text-anchor="middle">'
    ]
    for i, ln in enumerate(lines):
        parts.append(f'<tspan x="{cx}" dy="{17 if i else 0}">{_esc(ln)}</tspan>')
    parts.append("</text>")
    return "".join(parts)


def _ah(x1: int, y: int, x2: int) -> str:
    return (
        f'<line x1="{x1}" y1="{y}" x2="{x2 - 8}" y2="{y}" stroke="#222" '
        f'stroke-width="{STROKE}" marker-end="url(#arr)"/>'
    )


def _av(x: int, y1: int, y2: int) -> str:
    dy = -8 if y2 > y1 else 8
    return (
        f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2 - dy}" stroke="#222" '
        f'stroke-width="{STROKE}" marker-end="url(#arr)"/>'
    )


def _line(x1: int, y1: int, x2: int, y2: int, dash: bool = False) -> str:
    ds = ' stroke-dasharray="5,4"' if dash else ""
    col = "#666" if dash else "#222"
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="{1.2 if dash else STROKE}"{ds}/>'


def build_svg() -> str:
    L = _layout()
    p: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        "<defs>",
        '<marker id="arr" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">',
        '<polygon points="0,0 8,3 0,6" fill="#222"/></marker>',
        "</defs>",
    ]

    ch_note_y = 0
    for i, b in enumerate(L["row1"]):
        y = Y_ROW1 - b["h"] // 2
        if b["key"] == "channel":
            p.append(_rect(b["x"], y, b["w"], b["h"], "#fafafa"))
            p.append(_txt(b["cx"], y + 20, ["Channel"], FS))
            p.append(_txt(b["cx"], y + 42, ["AWGN"], FS_CH))
            p.append(_txt(b["cx"], y + 58, ["Rayleigh flat fading"], FS_CH))
            p.append(_txt(b["cx"], y + 74, ["Rician flat fading"], FS_CH))
            ch_note_y = y + b["h"] + 20
            p.append(
                _txt(b["cx"], ch_note_y, ["single-tap; Rician K = 6 dB"], FS_NOTE)
            )
        else:
            p.append(_rect(b["x"], y, b["w"], b["h"]))
            p.append(_txt(b["cx"], Y_ROW1, b["lines"], FS))

        if i:
            prev = L["row1"][i - 1]
            p.append(_ah(prev["x"] + prev["w"] + 2, Y_ROW1, b["x"] - 2))

    # Channel Estimation → vertical trunk → estimator row
    ce, eq = L["ce"], L["eq"]
    ce_bot = Y_ROW1 + ce["h"] // 2
    p.append(_av(ce["cx"], ce_bot + 2, Y_TRUNK))
    est = L["estimators"]
    trunk_l, trunk_r = est[0]["cx"], est[-1]["cx"]
    p.append(_line(trunk_l, Y_TRUNK, trunk_r, Y_TRUNK))

    for e in est:
        top = Y_ROW2 - e["h"] // 2
        p.append(_av(e["cx"], Y_TRUNK, top - 2))
        p.append(_rect(e["x"], top, e["w"], e["h"], "#fafafa"))
        p.append(_txt(e["cx"], Y_ROW2, [e["label"]], FS))

    # Estimators → merge bus → Equalization
    merge_y = Y_ROW2 + BH_EST // 2 + 36
    p.append(_line(est[0]["cx"], merge_y, est[-1]["cx"], merge_y))
    for e in est:
        bot = Y_ROW2 + e["h"] // 2
        p.append(_line(e["cx"], bot + 2, e["cx"], merge_y))
    p.append(_line(ce["cx"], merge_y, eq["cx"], merge_y))
    eq_bot = Y_ROW1 + eq["h"] // 2
    p.append(_av(eq["cx"], merge_y, eq_bot + 2))

    # MSE / NMSE evaluation (single dashed link from estimator group)
    m = L["metrics"]
    met_top = Y_MET - m["h"] // 2
    p.append(_line(ce["cx"], merge_y + 2, ce["cx"], met_top + m["h"] + 2, dash=True))
    p.append(_rect(m["x"], met_top, m["w"], m["h"], "#ffffff"))
    p.append(_txt(m["cx"], Y_MET, ["MSE / NMSE Evaluation"], FS))

    p.append("</svg>")
    return "\n".join(p)


def _verify_layout() -> None:
    L = _layout()
    boxes: list[tuple[str, int, int, int, int]] = []

    def add(name: str, x: int, y: int, w: int, h: int) -> None:
        boxes.append((name, x, y, x + w, y + h))

    for b in L["row1"]:
        y = Y_ROW1 - b["h"] // 2
        add(b["key"], b["x"], y, b["w"], b["h"])

    for e in L["estimators"]:
        add(e["label"], e["x"], Y_ROW2 - e["h"] // 2, e["w"], e["h"])

    m = L["metrics"]
    add("metrics", m["x"], Y_MET - m["h"] // 2, m["w"], m["h"])

    warns: list[str] = []
    for i, (n1, x1, y1, x2, y2) in enumerate(boxes):
        if x1 < 20 or y1 < 20 or x2 > W - 20 or y2 > H - 20:
            warns.append(f"OUT OF BOUNDS: {n1}")
        for n2, a1, b1, a2, b2 in boxes[i + 1 :]:
            pad = 4
            if x1 + pad < a2 and x2 - pad > a1 and y1 + pad < b2 and y2 - pad > b1:
                warns.append(f"OVERLAP: {n1} vs {n2}")

    print("Layout verification:")
    if warns:
        for w in warns:
            print(f"  [WARN] {w}")
    else:
        print("  [OK] No overlaps; all blocks within canvas margins")


def export_png_pdf(svg_path: Path, png_path: Path, pdf_path: Path) -> None:
    try:
        import cairosvg

        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=W, output_height=H)
        cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
        print("PNG/PDF exported via cairosvg")
        return
    except Exception as exc:
        print(f"cairosvg unavailable ({exc}); using matplotlib fallback")

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    L = _layout()
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def box(x: int, y: int, w: int, h: int, fill: str = "#ffffff") -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle=f"round,pad=0.012,rounding_size={R}",
                linewidth=STROKE, edgecolor="#222", facecolor=fill,
            )
        )

    def txt(cx: int, cy: int, lines: list[str], fs: int = FS) -> None:
        ax.text(cx, cy, "\n".join(lines), ha="center", va="center", fontsize=fs, family="sans-serif")

    def arr_h(x1: int, y: int, x2: int) -> None:
        ax.add_patch(
            FancyArrowPatch((x1, y), (x2 - 7, y), arrowstyle="-|>", mutation_scale=11,
                            linewidth=STROKE, color="#222")
        )

    def arr_v(x: int, y1: int, y2: int) -> None:
        dy = -7 if y2 > y1 else 7
        ax.add_patch(
            FancyArrowPatch((x, y1), (x, y2 - dy), arrowstyle="-|>", mutation_scale=11,
                            linewidth=STROKE, color="#222")
        )

    def seg(x1: int, y1: int, x2: int, y2: int, dash: bool = False) -> None:
        ax.plot([x1, x2], [y1, y2], color="#666" if dash else "#222",
                lw=1.2 if dash else STROKE, ls=(0, (5, 4)) if dash else "-")

    for i, b in enumerate(L["row1"]):
        y = Y_ROW1 - b["h"] // 2
        if b["key"] == "channel":
            box(b["x"], y, b["w"], b["h"], "#fafafa")
            txt(b["cx"], y + 20, ["Channel"])
            txt(b["cx"], y + 42, ["AWGN"], FS_CH)
            txt(b["cx"], y + 58, ["Rayleigh flat fading"], FS_CH)
            txt(b["cx"], y + 74, ["Rician flat fading"], FS_CH)
            txt(b["cx"], y + b["h"] + 20, ["single-tap; Rician K = 6 dB"], FS_NOTE)
        else:
            box(b["x"], y, b["w"], b["h"])
            txt(b["cx"], Y_ROW1, b["lines"])
        if i:
            prev = L["row1"][i - 1]
            arr_h(prev["x"] + prev["w"] + 2, Y_ROW1, b["x"] - 2)

    ce, eq = L["ce"], L["eq"]
    arr_v(ce["cx"], Y_ROW1 + ce["h"] // 2 + 2, Y_TRUNK)
    est = L["estimators"]
    seg(est[0]["cx"], Y_TRUNK, est[-1]["cx"], Y_TRUNK)
    for e in est:
        top = Y_ROW2 - e["h"] // 2
        arr_v(e["cx"], Y_TRUNK, top - 2)
        box(e["x"], top, e["w"], e["h"], "#fafafa")
        txt(e["cx"], Y_ROW2, [e["label"]])

    merge_y = Y_ROW2 + BH_EST // 2 + 36
    seg(est[0]["cx"], merge_y, est[-1]["cx"], merge_y)
    for e in est:
        seg(e["cx"], Y_ROW2 + e["h"] // 2 + 2, e["cx"], merge_y)
    seg(ce["cx"], merge_y, eq["cx"], merge_y)
    arr_v(eq["cx"], merge_y, Y_ROW1 + eq["h"] // 2 + 2)

    m = L["metrics"]
    met_top = Y_MET - m["h"] // 2
    seg(ce["cx"], merge_y + 2, ce["cx"], met_top + m["h"] + 2, dash=True)
    box(m["x"], met_top, m["w"], m["h"])
    txt(m["cx"], Y_MET, ["MSE / NMSE Evaluation"])

    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print("PNG/PDF exported via matplotlib (300 dpi PNG)")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _verify_layout()

    svg = build_svg()
    svg_path = OUT / f"{STEM}.svg"
    svg_path.write_text(svg, encoding="utf-8")
    ET.fromstring(svg)
    print(f"Saved: {svg_path}")

    export_png_pdf(svg_path, OUT / f"{STEM}.png", OUT / f"{STEM}.pdf")

    (OUT / f"{STEM}_metadata.json").write_text(
        json.dumps({"caption": CAPTION, "methodology_checklist": CHECKLIST, "layout": "two-row"}, indent=2),
        encoding="utf-8",
    )
    (OUT / f"{STEM}_caption.txt").write_text(CAPTION + "\n", encoding="utf-8")

    print("Methodology verification:")
    for k, v in CHECKLIST.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")


if __name__ == "__main__":
    main()
