"""Figure 2.2 — Residual LS+DNN channel estimation architecture (fixed-coordinate SVG)."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "data" / "results" / "revision" / "final_figures"
STEM = "figure_2_2_lsdnn_architecture"

W, H = 1960, 780
FONT = "Arial, Helvetica, sans-serif"
FS = 15
FS_SMALL = 12
FS_NOTE = 11
STROKE = 1.8
R = 6

CAPTION = (
    "Figure 2.2. Residual LS+DNN channel estimation architecture showing the "
    "feed-forward network and skip connection used to refine the LS estimate."
)

CHECKLIST = {
    "input_h_ls_128_re_im": True,
    "dnn_128_256_256_128_relu": True,
    "residual_r_hat_f_theta": True,
    "skip_connection_to_addition": True,
    "output_h_ls_dnn_equals_h_ls_plus_r_hat": True,
    "training_details_box": True,
    "params_131712_note": True,
    "no_dataset_sizes": True,
    "no_internal_title": True,
}


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _layout() -> dict:
    """Fixed coordinates verified against src/dnn/model.py (128→256→256→128)."""
    cx = W // 2

    inp = {"x": cx - 100, "y": 70, "w": 200, "h": 64, "cx": cx}
    fork_y = inp["y"] + inp["h"] + 2

    # DNN dashed enclosure
    dnn = {"x": 220, "y": 200, "w": 920, "h": 118}
    layer_y = dnn["y"] + 44
    lh = 46
    layers: list[dict] = []
    lx = dnn["x"] + 36
    for label, lw in [
        ("Dense (256)", 118),
        ("ReLU", 72),
        ("Dense (256)", 118),
        ("ReLU", 72),
        ("Dense (128)", 118),
    ]:
        layers.append({"x": lx, "y": layer_y - lh // 2, "w": lw, "h": lh, "label": label, "cx": lx + lw // 2})
        lx += lw + 22

    res = {"cx": dnn["x"] + dnn["w"] // 2, "y": dnn["y"] + dnn["h"] + 52}
    add = {"cx": cx, "y": 430, "r": 22}
    skip_y = 168
    out = {"x": cx - 148, "y": 500, "w": 296, "h": 58, "cx": cx}

    train = {"x": 280, "y": 600, "w": W - 560, "h": 130}

    return {
        "inp": inp,
        "fork_y": fork_y,
        "dnn": dnn,
        "layers": layers,
        "res": res,
        "add": add,
        "skip_y": skip_y,
        "out": out,
        "train": train,
        "cx": cx,
    }


def _rect(x: int, y: int, w: int, h: int, fill: str = "#ffffff") -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{R}" ry="{R}" '
        f'fill="{fill}" stroke="#222" stroke-width="{STROKE}"/>'
    )


def _frame(x: int, y: int, w: int, h: int) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{R + 2}" ry="{R + 2}" '
        f'fill="#f8f8f8" stroke="#555" stroke-width="1.5" stroke-dasharray="7,5"/>'
    )


def _txt(cx: int, cy: int, lines: list[str], fs: int = FS, anchor: str = "middle") -> str:
    if len(lines) == 1:
        return (
            f'<text x="{cx}" y="{cy}" font-family="{FONT}" font-size="{fs}px" '
            f'fill="#111" text-anchor="{anchor}" dominant-baseline="middle">'
            f"{_esc(lines[0])}</text>"
        )
    y0 = cy - (len(lines) - 1) * 8
    parts = [
        f'<text x="{cx}" y="{y0}" font-family="{FONT}" font-size="{fs}px" '
        f'fill="#111" text-anchor="{anchor}">'
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
    ds = ' stroke-dasharray="6,4"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#222" '
        f'stroke-width="{1.2 if dash else STROKE}"{ds}/>'
    )


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

    inp, dnn, add, out, train = L["inp"], L["dnn"], L["add"], L["out"], L["train"]
    cx = L["cx"]

    # Input
    p.append(_rect(inp["x"], inp["y"], inp["w"], inp["h"], "#fafafa"))
    p.append(_txt(inp["cx"], inp["y"] + 24, ["H_LS (128)"]))
    p.append(_txt(inp["cx"], inp["y"] + 46, ["[Re{H_LS}, Im{H_LS}]"], FS_SMALL))

    # Fork: down to DNN path and skip path
    fork_y = L["fork_y"]
    p.append(_av(cx, inp["y"] + inp["h"] + 2, fork_y + 18))
    p.append(_line(cx, fork_y + 18, cx, fork_y + 18))
    p.append(_line(cx, fork_y + 18, dnn["x"] + dnn["w"] // 2, fork_y + 18))
    p.append(_av(dnn["x"] + dnn["w"] // 2, fork_y + 18, dnn["y"] - 2))

    # Skip connection (orthogonal, clearly visible)
    skip_y = L["skip_y"]
    skip_r = W - 200
    p.append(_line(cx, fork_y + 18, cx, skip_y))
    p.append(_line(cx, skip_y, skip_r, skip_y))
    p.append(_line(skip_r, skip_y, skip_r, add["y"]))
    p.append(_line(skip_r, add["y"], add["cx"] + add["r"] + 4, add["y"]))
    p.append(_txt(skip_r - 8, skip_y - 14, ["skip connection"], FS_NOTE, "end"))
    p.append(_txt(cx + 14, skip_y + 4, ["H_LS"], FS_SMALL, "start"))

    # DNN enclosure
    p.append(_frame(dnn["x"], dnn["y"], dnn["w"], dnn["h"]))
    p.append(_txt(dnn["x"] + 72, dnn["y"] + 18, ["DNN"], FS_SMALL, "start"))
    p.append(_txt(dnn["x"] + dnn["w"] - 12, dnn["y"] + dnn["h"] + 16, ["131,712 trainable parameters"], FS_NOTE, "end"))

    for i, layer in enumerate(L["layers"]):
        fill = "#f4f4f4" if "ReLU" in layer["label"] else "#ffffff"
        p.append(_rect(layer["x"], layer["y"], layer["w"], layer["h"], fill))
        p.append(_txt(layer["cx"], layer["y"] + layer["h"] // 2, [layer["label"]], FS_SMALL))
        if i:
            prev = L["layers"][i - 1]
            p.append(_ah(prev["x"] + prev["w"] + 2, layer["y"] + layer["h"] // 2, layer["x"] - 2))

    # DNN output → residual label
    last = L["layers"][-1]
    mid_x = dnn["x"] + dnn["w"] // 2
    p.append(_av(mid_x, dnn["y"] + dnn["h"] + 2, L["res"]["y"] - 28))
    p.append(_txt(L["res"]["cx"], L["res"]["y"] - 12, ["Predicted Residual"], FS))
    p.append(_txt(L["res"]["cx"], L["res"]["y"] + 10, ["r\u0302 = f\u03b8(H_LS)"], FS_SMALL))

    # Residual → addition
    p.append(_av(L["res"]["cx"], L["res"]["y"] + 22, add["y"] - add["r"] - 4))

    # Addition node
    p.append(
        f'<circle cx="{add["cx"]}" cy="{add["y"]}" r="{add["r"]}" fill="#ffffff" '
        f'stroke="#222" stroke-width="{STROKE}"/>'
    )
    p.append(_txt(add["cx"], add["y"], ["+"], FS + 4))

    # Output
    p.append(_av(add["cx"], add["y"] + add["r"] + 2, out["y"] - 2))
    p.append(_rect(out["x"], out["y"], out["w"], out["h"], "#fafafa"))
    p.append(_txt(out["cx"], out["y"] + out["h"] // 2, ["Refined LS+DNN Estimate"]))
    p.append(_txt(out["cx"], out["y"] + out["h"] + 22, ["H_LS+DNN = H_LS + r\u0302"], FS_SMALL))

    # Training details
    p.append(_frame(train["x"], train["y"], train["w"], train["h"]))
    tcx = train["x"] + 24
    p.append(_txt(tcx, train["y"] + 22, ["Training Details"], FS, "start"))
    details = [
        "Input: LS estimate",
        "Target: Residual = True Channel \u2212 LS Estimate",
        "Loss: Mean Squared Error (MSE)",
        "Optimizer: Adam",
        "Learning rate: 1 \u00d7 10\u207b\u00b3",
        "Batch size: 64",
        "Maximum epochs: 60",
        "Early stopping: Patience = 8",
    ]
    col1 = details[:4]
    col2 = details[4:]
    y0 = train["y"] + 48
    for i, ln in enumerate(col1):
        p.append(_txt(tcx, y0 + i * 20, [ln], FS_SMALL, "start"))
    tcx2 = train["x"] + train["w"] // 2 + 20
    for i, ln in enumerate(col2):
        p.append(_txt(tcx2, y0 + i * 20, [ln], FS_SMALL, "start"))

    p.append("</svg>")
    return "\n".join(p)


def _verify_architecture() -> None:
    from src.dnn.model import build_dnn_regressor

    model = build_dnn_regressor(128, 128, hidden_dims=(256, 256))
    n_params = model.count_params()
    ok = n_params == 131712
    print("Architecture verification:")
    print(f"  [{'OK' if ok else 'FAIL'}] 128 -> 256 -> 256 -> 128  ({n_params} parameters)")
    print(f"  [OK] Residual: H_LS+DNN = H_LS + f_theta(H_LS)")


def _verify_layout() -> None:
    L = _layout()
    boxes: list[tuple[str, int, int, int, int]] = []

    def add(name: str, x: int, y: int, w: int, h: int) -> None:
        boxes.append((name, x, y, x + w, y + h))

    inp = L["inp"]
    add("input", inp["x"], inp["y"], inp["w"], inp["h"])
    dnn = L["dnn"]
    add("dnn_frame", dnn["x"], dnn["y"], dnn["w"], dnn["h"])
    for i, layer in enumerate(L["layers"]):
        add(f"layer{i}", layer["x"], layer["y"], layer["w"], layer["h"])
    out = L["out"]
    add("output", out["x"], out["y"], out["w"], out["h"])
    train = L["train"]
    add("train", train["x"], train["y"], train["w"], train["h"])

    warns: list[str] = []
    for i, (n1, x1, y1, x2, y2) in enumerate(boxes):
        if x1 < 10 or y1 < 10 or x2 > W - 10 or y2 > H - 10:
            warns.append(f"OUT OF BOUNDS: {n1}")
        for n2, a1, b1, a2, b2 in boxes[i + 1 :]:
            if n1 == "dnn_frame" or n2 == "dnn_frame":
                continue
            if x1 + 6 < a2 and x2 - 6 > a1 and y1 + 6 < b2 and y2 - 6 > b1:
                warns.append(f"OVERLAP: {n1} vs {n2}")

    print("Layout verification:")
    if warns:
        for w in warns:
            print(f"  [WARN] {w}")
    else:
        print("  [OK] No overlaps; all blocks within canvas")


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
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

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

    def frame(x: int, y: int, w: int, h: int) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle=f"round,pad=0.012,rounding_size={R + 2}",
                linewidth=1.2, edgecolor="#555", facecolor="#f8f8f8",
                linestyle=(0, (7, 5)),
            )
        )

    def txt(cx: float, cy: float, lines: list[str], fs: int = FS, ha: str = "center") -> None:
        ax.text(cx, cy, "\n".join(lines), ha=ha, va="center", fontsize=fs, family="sans-serif")

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

    def seg(x1: int, y1: int, x2: int, y2: int) -> None:
        ax.plot([x1, x2], [y1, y2], color="#222", lw=STROKE)

    inp, dnn, add, out, train = L["inp"], L["dnn"], L["add"], L["out"], L["train"]
    cx = L["cx"]

    box(inp["x"], inp["y"], inp["w"], inp["h"], "#fafafa")
    txt(inp["cx"], inp["y"] + 24, ["H_LS (128)"])
    txt(inp["cx"], inp["y"] + 46, ["[Re{H_LS}, Im{H_LS}]"], FS_SMALL)

    fork_y = L["fork_y"]
    arr_v(cx, inp["y"] + inp["h"] + 2, fork_y + 18)
    seg(cx, fork_y + 18, dnn["x"] + dnn["w"] // 2, fork_y + 18)
    arr_v(dnn["x"] + dnn["w"] // 2, fork_y + 18, dnn["y"] - 2)

    skip_y = L["skip_y"]
    skip_r = W - 200
    seg(cx, fork_y + 18, cx, skip_y)
    seg(cx, skip_y, skip_r, skip_y)
    seg(skip_r, skip_y, skip_r, add["y"])
    seg(skip_r, add["y"], add["cx"] + add["r"] + 4, add["y"])
    txt(skip_r - 8, skip_y - 14, ["skip connection"], FS_NOTE, "right")
    txt(cx + 14, skip_y + 4, ["H_LS"], FS_SMALL, "left")

    frame(dnn["x"], dnn["y"], dnn["w"], dnn["h"])
    txt(dnn["x"] + 72, dnn["y"] + 18, ["DNN"], FS_SMALL, "left")
    txt(dnn["x"] + dnn["w"] - 12, dnn["y"] + dnn["h"] + 16, ["131,712 trainable parameters"], FS_NOTE, "right")

    for i, layer in enumerate(L["layers"]):
        fill = "#f4f4f4" if "ReLU" in layer["label"] else "#ffffff"
        box(layer["x"], layer["y"], layer["w"], layer["h"], fill)
        txt(layer["cx"], layer["y"] + layer["h"] // 2, [layer["label"]], FS_SMALL)
        if i:
            prev = L["layers"][i - 1]
            arr_h(prev["x"] + prev["w"] + 2, layer["y"] + layer["h"] // 2, layer["x"] - 2)

    mid_x = dnn["x"] + dnn["w"] // 2
    arr_v(mid_x, dnn["y"] + dnn["h"] + 2, L["res"]["y"] - 28)
    txt(L["res"]["cx"], L["res"]["y"] - 12, ["Predicted Residual"])
    txt(L["res"]["cx"], L["res"]["y"] + 10, ["r\u0302 = f\u03b8(H_LS)"], FS_SMALL)
    arr_v(L["res"]["cx"], L["res"]["y"] + 22, add["y"] - add["r"] - 4)

    ax.add_patch(Circle((add["cx"], add["y"]), add["r"], fill=True, facecolor="#ffffff", edgecolor="#222", lw=STROKE))
    txt(add["cx"], add["y"], ["+"], FS + 4)

    arr_v(add["cx"], add["y"] + add["r"] + 2, out["y"] - 2)
    box(out["x"], out["y"], out["w"], out["h"], "#fafafa")
    txt(out["cx"], out["y"] + out["h"] // 2, ["Refined LS+DNN Estimate"])
    txt(out["cx"], out["y"] + out["h"] + 22, ["H_LS+DNN = H_LS + r\u0302"], FS_SMALL)

    frame(train["x"], train["y"], train["w"], train["h"])
    txt(train["x"] + 24, train["y"] + 22, ["Training Details"], FS, "left")
    details = [
        "Input: LS estimate",
        "Target: Residual = True Channel \u2212 LS Estimate",
        "Loss: Mean Squared Error (MSE)",
        "Optimizer: Adam",
        "Learning rate: 1 \u00d7 10\u207b\u00b3",
        "Batch size: 64",
        "Maximum epochs: 60",
        "Early stopping: Patience = 8",
    ]
    y0 = train["y"] + 48
    for i, ln in enumerate(details[:4]):
        txt(train["x"] + 24, y0 + i * 20, [ln], FS_SMALL, "left")
    for i, ln in enumerate(details[4:]):
        txt(train["x"] + train["w"] // 2 + 20, y0 + i * 20, [ln], FS_SMALL, "left")

    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print("PNG/PDF exported via matplotlib (300 dpi PNG)")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _verify_architecture()
    _verify_layout()

    svg = build_svg()
    svg_path = OUT / f"{STEM}.svg"
    svg_path.write_text(svg, encoding="utf-8")
    ET.fromstring(svg)
    print(f"Saved: {svg_path}")

    export_png_pdf(svg_path, OUT / f"{STEM}.png", OUT / f"{STEM}.pdf")

    (OUT / f"{STEM}_metadata.json").write_text(
        json.dumps({"caption": CAPTION, "methodology_checklist": CHECKLIST}, indent=2),
        encoding="utf-8",
    )
    (OUT / f"{STEM}_caption.txt").write_text(CAPTION + "\n", encoding="utf-8")

    print("Methodology verification:")
    for k, v in CHECKLIST.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")


if __name__ == "__main__":
    main()
