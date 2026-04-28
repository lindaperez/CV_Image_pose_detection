#!/usr/bin/env python3
"""Generate result visualizations for the exercise counting project.

The script intentionally uses only the Python standard library so the figures
can be regenerated in a clean environment without installing plotting packages.
"""

from __future__ import annotations

import csv
import html
import math
from pathlib import Path
from typing import Iterable


OUT_DIR = Path(__file__).resolve().parent

EXERCISE_LABELS = {
    "squat": "Squat",
    "pull_up": "Pull-up",
    "push_up": "Push-up",
}

EXERCISE_COLORS = {
    "squat": "#4477AA",
    "pull_up": "#228833",
    "push_up": "#CC6677",
}

ROUTED_RESULTS = [
    {
        "exercise": "squat",
        "selected_branch": "Dedicated squat pose TCN",
        "n": 16,
        "mae": 2.1405,
        "mae_ci_low": 1.1266,
        "mae_ci_high": 3.3313,
        "rmse": 3.1016,
        "rmse_ci_low": 1.6982,
        "rmse_ci_high": 4.2837,
        "within_1": 0.5625,
        "within_1_ci_low": 0.3125,
        "within_1_ci_high": 0.8125,
    },
    {
        "exercise": "pull_up",
        "selected_branch": "Shared pose TCN",
        "n": 14,
        "mae": 4.6088,
        "mae_ci_low": 2.0863,
        "mae_ci_high": 7.5386,
        "rmse": 7.0169,
        "rmse_ci_low": 3.5909,
        "rmse_ci_high": 9.7687,
        "within_1": 0.4286,
        "within_1_ci_low": 0.2143,
        "within_1_ci_high": 0.7143,
    },
    {
        "exercise": "push_up",
        "selected_branch": "RGB ResNet18 + TCN",
        "n": 18,
        "mae": 6.6018,
        "mae_ci_low": 3.3063,
        "mae_ci_high": 10.4238,
        "rmse": 10.2865,
        "rmse_ci_low": 5.1748,
        "rmse_ci_high": 14.8974,
        "within_1": 0.2778,
        "within_1_ci_low": 0.0556,
        "within_1_ci_high": 0.5000,
    },
]

ARCHITECTURE_RESULTS = [
    {
        "architecture": "FSM baseline",
        "exercise": "squat",
        "mae": 3.0625,
        "rmse": 4.9181,
        "within_1": 0.5625,
        "selected": False,
    },
    {
        "architecture": "Dedicated squat TCN",
        "exercise": "squat",
        "mae": 2.1405,
        "rmse": 3.1016,
        "within_1": 0.5625,
        "selected": True,
    },
    {
        "architecture": "Shared pose TCN",
        "exercise": "squat",
        "mae": 8.0430,
        "rmse": 11.1896,
        "within_1": 0.2500,
        "selected": False,
    },
    {
        "architecture": "Shared pose TCN",
        "exercise": "pull_up",
        "mae": 4.6088,
        "rmse": 7.0169,
        "within_1": 0.4286,
        "selected": True,
    },
    {
        "architecture": "Shared pose TCN",
        "exercise": "push_up",
        "mae": 8.8724,
        "rmse": 11.2798,
        "within_1": 0.0000,
        "selected": False,
    },
    {
        "architecture": "Pose Transformer",
        "exercise": "squat",
        "mae": 9.1502,
        "rmse": None,
        "within_1": 0.1250,
        "selected": False,
    },
    {
        "architecture": "Pose Transformer",
        "exercise": "pull_up",
        "mae": 5.0210,
        "rmse": None,
        "within_1": 0.0714,
        "selected": False,
    },
    {
        "architecture": "Pose Transformer",
        "exercise": "push_up",
        "mae": 7.4561,
        "rmse": None,
        "within_1": 0.0556,
        "selected": False,
    },
    {
        "architecture": "RGB ResNet18 TCN",
        "exercise": "squat",
        "mae": 6.5446,
        "rmse": 8.2765,
        "within_1": 0.0625,
        "selected": False,
    },
    {
        "architecture": "RGB ResNet18 TCN",
        "exercise": "pull_up",
        "mae": 4.8686,
        "rmse": None,
        "within_1": 0.1429,
        "selected": False,
    },
    {
        "architecture": "RGB ResNet18 TCN",
        "exercise": "push_up",
        "mae": 6.6018,
        "rmse": 10.2865,
        "within_1": 0.2778,
        "selected": True,
    },
    {
        "architecture": "RGB ResNet50 TCN",
        "exercise": "squat",
        "mae": 5.4245,
        "rmse": 6.8711,
        "within_1": 0.1875,
        "selected": False,
    },
    {
        "architecture": "RGB ResNet50 TCN",
        "exercise": "pull_up",
        "mae": 4.1992,
        "rmse": 5.8931,
        "within_1": 0.3571,
        "selected": False,
    },
    {
        "architecture": "RGB ResNet50 TCN",
        "exercise": "push_up",
        "mae": 7.3768,
        "rmse": None,
        "within_1": 0.0556,
        "selected": False,
    },
    {
        "architecture": "Multimodal fusion",
        "exercise": "squat",
        "mae": 6.6988,
        "rmse": 8.2765,
        "within_1": 0.0000,
        "selected": False,
    },
    {
        "architecture": "Multimodal fusion",
        "exercise": "pull_up",
        "mae": 4.1193,
        "rmse": 5.2182,
        "within_1": 0.1429,
        "selected": False,
    },
    {
        "architecture": "Multimodal fusion",
        "exercise": "push_up",
        "mae": 6.1691,
        "rmse": 9.8993,
        "within_1": 0.1111,
        "selected": False,
    },
    {
        "architecture": "Dedicated pull-up pose",
        "exercise": "pull_up",
        "mae": 3.5463,
        "rmse": None,
        "within_1": 0.2857,
        "selected": False,
    },
]

ARCHITECTURE_ORDER = [
    "FSM baseline",
    "Dedicated squat TCN",
    "Shared pose TCN",
    "Pose Transformer",
    "RGB ResNet18 TCN",
    "RGB ResNet50 TCN",
    "Multimodal fusion",
    "Dedicated pull-up pose",
]

EXERCISE_ORDER = ["squat", "pull_up", "push_up"]


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, Helvetica, sans-serif; fill: #1f2933; }",
        ".title { font-size: 24px; font-weight: 700; }",
        ".subtitle { font-size: 14px; fill: #52606d; }",
        ".axis { stroke: #9aa5b1; stroke-width: 1; }",
        ".grid { stroke: #e4e7eb; stroke-width: 1; }",
        ".label { font-size: 12px; }",
        ".small { font-size: 11px; fill: #52606d; }",
        ".bold { font-size: 12px; font-weight: 700; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def svg_footer() -> list[str]:
    return ["</svg>"]


def write_svg(path: Path, width: int, height: int, body: Iterable[str]) -> None:
    lines = svg_header(width, height)
    lines.extend(body)
    lines.extend(svg_footer())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def text(x: float, y: float, value: object, class_name: str = "label", **attrs: object) -> str:
    attr = " ".join(f'{key.replace("_", "-")}="{esc(val)}"' for key, val in attrs.items())
    if attr:
        attr = " " + attr
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{class_name}"{attr}>{esc(value)}</text>'


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = "#1f2933", width: float = 1.0, **attrs: object) -> str:
    attr = " ".join(f'{key.replace("_", "-")}="{esc(val)}"' for key, val in attrs.items())
    if attr:
        attr = " " + attr
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"{attr}/>'


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", width: float = 1.0, **attrs: object) -> str:
    attr = " ".join(f'{key.replace("_", "-")}="{esc(val)}"' for key, val in attrs.items())
    if attr:
        attr = " " + attr
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"{attr}/>'


def circle(cx: float, cy: float, r: float, fill: str, stroke: str = "#ffffff", width: float = 1.0, **attrs: object) -> str:
    attr = " ".join(f'{key.replace("_", "-")}="{esc(val)}"' for key, val in attrs.items())
    if attr:
        attr = " " + attr
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"{attr}/>'


def polygon(points: list[tuple[float, float]], fill: str, stroke: str = "#ffffff", width: float = 1.0) -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'


def wrap_lines(value: str, max_chars: int) -> list[str]:
    words = value.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if len(candidate) <= max_chars or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def wrapped_text(x: float, y: float, value: str, max_chars: int, class_name: str = "label", line_height: int = 15, **attrs: object) -> list[str]:
    lines = wrap_lines(value, max_chars)
    output: list[str] = []
    for idx, part in enumerate(lines):
        output.append(text(x, y + idx * line_height, part, class_name, **attrs))
    return output


def color_interp(low: str, high: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    low_rgb = tuple(int(low[i : i + 2], 16) for i in (1, 3, 5))
    high_rgb = tuple(int(high[i : i + 2], 16) for i in (1, 3, 5))
    rgb = tuple(round(low_rgb[i] + (high_rgb[i] - low_rgb[i]) * t) for i in range(3))
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def write_csv_outputs() -> None:
    routed_fields = [
        "exercise",
        "selected_branch",
        "n",
        "mae",
        "mae_ci_low",
        "mae_ci_high",
        "rmse",
        "rmse_ci_low",
        "rmse_ci_high",
        "within_1",
        "within_1_ci_low",
        "within_1_ci_high",
    ]
    with (OUT_DIR / "selected_routed_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=routed_fields)
        writer.writeheader()
        writer.writerows(ROUTED_RESULTS)

    architecture_fields = ["architecture", "exercise", "mae", "rmse", "within_1", "selected"]
    with (OUT_DIR / "architecture_result_visualization_data.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=architecture_fields)
        writer.writeheader()
        writer.writerows(ARCHITECTURE_RESULTS)


def draw_axis_chart(
    body: list[str],
    x: float,
    y: float,
    width: float,
    height: float,
    y_max: float,
    title: str,
    y_label: str,
    metric: str,
    ci_low: str,
    ci_high: str,
    tick_step: float,
) -> None:
    body.append(text(x, y - 36, title, "bold"))
    body.append(text(x, y - 18, y_label, "small"))
    chart_bottom = y + height

    tick = 0.0
    while tick <= y_max + 1e-9:
        ty = chart_bottom - (tick / y_max) * height
        body.append(line(x, ty, x + width, ty, "#e4e7eb"))
        body.append(text(x - 8, ty + 4, f"{tick:.1f}" if y_max <= 1.1 else f"{tick:.0f}", "small", text_anchor="end"))
        tick += tick_step

    body.append(line(x, y, x, chart_bottom, "#9aa5b1"))
    body.append(line(x, chart_bottom, x + width, chart_bottom, "#9aa5b1"))

    bar_w = 74
    gap = (width - bar_w * len(ROUTED_RESULTS)) / (len(ROUTED_RESULTS) + 1)
    for idx, row in enumerate(ROUTED_RESULTS):
        bx = x + gap + idx * (bar_w + gap)
        value = float(row[metric])
        low = float(row[ci_low])
        high = float(row[ci_high])
        bar_h = (value / y_max) * height
        by = chart_bottom - bar_h
        color = EXERCISE_COLORS[row["exercise"]]
        body.append(rect(bx, by, bar_w, bar_h, color, rx=4))

        err_low_y = chart_bottom - (low / y_max) * height
        err_high_y = chart_bottom - (high / y_max) * height
        cx = bx + bar_w / 2
        body.append(line(cx, err_low_y, cx, err_high_y, "#1f2933", 1.5))
        body.append(line(cx - 11, err_low_y, cx + 11, err_low_y, "#1f2933", 1.5))
        body.append(line(cx - 11, err_high_y, cx + 11, err_high_y, "#1f2933", 1.5))
        value_text = f"{value:.2f}" if y_max > 1.1 else f"{value:.2f}"
        body.append(text(cx, by - 8, value_text, "bold", text_anchor="middle"))
        body.append(text(cx, chart_bottom + 22, EXERCISE_LABELS[row["exercise"]], "label", text_anchor="middle"))
        body.append(text(cx, chart_bottom + 38, f"n={row['n']}", "small", text_anchor="middle"))


def figure_1_routed_performance() -> None:
    width, height = 1120, 560
    body: list[str] = [
        text(48, 42, "Final Routed Performance With 95% Bootstrap Confidence Intervals", "title"),
        text(48, 66, "Selected branch per exercise. Lower MAE is better; higher Within-1 is better.", "subtitle"),
    ]
    draw_axis_chart(
        body,
        x=88,
        y=140,
        width=420,
        height=290,
        y_max=11,
        title="A. Mean Absolute Error",
        y_label="Repetition count error",
        metric="mae",
        ci_low="mae_ci_low",
        ci_high="mae_ci_high",
        tick_step=2,
    )
    draw_axis_chart(
        body,
        x=650,
        y=140,
        width=360,
        height=290,
        y_max=1.0,
        title="B. Within-1 Accuracy",
        y_label="Fraction of videos within one repetition",
        metric="within_1",
        ci_low="within_1_ci_low",
        ci_high="within_1_ci_high",
        tick_step=0.2,
    )
    body.append(text(48, 510, "Caption: Final routed counting results across the three reportable exercises. Error bars show 95% bootstrap confidence intervals.", "small"))
    write_svg(OUT_DIR / "figure_1_routed_performance_ci.svg", width, height, body)


def architecture_lookup() -> dict[tuple[str, str], dict[str, object]]:
    return {(row["architecture"], row["exercise"]): row for row in ARCHITECTURE_RESULTS}


def figure_2_architecture_heatmap() -> None:
    width, height = 1120, 650
    left, top = 275, 125
    cell_w, cell_h = 170, 48
    body: list[str] = [
        text(48, 42, "Architecture Comparison Heatmap", "title"),
        text(48, 66, "Cell color encodes MAE. Green is lower error; orange is higher error. Black border marks the selected routed branch.", "subtitle"),
    ]
    lookup = architecture_lookup()
    values = [float(row["mae"]) for row in ARCHITECTURE_RESULTS if row["mae"] is not None]
    min_v, max_v = min(values), max(values)

    for col_idx, exercise in enumerate(EXERCISE_ORDER):
        cx = left + col_idx * cell_w + cell_w / 2
        body.append(text(cx, top - 18, EXERCISE_LABELS[exercise], "bold", text_anchor="middle"))

    for row_idx, architecture in enumerate(ARCHITECTURE_ORDER):
        y = top + row_idx * cell_h
        body.append(text(left - 18, y + 30, architecture, "label", text_anchor="end"))
        for col_idx, exercise in enumerate(EXERCISE_ORDER):
            x = left + col_idx * cell_w
            item = lookup.get((architecture, exercise))
            if item is None:
                body.append(rect(x, y, cell_w - 2, cell_h - 2, "#f3f4f6", "#ffffff"))
                body.append(text(x + cell_w / 2, y + 30, "not run", "small", text_anchor="middle"))
                continue
            mae = float(item["mae"])
            t = (mae - min_v) / (max_v - min_v)
            fill = color_interp("#2f9e44", "#f08c00", t)
            stroke = "#111827" if item["selected"] else "#ffffff"
            stroke_width = 3 if item["selected"] else 1
            body.append(rect(x, y, cell_w - 2, cell_h - 2, fill, stroke, stroke_width))
            body.append(text(x + cell_w / 2, y + 24, f"MAE {mae:.2f}", "bold", text_anchor="middle"))
            body.append(text(x + cell_w / 2, y + 40, f"W1 {float(item['within_1']):.2f}", "small", text_anchor="middle"))

    legend_x, legend_y = 48, 550
    body.append(text(legend_x, legend_y, "Legend", "bold"))
    body.append(rect(legend_x, legend_y + 18, 34, 18, "#2f9e44"))
    body.append(text(legend_x + 44, legend_y + 32, "lower MAE", "small"))
    body.append(rect(legend_x + 145, legend_y + 18, 34, 18, "#f08c00"))
    body.append(text(legend_x + 189, legend_y + 32, "higher MAE", "small"))
    body.append(rect(legend_x + 306, legend_y + 18, 34, 18, "#ffffff", "#111827", 3))
    body.append(text(legend_x + 350, legend_y + 32, "selected routed branch", "small"))
    body.append(text(48, 615, "Caption: The heatmap shows why the final result is exercise-dependent instead of one universal architecture.", "small"))
    write_svg(OUT_DIR / "figure_2_architecture_mae_heatmap.svg", width, height, body)


def marker(body: list[str], x: float, y: float, row: dict[str, object], selected: bool) -> None:
    exercise = str(row["exercise"])
    color = EXERCISE_COLORS[exercise]
    arch = str(row["architecture"])
    r = 8 if selected else 6
    if "RGB" in arch:
        body.append(rect(x - r, y - r, 2 * r, 2 * r, color, "#111827" if selected else "#ffffff", 2 if selected else 1))
    elif "Transformer" in arch:
        body.append(polygon([(x, y - r), (x + r, y + r), (x - r, y + r)], color, "#111827" if selected else "#ffffff", 2 if selected else 1))
    else:
        body.append(circle(x, y, r, color, "#111827" if selected else "#ffffff", 2 if selected else 1))


def figure_3_tradeoff_scatter() -> None:
    width, height = 1080, 650
    x0, y0, chart_w, chart_h = 105, 105, 800, 420
    x_max, y_max = 12.0, 0.65
    body: list[str] = [
        text(48, 42, "MAE vs Within-1 Tradeoff", "title"),
        text(48, 66, "Selected branches are outlined. The plot shows why routing should not be described as lowest-MAE only.", "subtitle"),
    ]
    bottom = y0 + chart_h
    for tick in range(0, 13, 2):
        tx = x0 + (tick / x_max) * chart_w
        body.append(line(tx, y0, tx, bottom, "#e4e7eb"))
        body.append(text(tx, bottom + 24, str(tick), "small", text_anchor="middle"))
    for i in range(0, 8):
        value = i * 0.1
        ty = bottom - (value / y_max) * chart_h
        body.append(line(x0, ty, x0 + chart_w, ty, "#e4e7eb"))
        body.append(text(x0 - 10, ty + 4, f"{value:.1f}", "small", text_anchor="end"))
    body.append(line(x0, y0, x0, bottom, "#9aa5b1"))
    body.append(line(x0, bottom, x0 + chart_w, bottom, "#9aa5b1"))
    body.append(text(x0 + chart_w / 2, bottom + 55, "MAE, lower is better", "label", text_anchor="middle"))
    body.append(text(24, y0 + chart_h / 2, "Within-1, higher is better", "label", transform=f"rotate(-90 24 {y0 + chart_h / 2:.1f})", text_anchor="middle"))

    for row in ARCHITECTURE_RESULTS:
        x = x0 + (float(row["mae"]) / x_max) * chart_w
        y = bottom - (float(row["within_1"]) / y_max) * chart_h
        marker(body, x, y, row, bool(row["selected"]))

    labels = [
        ("Dedicated squat TCN", "squat", 2.1405, 0.5625, -5, -20),
        ("Shared pose TCN", "pull_up", 4.6088, 0.4286, 12, -10),
        ("RGB ResNet18 TCN", "push_up", 6.6018, 0.2778, 12, -8),
        ("Lower-MAE pull-up alternatives", "pull_up", 3.5463, 0.2857, 12, 24),
        ("Fusion lower MAE, weaker W1", "push_up", 6.1691, 0.1111, 12, 24),
    ]
    for label, exercise, mae, w1, dx, dy in labels:
        lx = x0 + (mae / x_max) * chart_w + dx
        ly = bottom - (w1 / y_max) * chart_h + dy
        body.append(text(lx, ly, label, "small"))

    legend_x, legend_y = 930, 130
    body.append(text(legend_x, legend_y, "Exercise", "bold"))
    for idx, exercise in enumerate(EXERCISE_ORDER):
        yy = legend_y + 28 + idx * 26
        body.append(circle(legend_x + 8, yy - 5, 6, EXERCISE_COLORS[exercise]))
        body.append(text(legend_x + 24, yy, EXERCISE_LABELS[exercise], "small"))
    body.append(text(legend_x, legend_y + 130, "Marker", "bold"))
    body.append(circle(legend_x + 8, legend_y + 158, 6, "#6b7280"))
    body.append(text(legend_x + 24, legend_y + 163, "Pose/FSM", "small"))
    body.append(rect(legend_x + 2, legend_y + 178, 12, 12, "#6b7280"))
    body.append(text(legend_x + 24, legend_y + 189, "RGB", "small"))
    body.append(polygon([(legend_x + 8, legend_y + 204), (legend_x + 15, legend_y + 218), (legend_x + 1, legend_y + 218)], "#6b7280"))
    body.append(text(legend_x + 24, legend_y + 218, "Transformer", "small"))
    body.append(text(48, 610, "Caption: Exact-count reliability can change the selected branch even when another model has a lower MAE.", "small"))
    write_svg(OUT_DIR / "figure_3_mae_within1_tradeoff.svg", width, height, body)


def figure_4_per_exercise_mae_bars() -> None:
    width, height = 1280, 720
    body: list[str] = [
        text(48, 42, "Per-Exercise Architecture Comparison", "title"),
        text(48, 66, "Horizontal bars show MAE. Colored bars are the selected routed branches.", "subtitle"),
    ]
    panel_w, panel_h = 365, 500
    panel_top = 120
    panel_gap = 35
    panel_lefts = [60, 60 + panel_w + panel_gap, 60 + 2 * (panel_w + panel_gap)]
    x_max = 11.5

    for panel_idx, exercise in enumerate(EXERCISE_ORDER):
        px = panel_lefts[panel_idx]
        body.append(text(px, panel_top - 22, EXERCISE_LABELS[exercise], "bold"))
        rows = [row for row in ARCHITECTURE_RESULTS if row["exercise"] == exercise]
        rows = sorted(rows, key=lambda item: float(item["mae"]))
        bar_h = 26
        gap = 15
        label_w = 148
        chart_x = px + label_w
        chart_w = panel_w - label_w - 20
        for tick in range(0, 13, 4):
            tx = chart_x + (tick / x_max) * chart_w
            body.append(line(tx, panel_top, tx, panel_top + len(rows) * (bar_h + gap), "#eef0f2"))
            body.append(text(tx, panel_top + len(rows) * (bar_h + gap) + 18, str(tick), "small", text_anchor="middle"))
        for row_idx, row in enumerate(rows):
            y = panel_top + row_idx * (bar_h + gap)
            label_lines = wrap_lines(str(row["architecture"]), 20)
            for line_idx, label in enumerate(label_lines[:2]):
                body.append(text(px, y + 14 + line_idx * 12, label, "small"))
            mae = float(row["mae"])
            bw = (mae / x_max) * chart_w
            fill = EXERCISE_COLORS[exercise] if row["selected"] else "#cbd2d9"
            stroke = "#111827" if row["selected"] else "none"
            sw = 2 if row["selected"] else 1
            body.append(rect(chart_x, y, bw, bar_h, fill, stroke, sw, rx=4))
            body.append(text(chart_x + bw + 5, y + 18, f"{mae:.2f}", "small"))
            body.append(text(chart_x + 5, y + 18, f"W1 {float(row['within_1']):.2f}", "small"))
        body.append(text(chart_x + chart_w / 2, panel_top + len(rows) * (bar_h + gap) + 42, "MAE", "small", text_anchor="middle"))

    body.append(text(48, 678, "Caption: The selected branch differs by exercise, supporting a routed result rather than a universal winner.", "small"))
    write_svg(OUT_DIR / "figure_4_per_exercise_mae_comparison.svg", width, height, body)


def draw_box(body: list[str], x: float, y: float, w: float, h: float, label: str, fill: str, stroke: str = "#1f2933") -> None:
    body.append(rect(x, y, w, h, fill, stroke, 1.4, rx=8))
    lines = wrap_lines(label, max(12, int(w / 8)))
    start_y = y + h / 2 - (len(lines) - 1) * 7 + 5
    for idx, part in enumerate(lines):
        body.append(text(x + w / 2, start_y + idx * 15, part, "label", text_anchor="middle"))


def arrow(body: list[str], x1: float, y1: float, x2: float, y2: float) -> None:
    body.append(line(x1, y1, x2, y2, "#52606d", 1.8, marker_end="url(#arrowhead)"))


def figure_5_routed_architecture() -> None:
    width, height = 1180, 620
    body: list[str] = [
        "<defs>",
        '<marker id="arrowhead" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">',
        '<polygon points="0 0, 9 3.5, 0 7" fill="#52606d"/>',
        "</marker>",
        "</defs>",
        text(48, 42, "Result-Driven Routed Architecture", "title"),
        text(48, 66, "The final system uses the exercise label to select the representation branch supported by validation evidence.", "subtitle"),
    ]
    draw_box(body, 60, 116, 150, 58, "Input video", "#f8fafc")
    draw_box(body, 260, 116, 180, 58, "Known exercise label", "#f8fafc")
    draw_box(body, 505, 116, 150, 58, "Routing", "#fff7ed")
    arrow(body, 210, 145, 260, 145)
    arrow(body, 440, 145, 505, 145)

    route_x = 580
    bus_y = 218
    body.append(line(route_x, 174, route_x, bus_y, "#52606d", 1.8))
    body.append(line(195, bus_y, 985, bus_y, "#52606d", 1.8))

    columns = [
        (70, "Squat", "#4477AA", "YOLO pose", "Squat features", "Dedicated squat TCN", "MAE 2.14 / W1 0.56"),
        (465, "Pull-up", "#228833", "YOLO pose", "Normalized pose sequence", "Shared pose TCN", "MAE 4.61 / W1 0.43"),
        (860, "Push-up", "#CC6677", "RGB frames", "Frozen ResNet18 features", "RGB TCN", "MAE 6.60 / W1 0.28"),
    ]
    box_w = 250
    box_h = 42
    y_values = [252, 313, 374, 435, 500]
    for x, exercise, color, step1, step2, step3, metric in columns:
        center_x = x + box_w / 2
        arrow(body, center_x, bus_y, center_x, y_values[0])
        draw_box(body, x, y_values[0], box_w, box_h, exercise, color)
        draw_box(body, x, y_values[1], box_w, box_h, step1, "#f8fafc")
        draw_box(body, x, y_values[2], box_w, box_h, step2, "#f8fafc")
        draw_box(body, x, y_values[3], box_w, box_h, step3, "#f8fafc")
        draw_box(body, x, y_values[4], box_w, box_h, metric, "#ecfdf5")
        for top_y, next_y in zip(y_values, y_values[1:]):
            arrow(body, center_x, top_y + box_h, center_x, next_y)

    body.append(text(48, 585, "Caption: The routed architecture mirrors the result finding: squat uses engineered pose, pull-up uses shared pose, and push-up uses RGB appearance.", "small"))
    write_svg(OUT_DIR / "figure_5_routed_architecture.svg", width, height, body)


def write_dashboard() -> None:
    figures = [
        ("figure_1_routed_performance_ci.svg", "Final routed performance with confidence intervals"),
        ("figure_2_architecture_mae_heatmap.svg", "Architecture comparison heatmap"),
        ("figure_3_mae_within1_tradeoff.svg", "MAE versus Within-1 tradeoff"),
        ("figure_4_per_exercise_mae_comparison.svg", "Per-exercise architecture comparison"),
        ("figure_5_routed_architecture.svg", "Result-driven routed architecture"),
    ]
    cards = []
    for filename, caption in figures:
        cards.append(
            f"""
      <section>
        <h2>{esc(caption)}</h2>
        <img src="{esc(filename)}" alt="{esc(caption)}">
      </section>"""
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Exercise Counting Results Visualizations</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: #1f2933;
      background: #f8fafc;
    }}
    header {{
      padding: 32px 40px 20px;
      background: #ffffff;
      border-bottom: 1px solid #d9e2ec;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 24px 28px 48px;
    }}
    section {{
      margin-bottom: 32px;
      background: #ffffff;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      padding: 20px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    h2 {{
      margin: 0 0 16px;
      font-size: 18px;
    }}
    p {{
      margin: 0;
      color: #52606d;
    }}
    img {{
      width: 100%;
      height: auto;
      display: block;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Exercise Counting Results Visualizations</h1>
    <p>Generated from the project result matrix and reportable routed metrics.</p>
  </header>
  <main>
{''.join(cards)}
  </main>
</body>
</html>
"""
    (OUT_DIR / "results_dashboard.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    write_csv_outputs()
    figure_1_routed_performance()
    figure_2_architecture_heatmap()
    figure_3_tradeoff_scatter()
    figure_4_per_exercise_mae_bars()
    figure_5_routed_architecture()
    write_dashboard()


if __name__ == "__main__":
    main()
