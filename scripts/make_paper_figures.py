"""Generate the method diagrams and vector experiment figures for the paper."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/zyp-paper-figures-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper" / "figures"
DEEPPCB = ROOT / "data" / "raw" / "DeepPCB" / "PCBData"

FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
CJK = font_manager.FontProperties(fname=FONT_REGULAR)
CJK_BOLD = font_manager.FontProperties(fname=FONT_BOLD)

INK = "#263238"
BLUE = "#2F6F9F"
PALE_BLUE = "#EAF3F8"
ORANGE = "#D8842F"
PALE_ORANGE = "#FFF2E3"
GREEN = "#4D7C66"
PALE_GREEN = "#EDF5F0"
GRAY = "#F3F5F6"

CLASS_LABELS = ("open", "short", "mousebite", "spur", "pinhole", "spurious_copper")
STRATEGIES = (
    "random",
    "low_confidence",
    "manual_linear",
    "logistic",
    "lightgbm",
    "review_acceptance",
)
GEOMETRIC_ORDER = (
    "baseline",
    "shift_m8px",
    "shift_m4px",
    "shift_m2px",
    "shift_m1px",
    "shift_1px",
    "shift_2px",
    "shift_4px",
    "shift_8px",
    "rotate_m2deg",
    "rotate_m1deg",
    "rotate_m0.5deg",
    "rotate_0.5deg",
    "rotate_1deg",
    "rotate_2deg",
    "scale_0.98",
    "scale_1.02",
)
SETTINGS = (
    "no_reg_direct",
    "ecc_direct",
    "ecc_direct_gate",
    "ecc_tolerant",
    "ecc_tolerant_gate",
)


def configure_matplotlib() -> None:
    """Use embedded TrueType fonts and a restrained paper style."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.25,
            "savefig.pad_inches": 0.025,
        }
    )


def save_pdf(fig: plt.Figure, filename: str) -> None:
    fig.savefig(OUTPUT / filename, format="pdf")
    plt.close(fig)


def add_node(
    ax: plt.Axes,
    center: tuple[float, float],
    size: tuple[float, float],
    text: str,
    *,
    facecolor: str = "white",
    edgecolor: str = BLUE,
    fontsize: float = 7.2,
    bold: bool = False,
) -> FancyBboxPatch:
    x, y = center
    width, height = size
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.035,rounding_size=0.075",
        linewidth=0.9,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        fontproperties=CJK_BOLD if bold else CJK,
        zorder=4,
    )
    return patch


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    dashed: bool = False,
    connectionstyle: str = "arc3",
    linewidth: float = 0.85,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7.0,
            linewidth=linewidth,
            linestyle=(0, (4, 2.5)) if dashed else "solid",
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=0,
            shrinkB=0,
            zorder=2,
        )
    )


def make_framework() -> None:
    """Draw the current system dataflow from development specification section 2."""
    fig, ax = plt.subplots(figsize=(7.16, 5.35))
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0.25, 10.35)
    ax.axis("off")

    # Registration and difference-candidate branch.
    ax.text(0.28, 9.72, "待检图 I", fontsize=7.5, color=INK, fontproperties=CJK)
    ax.text(0.28, 8.82, "正常模板 R", fontsize=7.5, color=INK, fontproperties=CJK)
    add_node(ax, (3.05, 9.27), (2.25, 0.68), "M1  配准与归一", facecolor=PALE_BLUE, bold=True)
    add_node(ax, (5.95, 9.88), (1.72, 0.54), "配准质量  q_reg", edgecolor=GREEN, facecolor=PALE_GREEN, fontsize=6.8)
    add_node(ax, (5.75, 8.88), (1.92, 0.64), "M2  外观差异", facecolor=PALE_BLUE)
    add_node(ax, (5.75, 7.78), (1.92, 0.64), "M3  结构差异", facecolor=PALE_BLUE)
    add_node(ax, (8.78, 8.33), (2.16, 0.72), "M4  差异候选生成", facecolor=PALE_ORANGE, edgecolor=ORANGE, bold=True)
    add_arrow(ax, (1.08, 9.69), (1.93, 9.42), connectionstyle="angle3,angleA=0,angleB=90")
    add_arrow(ax, (1.22, 8.84), (1.93, 9.12), connectionstyle="angle3,angleA=0,angleB=-90")
    add_arrow(ax, (4.18, 9.39), (5.09, 9.78))
    add_arrow(ax, (4.18, 9.20), (4.79, 8.88), connectionstyle="angle3,angleA=0,angleB=90")
    add_arrow(ax, (4.18, 9.10), (4.79, 7.78), connectionstyle="angle3,angleA=0,angleB=90")
    add_arrow(ax, (6.71, 8.88), (7.70, 8.47), connectionstyle="angle3,angleA=0,angleB=90")
    add_arrow(ax, (6.71, 7.78), (7.70, 8.19), connectionstyle="angle3,angleA=0,angleB=-90")

    # Detector branch and fusion.
    ax.text(0.28, 6.72, "待检图 I", fontsize=7.5, color=INK, fontproperties=CJK)
    add_node(ax, (3.08, 6.73), (2.82, 0.72), "M0  检测器 + 置信度校准", facecolor=PALE_BLUE, bold=True)
    add_node(ax, (6.15, 6.73), (1.72, 0.58), "检测候选", edgecolor=GREEN, facecolor=PALE_GREEN)
    add_node(
        ax,
        (9.98, 6.73),
        (2.78, 0.78),
        "M5  候选融合\nDET / DIFF / BOTH",
        facecolor=PALE_ORANGE,
        edgecolor=ORANGE,
        bold=True,
    )
    add_arrow(ax, (1.06, 6.72), (1.67, 6.72))
    add_arrow(ax, (4.49, 6.73), (5.29, 6.73))
    add_arrow(ax, (7.01, 6.73), (8.59, 6.73))
    add_arrow(ax, (8.78, 7.97), (9.55, 7.13), connectionstyle="angle3,angleA=-90,angleB=0")

    # Review and action branch.
    add_node(ax, (9.98, 5.32), (1.78, 0.65), "M6  复检门", facecolor=PALE_ORANGE, edgecolor=ORANGE, bold=True)
    add_node(ax, (12.85, 5.32), (1.76, 0.60), "默认动作", edgecolor=GREEN, facecolor=PALE_GREEN)
    add_node(ax, (8.42, 4.17), (1.92, 0.65), "M7  证据卡 A2", facecolor=PALE_BLUE, bold=True)
    add_node(ax, (11.36, 4.17), (2.32, 0.65), "M8  大模型独立判断", facecolor=PALE_BLUE, bold=True)
    add_node(ax, (9.98, 3.08), (1.90, 0.62), "M9  动作映射", facecolor=PALE_ORANGE, edgecolor=ORANGE)
    add_node(ax, (9.98, 2.04), (2.80, 0.70), "M10  接受门 + 安全规则", facecolor=PALE_ORANGE, edgecolor=ORANGE, bold=True)
    add_node(ax, (9.98, 0.95), (2.34, 0.66), "M11  类别感知 NMS", facecolor=PALE_BLUE, bold=True)
    add_node(ax, (13.00, 0.95), (2.00, 0.70), "最终结果\n+ 审计日志", edgecolor=GREEN, facecolor=PALE_GREEN, bold=True)
    add_arrow(ax, (9.98, 6.34), (9.98, 5.65))
    add_arrow(ax, (10.87, 5.32), (11.97, 5.32), color=GREEN)
    ax.text(11.40, 5.49, "低风险", fontsize=6.3, color=GREEN, ha="center", fontproperties=CJK)
    add_arrow(ax, (9.70, 5.00), (8.76, 4.49), color=ORANGE, connectionstyle="angle3,angleA=-90,angleB=0")
    ax.text(9.20, 4.78, "高风险", fontsize=6.3, color=ORANGE, ha="center", fontproperties=CJK)
    add_arrow(ax, (9.38, 4.17), (10.20, 4.17))
    add_arrow(ax, (11.36, 3.84), (10.35, 3.39), connectionstyle="angle3,angleA=-90,angleB=0")
    add_arrow(ax, (9.98, 2.77), (9.98, 2.39))
    add_arrow(ax, (9.98, 1.69), (9.98, 1.28))
    add_arrow(ax, (11.15, 0.95), (12.00, 0.95))

    # The topology map itself goes only to M4. Its separate 8-D vector is a
    # gate feature, shown as a dashed bus into M6 and M10 (never into M7).
    ax.plot([6.72, 13.78, 13.78], [7.78, 7.78, 2.04], color=ORANGE, linewidth=0.9, linestyle=(0, (4, 2.5)), zorder=1)
    add_arrow(
        ax,
        (13.78, 5.68),
        (10.52, 5.65),
        color=ORANGE,
        dashed=True,
        linewidth=0.9,
        connectionstyle="angle3,angleA=180,angleB=90",
    )
    add_arrow(ax, (13.78, 2.04), (11.38, 2.04), color=ORANGE, dashed=True, linewidth=0.9)
    ax.text(
        13.94,
        6.55,
        "8 维拓扑向量\n（门控特征）",
        fontsize=6.4,
        color=ORANGE,
        ha="right",
        va="center",
        fontproperties=CJK,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
        zorder=5,
    )

    ax.text(
        0.28,
        0.33,
        "实线：图像 / 候选 / 决策流    虚线：仅用于门控的拓扑特征",
        fontsize=6.1,
        color="#59666D",
        fontproperties=CJK,
    )
    fig.subplots_adjust(left=0.008, right=0.992, bottom=0.008, top=0.992)
    save_pdf(fig, "framework.pdf")


def crop_square(image: np.ndarray, box: tuple[int, int, int, int], scale: float) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * scale
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    right = int(round(cx + side / 2.0))
    bottom = int(round(cy + side / 2.0))
    height, width = image.shape[:2]
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))
    return image[top:bottom, left:right], (left, top, right, bottom)


def make_evidence_card() -> None:
    """Create the final A2 three-panel evidence card from a real DeepPCB short."""
    group = DEEPPCB / "group20085" / "20085"
    current = np.asarray(Image.open(group / "20085117_test.jpg").convert("L"))
    template = np.asarray(Image.open(group / "20085117_temp.jpg").convert("L"))
    box = (265, 368, 305, 408)

    current_tight, tight_window = crop_square(current, box, 1.75)
    template_tight = template[tight_window[1] : tight_window[3], tight_window[0] : tight_window[2]]
    context, context_window = crop_square(current, box, 3.5)
    current_tight = np.asarray(
        Image.fromarray(current_tight).resize((256, 256), Image.Resampling.NEAREST)
    )
    template_tight = np.asarray(
        Image.fromarray(template_tight).resize((256, 256), Image.Resampling.NEAREST)
    )
    context = np.asarray(Image.fromarray(context).resize((256, 256), Image.Resampling.NEAREST))

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.48), gridspec_kw={"wspace": 0.055})
    panels = (
        (current_tight, "P1  当前候选紧裁剪"),
        (template_tight, "P2  同位置正常模板"),
        (context, "P3  更大上下文"),
    )
    for ax, (panel, label) in zip(axes, panels):
        ax.imshow(panel, cmap="gray", vmin=0, vmax=255, interpolation="none")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(INK)
            spine.set_linewidth(0.7)
        ax.set_xlabel(label, fontsize=7.4, labelpad=4.2, fontproperties=CJK)

    left, top, _, _ = context_window
    scale_x = 256.0 / (context_window[2] - context_window[0])
    scale_y = 256.0 / (context_window[3] - context_window[1])
    axes[2].add_patch(
        Rectangle(
            ((box[0] - left) * scale_x, (box[1] - top) * scale_y),
            (box[2] - box[0]) * scale_x,
            (box[3] - box[1]) * scale_y,
            fill=False,
            edgecolor="#D62728",
            linewidth=1.35,
        )
    )
    fig.subplots_adjust(left=0.006, right=0.994, bottom=0.17, top=0.995)
    save_pdf(fig, "evidence_card.pdf")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_call_rate_curves() -> None:
    summary = load_json(ROOT / "experiments" / "rq3_gating" / "full_results.json")["summary"]
    display = {
        "random": "Random",
        "low_confidence": "Low confidence",
        "manual_linear": "Manual linear",
        "logistic": "Logistic",
        "lightgbm": "LightGBM",
        "review_acceptance": "Review + acceptance",
    }
    colors = plt.get_cmap("tab10").colors
    for metric, filename, ylabel in (
        ("f1", "f1_call_rate.pdf", "Final F1"),
        ("hcr", "hcr_call_rate.pdf", "Harmful correction rate"),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        for index, strategy in enumerate(STRATEGIES):
            rows = [row for row in summary if row["strategy"] == strategy]
            ax.plot(
                [row["budget_percent"] for row in rows],
                [row[f"{metric}_mean"] for row in rows],
                marker="o",
                markersize=3.8,
                color=colors[index],
                label=display[strategy],
            )
        ax.set_xlabel("Call budget (%)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        fig.tight_layout()
        save_pdf(fig, filename)


def plot_error_transfer() -> None:
    folds = load_json(ROOT / "experiments" / "rq2_correction" / "full_results.json")["folds"]
    counts: Counter[tuple[str, str]] = Counter()
    for fold in folds:
        for key, value in fold["transfer"].items():
            original, final = key.split("|", maxsplit=1)
            counts[(original, final)] += value
    rows = sorted({key[0] for key in counts})
    columns = sorted({key[1] for key in counts})
    matrix = np.asarray([[counts[(row, column)] for column in columns] for row in rows], dtype=int)

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    color_steps = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.05, 0.95, 16)))
    color_bounds = np.linspace(0, matrix.max(), color_steps.N + 1)
    color_norm = BoundaryNorm(color_bounds, color_steps.N)
    mesh = ax.pcolormesh(
        np.arange(len(columns) + 1),
        np.arange(len(rows) + 1),
        matrix,
        cmap=color_steps,
        norm=color_norm,
        shading="flat",
    )
    ax.invert_yaxis()
    ax.set_aspect("auto")
    fig.colorbar(mesh, ax=ax, label="Candidates")
    ax.set_xticks(np.arange(len(columns)) + 0.5, columns, rotation=35, ha="right", fontproperties=CJK)
    ax.set_yticks(np.arange(len(rows)) + 0.5, rows, fontproperties=CJK)
    threshold = matrix.max() * 0.55
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column + 0.5,
                row + 0.5,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if matrix[row, column] > threshold else INK,
            )
    ax.set_xlabel("Final state")
    ax.set_ylabel("Original error type")
    fig.tight_layout()
    save_pdf(fig, "error_transfer_heatmap.pdf")


def plot_per_class_gain() -> None:
    metrics = load_json(ROOT / "experiments" / "rq1_evidence" / "full_results.json")["metrics"]
    a0 = metrics["A0"]["overall"]["classification"]["per_class_f1"]
    a2 = metrics["A2"]["overall"]["classification"]["per_class_f1"]
    gains = [a2[name] - a0[name] for name in CLASS_LABELS]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.bar(CLASS_LABELS, gains, color=["#2B8CBE" if value >= 0 else "#DE2D26" for value in gains])
    ax.axhline(0.0, color=INK, linewidth=0.8)
    ax.set_ylabel("A2 minus A0 class F1")
    ax.tick_params(axis="x", rotation=25)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    save_pdf(fig, "per_class_evidence_gain.pdf")


def plot_acceptance_reliability() -> None:
    # Exact bin means recovered from the completed RQ2 run's cached responses.
    predicted = np.asarray(
        [0.0030779656, 0.1425264807, 0.2365908748, 0.3580422722, 0.4346367227,
         0.5827264489, 0.7233627521, 0.8360410411, 0.9686794499]
    )
    observed = np.asarray([0.0519662921, 0.0909090909, 0.0, 0.5, 1.0, 1.0, 1.0, 2.0 / 3.0, 1.0 / 3.0])
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot(predicted, observed, marker="o", label="Acceptance gate")
    ax.plot([0, 1], [0, 1], linestyle=":", color="gray", label="Ideal")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed positive utility")
    ax.legend()
    fig.tight_layout()
    save_pdf(fig, "acceptance_reliability.pdf")


def plot_rq4() -> None:
    result = load_json(ROOT / "experiments" / "rq4_robustness" / "full_results.json")
    geometry = [row for row in result["rows"] if row["family"] == "geometry"]
    available = {row["perturbation"] for row in geometry}
    order = [name for name in GEOMETRIC_ORDER if name in available]
    x = np.arange(len(order))
    display = {
        "no_reg_direct": "无二次配准＋直接 XOR",
        "ecc_direct": "ECC＋直接 XOR",
        "ecc_direct_gate": "ECC＋直接 XOR＋质量门",
        "ecc_tolerant": "ECC＋容差 XOR",
        "ecc_tolerant_gate": "ECC＋容差 XOR＋质量门",
    }

    fig, ax = plt.subplots(figsize=(12.0, 5.6))
    for setting in SETTINGS:
        subset = {row["perturbation"]: row for row in geometry if row["setting"] == setting}
        ax.plot(x, [subset[name]["f1"] for name in order], marker="o", markersize=3.5, label=display[setting])
    ax.set_xticks(x, order, rotation=45, ha="right")
    ax.set_ylabel("最终 F1", fontproperties=CJK)
    ax.legend(prop=CJK, fontsize=8)
    fig.tight_layout()
    save_pdf(fig, "rq4_registration_performance.pdf")

    by_perturbation: dict[str, list[float]] = defaultdict(list)
    details_path = ROOT / "experiments" / "rq4_robustness" / "perturbation_details.csv"
    with details_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["family"] == "geometry":
                by_perturbation[row["perturbation"]].append(float(row["q_reg_ecc"]))
    fig, ax = plt.subplots(figsize=(12.0, 5.6))
    ax.boxplot([by_perturbation[name] for name in order], showfliers=False)
    ax.set_xticks(x + 1, order, rotation=45, ha="right")
    ax.set_ylabel("q_reg")
    fig.tight_layout()
    save_pdf(fig, "rq4_qreg_distribution.pdf")

    direct = {row["perturbation"]: row for row in geometry if row["setting"] == "ecc_direct"}
    tolerant = {row["perturbation"]: row for row in geometry if row["setting"] == "ecc_tolerant"}
    fig, ax = plt.subplots(figsize=(12.0, 5.6))
    ax.plot(x, [direct[name]["xor_rate_mean"] for name in order], marker="o", markersize=3.5, label="直接 XOR")
    ax.plot(x, [tolerant[name]["xor_rate_mean"] for name in order], marker="o", markersize=3.5, label="容差 XOR")
    ax.set_xticks(x, order, rotation=45, ha="right")
    ax.set_ylabel("骨架差异像素占比", fontproperties=CJK)
    ax.legend(prop=CJK)
    fig.tight_layout()
    save_pdf(fig, "rq4_xor_comparison.pdf")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    make_framework()
    make_evidence_card()
    plot_call_rate_curves()
    plot_error_transfer()
    plot_per_class_gain()
    plot_acceptance_reliability()
    plot_rq4()


if __name__ == "__main__":
    main()
