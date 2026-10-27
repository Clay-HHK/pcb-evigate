"""Render fixed 2x3 evidence cards for candidate verification."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


PANEL_TITLES = (
    "P1 Current candidate",
    "P2 Normal reference",
    "P3 Larger context",
    "P4 Appearance difference",
    "P5 Skeleton tolerance XOR",
    "P6 Numeric topology summary",
)


def _bounds(box: list[float], scale: float, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    width, height = (x2 - x1) * scale, (y2 - y1) * scale
    left = max(0, int(round(cx - width / 2.0)))
    top = max(0, int(round(cy - height / 2.0)))
    right = min(shape[1], int(round(cx + width / 2.0)))
    bottom = min(shape[0], int(round(cy + height / 2.0)))
    return left, top, right, bottom


def _gray_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()


def _crop_panel(image: np.ndarray, bounds: tuple[int, int, int, int], size: int) -> np.ndarray:
    x1, y1, x2, y2 = bounds
    crop = image[y1:y2, x1:x2]
    return cv2.resize(_gray_bgr(crop), (size, size), interpolation=cv2.INTER_CUBIC)


def _with_title(image: np.ndarray, title: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(output, title, (7, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def _disabled_panel(size: int) -> np.ndarray:
    panel = np.full((size, size, 3), 238, dtype=np.uint8)
    cv2.putText(panel, "Not provided", (size // 4, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1, cv2.LINE_AA)
    return panel


def create_evidence_card(
    current: np.ndarray,
    aligned_template: np.ndarray,
    candidate: dict,
    appearance_map: np.ndarray,
    topology: dict,
    q_reg: float,
    output_path: Path,
    config: dict,
    enabled_panels: set[str] | None = None,
) -> Path:
    size = int(config.get("panel_size", 256))
    enabled = enabled_panels or {"P1", "P2", "P3", "P4", "P5", "P6"}
    tight = _bounds(candidate["box"], float(config.get("tight_scale", 1.8)), current.shape[:2])
    context = _bounds(candidate["box"], float(config.get("context_scale", 3.5)), current.shape[:2])
    panels: list[np.ndarray] = []
    panels.append(_crop_panel(current, tight, size))
    panels.append(_crop_panel(aligned_template, tight, size))

    context_panel = _crop_panel(current, context, size)
    bx1, by1, bx2, by2 = candidate["box"]
    cx1, cy1, cx2, cy2 = context
    sx, sy = size / (cx2 - cx1), size / (cy2 - cy1)
    cv2.rectangle(
        context_panel,
        (int((bx1 - cx1) * sx), int((by1 - cy1) * sy)),
        (int((bx2 - cx1) * sx), int((by2 - cy1) * sy)),
        (0, 0, 255),
        2,
    )
    panels.append(context_panel)

    if config.get("appearance_mode", "continuous") == "binary":
        app_color = np.zeros((*appearance_map.shape, 3), dtype=np.uint8)
        app_color[appearance_map > float(config["appearance_threshold"])] = (0, 0, 255)
    else:
        app_uint8 = np.clip(appearance_map * 255.0, 0, 255).astype(np.uint8)
        app_color = cv2.applyColorMap(app_uint8, cv2.COLORMAP_TURBO)
    panels.append(_crop_panel(app_color, context, size))

    test_skeleton = topology["test_skeleton"]
    template_skeleton = topology["template_skeleton"]
    xor_map = topology["M_topo"] > 0
    topo_color = np.zeros((*xor_map.shape, 3), dtype=np.uint8)
    topo_color[template_skeleton] = (255, 120, 0)
    topo_color[test_skeleton] = (0, 220, 0)
    topo_color[xor_map] = (0, 0, 255)
    panels.append(_crop_panel(topo_color, context, size))

    summary = np.full((size, size, 3), 248, dtype=np.uint8)
    vector = candidate.get("topology_vector", topology["topology_vector"])
    lines = [
        f"q_reg: {q_reg:.3f}",
        f"source: {candidate['source']}",
    ]
    if config.get("include_appearance_numeric", True):
        lines.append(f"app mean: {candidate.get('appearance_residual', {}).get('mean', 0.0):.3f}")
    lines.extend([
        "topology delta:",
        f"components {vector[0]:.2f}  endpoints {vector[1]:.2f}",
        f"branches {vector[2]:.2f}  euler {vector[3]:.2f}",
        f"length {vector[4]:.2f}  xor {vector[5]:.3f}",
        f"area {vector[6]:.2f}  perimeter {vector[7]:.2f}",
        "P5: blue ref, green current, red XOR",
    ])
    for index, line in enumerate(lines):
        cv2.putText(summary, line, (8, 48 + index * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (25, 25, 25), 1, cv2.LINE_AA)
    panels.append(summary)

    rendered = [
        _with_title(panel if f"P{index + 1}" in enabled else _disabled_panel(size), PANEL_TITLES[index])
        for index, panel in enumerate(panels)
    ]
    card = np.vstack([np.hstack(rendered[:3]), np.hstack(rendered[3:])])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), card)
    return output_path
