"""Feature construction for review and action-acceptance gates."""

from __future__ import annotations

import math

import numpy as np


CLASS_NAMES = ["open", "short", "mousebite", "spur", "pinhole", "spurious_copper", "normal", "uncertain"]

REVIEW_FEATURE_NAMES = [
    "calibrated_confidence", "class_margin", "entropy", "source_det", "source_diff", "source_both",
    "det_diff_iou", "appearance_mean", "appearance_p95", "above_threshold_fraction",
    "topology_components", "topology_endpoints", "topology_branches", "topology_euler",
    "topology_length", "topology_xor", "topology_area", "topology_perimeter",
    "q_reg", "box_width", "box_height", "aspect_ratio",
]


def review_features(candidate: dict) -> np.ndarray:
    confidence = float(candidate.get("calibrated_score", candidate.get("score", 0.0)))
    probability = min(max(confidence, 1e-6), 1.0 - 1e-6)
    entropy = -(probability * math.log(probability) + (1.0 - probability) * math.log(1.0 - probability))
    source = candidate["source"]
    residual = candidate.get("appearance_residual", {})
    topology = list(candidate.get("topology_vector", [0.0] * 8))
    topology = (topology + [0.0] * 8)[:8]
    x1, y1, x2, y2 = candidate["box"]
    width, height = max(1.0, x2 - x1), max(1.0, y2 - y1)
    return np.asarray(
        [
            confidence,
            float(candidate.get("class_margin", confidence)),
            entropy,
            float(source == "DET"),
            float(source == "DIFF"),
            float(source == "BOTH"),
            float(candidate.get("iou", 0.0)),
            float(residual.get("mean", candidate.get("diff_score", 0.0))),
            float(residual.get("p95", candidate.get("diff_score", 0.0))),
            float(candidate.get("above_threshold_fraction", 0.0)),
            *[float(value) for value in topology],
            float(candidate.get("q_reg", 1.0)),
            width,
            height,
            width / height,
        ],
        dtype=np.float64,
    )


def topology_compatibility(conclusion: dict) -> float:
    patterns = set(conclusion["topology_pattern"])
    expected = {
        "open": {"disconnection", "more_endpoints"},
        "short": {"new_connection"},
        "mousebite": {"more_endpoints"},
        "spur": {"extra_branch", "more_endpoints"},
        "pinhole": {"hole_change"},
        "spurious_copper": {"new_component"},
    }
    predicted = conclusion["predicted_class"]
    return float(bool(patterns & expected.get(predicted, set())))


def acceptance_features(candidate: dict, conclusion: dict, json_valid: bool = True) -> np.ndarray:
    class_vector = [float(conclusion["predicted_class"] == name) for name in CLASS_NAMES]
    presence_vector = [
        float(conclusion["defect_present"] is True),
        float(conclusion["defect_present"] is False),
        float(conclusion["defect_present"] is None),
    ]
    reliability = {None: 0.5, "low": 0.0, "medium": 0.5, "high": 1.0}[conclusion["reference_reliability"]]
    appearance_compatible = float("none" not in conclusion["appearance_pattern"] and "uncertain" not in conclusion["appearance_pattern"])
    extra = np.asarray(
        [
            float(candidate.get("q_reg", 1.0)),
            *class_vector,
            *presence_vector,
            float(conclusion["evidence_strength"]) / 3.0,
            reliability,
            topology_compatibility(conclusion),
            appearance_compatible,
            float(json_valid),
        ],
        dtype=np.float64,
    )
    return np.concatenate([review_features(candidate), extra])
