"""Generate connected-component difference candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.evidence.topology import topology_features


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    return intersection / (area_a + area_b - intersection)


def propose_candidates(
    appearance: dict,
    topology: dict,
    test_image: np.ndarray,
    aligned_template: np.ndarray,
    detections: list[dict] | None,
    config: dict,
    fold: int | None = None,
) -> dict:
    m_app = appearance["M_app"]
    tolerance = int(config.get("tolerance_px", 0))
    if tolerance:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1))
        tolerant_test = cv2.dilate(topology["test_skeleton"].astype(np.uint8), kernel) > 0
        tolerant_template = cv2.dilate(topology["template_skeleton"].astype(np.uint8), kernel) > 0
        m_topo = np.logical_or(
            np.logical_and(topology["test_skeleton"], np.logical_not(tolerant_template)),
            np.logical_and(topology["template_skeleton"], np.logical_not(tolerant_test)),
        ).astype(np.float32)
    else:
        m_topo = topology["M_topo"]
    combined = np.maximum(m_app, m_topo)

    def robust_threshold(residual: np.ndarray) -> float:
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        return median + float(config["mad_k"]) * 1.4826 * mad

    threshold_mode = config.get("threshold_mode", "image_mad")
    merge_mode = config.get("merge_mode", "max")
    image_thresholds = {
        "appearance": robust_threshold(m_app),
        "topology": robust_threshold(m_topo),
        "combined": robust_threshold(combined),
    }
    if fold is not None and "global_thresholds_by_fold" in config:
        global_thresholds = config["global_thresholds_by_fold"][str(fold)]
    else:
        global_thresholds = config.get("global_thresholds", image_thresholds)
    if threshold_mode == "global_quantile":
        thresholds = global_thresholds
    elif threshold_mode == "min_global_image":
        thresholds = {
            name: min(float(global_thresholds[name]), image_thresholds[name])
            for name in image_thresholds
        }
    else:
        thresholds = image_thresholds
    if merge_mode == "independent_or":
        binary = np.logical_or(
            m_app > float(thresholds["appearance"]),
            m_topo > float(thresholds["topology"]),
        ).astype(np.uint8)
        threshold: float | dict[str, float] = {
            "appearance": float(thresholds["appearance"]),
            "topology": float(thresholds["topology"]),
        }
    else:
        binary = (combined > float(thresholds["combined"])).astype(np.uint8)
        threshold = float(thresholds["combined"])
    dilate_kernel = np.ones((int(config["dilate_kernel"]), int(config["dilate_kernel"])), np.uint8)
    open_kernel = np.ones((int(config["open_kernel"]), int(config["open_kernel"])), np.uint8)
    close_kernel = np.ones((int(config["close_kernel"]), int(config["close_kernel"])), np.uint8)
    binary = cv2.dilate(binary, dilate_kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    candidates = []
    for label in range(1, count):
        x, y, width, height, area = stats[label].tolist()
        if area < int(config["min_area"]) or area > int(config["max_area"]):
            continue
        x2, y2 = x + width, y + height
        region = labels[y:y2, x:x2] == label
        test_mask = topology["test_mask"][y:y2, x:x2]
        template_mask = topology["template_mask"][y:y2, x:x2]
        test_skeleton = topology["test_skeleton"][y:y2, x:x2]
        template_skeleton = topology["template_skeleton"][y:y2, x:x2]
        topo_vector = np.abs(
            topology_features(test_mask, test_skeleton, region)
            - topology_features(template_mask, template_skeleton, region)
        ).tolist()
        residual = m_app[y:y2, x:x2][region]
        diff_box = [int(x), int(y), int(x2), int(y2)]
        overlap = [box_iou(diff_box, item["box"]) for item in (detections or [])]
        candidates.append(
            {
                "diff_id": 0,
                "box": diff_box,
                "area": int(area),
                "score": float(combined[y:y2, x:x2][region].mean()),
                "appearance_residual": {
                    "mean": float(residual.mean()),
                    "max": float(residual.max()),
                    "p95": float(np.quantile(residual, 0.95)),
                },
                "topology_vector": [float(value) for value in topo_vector],
                "det_overlap": [float(value) for value in overlap],
            }
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    candidates = candidates[: int(config["max_candidates_per_image"])]
    for diff_id, candidate in enumerate(candidates):
        candidate["diff_id"] = diff_id
    return {"threshold": threshold, "mask": binary, "candidates": candidates}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("test", type=Path)
    parser.add_argument("template", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/candidates.yaml"))
    parser.add_argument("--output", type=Path, default=Path("cache/diff_candidates.json"))
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    image = cv2.imread(str(args.test), cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(str(args.template), cv2.IMREAD_GRAYSCALE)
    from src.evidence.appearance import compute_appearance
    from src.evidence.topology import compute_topology

    appearance = compute_appearance(image, template, yaml.safe_load(Path("configs/evidence/appearance.yaml").read_text(encoding="utf-8")))
    topology = compute_topology(image, template, yaml.safe_load(Path("configs/evidence/topology.yaml").read_text(encoding="utf-8")))
    result = propose_candidates(appearance, topology, image, template, [], config, args.fold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(__import__("json").dumps({"threshold": result["threshold"], "candidates": result["candidates"]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
