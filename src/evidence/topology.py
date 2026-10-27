"""Topology difference maps and local eight-dimensional topology features."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from skimage.morphology import skeletonize


def binary_mask(image: np.ndarray, config: dict) -> np.ndarray:
    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_size = int(config["morphology_kernel"])
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask > 0


def _components(mask: np.ndarray) -> int:
    return int(cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1)


def _endpoints_and_branches(skeleton: np.ndarray) -> tuple[int, int]:
    padded = np.pad(skeleton.astype(np.uint8), 1)
    neighbours = sum(np.roll(np.roll(padded, dy, axis=0), dx, axis=1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))
    centre = padded[1:-1, 1:-1]
    return int(np.sum(skeleton & (neighbours[1:-1, 1:-1] == 1))), int(np.sum(skeleton & (neighbours[1:-1, 1:-1] >= 3)))


def _euler(mask: np.ndarray) -> int:
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    return int(sum(1 if item[3] < 0 else -1 for item in hierarchy[0]))


def _perimeter(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return float(sum(cv2.arcLength(contour, True) for contour in contours))


def topology_features(mask: np.ndarray, skeleton: np.ndarray, xor_mask: np.ndarray) -> np.ndarray:
    endpoints, branches = _endpoints_and_branches(skeleton)
    values = np.array(
        [
            _components(mask),
            endpoints,
            branches,
            _euler(mask),
            int(skeleton.sum()),
            float(xor_mask.mean()),
            int(mask.sum()),
            _perimeter(mask),
        ],
        dtype=np.float32,
    )
    return values


def compute_topology(test: np.ndarray, aligned_template: np.ndarray, config: dict) -> dict:
    test_mask = binary_mask(test, config)
    template_mask = binary_mask(aligned_template, config)
    test_skeleton = skeletonize(test_mask)
    template_skeleton = skeletonize(template_mask)
    tolerance = int(config["tolerance_px"])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1))
    tolerant_test = cv2.dilate(test_skeleton.astype(np.uint8), kernel) > 0
    tolerant_template = cv2.dilate(template_skeleton.astype(np.uint8), kernel) > 0
    xor_map = np.logical_or(
        np.logical_and(test_skeleton, np.logical_not(tolerant_template)),
        np.logical_and(template_skeleton, np.logical_not(tolerant_test)),
    )
    local_vector = np.abs(topology_features(test_mask, test_skeleton, xor_map) - topology_features(template_mask, template_skeleton, xor_map))
    return {
        "test_mask": test_mask,
        "template_mask": template_mask,
        "test_skeleton": test_skeleton,
        "template_skeleton": template_skeleton,
        "M_topo": xor_map.astype(np.float32),
        "topology_vector": local_vector,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("test", type=Path)
    parser.add_argument("template", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/evidence/topology.yaml"))
    parser.add_argument("--output", type=Path, default=Path("cache/topology.png"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    test = cv2.imread(str(args.test), cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(str(args.template), cv2.IMREAD_GRAYSCALE)
    result = compute_topology(test, template, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), (result["M_topo"] * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
