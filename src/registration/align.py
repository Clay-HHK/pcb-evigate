"""ECC/ORB template alignment and photometric normalization."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_gray(path: Path) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def _to_full_transform(transform: np.ndarray, scale: float) -> np.ndarray:
    homography = np.vstack([transform, [0.0, 0.0, 1.0]])
    scale_matrix = np.diag([scale, scale, 1.0])
    full = np.linalg.inv(scale_matrix) @ homography @ scale_matrix
    return full[:2].astype(np.float32)


def _to_level_transform(transform: np.ndarray, scale: float) -> np.ndarray:
    homography = np.vstack([transform, [0.0, 0.0, 1.0]])
    scale_matrix = np.diag([scale, scale, 1.0])
    level = scale_matrix @ homography @ np.linalg.inv(scale_matrix)
    return level[:2].astype(np.float32)


def _ecc(test: np.ndarray, template: np.ndarray, config: dict) -> tuple[np.ndarray, float]:
    levels = int(config["pyramid_levels"])
    scales = [1.0 / (2 ** index) for index in range(levels - 1, -1, -1)]
    transform = np.eye(2, 3, dtype=np.float32)
    correlation = 0.0
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(config["ecc_iterations"]), float(config["ecc_epsilon"]))
    for scale in scales:
        test_level = cv2.resize(test, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        template_level = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        transform_level = _to_level_transform(transform, scale)
        correlation, transform_level = cv2.findTransformECC(
            test_level,
            template_level,
            transform_level,
            cv2.MOTION_AFFINE,
            criteria,
            None,
            int(config["ecc_gaussian_size"]),
        )
        transform = _to_full_transform(transform_level, scale)
    return transform, float(correlation)


def _orb_ransac(test: np.ndarray, template: np.ndarray, config: dict) -> tuple[np.ndarray, float, float]:
    orb_config = config["orb"]
    orb = cv2.ORB_create(nfeatures=int(orb_config["nfeatures"]))
    keypoints_test, descriptors_test = orb.detectAndCompute(test, None)
    keypoints_template, descriptors_template = orb.detectAndCompute(template, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(descriptors_template, descriptors_test, k=2)
    good = [first for first, second in matches if first.distance < float(orb_config["ratio"]) * second.distance]
    source = np.float32([keypoints_template[item.queryIdx].pt for item in good])
    target = np.float32([keypoints_test[item.trainIdx].pt for item in good])
    homography, mask = cv2.findHomography(source, target, cv2.RANSAC, float(orb_config["ransac_reproj_threshold"]))
    if homography is None:
        translation = np.median(target - source, axis=0)
        affine = np.float32([[1.0, 0.0, translation[0]], [0.0, 1.0, translation[1]]])
        residuals = np.linalg.norm((source + translation) - target, axis=1)
        inlier_rate = float((residuals <= float(orb_config["ransac_reproj_threshold"])).mean())
    else:
        affine = homography[:2].astype(np.float32)
        inlier_rate = float(mask.ravel().mean())
    return affine, 0.0, inlier_rate


def _edge_chamfer(test: np.ndarray, aligned_template: np.ndarray) -> float:
    test_edges = cv2.Canny(test, 50, 150) > 0
    template_edges = cv2.Canny(aligned_template, 50, 150) > 0
    distance = cv2.distanceTransform((~test_edges).astype(np.uint8), cv2.DIST_L2, 3)
    return float(distance[template_edges].mean()) if template_edges.any() else float(test.shape[0])


def _transform_quality(transform: np.ndarray, config: dict) -> float:
    tx, ty = float(transform[0, 2]), float(transform[1, 2])
    rotation = np.degrees(np.arctan2(transform[1, 0], transform[0, 0]))
    translation_score = max(0.0, 1.0 - np.hypot(tx, ty) / float(config["max_translation"]))
    rotation_score = max(0.0, 1.0 - abs(rotation) / float(config["max_rotation_deg"]))
    determinant_score = max(0.0, min(1.0, float(np.linalg.det(transform[:, :2]))))
    return float((translation_score + rotation_score + determinant_score) / 3.0)


def photometric_normalize(test: np.ndarray, aligned_template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    test_float = test.astype(np.float32)
    template_float = aligned_template.astype(np.float32)
    test_median, template_median = np.median(test_float), np.median(template_float)
    test_mad = np.median(np.abs(test_float - test_median)) + 1.0
    template_mad = np.median(np.abs(template_float - template_median)) + 1.0
    normalized_test = np.clip((test_float - test_median) / test_mad, -5.0, 5.0)
    normalized_template = np.clip((template_float - template_median) / template_mad, -5.0, 5.0)
    return normalized_test.astype(np.float32), normalized_template.astype(np.float32)


def align_images(test: np.ndarray, template: np.ndarray, config: dict) -> dict:
    method = "ECC"
    try:
        transform, ecc_correlation = _ecc(test, template, config)
        inlier_rate = 1.0
        aligned_template = cv2.warpAffine(template, transform, (test.shape[1], test.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    except cv2.error:
        transform, ecc_correlation, inlier_rate = _orb_ransac(test, template, config)
        method = "ORB_RANSAC"
        aligned_template = cv2.warpPerspective(template, np.vstack([transform, [0, 0, 1]]), (test.shape[1], test.shape[0]))
    chamfer = _edge_chamfer(test, aligned_template)
    chamfer_score = max(0.0, 1.0 - chamfer / 20.0)
    transform_score = _transform_quality(transform, config)
    weights = config["weights"]
    q_reg = float(
        np.clip(
            weights["ecc"] * max(0.0, min(1.0, ecc_correlation))
            + weights["inlier"] * max(0.0, min(1.0, inlier_rate))
            + weights["chamfer"] * chamfer_score
            + weights["transform"] * transform_score,
            0.0,
            1.0,
        )
    )
    normalized_test, normalized_template = photometric_normalize(test, aligned_template)
    return {
        "aligned_template": aligned_template,
        "normalized_test": normalized_test,
        "normalized_template": normalized_template,
        "q_reg": q_reg,
        "method": method,
        "transform": transform,
        "ecc_correlation": float(ecc_correlation),
        "inlier_rate": float(inlier_rate),
        "chamfer": chamfer,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("test", type=Path)
    parser.add_argument("template", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/evidence/registration.yaml"))
    parser.add_argument("--output", type=Path, default=Path("cache/alignment.png"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = align_images(load_gray(args.test), load_gray(args.template), config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), result["aligned_template"])
    print(f"q_reg={result['q_reg']:.6f} method={result['method']}")


if __name__ == "__main__":
    main()
