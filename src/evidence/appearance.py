"""Appearance residual from absolute intensity and SSIM differences."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from skimage.metrics import structural_similarity


def compute_appearance(test: np.ndarray, aligned_template: np.ndarray, config: dict) -> dict:
    if np.issubdtype(test.dtype, np.integer):
        test_float = test.astype(np.float32) / 255.0
        template_float = aligned_template.astype(np.float32) / 255.0
    else:
        test_float = np.clip((test.astype(np.float32) + 5.0) / 10.0, 0.0, 1.0)
        template_float = np.clip((aligned_template.astype(np.float32) + 5.0) / 10.0, 0.0, 1.0)
    absolute = np.abs(test_float - template_float)
    _, ssim_map = structural_similarity(test_float, template_float, data_range=1.0, full=True)
    ssim_difference = 1.0 - ssim_map.astype(np.float32)
    weights = config["weights"]
    m_app = weights["absolute"] * absolute + weights["ssim"] * ssim_difference
    return {
        "absolute": absolute.astype(np.float32),
        "ssim_difference": ssim_difference,
        "M_app": np.clip(m_app, 0.0, 1.0).astype(np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("test", type=Path)
    parser.add_argument("template", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/evidence/appearance.yaml"))
    parser.add_argument("--output", type=Path, default=Path("cache/appearance.png"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    test = cv2.imread(str(args.test), cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(str(args.template), cv2.IMREAD_GRAYSCALE)
    result = compute_appearance(test, template, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), (result["M_app"] * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
