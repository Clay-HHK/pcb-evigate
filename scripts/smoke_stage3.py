"""Run M1 through M5 on twenty deterministic DeepPCB pairs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.candidates.fuse import fuse_candidates
from src.candidates.propose import propose_candidates
from src.evidence.appearance import compute_appearance
from src.evidence.topology import compute_topology
from src.registration.align import align_images, load_gray

SEED = 20260811


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--pairs", type=int, default=20)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "data/manifests/deeppcb.csv"
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))[: args.pairs]
    registration_config = yaml.safe_load((root / "configs/evidence/registration.yaml").read_text(encoding="utf-8"))
    appearance_config = yaml.safe_load((root / "configs/evidence/appearance.yaml").read_text(encoding="utf-8"))
    topology_config = yaml.safe_load((root / "configs/evidence/topology.yaml").read_text(encoding="utf-8"))
    candidate_config = yaml.safe_load((root / "configs/candidates.yaml").read_text(encoding="utf-8"))
    prediction_path = root / f"cache/detector_predictions/fold{args.fold}.jsonl"
    predictions = {}
    if prediction_path.exists():
        predictions = {item["pair_id"]: item for item in map(json.loads, prediction_path.read_text(encoding="utf-8").splitlines())}
    output = root / "experiments/stage3_smoke"
    output.mkdir(parents=True, exist_ok=True)
    totals = {"aligned": 0, "appearance": 0, "topology": 0, "diff_candidates": 0, "fused_candidates": 0}
    total_times = {"M1": 0.0, "M2": 0.0, "M3": 0.0, "M4": 0.0, "M5": 0.0}
    for index, row in enumerate(rows):
        test = load_gray(root / row["test_path"])
        template = load_gray(root / row["temp_path"])
        start = time.perf_counter()
        aligned = align_images(test, template, registration_config)
        total_times["M1"] += time.perf_counter() - start
        start = time.perf_counter()
        appearance = compute_appearance(aligned["normalized_test"], aligned["normalized_template"], appearance_config)
        total_times["M2"] += time.perf_counter() - start
        start = time.perf_counter()
        topology = compute_topology((aligned["normalized_test"] * 32 + 128).astype(np.uint8), (aligned["normalized_template"] * 32 + 128).astype(np.uint8), topology_config)
        total_times["M3"] += time.perf_counter() - start
        detections = []
        if row["pair_id"] in predictions:
            prediction = predictions[row["pair_id"]]
            detections = [{"box": box, "class": class_id, "score": score} for box, class_id, score in zip(prediction["boxes"], prediction["classes"], prediction["scores"])]
        start = time.perf_counter()
        differences = propose_candidates(appearance, topology, test, aligned["aligned_template"], detections, candidate_config)
        total_times["M4"] += time.perf_counter() - start
        start = time.perf_counter()
        fused = fuse_candidates(row["pair_id"], detections, differences["candidates"], int(candidate_config["center_expand_px"]))
        total_times["M5"] += time.perf_counter() - start
        totals["aligned"] += 1
        totals["appearance"] += 1
        totals["topology"] += 1
        totals["diff_candidates"] += len(differences["candidates"])
        totals["fused_candidates"] += len(fused)
        if index < 3:
            cv2.imwrite(str(output / f"{row['pair_id']}_registration.png"), cv2.addWeighted(test, 0.5, aligned["aligned_template"], 0.5, 0))
            cv2.imwrite(str(output / f"{row['pair_id']}_appearance.png"), (appearance["M_app"] * 255).astype(np.uint8))
            cv2.imwrite(str(output / f"{row['pair_id']}_topology.png"), (topology["M_topo"] * 255).astype(np.uint8))
    print(f"样本对数量: {len(rows)}")
    for module in ("M1", "M2", "M3", "M4", "M5"):
        print(f"{module} 耗时: {total_times[module]:.3f} 秒")
    print(f"M1 配准产出: {totals['aligned']}")
    print(f"M2 外观差异产出: {totals['appearance']}")
    print(f"M3 拓扑差异产出: {totals['topology']}")
    print(f"M4 差异候选数量: {totals['diff_candidates']}")
    print(f"M5 融合候选数量: {totals['fused_candidates']}")
    print(f"可视化目录: {output}")


if __name__ == "__main__":
    main()
