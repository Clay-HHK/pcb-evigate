#!/usr/bin/env python3
"""在 GroupKFold val 部分扫描差异候选参数并生成阶段 A 产物。"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detector.error_decompose import decompose_record
from src.evidence.appearance import compute_appearance
from src.evidence.topology import binary_mask
from src.registration.align import align_images, load_gray, photometric_normalize


SEED = 20260811
FOLDS = range(5)
INITIAL_SAMPLE_SIZE = 300
EXTENDED_SAMPLE_SIZE = 500
MIN_MISSED_COUNT = 150
HIST_BINS = 65536
OUTPUT = ROOT / "experiments/stageA_sweep"
PREPARED = OUTPUT / "prepared_val"
PREDICTIONS = ROOT / "cache/detector_predictions"


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_inputs() -> tuple[list[dict[str, str]], dict[str, dict[str, list[str]]]]:
    with (ROOT / "data/manifests/deeppcb.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    splits = json.loads((ROOT / "data/splits/groupkfold5.json").read_text(encoding="utf-8"))
    return rows, splits


def read_gt(row: dict[str, str]) -> list[list[float]]:
    return [list(map(float, line.split()[:4])) for line in (ROOT / row["anno_path"]).read_text().splitlines()]


def iou(box_a: list[float], box_b: list[float]) -> float:
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def select_sample(
    rows: list[dict[str, str]], splits: dict[str, dict[str, list[str]]], sample_size: int
) -> list[dict[str, str | int]]:
    rng = random.Random(SEED)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    for values in grouped.values():
        rng.shuffle(values)
    selected: list[dict[str, str]] = []
    group_names = sorted(grouped)
    cursor = {group: 0 for group in group_names}
    while len(selected) < sample_size:
        for group in group_names:
            if cursor[group] < len(grouped[group]) and len(selected) < sample_size:
                selected.append(grouped[group][cursor[group]])
                cursor[group] += 1
    sample = []
    for row in selected:
        fold = next(fold for fold in FOLDS if row["group"] in splits[f"fold_{fold}"]["val"])
        sample.append({**row, "fold": fold})
    return sample


def topology_map(test: np.ndarray, template: np.ndarray, tolerance: int) -> np.ndarray:
    from skimage.morphology import skeletonize

    config = read_yaml(ROOT / "configs/evidence/topology.yaml")
    test_skeleton = skeletonize(binary_mask(test, config))
    template_skeleton = skeletonize(binary_mask(template, config))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1))
    tolerant_test = cv2.dilate(test_skeleton.astype(np.uint8), kernel) > 0
    tolerant_template = cv2.dilate(template_skeleton.astype(np.uint8), kernel) > 0
    return np.logical_or(
        np.logical_and(test_skeleton, np.logical_not(tolerant_template)),
        np.logical_and(template_skeleton, np.logical_not(tolerant_test)),
    ).astype(np.uint8)


def compute_maps(row: dict[str, str]) -> tuple[np.ndarray, dict[int, np.ndarray], float]:
    registration_config = read_yaml(ROOT / "configs/evidence/registration.yaml")
    appearance_config = read_yaml(ROOT / "configs/evidence/appearance.yaml")
    test = load_gray(ROOT / row["test_path"])
    template = load_gray(ROOT / row["temp_path"])
    orb_config = registration_config["orb"]
    orb = cv2.ORB_create(nfeatures=int(orb_config["nfeatures"]))
    keypoints_test, descriptors_test = orb.detectAndCompute(test, None)
    keypoints_template, descriptors_template = orb.detectAndCompute(template, None)
    if descriptors_test is None or descriptors_template is None:
        observable_matches = 0
    else:
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptors_template, descriptors_test, k=2)
        observable_matches = sum(
            pair[0].distance < float(orb_config["ratio"]) * pair[1].distance
            for pair in matches if len(pair) == 2
        )
    if observable_matches < 4:
        normalized_test, normalized_template = photometric_normalize(test, template)
        q_reg = 0.0
    else:
        aligned = align_images(test, template, registration_config)
        normalized_test = aligned["normalized_test"]
        normalized_template = aligned["normalized_template"]
        q_reg = float(aligned["q_reg"])
    appearance = compute_appearance(normalized_test, normalized_template, appearance_config)["M_app"]
    topo_test = (normalized_test * 32 + 128).astype(np.uint8)
    topo_template = (normalized_template * 32 + 128).astype(np.uint8)
    topology = {delta: topology_map(topo_test, topo_template, delta) for delta in (1, 2, 3)}
    return appearance, topology, q_reg


def quantized_hist(values: np.ndarray) -> np.ndarray:
    quantized = np.clip(
        np.rint(values.astype(np.float32) * (HIST_BINS - 1)), 0, HIST_BINS - 1
    ).astype(np.uint16)
    return np.bincount(quantized.ravel(), minlength=HIST_BINS).astype(np.int64)


def histogram_worker(row: dict[str, str]) -> tuple[str, str, dict[str, np.ndarray]]:
    appearance, topology, _ = compute_maps(row)
    outside = np.ones(appearance.shape, dtype=bool)
    for x1, y1, x2, y2 in read_gt(row):
        outside[int(y1):int(y2), int(x1):int(x2)] = False
    histograms = {"appearance": quantized_hist(appearance[outside])}
    for delta, topo in topology.items():
        histograms[f"topology_{delta}"] = quantized_hist(topo[outside])
        histograms[f"combined_{delta}"] = quantized_hist(np.maximum(appearance, topo)[outside])
    return row["pair_id"], row["group"], histograms


def prepare_worker(row: dict[str, str | int]) -> tuple[str, float]:
    string_row = {key: str(value) for key, value in row.items() if key != "fold"}
    appearance, topology, q_reg = compute_maps(string_row)
    path = PREPARED / f"{row['pair_id']}_fold{row['fold']}.npz"
    np.savez_compressed(
        path,
        appearance=appearance.astype(np.float16),
        topology_1=topology[1],
        topology_2=topology[2],
        topology_3=topology[3],
        q_reg=np.float32(q_reg),
    )
    return str(row["pair_id"]), q_reg


def build_global_thresholds(
    rows: list[dict[str, str]], splits: dict[str, dict[str, list[str]]], workers: int
) -> dict[str, dict[str, dict[str, float]]]:
    cache_path = OUTPUT / "global_thresholds_val.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    group_histograms: dict[str, dict[str, np.ndarray]] = {}
    with mp.get_context("spawn").Pool(workers) as pool:
        for index, (_, group, histograms) in enumerate(pool.imap_unordered(histogram_worker, rows), 1):
            if group not in group_histograms:
                group_histograms[group] = {name: np.zeros(HIST_BINS, dtype=np.int64) for name in histograms}
            for name, histogram in histograms.items():
                group_histograms[group][name] += histogram
            if index % 100 == 0:
                print(f"全局阈值残差统计 {index}/{len(rows)}", flush=True)
    output: dict[str, dict[str, dict[str, float]]] = {}
    for fold in FOLDS:
        fold_histograms = {
            name: sum((group_histograms[group][name] for group in splits[f"fold_{fold}"]["val"]), np.zeros(HIST_BINS, dtype=np.int64))
            for name in next(iter(group_histograms.values()))
        }
        output[str(fold)] = {}
        for alpha in (0.05, 0.02, 0.01, 0.005):
            values = {}
            for name, histogram in fold_histograms.items():
                target = (1.0 - alpha) * int(histogram.sum())
                index = int(np.searchsorted(np.cumsum(histogram), target, side="left"))
                values[name] = index / (HIST_BINS - 1)
            output[str(fold)][str(alpha)] = values
    cache_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def prepare_sample(sample: list[dict[str, str | int]], workers: int) -> None:
    PREPARED.mkdir(parents=True, exist_ok=True)
    missing = [row for row in sample if not (PREPARED / f"{row['pair_id']}_fold{row['fold']}.npz").exists()]
    if missing:
        with mp.get_context("spawn").Pool(workers) as pool:
            for index, _ in enumerate(pool.imap_unordered(prepare_worker, missing), 1):
                if index % 25 == 0:
                    print(f"样本差异图准备 {index}/{len(missing)}", flush=True)


def predict_sample(
    sample: list[dict[str, str | int]],
    rows: list[dict[str, str]],
    splits: dict[str, dict[str, list[str]]],
) -> dict[tuple[int, str], dict]:
    from ultralytics import YOLO

    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    output: dict[tuple[int, str], dict] = {}
    for fold in FOLDS:
        fold_path = PREDICTIONS / f"fold{fold}_val.jsonl"
        fold_rows = [row for row in rows if row["group"] in splits[f"fold_{fold}"]["val"]]
        if not fold_path.exists():
            working = json.loads((ROOT / f"cache/detector_predictions/working_point_fold{fold}.json").read_text())
            model = YOLO(str(ROOT / f"cache/detector_runs/fold{fold}/weights/best.pt"))
            results = model.predict(
                [str(ROOT / str(row["test_path"])) for row in fold_rows],
                conf=float(working["working_confidence"]),
                batch=16,
                verbose=False,
            )
            records = []
            for row, result in zip(fold_rows, results):
                records.append(
                    {
                        "pair_id": row["pair_id"],
                        "boxes": result.boxes.xyxy.cpu().numpy().tolist(),
                        "classes": result.boxes.cls.cpu().numpy().astype(int).tolist(),
                        "scores": result.boxes.conf.cpu().numpy().tolist(),
                    }
                )
            fold_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n", encoding="utf-8")
        selected_ids = {str(row["pair_id"]) for row in sample if int(row["fold"]) == fold}
        for record in map(json.loads, fold_path.read_text(encoding="utf-8").splitlines()):
            if record["pair_id"] in selected_ids:
                output[(fold, record["pair_id"])] = record
    return output


def count_misses(
    sample: list[dict[str, str | int]], predictions: dict[tuple[int, str], dict]
) -> int:
    return sum(
        len(missed_boxes(predictions[(int(row["fold"]), str(row["pair_id"]))], row))
        for row in sample
    )


def count_full_val_misses(
    rows: list[dict[str, str]], splits: dict[str, dict[str, list[str]]]
) -> int:
    total = 0
    for fold in FOLDS:
        records = {
            record["pair_id"]: record
            for record in map(
                json.loads,
                (PREDICTIONS / f"fold{fold}_val.jsonl").read_text(encoding="utf-8").splitlines(),
            )
        }
        for row in rows:
            if row["group"] in splits[f"fold_{fold}"]["val"]:
                total += len(missed_boxes(records[row["pair_id"]], row))
    return total


def robust_threshold(residual: np.ndarray, k: float, nonzero: bool = False) -> float:
    values = residual[residual > 0] if nonzero else residual.ravel()
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median + k * 1.4826 * mad


def make_binary(
    appearance: np.ndarray,
    topology: np.ndarray,
    config: dict,
    global_values: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    combined = np.maximum(appearance, topology)
    if config["threshold_mode"] == "legacy":
        threshold = robust_threshold(appearance, float(config["mad_k"]), nonzero=True)
        return (appearance >= threshold).astype(np.uint8), combined
    image_values = {
        "appearance": robust_threshold(appearance, float(config["mad_k"])),
        "topology": robust_threshold(topology, float(config["mad_k"])),
        "combined": robust_threshold(combined, float(config["mad_k"])),
    }
    if config["threshold_mode"] == "global_quantile":
        thresholds = global_values
    elif config["threshold_mode"] == "min_global_image":
        thresholds = {name: min(global_values[name], image_values[name]) for name in image_values}
    else:
        thresholds = image_values
    if config["merge_mode"] == "independent_or":
        binary = np.logical_or(
            appearance > thresholds["appearance"], topology > thresholds["topology"]
        )
    else:
        binary = combined > thresholds["combined"]
    return binary.astype(np.uint8), combined


def extract_candidates(binary: np.ndarray, combined: np.ndarray, config: dict) -> list[dict]:
    dilate = np.ones((int(config["dilate_kernel"]),) * 2, np.uint8)
    opening = np.ones((int(config["open_kernel"]),) * 2, np.uint8)
    closing = np.ones((int(config["close_kernel"]),) * 2, np.uint8)
    mask = cv2.dilate(binary, dilate)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    for label in range(1, count):
        x, y, width, height, area = stats[label].tolist()
        if int(config["min_area"]) <= area <= int(config["max_area"]):
            region = labels[y:y + height, x:x + width] == label
            candidates.append(
                {
                    "box": [float(x), float(y), float(x + width), float(y + height)],
                    "area": int(area),
                    "score": float(combined[y:y + height, x:x + width][region].mean()),
                }
            )
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def missed_boxes(record: dict, row: dict[str, str | int]) -> list[list[float]]:
    errors = decompose_record(record, {key: str(value) for key, value in row.items() if key != "fold"})
    return [json.loads(item["gt_box"]) for item in errors if item["error_type"] == "漏检"]


def evaluate_config(
    config: dict,
    sample: list[dict[str, str | int]],
    predictions: dict[tuple[int, str], dict],
    global_thresholds: dict[str, dict[str, dict[str, float]]],
) -> tuple[dict, list[dict]]:
    totals = {"candidates": 0, "noise": 0, "missed": 0, "recovered_03": 0, "recovered_05": 0}
    folds = {fold: {"missed": 0, "recovered_03": 0, "recovered_05": 0} for fold in FOLDS}
    details = []
    for row in sample:
        fold = int(row["fold"])
        record = predictions[(fold, str(row["pair_id"]))]
        data = np.load(PREPARED / f"{row['pair_id']}_fold{fold}.npz")
        appearance = data["appearance"].astype(np.float32)
        topology = data[f"topology_{config['tolerance_px']}"]
        alpha_values = global_thresholds[str(fold)][str(config["alpha"])]
        global_values = {
            "appearance": alpha_values["appearance"],
            "topology": alpha_values[f"topology_{config['tolerance_px']}"],
            "combined": alpha_values[f"combined_{config['tolerance_px']}"],
        }
        binary, combined = make_binary(appearance, topology, config, global_values)
        candidates = extract_candidates(binary, combined, config)
        gt_boxes = read_gt({key: str(value) for key, value in row.items() if key != "fold"})
        misses = missed_boxes(record, row)
        detection_boxes = record["boxes"]
        recovered_03 = sum(any(iou(miss, item["box"]) >= 0.3 for item in candidates) for miss in misses)
        recovered_05 = sum(any(iou(miss, item["box"]) >= 0.5 for item in candidates) for miss in misses)
        noise = sum(
            not any(iou(item["box"], gt) >= 0.3 for gt in gt_boxes)
            and not any(iou(item["box"], det) > 0.0 for det in detection_boxes)
            for item in candidates
        )
        totals["candidates"] += len(candidates)
        totals["noise"] += noise
        totals["missed"] += len(misses)
        totals["recovered_03"] += recovered_03
        totals["recovered_05"] += recovered_05
        folds[fold]["missed"] += len(misses)
        folds[fold]["recovered_03"] += recovered_03
        folds[fold]["recovered_05"] += recovered_05
        details.append(
            {
                "row": row,
                "candidates": candidates,
                "misses": misses,
                "detections": detection_boxes,
                "gt": gt_boxes,
                "recovered_03": recovered_03,
                "noise": noise,
                "q_reg": float(data["q_reg"]),
            }
        )
    missed = totals["missed"]
    recovered = totals["recovered_03"]
    metrics = {
        "sample_images": len(sample),
        "missed_count": missed,
        "recovered_count_iou03": recovered,
        "recovered_count_iou05": totals["recovered_05"],
        "miss_recovery_iou03": recovered / missed if missed else 0.0,
        "miss_recovery_iou05": totals["recovered_05"] / missed if missed else 0.0,
        "candidates_per_image": totals["candidates"] / len(sample),
        "noise_candidates_per_image": totals["noise"] / len(sample),
        "candidates_per_recovered_miss": totals["candidates"] / recovered if recovered else float("inf"),
    }
    for fold, values in folds.items():
        metrics[f"fold{fold}_missed"] = values["missed"]
        metrics[f"fold{fold}_recovery_iou03"] = values["recovered_03"] / values["missed"] if values["missed"] else 0.0
        metrics[f"fold{fold}_recovery_iou05"] = values["recovered_05"] / values["missed"] if values["missed"] else 0.0
    return metrics, details


def config_key(config: dict) -> tuple:
    return tuple((key, config[key]) for key in sorted(config))


def rank(rows: list[dict]) -> list[dict]:
    feasible = [row for row in rows if row["noise_candidates_per_image"] <= 20.0]
    candidates = feasible or rows
    return sorted(
        candidates,
        key=lambda row: (
            -float(row["miss_recovery_iou03"]),
            float(row["noise_candidates_per_image"]),
            float(row["candidates_per_recovered_miss"]),
        ),
    )


def pareto(rows: list[dict]) -> list[dict]:
    return [
        row for row in rows
        if not any(
            other is not row
            and float(other["miss_recovery_iou03"]) >= float(row["miss_recovery_iou03"])
            and float(other["noise_candidates_per_image"]) <= float(row["noise_candidates_per_image"])
            and (
                float(other["miss_recovery_iou03"]) > float(row["miss_recovery_iou03"])
                or float(other["noise_candidates_per_image"]) < float(row["noise_candidates_per_image"])
            )
            for other in rows
        )
    ]


def run_sweep(
    sample: list[dict[str, str | int]],
    predictions: dict[tuple[int, str], dict],
    global_thresholds: dict[str, dict[str, dict[str, float]]],
    train_global_thresholds: dict[str, dict[str, dict[str, float]]],
) -> tuple[list[dict], dict, list[dict], dict, dict]:
    base = {
        "threshold_mode": "image_mad", "alpha": 0.02, "mad_k": 3.0,
        "dilate_kernel": 3, "open_kernel": 1, "close_kernel": 1,
        "min_area": 300, "max_area": 8000, "tolerance_px": 2, "merge_mode": "max",
    }
    current_default = {
        **base, "threshold_mode": "legacy", "mad_k": 0.4, "min_area": 300,
        "open_kernel": 1, "close_kernel": 1, "max_area": 20000, "merge_mode": "max",
    }
    train_recommended = {
        **base, "threshold_mode": "global_quantile", "alpha": 0.02,
        "open_kernel": 1, "close_kernel": 5, "min_area": 150,
        "max_area": 8000, "tolerance_px": 3, "merge_mode": "max",
    }
    results: list[dict] = []
    details_by_id: dict[str, list[dict]] = {}

    def evaluate(stage: str, label: str, config: dict) -> dict:
        metrics, details = evaluate_config(config, sample, predictions, global_thresholds)
        row = {"config_id": f"{stage}_{len(results):03d}_{label}", "scan_stage": stage, **config, **metrics}
        results.append(row)
        details_by_id[row["config_id"]] = details
        print(f"{row['config_id']}: R@0.3={metrics['miss_recovery_iou03']:.3f}, 噪声/图={metrics['noise_candidates_per_image']:.2f}", flush=True)
        return row

    default_row = evaluate("baseline", "current_default", current_default)
    train_metrics, train_details = evaluate_config(
        train_recommended, sample, predictions, train_global_thresholds
    )
    train_row = {
        "config_id": "transfer_001_train_recommended",
        "scan_stage": "transfer",
        **train_recommended,
        **train_metrics,
    }
    results.append(train_row)
    details_by_id[train_row["config_id"]] = train_details
    print(
        f"{train_row['config_id']}: R@0.3={train_metrics['miss_recovery_iou03']:.3f}, "
        f"噪声/图={train_metrics['noise_candidates_per_image']:.2f}",
        flush=True,
    )
    threshold_rows = []
    for alpha in (0.05, 0.02, 0.01, 0.005):
        threshold_rows.append(evaluate("threshold", f"global_a{alpha}", {**base, "threshold_mode": "global_quantile", "alpha": alpha}))
    for k in (2.0, 3.0, 4.0, 6.0):
        threshold_rows.append(evaluate("threshold", f"image_k{k:g}", {**base, "threshold_mode": "image_mad", "mad_k": k}))
    for alpha in (0.05, 0.02, 0.01, 0.005):
        for k in (2.0, 3.0, 4.0, 6.0):
            threshold_rows.append(evaluate("threshold", f"min_a{alpha}_k{k:g}", {**base, "threshold_mode": "min_global_image", "alpha": alpha, "mad_k": k}))
    working = {key: rank(threshold_rows)[0][key] for key in base}

    stages: list[tuple[str, list[dict]]] = []
    morphology = [{"open_kernel": opening, "close_kernel": closing} for opening in (1, 3, 5) for closing in (3, 5, 7)]
    stages.append(("morphology", morphology))
    stages.append(("min_area", [{"min_area": value} for value in (150, 250, 400, 600)]))
    stages.append(("tolerance", [{"tolerance_px": value} for value in (1, 2, 3)]))
    stages.append(("merge", [{"merge_mode": value} for value in ("max", "independent_or")]))
    stage_rows: dict[str, list[dict]] = {"threshold": threshold_rows}
    for stage, changes in stages:
        scanned = [evaluate(stage, "_".join(f"{k}{v}" for k, v in change.items()), {**working, **change}) for change in changes]
        stage_rows[stage] = scanned
        best = rank(scanned)[0]
        working = {key: best[key] for key in base}

    effects = sorted(
        stage_rows,
        key=lambda stage: max(float(row["miss_recovery_iou03"]) for row in stage_rows[stage])
        - min(float(row["miss_recovery_iou03"]) for row in stage_rows[stage]),
        reverse=True,
    )
    first_stage, second_stage = effects[:2]
    top_first, top_second = rank(stage_rows[first_stage])[:2], rank(stage_rows[second_stage])[:2]
    stage_keys = {
        "threshold": ("threshold_mode", "alpha", "mad_k"),
        "morphology": ("open_kernel", "close_kernel"),
        "min_area": ("min_area",),
        "max_area": ("max_area",),
        "tolerance": ("tolerance_px",),
        "merge": ("merge_mode",),
    }
    joint_rows = []
    for first in top_first:
        for second in top_second:
            config = dict(working)
            for key in stage_keys[first_stage]:
                config[key] = first[key]
            for key in stage_keys[second_stage]:
                config[key] = second[key]
            joint_rows.append(evaluate("joint_top2", f"{first_stage}_{second_stage}", config))
    recommended = rank([
        row for row in results if row["scan_stage"] not in ("baseline", "transfer")
    ])[0]
    return results, recommended, details_by_id[recommended["config_id"]], default_row, train_row


def write_csv(results: list[dict]) -> None:
    path = OUTPUT / "sweep_results_val.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def format_table(rows: list[dict]) -> list[str]:
    lines = [
        "| 配置 | 阈值方式 | α | k | 开/闭 | 面积范围 | δ | 合并 | R@0.3 | R@0.5 | 候选/图 | 噪声/图 | 候选效率 |",
        "| --- | --- | ---: | ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['config_id']} | {row['threshold_mode']} | {row['alpha']} | {row['mad_k']} | "
            f"{row['open_kernel']}/{row['close_kernel']} | {row['min_area']}至{row['max_area']} | "
            f"{row['tolerance_px']} | {row['merge_mode']} | {row['miss_recovery_iou03']:.3f} | "
            f"{row['miss_recovery_iou05']:.3f} | {row['candidates_per_image']:.2f} | "
            f"{row['noise_candidates_per_image']:.2f} | {row['candidates_per_recovered_miss']:.2f} |"
        )
    return lines


def write_report(
    results: list[dict],
    recommended: dict,
    default: dict,
    train_recommended: dict,
    sample: list[dict[str, str | int]],
    full_val_missed: int,
) -> None:
    scan_results = [
        row for row in results if row["scan_stage"] not in ("baseline", "transfer")
    ]
    front = sorted(pareto(scan_results), key=lambda row: float(row["noise_candidates_per_image"]))
    stages = ["threshold", "morphology", "min_area", "tolerance", "merge", "joint_top2"]
    effects = {}
    for stage in stages[:-1]:
        values = [float(row["miss_recovery_iou03"]) for row in results if row["scan_stage"] == stage]
        effects[stage] = max(values) - min(values)
    fold_counts = {fold: sum(int(row["fold"]) == fold for row in sample) for fold in FOLDS}
    lines = [
        "# 阶段 A：差异候选参数扫描",
        "",
        "## 数据与口径",
        "",
        f"固定种子为 {SEED}。从五折 val 组中按模板组轮转抽取 {len(sample)} 张不重复图像，各折分配为 "
        + "、".join(f"fold{fold} {count} 张" for fold, count in fold_counts.items())
        + "。检测器使用对应折权重和由 val 确定的工作点。每张样本只按其所属 val 角色进入相应折，未使用对应折的 test 成员或 test 预测缓存。",
        "",
        "漏检清单严格复用五类错误分解口径。差异候选匹配漏检分别使用 IoU 0.3 和 0.5。噪声候选定义为对所有真值框 IoU 均低于 0.3，且与所有检测框交集为零。候选效率为差异候选总数除以 IoU 0.3 下恢复的漏检数。全局分位数仅统计对应折全部 val 图像的真值框外像素。",
        "",
        "扫描按形态学与面积过滤后的全部连通域计数，不施加旧配置每图 5 个的提前截断。面积上界固定为上一轮推荐值 8000。为保证部署口径一致，推荐配置把接口上限提高到 1000。配准可观测性低于 4 对 ORB 匹配的图像采用恒等对齐并记 q_reg=0，样本不被静默删除，仍进入对应 val 统计。",
        "",
        "## 逐项扫描",
    ]
    names = {"threshold": "阈值估计", "morphology": "形态学", "min_area": "面积下界", "tolerance": "容差 δ", "merge": "合并方式", "joint_top2": "影响最大的两项前两名联合验证"}
    for stage in stages:
        lines.extend(["", f"### {names[stage]}", "", *format_table([row for row in results if row["scan_stage"] == stage])])
    lines.extend(["", "## 召回率与噪声数的帕累托前沿", "", *format_table(front)])
    lines.extend(
        [
            "",
            f"推荐配置为 `{recommended['config_id']}`。选择规则是在每图噪声不超过 20 的配置中，先最大化漏检恢复率，再最小化噪声和候选效率。",
            "",
            "## 推荐配置与当前默认对比",
            "",
            "| 配置 | R@0.3 | R@0.5 | 候选/图 | 噪声/图 | 候选效率 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            f"| 当前默认实测 | {default['miss_recovery_iou03']:.3f} | {default['miss_recovery_iou05']:.3f} | {default['candidates_per_image']:.2f} | {default['noise_candidates_per_image']:.2f} | {default['candidates_per_recovered_miss']:.2f} |",
            f"| 推荐配置 | {recommended['miss_recovery_iou03']:.3f} | {recommended['miss_recovery_iou05']:.3f} | {recommended['candidates_per_image']:.2f} | {recommended['noise_candidates_per_image']:.2f} | {recommended['candidates_per_recovered_miss']:.2f} |",
            "",
            "## train 折推荐配置迁移表现",
            "",
            "| 配置 | R@0.3 | R@0.5 | 候选/图 | 噪声/图 | 候选效率 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            f"| 上一轮 train 折推荐在 train 折 | 0.595 | 0.270 | 24.91 | 18.89 | 226.50 |",
            f"| 上一轮 train 折推荐在 val 折 | {train_recommended['miss_recovery_iou03']:.3f} | {train_recommended['miss_recovery_iou05']:.3f} | {train_recommended['candidates_per_image']:.2f} | {train_recommended['noise_candidates_per_image']:.2f} | {train_recommended['candidates_per_recovered_miss']:.2f} |",
            "",
            migration_statement(train_recommended),
            "",
            "## 漏检分母",
            "",
            f"本样本共有 {recommended['missed_count']} 个检测器漏检，即每图 {recommended['missed_count'] / len(sample):.3f} 个。五折 test 统计为每图 0.631 个，本次高出 {recommended['missed_count'] / len(sample) - 947 / 1500:.3f} 个每图，相对高出 {(recommended['missed_count'] / len(sample) / (947 / 1500) - 1.0) * 100:.1f}%，因此不一致。对应检测器在五折完整 val 上共有 {full_val_missed} 个漏检，即每图 {full_val_missed / 1500:.3f} 个，说明差异主要来自 val 的折模型组合，而不是 300 张抽样偶然性。按模板组均衡后，本样本对不同规模模板组赋予近似相同权重，所以样本率与完整 val 率仍有差别。推荐配置在 IoU 0.3 下恢复 {recommended['recovered_count_iou03']} 个，在 IoU 0.5 下恢复 {recommended['recovered_count_iou05']} 个。",
            "",
            "## 与 train 折推荐的参数差异",
            "",
            parameter_difference(recommended),
            "",
            "## 参数影响",
            "",
        ]
    )
    for stage, effect in sorted(effects.items(), key=lambda item: item[1], reverse=True):
        conclusion = "几乎没有影响，后续无需优先调节" if effect < 0.01 else "有可观察影响"
        lines.append(f"* {names[stage]}：R@0.3 极差为 {effect:.3f}，{conclusion}。")
    hit_target = float(recommended["miss_recovery_iou03"]) >= 0.6 and float(recommended["noise_candidates_per_image"]) <= 20
    lines.extend(["", "## 结论与止损", ""])
    if hit_target:
        lines.append("推荐配置达到 R@0.3 不低于 0.6 且每图噪声候选不超过 20 的目标。")
    elif float(recommended["miss_recovery_iou03"]) < 0.4:
        lines.append("全部扫描配置的 R@0.3 均未达到 0.4，已按止损规则停止，不继续扩大参数搜索。候选生成能力边界及原因诊断见下文。")
    else:
        lines.append("推荐配置未同时达到目标，但已超过 0.4 止损线。本阶段不扩大搜索空间，如实保留当前最优结果。")
    lines.extend(
        [
            "",
            "可视化位于 `experiments/stageA_sweep/vis_val/`。颜色为真值红色、检测框蓝色、差异候选黄色，检测器漏掉的真值框另用洋红色加粗。",
            "",
        ]
    )
    (OUTPUT / "report_val.md").write_text("\n".join(lines), encoding="utf-8")


def parameter_difference(recommended: dict) -> str:
    train_values = {
        "threshold_mode": "global_quantile", "alpha": 0.02, "mad_k": 3.0,
        "dilate_kernel": 3, "open_kernel": 1, "close_kernel": 5,
        "min_area": 150, "max_area": 8000, "tolerance_px": 3, "merge_mode": "max",
    }
    labels = {
        "threshold_mode": "阈值方式", "alpha": "α", "mad_k": "k",
        "dilate_kernel": "膨胀核", "open_kernel": "开运算核", "close_kernel": "闭运算核",
        "min_area": "面积下界", "max_area": "面积上界", "tolerance_px": "容差 δ",
        "merge_mode": "合并方式",
    }
    differences = [
        f"{labels[key]}从 {value} 改为 {recommended[key]}"
        for key, value in train_values.items() if recommended[key] != value
    ]
    return "val 折最优配置与 train 折推荐配置相同。" if not differences else "差异为：" + "；".join(differences) + "。"


def migration_statement(train_recommended: dict) -> str:
    difference_03 = train_recommended["miss_recovery_iou03"] - 0.595
    difference_05 = train_recommended["miss_recovery_iou05"] - 0.270
    direction = "没有下降，反而提升" if difference_03 >= 0 else "下降"
    return (
        f"train 折调出的参数和 train 拟合阈值原样迁移到未见过的 val 数据后，"
        f"R@0.3 {direction} {abs(difference_03):.3f}，相对变化 "
        f"{difference_03 / 0.595 * 100:+.1f}%。R@0.5 绝对变化 {difference_05:+.3f}。"
    )


def write_visualizations(details: list[dict]) -> None:
    vis = OUTPUT / "vis_val"
    vis.mkdir(parents=True, exist_ok=True)
    ordered = sorted(details, key=lambda item: (-item["recovered_03"], -len(item["misses"]), item["noise"], str(item["row"]["pair_id"])))
    chosen = ordered[:3] + sorted(details, key=lambda item: (-item["noise"], -len(item["misses"]), str(item["row"]["pair_id"])))[:3]
    unique = []
    for item in chosen + ordered:
        if item["row"]["pair_id"] not in {entry["row"]["pair_id"] for entry in unique}:
            unique.append(item)
        if len(unique) == 6:
            break
    for index, item in enumerate(unique, 1):
        image = cv2.imread(str(ROOT / str(item["row"]["test_path"])))
        for box in item["gt"]:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
        for box in item["detections"]:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        for candidate in item["candidates"]:
            x1, y1, x2, y2 = map(int, candidate["box"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 1)
        for box in item["misses"]:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.putText(image, "GT red | DET blue | DIFF yellow | MISS magenta", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        cv2.putText(image, "GT red | DET blue | DIFF yellow | MISS magenta", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.imwrite(str(vis / f"{index:02d}_{item['row']['pair_id']}_fold{item['row']['fold']}.jpg"), image)


def write_recommended_config(
    recommended: dict, global_thresholds: dict[str, dict[str, dict[str, float]]]
) -> None:
    path = ROOT / "configs/candidates.yaml"
    old = read_yaml(path)
    alpha = str(recommended["alpha"])
    tolerance = int(recommended["tolerance_px"])
    fitted = {
        str(fold): {
            "appearance": global_thresholds[str(fold)][alpha]["appearance"],
            "topology": global_thresholds[str(fold)][alpha][f"topology_{tolerance}"],
            "combined": global_thresholds[str(fold)][alpha][f"combined_{tolerance}"],
        }
        for fold in FOLDS
    }
    lines = [
        f"# 阶段 A 推荐配置，基于固定种子 20260811 的 {recommended['sample_images']} 张 val 样本。",
        "# 上一轮 train 折推荐：threshold_mode=global_quantile, alpha=0.02, mad_k=3.0, dilate_kernel=3, open_kernel=1, close_kernel=5, min_area=150, max_area=8000, tolerance_px=3, merge_mode=max。",
        f"threshold_mode: {recommended['threshold_mode']}",
        f"alpha: {recommended['alpha']}",
        f"mad_k: {recommended['mad_k']}",
        f"dilate_kernel: {recommended['dilate_kernel']}",
        f"open_kernel: {recommended['open_kernel']}",
        f"close_kernel: {recommended['close_kernel']}",
        f"min_area: {recommended['min_area']}",
        f"max_area: {recommended['max_area']}",
        f"tolerance_px: {recommended['tolerance_px']}",
        f"merge_mode: {recommended['merge_mode']}",
        "# 扫描按未截断候选计成本，1000 仅防止生产接口提前截断扫描推荐结果。",
        "max_candidates_per_image: 1000",
        f"center_expand_px: {old['center_expand_px']}",
        "global_thresholds_by_fold:",
    ]
    for fold, thresholds in fitted.items():
        lines.extend(
            [
                f"  '{fold}':",
                f"    appearance: {thresholds['appearance']:.10f}",
                f"    topology: {thresholds['topology']:.10f}",
                f"    combined: {thresholds['combined']:.10f}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, min(12, mp.cpu_count() - 1)))
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    cv2.setNumThreads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rows, splits = read_inputs()
    sample = select_sample(rows, splits, INITIAL_SAMPLE_SIZE)
    predictions = predict_sample(sample, rows, splits)
    if count_misses(sample, predictions) < MIN_MISSED_COUNT:
        sample = select_sample(rows, splits, EXTENDED_SAMPLE_SIZE)
        predictions = predict_sample(sample, rows, splits)
    full_val_missed = count_full_val_misses(rows, splits)
    (OUTPUT / "sample_manifest_val.json").write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    global_thresholds = build_global_thresholds(rows, splits, args.workers)
    train_global_thresholds = json.loads(
        (OUTPUT / "global_thresholds.json").read_text(encoding="utf-8")
    )
    prepare_sample(sample, args.workers)
    results, recommended, details, default, train_recommended = run_sweep(
        sample, predictions, global_thresholds, train_global_thresholds
    )
    write_csv(results)
    write_visualizations(details)
    write_report(results, recommended, default, train_recommended, sample, full_val_missed)
    write_recommended_config(recommended, global_thresholds)
    print(f"完成，耗时 {(time.monotonic() - started) / 60:.1f} 分钟。推荐 R@0.3={recommended['miss_recovery_iou03']:.3f}，噪声/图={recommended['noise_candidates_per_image']:.2f}")


if __name__ == "__main__":
    main()
