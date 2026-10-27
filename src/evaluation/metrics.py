"""Conventional detection, correction, and cost metrics."""

from __future__ import annotations

import numpy as np

from src.detector.calibrate import box_iou


def _class_ap(predictions: list[dict], ground_truth: dict[str, list[dict]], class_id: int, iou_threshold: float) -> float:
    ranked = []
    total_gt = 0
    for pair_id, objects in ground_truth.items():
        total_gt += sum(int(item["class"] == class_id) for item in objects)
    for record in predictions:
        for box, predicted_class, score in zip(record["boxes"], record["classes"], record["scores"]):
            if int(predicted_class) == class_id:
                ranked.append((float(score), record["pair_id"], box))
    ranked.sort(key=lambda item: -item[0])
    matched: dict[str, set[int]] = {}
    true_positive, false_positive = [], []
    for _, pair_id, box in ranked:
        used = matched.setdefault(pair_id, set())
        candidates = [
            (index, box_iou(box, item["box"]))
            for index, item in enumerate(ground_truth[pair_id])
            if item["class"] == class_id and index not in used
        ]
        index, iou = max(candidates, key=lambda item: item[1]) if candidates else (-1, 0.0)
        is_true = iou >= iou_threshold
        if is_true:
            used.add(index)
        true_positive.append(float(is_true))
        false_positive.append(float(not is_true))
    if total_gt == 0:
        return float("nan")
    recall = np.cumsum(true_positive) / total_gt
    precision = np.cumsum(true_positive) / np.maximum(np.cumsum(true_positive) + np.cumsum(false_positive), 1.0)
    recall_grid = np.linspace(0.0, 1.0, 101)
    interpolated = [max(precision[recall >= level], default=0.0) for level in recall_grid]
    return float(np.mean(interpolated))


def detection_metrics(predictions: list[dict], ground_truth: dict[str, list[dict]], class_count: int = 6) -> dict:
    true_positive = 0
    prediction_count = sum(len(record["boxes"]) for record in predictions)
    ground_truth_count = sum(len(items) for items in ground_truth.values())
    for record in predictions:
        used: set[int] = set()
        for box, class_id, score in sorted(
            zip(record["boxes"], record["classes"], record["scores"]), key=lambda item: -float(item[2])
        ):
            candidates = [
                (index, box_iou(box, gt["box"]))
                for index, gt in enumerate(ground_truth[record["pair_id"]])
                if index not in used and int(class_id) == int(gt["class"])
            ]
            index, iou = max(candidates, key=lambda item: item[1]) if candidates else (-1, 0.0)
            if iou >= 0.5:
                used.add(index)
                true_positive += 1
    precision = true_positive / prediction_count if prediction_count else 0.0
    recall = true_positive / ground_truth_count if ground_truth_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    thresholds = np.arange(0.5, 1.0, 0.05)
    per_class = {class_id: _class_ap(predictions, ground_truth, class_id, 0.5) for class_id in range(class_count)}
    per_class_50_95 = {
        class_id: float(np.nanmean([
            _class_ap(predictions, ground_truth, class_id, float(threshold))
            for threshold in thresholds
        ]))
        for class_id in range(class_count)
    }
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": float(np.nanmean(list(per_class.values()))),
        "map50_95": float(np.nanmean(list(per_class_50_95.values()))),
        "per_class_ap50": per_class,
        "per_class_ap50_95": per_class_50_95,
    }


def correction_metrics(audits: list[dict], original_counts: dict[str, int]) -> dict:
    accepted = [item for item in audits if item.get("accepted", False)]
    positive = [item for item in accepted if item.get("positive_utility", False)]
    correct_remove = sum(item["action"] == "remove" and item.get("positive_utility", False) for item in accepted)
    correct_add = sum(item["action"] == "add" and item.get("positive_utility", False) for item in accepted)
    correct_relabel = sum(item["action"] == "relabel" and item.get("positive_utility", False) for item in accepted)
    harmful = sum(not item.get("positive_utility", False) for item in accepted)
    available_actions = sum(int(item.get("positive_utility", False)) for item in audits)
    return {
        "fpcr": correct_remove / original_counts.get("FP", 1),
        "mdrr": correct_add / original_counts.get("漏检", 1),
        "mcca": correct_relabel / original_counts.get("错分类", 1),
        "tprr": 1.0 - sum(item["action"] in {"remove", "relabel"} and not item.get("positive_utility", False) for item in accepted) / original_counts.get("TP", 1),
        "hcr": harmful / len(accepted) if accepted else 0.0,
        "action_precision": len(positive) / len(accepted) if accepted else 0.0,
        "action_recall": len(positive) / available_actions if available_actions else 0.0,
    }


def cost_metrics(audits: list[dict], image_count: int) -> dict:
    called = [item for item in audits if item.get("mllm_called", False)]
    latencies = [float(item.get("latency_seconds", 0.0)) for item in called]
    return {
        "call_rate": len(called) / len(audits) if audits else 0.0,
        "calls_per_image": len(called) / image_count if image_count else 0.0,
        "mean_latency_seconds": float(np.mean(latencies)) if latencies else 0.0,
        "p95_latency_seconds": float(np.quantile(latencies, 0.95)) if latencies else 0.0,
    }
