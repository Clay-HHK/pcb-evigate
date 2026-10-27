"""Apply accepted actions and class-aware NMS with candidate audit records."""

from __future__ import annotations

import json
from pathlib import Path

from src.mllm.action_map import CLASS_NAMES
from src.detector.calibrate import box_iou


def class_aware_nms(boxes: list[list[float]], classes: list[int], scores: list[float], threshold: float = 0.5) -> list[int]:
    order = sorted(range(len(boxes)), key=lambda index: -float(scores[index]))
    keep = []
    for index in order:
        if all(classes[index] != classes[chosen] or box_iou(boxes[index], boxes[chosen]) < threshold for chosen in keep):
            keep.append(index)
    return keep


def apply_actions(prediction: dict, candidates: list[dict], decisions: list[dict], nms_iou: float = 0.5) -> tuple[dict, list[dict]]:
    boxes = [list(box) for box in prediction["boxes"]]
    classes = [int(value) for value in prediction["classes"]]
    scores = [float(value) for value in prediction["scores"]]
    audits = []
    for candidate, decision in zip(candidates, decisions):
        action = decision["action"]
        accepted = bool(decision.get("accepted", False))
        if accepted and action in {"remove", "relabel"}:
            target = candidate["action_box"]
            index = max(range(len(boxes)), key=lambda item: box_iou(boxes[item], target))
            if action == "remove":
                boxes.pop(index)
                classes.pop(index)
                scores.pop(index)
            else:
                classes[index] = CLASS_NAMES.index(decision["conclusion"]["predicted_class"])
        elif accepted and action == "add":
            boxes.append(list(candidate["diff_box"]))
            classes.append(CLASS_NAMES.index(decision["conclusion"]["predicted_class"]))
            scores.append(float(decision["accept_prob"]))
        audits.append({"pair_id": prediction["pair_id"], "candidate_id": candidate["candidate_id"], **decision})
    if not any(item["accepted"] for item in audits):
        return {
            "pair_id": prediction["pair_id"],
            "boxes": [list(box) for box in prediction["boxes"]],
            "classes": [int(value) for value in prediction["classes"]],
            "scores": [float(value) for value in prediction["scores"]],
        }, audits
    keep = class_aware_nms(boxes, classes, scores, nms_iou)
    final = {
        "pair_id": prediction["pair_id"],
        "boxes": [boxes[index] for index in keep],
        "classes": [classes[index] for index in keep],
        "scores": [scores[index] for index in keep],
    }
    return final, audits


def write_outputs(final_predictions: list[dict], audits: list[dict], output_directory: Path) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    prediction_path = output_directory / "final_predictions.jsonl"
    audit_path = output_directory / "candidate_audit.jsonl"
    prediction_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in final_predictions) + "\n", encoding="utf-8")
    audit_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in audits) + "\n", encoding="utf-8")
    return prediction_path, audit_path
