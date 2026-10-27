"""Shared deterministic small-sample pipeline for experiment smoke runs."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.candidates.fuse import fuse_candidates
from src.candidates.propose import propose_candidates
from src.detector.calibrate import box_iou, read_gt
from src.detector.dataset import rows_for_role
from src.evidence.appearance import compute_appearance
from src.evidence.card import create_evidence_card
from src.evidence.topology import compute_topology, topology_features
from src.registration.align import align_images, load_gray


PANEL_CONFIGS = {
    "A0": {"P1"},
    "A1": {"P1", "P3"},
    "A2": {"P1", "P2", "P3"},
    "A3": {"P1", "P2", "P3", "P4"},
    "A4": {"P1", "P2", "P3", "P5"},
    "A5": {"P1", "P2", "P3", "P4", "P5"},
    "A6": {"P1", "P2", "P3", "P4", "P5", "P6"},
}


def _configs() -> tuple[dict, dict, dict, dict, dict]:
    return tuple(
        yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        for path in (
            "configs/evidence/registration.yaml",
            "configs/evidence/appearance.yaml",
            "configs/evidence/topology.yaml",
            "configs/candidates.yaml",
            "configs/evidence/card.yaml",
        )
    )


def _prediction_map(fold: int) -> dict[str, dict]:
    path = Path(f"cache/detector_predictions/fold{fold}.jsonl")
    return {item["pair_id"]: item for item in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def _detector_items(record: dict) -> list[dict]:
    return [
        {"box": box, "class": int(class_id), "score": float(score)}
        for box, class_id, score in zip(record["boxes"], record["classes"], record["scores"])
    ]


def prepare_smoke_candidates(
    limit: int, evidence_names: list[str], fold: int = 0, complete_images: bool = False
) -> list[dict]:
    cache_path = Path(f"cache/smoke/candidates_v5_fold{fold}.jsonl")
    card_root = Path("cache/evidence_cards")
    if cache_path.exists():
        records = [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines()]
        selected = records[:limit]
        if complete_images and len(records) > limit:
            boundary_pair_id = selected[-1]["candidate"]["pair_id"]
            selected.extend(
                item for item in records[limit:] if item["candidate"]["pair_id"] == boundary_pair_id
            )
        required = [card_root / name / f"{item['candidate']['candidate_id']}.png" for item in selected for name in evidence_names]
        if len(records) >= limit and all(path.exists() for path in required):
            return selected

    registration_config, appearance_config, topology_config, candidate_config, card_config = _configs()
    predictions = _prediction_map(fold)
    output = []
    for row in rows_for_role(fold, "test"):
        prediction = predictions[row["pair_id"]]
        current = load_gray(Path(row["test_path"]))
        template = load_gray(Path(row["temp_path"]))
        aligned = align_images(current, template, registration_config)
        appearance = compute_appearance(aligned["normalized_test"], aligned["normalized_template"], appearance_config)
        normalized_current = ((aligned["normalized_test"] * 32.0) + 128.0).clip(0, 255).astype("uint8")
        normalized_template = ((aligned["normalized_template"] * 32.0) + 128.0).clip(0, 255).astype("uint8")
        topology = compute_topology(normalized_current, normalized_template, topology_config)
        detections = _detector_items(prediction)
        differences = propose_candidates(appearance, topology, current, aligned["aligned_template"], detections, candidate_config)
        candidates = fuse_candidates(row["pair_id"], detections, differences["candidates"], int(candidate_config["center_expand_px"]))
        for candidate in candidates:
            candidate["pair_id"] = row["pair_id"]
            candidate["q_reg"] = float(aligned["q_reg"])
            x1, y1, x2, y2 = [int(round(value)) for value in candidate["box"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(current.shape[1], x2), min(current.shape[0], y2)
            residual = appearance["M_app"][y1:y2, x1:x2]
            xor_region = topology["M_topo"][y1:y2, x1:x2] > 0
            test_vector = topology_features(
                topology["test_mask"][y1:y2, x1:x2], topology["test_skeleton"][y1:y2, x1:x2], xor_region
            )
            template_vector = topology_features(
                topology["template_mask"][y1:y2, x1:x2], topology["template_skeleton"][y1:y2, x1:x2], xor_region
            )
            candidate["appearance_residual"] = {
                "mean": float(residual.mean()),
                "p95": float(np.quantile(residual, 0.95)),
                "max": float(residual.max()),
            }
            candidate["above_threshold_fraction"] = float((residual >= differences["threshold"]).mean())
            candidate["topology_vector"] = np.abs(test_vector - template_vector).astype(float).tolist()
            for name in evidence_names:
                create_evidence_card(
                    current,
                    aligned["aligned_template"],
                    candidate,
                    appearance["M_app"],
                    topology,
                    aligned["q_reg"],
                    card_root / name / f"{candidate['candidate_id']}.png",
                    card_config,
                    PANEL_CONFIGS[name],
                )
            output.append(
                {
                    "fold": fold,
                    "row": row,
                    "prediction": prediction,
                    "candidate": candidate,
                    "ground_truth": [{"box": box, "class": class_id} for box, class_id in read_gt(Path(row["anno_path"]))],
                }
            )
        if len(output) >= limit:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in output) + "\n", encoding="utf-8")
            return output if complete_images else output[:limit]
    return output


def card_paths(records: list[dict], evidence_name: str) -> list[Path]:
    return [Path("cache/evidence_cards") / evidence_name / f"{item['candidate']['candidate_id']}.png" for item in records]


def review_label(record: dict) -> int:
    candidate = record["candidate"]
    matches = [(gt, box_iou(candidate["action_box"], gt["box"])) for gt in record["ground_truth"]]
    best_gt, best_iou = max(matches, key=lambda item: item[1]) if matches else ({"class": -1}, 0.0)
    if candidate["source"] == "DIFF":
        return int(best_iou >= 0.5)
    return int(best_iou < 0.5 or int(candidate["class"]) != int(best_gt["class"]))


def positive_utility(record: dict, action: str, conclusion: dict) -> bool:
    candidate = record["candidate"]
    matches = [(gt, box_iou(candidate["action_box"], gt["box"])) for gt in record["ground_truth"]]
    best_gt, best_iou = max(matches, key=lambda item: item[1]) if matches else ({"class": -1}, 0.0)
    if action == "remove":
        return best_iou < 0.5
    predicted_names = ["open", "short", "mousebite", "spur", "pinhole", "spurious_copper"]
    if action == "relabel":
        return best_iou >= 0.5 and predicted_names.index(conclusion["predicted_class"]) == int(best_gt["class"])
    if action == "add":
        return best_iou >= 0.5
    return False


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
