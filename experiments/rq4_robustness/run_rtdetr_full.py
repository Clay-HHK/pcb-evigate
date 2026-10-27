"""Run the fold0 RT-DETR-R18VD replacement through the 200-candidate A2 pipeline."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.rq1_evidence.run import (
    align_for_pilot,
    box_key,
    detector_items,
    enrich_candidate,
    load_configs,
)
from experiments.rq2_correction.run_full import (
    acceptance_probability,
    fit_fold_models,
    ground_truth_map,
    load_gate_records,
)
from src.candidates.fuse import fuse_candidates
from src.candidates.propose import propose_candidates
from src.detector.calibrate import box_iou, read_gt
from src.detector.dataset import rows_for_role
from src.detector.error_decompose import decompose_record
from src.evaluation.apply import apply_actions
from src.evaluation.metrics import correction_metrics, detection_metrics
from src.evaluation.utility import candidate_action_utility
from src.evidence.appearance import compute_appearance
from src.evidence.card import create_evidence_card
from src.evidence.topology import compute_topology
from src.gates.features import review_features
from src.gates.safety import accept_action
from src.mllm.action_map import map_action
from src.mllm.client import review_candidates
from src.registration.align import load_gray


SEED = 20260811
OUTPUT = Path("experiments/rq4_robustness")
CANDIDATES = OUTPUT / "rtdetr_candidates.jsonl"
SAMPLE = OUTPUT / "rtdetr_sample_200.jsonl"
CARDS = OUTPUT / "rtdetr_cards" / "A2"
MODEL_RESULT = OUTPUT / "model_replacement.json"
PREDICTIONS = Path("cache/rtdetr_r18_fold0/fold0.jsonl")
CLASS_NAMES = ("open", "short", "mousebite", "spur", "pinhole", "spurious_copper")


def append_progress(stage: str, completed: int, total: int, detail: str) -> None:
    with (OUTPUT / "progress.csv").open("a", encoding="utf-8") as handle:
        handle.write(
            f"{datetime.now().astimezone().isoformat(timespec='seconds')},{stage},{completed},{total},{detail}\n"
        )


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def prediction_map() -> dict[str, dict]:
    return {item["pair_id"]: item for item in load_jsonl(PREDICTIONS)}


def build_candidates() -> list[dict]:
    registration_config, appearance_config, topology_config, candidate_config, _ = load_configs()
    predictions = prediction_map()
    records = []
    rows = rows_for_role(0, "test")
    for completed, row in enumerate(rows, start=1):
        prediction = predictions[row["pair_id"]]
        current = load_gray(Path(row["test_path"]))
        template = load_gray(Path(row["temp_path"]))
        aligned = align_for_pilot(current, template, registration_config)
        appearance = compute_appearance(aligned["normalized_test"], aligned["normalized_template"], appearance_config)
        normalized_current = np.clip(aligned["normalized_test"] * 32.0 + 128.0, 0, 255).astype(np.uint8)
        normalized_template = np.clip(aligned["normalized_template"] * 32.0 + 128.0, 0, 255).astype(np.uint8)
        topology = compute_topology(normalized_current, normalized_template, topology_config)
        detections = detector_items(prediction)
        differences = propose_candidates(
            appearance, topology, current, aligned["aligned_template"], detections, candidate_config, 0
        )
        candidates = fuse_candidates(
            row["pair_id"], detections, differences["candidates"], int(candidate_config["center_expand_px"])
        )
        errors = decompose_record(prediction, row)
        error_by_box = {box_key(json.loads(item["pred_box"])): item for item in errors if item["pred_box"] != "[]"}
        missed = [item for item in errors if item["error_type"] == "漏检"]
        for candidate in candidates:
            candidate["pair_id"] = row["pair_id"]
            enrich_candidate(candidate, aligned, appearance, topology, differences["threshold"])
            if candidate["source"] in {"DET", "BOTH"}:
                error = error_by_box[box_key(candidate["det_box"])]
                review_label = int(error["error_type"] in {"FP", "错分类"})
                error_type = error["error_type"]
                gt_class = int(error["gt_class"])
                gt_iou = float(error["iou"])
            else:
                matches = [(item, box_iou(candidate["action_box"], json.loads(item["gt_box"]))) for item in missed]
                matched, gt_iou = max(matches, key=lambda item: item[1]) if matches else ({"gt_class": -1}, 0.0)
                review_label = int(gt_iou >= 0.3)
                error_type = "漏检" if review_label else "差异噪声"
                gt_class = int(matched["gt_class"]) if review_label else -1
            records.append(
                {
                    "fold": 0,
                    "pair_id": row["pair_id"],
                    "candidate": candidate,
                    "review_label": review_label,
                    "error_type": error_type,
                    "gt_class": gt_class,
                    "gt_iou": gt_iou,
                }
            )
        append_progress("rtdetr_candidates", completed, len(rows), row["pair_id"])
    CANDIDATES.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n", encoding="utf-8")
    return records


def sample_200(records: list[dict]) -> list[dict]:
    strata: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        strata[f"{record['error_type']}|{record['candidate']['source']}"] .append(record)
    exact = {name: 200 * len(items) / len(records) for name, items in strata.items()}
    counts = {name: int(value) for name, value in exact.items()}
    for name in sorted(strata, key=lambda item: (-(exact[item] - counts[item]), item))[: 200 - sum(counts.values())]:
        counts[name] += 1
    rng = np.random.default_rng(SEED)
    sampled = []
    for name in sorted(strata):
        indices = rng.permutation(len(strata[name]))[: counts[name]]
        sampled.extend(strata[name][int(index)] for index in indices)
    SAMPLE.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in sampled) + "\n", encoding="utf-8")
    return sampled


def render_cards(sampled: list[dict]) -> list[Path]:
    registration_config, appearance_config, topology_config, _, card_config = load_configs()
    row_map = {row["pair_id"]: row for row in rows_for_role(0, "test")}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in sampled:
        grouped[record["pair_id"]].append(record)
    CARDS.mkdir(parents=True, exist_ok=True)
    for pair_id, records in grouped.items():
        row = row_map[pair_id]
        current = load_gray(Path(row["test_path"]))
        template = load_gray(Path(row["temp_path"]))
        aligned = align_for_pilot(current, template, registration_config)
        appearance = compute_appearance(aligned["normalized_test"], aligned["normalized_template"], appearance_config)
        normalized_current = np.clip(aligned["normalized_test"] * 32.0 + 128.0, 0, 255).astype(np.uint8)
        normalized_template = np.clip(aligned["normalized_template"] * 32.0 + 128.0, 0, 255).astype(np.uint8)
        topology = compute_topology(normalized_current, normalized_template, topology_config)
        for record in records:
            candidate = record["candidate"]
            create_evidence_card(
                current,
                aligned["aligned_template"],
                candidate,
                appearance["M_app"],
                topology,
                float(candidate["q_reg"]),
                CARDS / f"{candidate['candidate_id']}.png",
                card_config,
                {"P1", "P2", "P3"},
            )
    return [CARDS / f"{record['candidate']['candidate_id']}.png" for record in sampled]


def evaluate(sampled: list[dict], responses: list[dict]) -> dict:
    models = fit_fold_models(load_gate_records(), 0)
    predictions = prediction_map()
    ground_truth = ground_truth_map(0)
    grouped_candidates: dict[str, list[dict]] = defaultdict(list)
    grouped_decisions: dict[str, list[dict]] = defaultdict(list)
    for record, response in zip(sampled, responses):
        candidate = record["candidate"]
        conclusion = response["result"]
        action = map_action(candidate, conclusion)
        utility_record = {"candidate": candidate, "prediction": predictions[record["pair_id"]], "ground_truth": ground_truth[record["pair_id"]]}
        positive_utility, before, after = candidate_action_utility(utility_record, action, conclusion)
        review_score = float(models["logistic"].predict_proba(review_features(candidate).reshape(1, -1))[0, 1])
        called = review_score >= models["logistic_threshold"]
        probability = acceptance_probability(models, action, candidate, conclusion)
        accepted = called and action in {"remove", "relabel", "add"} and probability >= models["acceptance_thresholds"].get(action, 1.0)
        if accepted:
            accepted = accept_action(action, candidate, conclusion, probability, response.get("json_parseable", True))
        grouped_candidates[record["pair_id"]].append(candidate)
        grouped_decisions[record["pair_id"]].append(
            {
                "action": action, "conclusion": conclusion, "accept_prob": probability,
                "positive_utility": positive_utility, "gt_match_before": before, "gt_match_after": after,
                "accepted": accepted, "mllm_called": called, "latency_seconds": response["latency_seconds"],
                "error_type": record["error_type"], "source": candidate["source"], "review_label": record["review_label"],
            }
        )
    corrected = []
    audits = []
    for pair_id, prediction in predictions.items():
        final, pair_audits = apply_actions(prediction, grouped_candidates[pair_id], grouped_decisions[pair_id])
        corrected.append(final)
        audits.extend(pair_audits)
    counts = Counter(record["error_type"] for record in sampled)
    baseline = detection_metrics(list(predictions.values()), ground_truth)
    full = detection_metrics(corrected, ground_truth)
    full.update(correction_metrics(audits, dict(counts)))
    full["calls"] = sum(item["mllm_called"] for item in audits)
    full["accepted_actions"] = sum(item["accepted"] for item in audits)
    return {"baseline": baseline, "full": full, "sample_candidates": len(sampled), "all_candidates": len(load_jsonl(CANDIDATES))}


def update_model_result(result: dict) -> None:
    payload = json.loads(MODEL_RESULT.read_text(encoding="utf-8"))
    rows = payload["rows"]
    for method, metrics in (("RT-DETR-R18 B1", result["baseline"]), ("RT-DETR-R18 Full", result["full"])):
        for metric in ("precision", "recall", "f1", "map50", "map50_95", "hcr", "calls", "accepted_actions"):
            if metric in metrics:
                rows.append({"model": method, "metric": metric, "value": metrics[metric]})
    payload["rtdetr_r18"] = result
    payload["rows"] = rows
    payload["report"] += (
        f" 检测器替换使用官方 PekingU RT-DETR-R18VD，在 fold0 训练后对全部 test 图像预测，并仅对 200 个比例分层候选调用 A2 Codex。"
        f"RT-DETR-R18 基线 F1={result['baseline']['f1']:.4f}，mAP@0.5={result['baseline']['map50']:.4f}；"
        f"完整流程 F1={result['full']['f1']:.4f}，HCR={result['full']['hcr']:.4f}，门控实际调用={result['full']['calls']}。"
    )
    MODEL_RESULT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    records = build_candidates()
    sampled = sample_200(records)
    cards = render_cards(sampled)
    responses = review_candidates(
        [record["candidate"] for record in sampled],
        cards,
        "A2",
        Path("configs/mllm/codex_cli.yaml"),
        progress_path=OUTPUT / "progress.csv",
    )
    result = evaluate(sampled, responses)
    update_model_result(result)
    (OUTPUT / "rtdetr_full_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
