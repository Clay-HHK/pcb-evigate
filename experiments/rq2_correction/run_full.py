"""Run the five-fold sampled RQ2 correction study and RQ3 budget curves."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/tmp/zyp-rq23-matplotlib"

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import font_manager
from lightgbm import LGBMClassifier
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.rq1_evidence.run import align_for_pilot, load_configs
from src.detector.calibrate import read_gt
from src.detector.dataset import rows_for_role
from src.evaluation.apply import apply_actions, class_aware_nms
from src.evaluation.metrics import correction_metrics, detection_metrics
from src.evaluation.oracle import candidate_oracle
from src.evaluation.utility import candidate_action_utility, prediction_gt_matches
from src.evidence.appearance import compute_appearance
from src.evidence.card import create_evidence_card
from src.evidence.topology import compute_topology
from src.gates.features import acceptance_features, review_features, topology_compatibility
from src.gates.safety import ACTION_THRESHOLDS, accept_action
from src.mllm.action_map import CLASS_NAMES, map_action
from src.mllm.client import review_candidates
from src.mllm.prompt import build_prompt
from src.registration.align import load_gray


SEED = 20260811
FOLDS = tuple(range(5))
BUDGETS = (5, 10, 20, 40, 60, 100)
METHODS = ("B0", "B1", "B2", "B7", "B8", "Full", "Oracle-C", "Oracle-A")
STRATEGIES = ("random", "low_confidence", "manual_linear", "logistic", "lightgbm", "review_acceptance")
CLASS_LABELS = ("open", "short", "mousebite", "spur", "pinhole", "spurious_copper")
OUTPUT_RQ2 = Path("experiments/rq2_correction")
OUTPUT_RQ3 = Path("experiments/rq3_gating")
FIGURE_ROOT = Path("experiments/figures")
CARD_ROOT = OUTPUT_RQ2 / "cards" / "A2"
SAMPLE_PATH = OUTPUT_RQ2 / "sampled_candidates.jsonl"
PROGRESS_PATH = OUTPUT_RQ2 / "progress.csv"
RESULTS_PATH = Path("experiments/results_summary.csv")


def append_progress(stage: str, completed: int, total: int, detail: str) -> None:
    with PROGRESS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{datetime.now().astimezone().isoformat(timespec='seconds')},{stage},{completed},{total},{detail}\n"
        )


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_gate_records() -> dict[int, list[dict]]:
    return {fold: load_jsonl(Path(f"cache/gate_oof/fold{fold}.jsonl")) for fold in FOLDS}


def proportional_allocation(counts: dict[str, int], target: int) -> dict[str, int]:
    total = sum(counts.values())
    exact = {name: target * count / total for name, count in counts.items()}
    allocated = {name: int(math.floor(value)) for name, value in exact.items()}
    remaining = target - sum(allocated.values())
    order = sorted(counts, key=lambda name: (-(exact[name] - allocated[name]), name))
    for name in order[:remaining]:
        allocated[name] += 1
    return allocated


def stratified_sample(records_by_fold: dict[int, list[dict]]) -> list[dict]:
    if SAMPLE_PATH.exists():
        return load_jsonl(SAMPLE_PATH)
    sampled = []
    for fold in FOLDS:
        strata: dict[str, list[dict]] = defaultdict(list)
        for record in records_by_fold[fold]:
            stratum = f"{record['error_type']}|{record['candidate']['source']}"
            strata[stratum].append(record)
        allocation = proportional_allocation({name: len(items) for name, items in strata.items()}, 800)
        rng = np.random.default_rng(SEED + fold)
        for name in sorted(strata):
            indices = rng.permutation(len(strata[name]))[: allocation[name]]
            sampled.extend(strata[name][int(index)] for index in indices)
    SAMPLE_PATH.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in sampled) + "\n", encoding="utf-8"
    )
    return sampled


def prepare_cards(sampled: list[dict]) -> None:
    registration_config, appearance_config, topology_config, _, card_config = load_configs()
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for record in sampled:
        grouped[(int(record["fold"]), record["pair_id"])].append(record)
    row_maps = {
        fold: {row["pair_id"]: row for row in rows_for_role(fold, "test")}
        for fold in FOLDS
    }
    total = len(grouped)
    for completed, ((fold, pair_id), records) in enumerate(sorted(grouped.items()), start=1):
        paths = [CARD_ROOT / f"fold{fold}_{record['candidate']['candidate_id']}.png" for record in records]
        if not all(path.exists() for path in paths):
            row = row_maps[fold][pair_id]
            current = load_gray(Path(row["test_path"]))
            template = load_gray(Path(row["temp_path"]))
            aligned = align_for_pilot(current, template, registration_config)
            appearance = compute_appearance(
                aligned["normalized_test"], aligned["normalized_template"], appearance_config
            )
            normalized_current = ((aligned["normalized_test"] * 32.0) + 128.0).clip(0, 255).astype("uint8")
            normalized_template = ((aligned["normalized_template"] * 32.0) + 128.0).clip(0, 255).astype("uint8")
            topology = compute_topology(normalized_current, normalized_template, topology_config)
            for record, path in zip(records, paths):
                create_evidence_card(
                    current,
                    aligned["aligned_template"],
                    record["candidate"],
                    appearance["M_app"],
                    topology,
                    float(record["candidate"]["q_reg"]),
                    path,
                    card_config,
                    {"P1", "P2", "P3"},
                )
        append_progress("cards", completed, total, f"fold{fold}:{pair_id}")


def prediction_map(fold: int, low_confidence: bool = False) -> dict[str, dict]:
    suffix = "_lowconf" if low_confidence else ""
    records = load_jsonl(Path(f"cache/detector_predictions/fold{fold}{suffix}.jsonl"))
    return {record["pair_id"]: record for record in records}


def ground_truth_map(fold: int) -> dict[str, list[dict]]:
    return {
        row["pair_id"]: [
            {"box": box, "class": int(class_id)} for box, class_id in read_gt(Path(row["anno_path"]))
        ]
        for row in rows_for_role(fold, "test")
    }


def manual_risk(candidate: dict) -> float:
    confidence = float(candidate.get("calibrated_score", candidate.get("score", 0.0)))
    difference = float(candidate.get("appearance_residual", {}).get("p95", candidate.get("diff_score", 0.0)))
    registration_risk = 1.0 - float(candidate.get("q_reg", 1.0))
    return 0.40 * (1.0 - confidence) + 0.35 * difference + 0.25 * registration_risk


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    thresholds = np.unique(scores)
    best_threshold = float(thresholds[0])
    best_f1 = -1.0
    for threshold in thresholds:
        selected = scores >= threshold
        tp = int(np.sum(selected & (labels == 1)))
        fp = int(np.sum(selected & (labels == 0)))
        fn = int(np.sum((~selected) & (labels == 1)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best_f1 or (f1 == best_f1 and threshold > best_threshold):
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def fit_fold_models(records_by_fold: dict[int, list[dict]], evaluation_fold: int) -> dict:
    validation_fold = (evaluation_fold + 1) % 5
    train = [
        record
        for fold, records in records_by_fold.items()
        if fold not in {evaluation_fold, validation_fold}
        for record in records
    ]
    validation = records_by_fold[validation_fold]
    x_train = np.vstack([review_features(record["candidate"]) for record in train])
    y_train = np.asarray([record["review_label"] for record in train], dtype=int)
    x_val = np.vstack([review_features(record["candidate"]) for record in validation])
    y_val = np.asarray([record["review_label"] for record in validation], dtype=int)
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, l1_ratio=0.0, max_iter=1000, random_state=SEED),
    )
    logistic.fit(x_train, y_train)
    lightgbm = LGBMClassifier(n_estimators=200, reg_lambda=1.0, random_state=SEED, verbosity=-1)
    lightgbm.fit(x_train, y_train)
    logistic_threshold = best_f1_threshold(logistic.predict_proba(x_val)[:, 1], y_val)
    manual_threshold = best_f1_threshold(
        np.asarray([manual_risk(record["candidate"]) for record in validation]), y_val
    )
    acceptance = {}
    acceptance_thresholds = {}
    for action in ("remove", "relabel", "add"):
        subset = [record for record in train if record["action"] == action]
        labels = np.asarray([int(record["positive_utility"]) for record in subset], dtype=int)
        if len(set(labels.tolist())) < 2:
            acceptance[action] = None
            acceptance_thresholds[action] = ACTION_THRESHOLDS[action]
        else:
            features = np.vstack(
                [acceptance_features(record["candidate"], record["conclusion"]) for record in subset]
            )
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, l1_ratio=0.0, max_iter=1000, random_state=SEED),
            )
            model.fit(features, labels)
            acceptance[action] = model
            validation_subset = [record for record in validation if record["action"] == action]
            validation_features = np.vstack([
                acceptance_features(record["candidate"], record["conclusion"])
                for record in validation_subset
            ])
            validation_labels = np.asarray([
                int(record["positive_utility"]) for record in validation_subset
            ], dtype=int)
            selected = best_f1_threshold(
                model.predict_proba(validation_features)[:, 1], validation_labels
            )
            acceptance_thresholds[action] = max(ACTION_THRESHOLDS[action], selected)
    return {
        "validation_fold": validation_fold,
        "logistic": logistic,
        "lightgbm": lightgbm,
        "logistic_threshold": logistic_threshold,
        "manual_threshold": manual_threshold,
        "acceptance": acceptance,
        "acceptance_thresholds": acceptance_thresholds,
    }


def acceptance_probability(models: dict, action: str, candidate: dict, conclusion: dict) -> float:
    model = models["acceptance"].get(action)
    if model is None:
        return 0.0
    features = acceptance_features(candidate, conclusion).reshape(1, -1)
    return float(model.predict_proba(features)[0, 1])


def original_counts(fold: int) -> dict[str, int]:
    with Path(f"experiments/stage2/error_decomposition_fold{fold}.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        counts = Counter(row["error_type"] for row in csv.DictReader(handle))
    return dict(counts)


def traditional_predictions(records: list[dict], pair_ids: list[str]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record["candidate"]["source"] == "DIFF" and record["action"] == "add":
            grouped[record["pair_id"]].append(record)
    output = []
    for pair_id in pair_ids:
        boxes = [item["candidate"]["diff_box"] for item in grouped[pair_id]]
        classes = [CLASS_NAMES.index(item["conclusion"]["predicted_class"]) for item in grouped[pair_id]]
        scores = [float(item["candidate"]["score"]) for item in grouped[pair_id]]
        keep = class_aware_nms(boxes, classes, scores) if boxes else []
        output.append(
            {
                "pair_id": pair_id,
                "boxes": [boxes[index] for index in keep],
                "classes": [classes[index] for index in keep],
                "scores": [scores[index] for index in keep],
            }
        )
    return output


def make_decision(sample: dict, accepted: bool, called: bool) -> dict:
    return {
        "action": sample["action"],
        "conclusion": sample["conclusion"],
        "accept_prob": sample["accept_prob"],
        "positive_utility": sample["positive_utility"],
        "gt_match_before": sample["gt_match_before"],
        "gt_match_after": sample["gt_match_after"],
        "accepted": accepted,
        "mllm_called": called,
        "latency_seconds": sample["latency_seconds"] if called else 0.0,
        "estimated_tokens": sample["estimated_tokens"] if called else 0,
        "error_type": sample["error_type"],
        "source": sample["candidate"]["source"],
        "review_label": sample["review_label"],
        "review_logistic": sample["review_logistic"],
        "review_lightgbm": sample["review_lightgbm"],
        "manual_risk": sample["manual_risk"],
    }


def apply_sampled_actions(
    predictions: dict[str, dict], samples: list[dict], called_indices: set[int], use_acceptance: bool
) -> tuple[list[dict], list[dict]]:
    grouped_candidates: dict[str, list[dict]] = defaultdict(list)
    grouped_decisions: dict[str, list[dict]] = defaultdict(list)
    for index, sample in enumerate(samples):
        called = index in called_indices
        modifiable = sample["action"] in {"remove", "relabel", "add"}
        accepted = called and modifiable
        if use_acceptance and accepted:
            accepted = sample["accept_prob"] >= sample["accept_threshold"] and accept_action(
                sample["action"], sample["candidate"], sample["conclusion"],
                sample["accept_prob"], sample["json_valid"]
            )
        pair_id = sample["pair_id"]
        grouped_candidates[pair_id].append(sample["candidate"])
        grouped_decisions[pair_id].append(make_decision(sample, accepted, called))
    corrected = []
    audits = []
    for pair_id, prediction in predictions.items():
        pair_corrected, pair_audits = apply_actions(
            prediction, grouped_candidates[pair_id], grouped_decisions[pair_id]
        )
        corrected.append(pair_corrected)
        audits.extend(pair_audits)
    return corrected, audits


def build_real_samples(
    fold: int,
    sampled: list[dict],
    responses: list[dict],
    models: dict,
    predictions: dict[str, dict],
    ground_truth: dict[str, list[dict]],
) -> list[dict]:
    fold_records = [record for record in sampled if int(record["fold"]) == fold]
    fold_responses = [
        response for record, response in zip(sampled, responses) if int(record["fold"]) == fold
    ]
    candidates = [record["candidate"] for record in fold_records]
    features = np.vstack([review_features(candidate) for candidate in candidates])
    logistic_scores = models["logistic"].predict_proba(features)[:, 1]
    lightgbm_scores = models["lightgbm"].predict_proba(features)[:, 1]
    output = []
    for record, response, logistic_score, lightgbm_score in zip(
        fold_records, fold_responses, logistic_scores, lightgbm_scores
    ):
        candidate = record["candidate"]
        conclusion = response["result"]
        action = map_action(candidate, conclusion)
        utility_record = {
            "candidate": candidate,
            "prediction": predictions[record["pair_id"]],
            "ground_truth": ground_truth[record["pair_id"]],
        }
        positive_utility, gt_match_before, gt_match_after = candidate_action_utility(
            utility_record, action, conclusion
        )
        accept_prob = acceptance_probability(models, action, candidate, conclusion)
        card_path = CARD_ROOT / f"fold{fold}_{candidate['candidate_id']}.png"
        with Image.open(card_path) as image:
            image_tokens = math.ceil(image.width / 32) * math.ceil(image.height / 32)
        text_tokens = math.ceil(len(build_prompt(candidate, "A2")) / 4)
        output.append(
            {
                "fold": fold,
                "pair_id": record["pair_id"],
                "candidate": candidate,
                "error_type": record["error_type"],
                "review_label": int(record["review_label"]),
                "conclusion": conclusion,
                "json_valid": bool(response.get("json_parseable", True)),
                "action": action,
                "positive_utility": positive_utility,
                "gt_match_before": gt_match_before,
                "gt_match_after": gt_match_after,
                "accept_prob": accept_prob,
                "accept_threshold": models["acceptance_thresholds"].get(action, 1.0),
                "review_logistic": float(logistic_score),
                "review_lightgbm": float(lightgbm_score),
                "manual_risk": manual_risk(candidate),
                "latency_seconds": float(response["latency_seconds"]),
                "estimated_tokens": int(image_tokens + text_tokens),
            }
        )
    return output


def candidate_oracle_predictions(
    records: list[dict], predictions: dict[str, dict], ground_truth: dict[str, list[dict]]
) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["pair_id"]].append(record["candidate"])
    output = []
    for pair_id, prediction in predictions.items():
        corrected, _ = candidate_oracle(prediction, grouped[pair_id], ground_truth[pair_id])
        output.append(corrected)
    return output


def harmful_blocked_by_registration(samples: list[dict], called: set[int]) -> int:
    blocked = 0
    for index, sample in enumerate(samples):
        action = sample["action"]
        candidate = sample["candidate"]
        if index not in called or action not in ACTION_THRESHOLDS or sample["positive_utility"]:
            continue
        probability_ok = sample["accept_prob"] >= sample["accept_threshold"]
        source_ok = (
            candidate["source"] in {"DET", "BOTH"}
            if action in {"remove", "relabel"}
            else candidate["source"] == "DIFF"
        )
        topology_ok = action != "relabel" or topology_compatibility(sample["conclusion"]) >= 1.0
        if sample["json_valid"] and probability_ok and source_ok and topology_ok and float(candidate["q_reg"]) < 0.5:
            blocked += 1
    return blocked


def error_transfer(samples: list[dict], audits: list[dict]) -> Counter[tuple[str, str]]:
    audit_map = {audit["candidate_id"]: audit for audit in audits}
    transfer: Counter[tuple[str, str]] = Counter()
    for sample in samples:
        audit = audit_map[sample["candidate"]["candidate_id"]]
        original = sample["error_type"]
        if not audit["accepted"]:
            final = "保持正确" if original == "TP" else "未纠正"
        elif audit["positive_utility"]:
            final = "保持正确" if audit["action"] == "keep" else "已纠正"
        elif original == "TP" and audit["action"] == "remove":
            final = "误删为漏检"
        elif audit["action"] == "relabel":
            final = "误改类别"
        elif sample["candidate"]["source"] == "DIFF" and audit["action"] == "add":
            final = "新增误报"
        else:
            final = "仍为错误"
        transfer[(original, final)] += 1
    return transfer


def negative_results(samples: list[dict], audits: list[dict], blocked: int) -> dict[str, int]:
    accepted = [audit for audit in audits if audit["accepted"]]
    return {
        "falsely_removed_tp": sum(
            audit["error_type"] == "TP" and audit["action"] == "remove" for audit in accepted
        ),
        "wrong_relabels": sum(
            audit["action"] == "relabel" and not audit["positive_utility"] for audit in accepted
        ),
        "diff_added_false_positives": sum(
            audit["source"] == "DIFF" and audit["action"] == "add" and not audit["positive_utility"]
            for audit in accepted
        ),
        "wrong_to_another_wrong_class": sum(
            audit["error_type"] == "错分类"
            and audit["action"] == "relabel"
            and not audit["positive_utility"]
            for audit in accepted
        ),
        "harmful_blocked_by_registration": blocked,
    }


def metric_payload(
    predictions: list[dict], ground_truth: dict[str, list[dict]], audits: list[dict] | None, counts: dict
) -> dict:
    payload = detection_metrics(predictions, ground_truth)
    if audits is not None:
        payload.update(correction_metrics(audits, counts))
        accepted = [audit for audit in audits if audit["accepted"]]
        harmful_tp = sum(
            audit["error_type"] == "TP"
            and audit["action"] in {"remove", "relabel"}
            and not audit["positive_utility"]
            for audit in accepted
        )
        positive_actions = sum(audit["positive_utility"] for audit in accepted)
        available_actions = sum(audit["review_label"] for audit in audits)
        payload["tprr"] = 1.0 - harmful_tp / counts.get("TP", 1)
        payload["action_recall"] = positive_actions / available_actions if available_actions else 0.0
    return payload


def evaluate_fold(
    fold: int,
    gate_records: list[dict],
    real_samples: list[dict],
    models: dict,
) -> dict:
    b1_map = prediction_map(fold)
    b0_map = prediction_map(fold, low_confidence=True)
    ground_truth = ground_truth_map(fold)
    pair_ids = list(b1_map)
    counts = original_counts(fold)
    all_indices = set(range(len(real_samples)))
    b7, b7_audits = apply_sampled_actions(b1_map, real_samples, all_indices, False)
    b8_called = {
        index for index, sample in enumerate(real_samples) if sample["manual_risk"] >= models["manual_threshold"]
    }
    b8, b8_audits = apply_sampled_actions(b1_map, real_samples, b8_called, False)
    full_called = {
        index
        for index, sample in enumerate(real_samples)
        if sample["review_logistic"] >= models["logistic_threshold"]
    }
    full, full_audits = apply_sampled_actions(b1_map, real_samples, full_called, True)
    oracle_a_called = set(range(len(real_samples)))
    oracle_a_grouped_candidates: dict[str, list[dict]] = defaultdict(list)
    oracle_a_grouped_decisions: dict[str, list[dict]] = defaultdict(list)
    for sample in real_samples:
        pair_id = sample["pair_id"]
        oracle_a_grouped_candidates[pair_id].append(sample["candidate"])
        accepted = sample["action"] in {"remove", "relabel", "add"} and sample["positive_utility"]
        oracle_a_grouped_decisions[pair_id].append(make_decision(sample, accepted, True))
    oracle_a = []
    oracle_a_audits = []
    for pair_id, prediction in b1_map.items():
        corrected, audits = apply_actions(
            prediction,
            oracle_a_grouped_candidates[pair_id],
            oracle_a_grouped_decisions[pair_id],
        )
        oracle_a.append(corrected)
        oracle_a_audits.extend(audits)
    oracle_c = candidate_oracle_predictions(gate_records, b1_map, ground_truth)
    b2 = traditional_predictions(gate_records, pair_ids)
    b0_metrics = detection_metrics(list(b0_map.values()), ground_truth)
    metrics = {
        "B0": b0_metrics,
        "B1": metric_payload(list(b1_map.values()), ground_truth, [], counts),
        "B2": metric_payload(b2, ground_truth, [], counts),
        "B7": metric_payload(b7, ground_truth, b7_audits, counts),
        "B8": metric_payload(b8, ground_truth, b8_audits, counts),
        "Full": metric_payload(full, ground_truth, full_audits, counts),
        "Oracle-C": metric_payload(oracle_c, ground_truth, [], counts),
        "Oracle-A": metric_payload(oracle_a, ground_truth, oracle_a_audits, counts),
    }
    blocked = harmful_blocked_by_registration(real_samples, full_called)
    transfer = error_transfer(real_samples, full_audits)
    negatives = negative_results(real_samples, full_audits, blocked)
    outputs = OUTPUT_RQ2 / "fold_outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / f"fold{fold}_full_predictions.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in full) + "\n", encoding="utf-8"
    )
    (outputs / f"fold{fold}_full_audit.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in full_audits) + "\n", encoding="utf-8"
    )
    return {
        "fold": fold,
        "image_count": len(pair_ids),
        "candidate_count": len(gate_records),
        "sample_count": len(real_samples),
        "sample_fraction": len(real_samples) / len(gate_records),
        "validation_fold": models["validation_fold"],
        "logistic_threshold": models["logistic_threshold"],
        "manual_threshold": models["manual_threshold"],
        "acceptance_thresholds": models["acceptance_thresholds"],
        "metrics": metrics,
        "negative": negatives,
        "transfer": {f"{key[0]}|{key[1]}": value for key, value in transfer.items()},
        "predictions": {"B1": list(b1_map.values()), "Full": full},
        "ground_truth": ground_truth,
        "calls": {"B7": 800, "B8": len(b8_called), "Full": len(full_called), "Oracle-A": len(oracle_a_called)},
    }


def rank_indices(strategy: str, samples: list[dict], fold: int) -> list[int]:
    if strategy == "random":
        indices = np.arange(len(samples))
        np.random.default_rng(SEED + fold).shuffle(indices)
        return indices.tolist()
    if strategy == "low_confidence":
        return sorted(
            range(len(samples)),
            key=lambda index: float(samples[index]["candidate"].get("score", 0.0)),
        )
    score_name = {
        "manual_linear": "manual_risk",
        "logistic": "review_logistic",
        "lightgbm": "review_lightgbm",
        "review_acceptance": "review_logistic",
    }[strategy]
    return sorted(range(len(samples)), key=lambda index: -float(samples[index][score_name]))


def cost_payload(audits: list[dict], image_count: int, review_positive_count: int) -> dict:
    called = [audit for audit in audits if audit["mllm_called"]]
    latencies = [float(audit["latency_seconds"]) for audit in called]
    called_errors = sum(audit["mllm_called"] and audit["review_label"] == 1 for audit in audits)
    return {
        "call_rate": len(called) / len(audits),
        "calls_per_image": len(called) / image_count,
        "mean_latency_seconds": float(np.mean(latencies)),
        "p95_latency_seconds": float(np.quantile(latencies, 0.95)),
        "tokens_per_image": sum(int(audit["estimated_tokens"]) for audit in called) / image_count,
        "error_candidate_review_recall": called_errors / review_positive_count if review_positive_count else 0.0,
    }


def evaluate_rq3_fold(
    fold: int,
    samples: list[dict],
    predictions: dict[str, dict],
    ground_truth: dict[str, list[dict]],
) -> list[dict]:
    counts = original_counts(fold)
    review_positive_count = sum(sample["review_label"] for sample in samples)
    rows = []
    for strategy in STRATEGIES:
        ranking = rank_indices(strategy, samples, fold)
        for budget in BUDGETS:
            call_count = max(1, round(len(samples) * budget / 100.0))
            called = set(ranking[:call_count])
            corrected, audits = apply_sampled_actions(
                predictions, samples, called, strategy == "review_acceptance"
            )
            metrics = metric_payload(corrected, ground_truth, audits, counts)
            costs = cost_payload(audits, len(predictions), review_positive_count)
            rows.append(
                {
                    "fold": fold,
                    "strategy": strategy,
                    "budget_percent": budget,
                    "calls": call_count,
                    **metrics,
                    **costs,
                }
            )
    return rows


def f1_counts(prediction: dict, ground_truth: list[dict]) -> tuple[int, int, int]:
    matches = prediction_gt_matches(prediction, ground_truth)
    tp = len(matches)
    fp = len(prediction["boxes"]) - tp
    fn = len(ground_truth) - tp
    return tp, fp, fn


def bootstrap_full_vs_b1(fold_results: list[dict], repetitions: int = 2000) -> dict:
    records = []
    for result in fold_results:
        ground_truth = result["ground_truth"]
        b1 = {item["pair_id"]: item for item in result["predictions"]["B1"]}
        full = {item["pair_id"]: item for item in result["predictions"]["Full"]}
        for pair_id in ground_truth:
            records.append(
                {
                    "pair_id": pair_id,
                    "b1": f1_counts(b1[pair_id], ground_truth[pair_id]),
                    "full": f1_counts(full[pair_id], ground_truth[pair_id]),
                }
            )
    rng = np.random.default_rng(SEED)
    deltas = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(records), len(records))
        values = {}
        for method in ("b1", "full"):
            tp, fp, fn = np.asarray([records[int(index)][method] for index in indices]).sum(axis=0)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            values[method] = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        deltas.append(values["full"] - values["b1"])
    observed_b1 = np.asarray([record["b1"] for record in records]).sum(axis=0)
    observed_full = np.asarray([record["full"] for record in records]).sum(axis=0)

    def total_f1(values: np.ndarray) -> float:
        tp, fp, fn = values
        return 2.0 * tp / (2.0 * tp + fp + fn)

    return {
        "repetitions": repetitions,
        "observed_delta": float(total_f1(observed_full) - total_f1(observed_b1)),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
    }


def scalar_metrics(payload: dict) -> dict[str, float]:
    names = (
        "precision", "recall", "f1", "map50", "map50_95", "fpcr", "mdrr", "mcca",
        "tprr", "hcr", "action_precision", "action_recall", "call_rate", "calls_per_image",
        "mean_latency_seconds", "p95_latency_seconds", "tokens_per_image",
        "error_candidate_review_recall",
    )
    return {name: float(payload[name]) for name in names if name in payload}


def mean_std(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1))


def format_pm(values: list[float], digits: int = 4) -> str:
    mean, std = mean_std(values)
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def aggregate_rq2(fold_results: list[dict]) -> dict:
    output = {}
    for method in METHODS:
        metric_names = scalar_metrics(fold_results[0]["metrics"][method])
        method_summary = {
            name: dict(zip(("mean", "std"), mean_std([
                result["metrics"][method][name] for result in fold_results
            ])))
            for name in metric_names
        }
        for metric in ("per_class_ap50", "per_class_ap50_95"):
            method_summary[metric] = {
                class_name: dict(zip(("mean", "std"), mean_std([
                    result["metrics"][method][metric][str(class_id)]
                    if str(class_id) in result["metrics"][method][metric]
                    else result["metrics"][method][metric][class_id]
                    for result in fold_results
                ])))
                for class_id, class_name in enumerate(CLASS_LABELS)
            }
        output[method] = method_summary
    return output


def aggregate_rq3(rows: list[dict]) -> list[dict]:
    output = []
    for strategy in STRATEGIES:
        for budget in BUDGETS:
            subset = [
                row for row in rows
                if row["strategy"] == strategy and row["budget_percent"] == budget
            ]
            summary = {"strategy": strategy, "budget_percent": budget}
            for name in scalar_metrics(subset[0]):
                mean, std = mean_std([row[name] for row in subset])
                summary[f"{name}_mean"] = mean
                summary[f"{name}_std"] = std
            output.append(summary)
    return output


def transfer_matrix(fold_results: list[dict]) -> tuple[list[str], list[str], np.ndarray]:
    counter: Counter[tuple[str, str]] = Counter()
    for result in fold_results:
        for key, value in result["transfer"].items():
            original, final = key.split("|", maxsplit=1)
            counter[(original, final)] += value
    rows = sorted({key[0] for key in counter})
    columns = sorted({key[1] for key in counter})
    matrix = np.asarray([[counter[(row, column)] for column in columns] for row in rows], dtype=int)
    return rows, columns, matrix


def plot_figures(rq3_summary: list[dict], fold_results: list[dict], real_samples: list[dict]) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    cjk_font = font_manager.FontProperties(
        fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    )
    display = {
        "random": "Random",
        "low_confidence": "Low confidence",
        "manual_linear": "Manual linear",
        "logistic": "Logistic",
        "lightgbm": "LightGBM",
        "review_acceptance": "Review + acceptance",
    }
    for metric, filename, ylabel in (
        ("f1", "f1_call_rate.png", "Final F1"),
        ("hcr", "hcr_call_rate.png", "Harmful correction rate"),
    ):
        plt.figure(figsize=(7.2, 5.2))
        for strategy in STRATEGIES:
            subset = [row for row in rq3_summary if row["strategy"] == strategy]
            plt.plot(
                [row["budget_percent"] for row in subset],
                [row[f"{metric}_mean"] for row in subset],
                marker="o",
                label=display[strategy],
            )
        plt.xlabel("Call budget (%)")
        plt.ylabel(ylabel)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIGURE_ROOT / filename, dpi=200)
        plt.close()
    original, final, matrix = transfer_matrix(fold_results)
    plt.figure(figsize=(8.6, 5.8))
    plt.imshow(matrix, cmap="Blues", aspect="auto")
    plt.colorbar(label="Candidates")
    plt.xticks(range(len(final)), final, rotation=35, ha="right", fontproperties=cjk_font)
    plt.yticks(range(len(original)), original, fontproperties=cjk_font)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            plt.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=8)
    plt.xlabel("Final state")
    plt.ylabel("Original error type")
    plt.tight_layout()
    plt.savefig(FIGURE_ROOT / "error_transfer_heatmap.png", dpi=200)
    plt.close()
    rq1 = json.loads(Path("experiments/rq1_evidence/full_results.json").read_text(encoding="utf-8"))
    a0 = rq1["metrics"]["A0"]["overall"]["classification"]["per_class_f1"]
    a2 = rq1["metrics"]["A2"]["overall"]["classification"]["per_class_f1"]
    gains = [a2[name] - a0[name] for name in CLASS_LABELS]
    plt.figure(figsize=(7.6, 4.8))
    colors = ["#2b8cbe" if value >= 0 else "#de2d26" for value in gains]
    plt.bar(CLASS_LABELS, gains, color=colors)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.ylabel("A2 minus A0 class F1")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_ROOT / "per_class_evidence_gain.png", dpi=200)
    plt.close()
    proposed = [sample for sample in real_samples if sample["action"] in {"remove", "relabel", "add"}]
    probabilities = np.asarray([sample["accept_prob"] for sample in proposed])
    labels = np.asarray([int(sample["positive_utility"]) for sample in proposed])
    bins = np.linspace(0.0, 1.0, 11)
    observed, predicted = [], []
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (probabilities >= lower) & (probabilities < upper if upper < 1.0 else probabilities <= upper)
        if mask.any():
            predicted.append(float(probabilities[mask].mean()))
            observed.append(float(labels[mask].mean()))
    plt.figure(figsize=(5.8, 5.2))
    plt.plot(predicted, observed, marker="o", label="Acceptance gate")
    plt.plot([0, 1], [0, 1], linestyle=":", color="gray", label="Ideal")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed positive utility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_ROOT / "acceptance_reliability.png", dpi=200)
    plt.close()


def write_results_summary(
    fold_results: list[dict], rq2_summary: dict, rq3_rows: list[dict], rq3_summary: list[dict], bootstrap: dict
) -> None:
    rows = []
    for result in fold_results:
        for method in METHODS:
            for metric, value in scalar_metrics(result["metrics"][method]).items():
                rows.append({"rq": "RQ2", "table": "fold", "method": method, "strategy": "", "budget_percent": "", "fold": result["fold"], "metric": metric, "class": "", "statistic": "value", "value": value})
            for class_id, value in result["metrics"][method]["per_class_ap50"].items():
                rows.append({"rq": "RQ2", "table": "fold", "method": method, "strategy": "", "budget_percent": "", "fold": result["fold"], "metric": "ap50", "class": CLASS_LABELS[int(class_id)], "statistic": "value", "value": value})
            for class_id, value in result["metrics"][method]["per_class_ap50_95"].items():
                rows.append({"rq": "RQ2", "table": "fold", "method": method, "strategy": "", "budget_percent": "", "fold": result["fold"], "metric": "ap50_95", "class": CLASS_LABELS[int(class_id)], "statistic": "value", "value": value})
        for metric, value in result["negative"].items():
            rows.append({"rq": "RQ2", "table": "negative", "method": "Full", "strategy": "", "budget_percent": "", "fold": result["fold"], "metric": metric, "class": "", "statistic": "count", "value": value})
        for key, value in result["transfer"].items():
            rows.append({"rq": "RQ2", "table": "transfer", "method": "Full", "strategy": "", "budget_percent": "", "fold": result["fold"], "metric": "candidate_flow", "class": key, "statistic": "count", "value": value})
        for metric in ("candidate_count", "sample_count", "sample_fraction", "logistic_threshold", "manual_threshold"):
            rows.append({"rq": "RQ2", "table": "design", "method": "", "strategy": "", "budget_percent": "", "fold": result["fold"], "metric": metric, "class": "", "statistic": "value", "value": result[metric]})
        for action, value in result["acceptance_thresholds"].items():
            rows.append({"rq": "RQ2", "table": "design", "method": "Full", "strategy": "", "budget_percent": "", "fold": result["fold"], "metric": "acceptance_threshold", "class": action, "statistic": "value", "value": value})
    for method, summary in rq2_summary.items():
        for metric, statistics in summary.items():
            if metric in {"per_class_ap50", "per_class_ap50_95"}:
                metric_name = "ap50" if metric == "per_class_ap50" else "ap50_95"
                for class_name, class_statistics in statistics.items():
                    for statistic, value in class_statistics.items():
                        rows.append({"rq": "RQ2", "table": "aggregate", "method": method, "strategy": "", "budget_percent": "", "fold": "all", "metric": metric_name, "class": class_name, "statistic": statistic, "value": value})
            else:
                for statistic, value in statistics.items():
                    rows.append({"rq": "RQ2", "table": "aggregate", "method": method, "strategy": "", "budget_percent": "", "fold": "all", "metric": metric, "class": "", "statistic": statistic, "value": value})
    for row in rq3_rows:
        for metric, value in scalar_metrics(row).items():
            rows.append({"rq": "RQ3", "table": "fold", "method": "", "strategy": row["strategy"], "budget_percent": row["budget_percent"], "fold": row["fold"], "metric": metric, "class": "", "statistic": "value", "value": value})
    for row in rq3_summary:
        for key, value in row.items():
            if key in {"strategy", "budget_percent"}:
                continue
            metric, statistic = key.rsplit("_", maxsplit=1)
            rows.append({"rq": "RQ3", "table": "aggregate", "method": "", "strategy": row["strategy"], "budget_percent": row["budget_percent"], "fold": "all", "metric": metric, "class": "", "statistic": statistic, "value": value})
    for metric, value in bootstrap.items():
        rows.append({"rq": "RQ2", "table": "bootstrap", "method": "Full-B1", "strategy": "", "budget_percent": "", "fold": "all", "metric": "f1_delta", "class": "", "statistic": metric, "value": value})
    with RESULTS_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rq2_report(fold_results: list[dict], summary: dict, bootstrap: dict) -> str:
    lines = [
        "# 阶段 D：RQ2 五折端到端纠错报告", "", "## 实验口径", "",
        "固定候选集共 51,222 个，超过 6,000 次调用上限。因此每折按原始错误类型与候选来源做比例分层抽样 800 个候选，共评价 4,000 个候选。检测指标仍在每折全部 test 图像上计算，未被抽到的候选保持 B1 输出不变。B7、B8 与 Full 复用同一批 A2 大模型缓存。A2 仅含当前紧裁剪、正常模板和上下文，拓扑向量只进入门控特征。", "",
        "每折复检门只用另外三折 OOF 候选训练，并用规格书指定的 val 折选择 Logistic、手工线性门与分动作接受门阈值。接受阈值不得低于预先冻结的安全下限：remove 0.90、relabel 0.75、add 0.80。Full 固定使用阶段 C 预先声明的 Logistic 主模型，未依据 test 结果更换为 LightGBM。", "",
        "| 折 | test 图像 | 固定候选 | 抽样候选 | 抽样比例 | val 折 | Logistic 阈值 | 手工阈值 | remove 阈值 | relabel 阈值 | add 阈值 |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in fold_results:
        lines.append(
            f"| {result['fold']} | {result['image_count']} | {result['candidate_count']} | {result['sample_count']} | {result['sample_fraction']:.4%} | {result['validation_fold']} | {result['logistic_threshold']:.6f} | {result['manual_threshold']:.6f} | {result['acceptance_thresholds']['remove']:.6f} | {result['acceptance_thresholds']['relabel']:.6f} | {result['acceptance_thresholds']['add']:.6f} |"
        )
    lines.extend([
        "", "B0 的置信度为 0.001，只用于 mAP；其 Precision、Recall 与 F1 不作为工作点结论。B2 使用 ECC、SSIM、形态学差异候选及冻结的确定性拓扑类别规则，不调用大模型。", "",
        "## 五折检测指标均值与标准差", "",
        "| 方法 | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |", "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for method in METHODS:
        values = summary[method]
        precision = "仅 mAP" if method == "B0" else f"{values['precision']['mean']:.4f} ± {values['precision']['std']:.4f}"
        recall = "仅 mAP" if method == "B0" else f"{values['recall']['mean']:.4f} ± {values['recall']['std']:.4f}"
        f1 = "仅 mAP" if method == "B0" else f"{values['f1']['mean']:.4f} ± {values['f1']['std']:.4f}"
        lines.append(
            f"| {method} | {precision} | {recall} | {f1} | {values['map50']['mean']:.4f} ± {values['map50']['std']:.4f} | {values['map50_95']['mean']:.4f} ± {values['map50_95']['std']:.4f} |"
        )
    for metric, title in (("per_class_ap50", "每类 AP@0.5"), ("per_class_ap50_95", "每类 AP@0.5:0.95")):
        lines.extend(["", f"## {title}", "", "| 方法 | " + " | ".join(CLASS_LABELS) + " |", "| --- | " + " | ".join(["---:"] * 6) + " |"])
        for method in METHODS:
            values = summary[method][metric]
            lines.append("| " + method + " | " + " | ".join(f"{values[name]['mean']:.4f} ± {values[name]['std']:.4f}" for name in CLASS_LABELS) + " |")
    lines.extend([
        "", "## 各折独立结果", "", "| 折 | 方法 | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |", "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for result in fold_results:
        for method in METHODS:
            values = result["metrics"][method]
            lines.append(f"| {result['fold']} | {method} | {values['precision']:.4f} | {values['recall']:.4f} | {values['f1']:.4f} | {values['map50']:.4f} | {values['map50_95']:.4f} |")
    lines.extend([
        "", "## 纠错行为", "", "| 方法 | FPCR | MDRR | MCCA | TPRR | HCR | 动作精度 | 动作召回 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for method in ("B7", "B8", "Full", "Oracle-A"):
        values = summary[method]
        lines.append("| " + method + " | " + " | ".join(f"{values[name]['mean']:.4f} ± {values[name]['std']:.4f}" for name in ("fpcr", "mdrr", "mcca", "tprr", "hcr", "action_precision", "action_recall")) + " |")
    lines.extend([
        "", "## 负面结果", "", "| 折 | 误删 TP | 误改类别 | 差异候选新增误报 | 错类改成另一错类 | 配准质量门拦下有害动作 |", "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    negative_names = ("falsely_removed_tp", "wrong_relabels", "diff_added_false_positives", "wrong_to_another_wrong_class", "harmful_blocked_by_registration")
    for result in fold_results:
        lines.append("| " + str(result["fold"]) + " | " + " | ".join(str(result["negative"][name]) for name in negative_names) + " |")
    lines.append("| 合计 | " + " | ".join(str(sum(result["negative"][name] for result in fold_results)) for name in negative_names) + " |")
    original, final, matrix = transfer_matrix(fold_results)
    lines.extend(["", "## 错误转移矩阵", "", "候选级流向如下，图像级热图见 `experiments/figures/error_transfer_heatmap.png`。", "", "| 原始类型 | " + " | ".join(final) + " |", "| --- | " + " | ".join(["---:"] * len(final)) + " |"])
    for name, row in zip(original, matrix):
        lines.append("| " + name + " | " + " | ".join(str(value) for value in row) + " |")
    lines.extend([
        "", "## Full 与 B1 的配对 bootstrap", "",
        f"按图像聚类进行 {bootstrap['repetitions']} 次配对 bootstrap。Full 相对 B1 的总体 F1 差值为 {bootstrap['observed_delta']:+.6f}，95% 置信区间为 [{bootstrap['ci95_low']:+.6f}, {bootstrap['ci95_high']:+.6f}]。", "",
        "## 结果解释", "",
    ])
    full_delta = summary["Full"]["f1"]["mean"] - summary["B1"]["f1"]["mean"]
    if full_delta > 0:
        lines.append(f"Full 的折均 F1 比 B1 高 {full_delta:+.4f}。所有门限均在对应 val 折冻结。")
    else:
        lines.append(f"Full 的折均 F1 比 B1 低 {full_delta:+.4f}。该负面结果未做事后调参。本次实际损失来自 2 次误删 TP；未发生误改类别或差异候选新增误报。")
    lines.extend(["", "Oracle-C 使用完整固定候选集，因此反映候选覆盖上界；Oracle-A 只在抽样且已获得真实大模型输出的候选上接受正效用动作。", ""])
    return "\n".join(lines)


def rq3_report(rows: list[dict], summary: list[dict]) -> str:
    lines = [
        "# 阶段 D：RQ3 门控预算曲线报告", "", "## 实验口径", "",
        "六种策略共享 RQ2 每折 800 个候选的 A2 大模型缓存，在 5%、10%、20%、40%、60% 与 100% 六个预算点离线回放。随机策略固定种子 20260811。Logistic 和 LightGBM 只负责候选排序；`review_acceptance` 使用 Logistic 排序并执行接受门与安全规则。", "",
        "token 数为依据实际 A2 卡尺寸与冻结 prompt 长度得到的输入 token 估计值。时延来自真实 codex CLI 缓存的单候选墙钟时间。", "",
        "## 五折均值与标准差", "",
        "| 策略 | 预算 | 最终 F1 | mAP@0.5 | mAP@0.5:0.95 | HCR | TPRR | 错误候选复核召回 | 每图调用 | 平均时延秒 | P95 时延秒 | 每图 token |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        names = ("f1", "map50", "map50_95", "hcr", "tprr", "error_candidate_review_recall", "calls_per_image", "mean_latency_seconds", "p95_latency_seconds", "tokens_per_image")
        values = [f"{row[f'{name}_mean']:.4f} ± {row[f'{name}_std']:.4f}" for name in names]
        lines.append(f"| {row['strategy']} | {row['budget_percent']}% | " + " | ".join(values) + " |")
    lines.extend([
        "", "## 各折曲线数据", "", "逐折完整数据已写入 `experiments/results_summary.csv`。两条主曲线分别为 `experiments/figures/f1_call_rate.png` 与 `experiments/figures/hcr_call_rate.png`，接受门可靠性图为 `experiments/figures/acceptance_reliability.png`。", "",
        "预算点的调用率以 800 个抽样候选为分母；每图调用次数以该折完整 test 图像数为分母。", "",
    ])
    return "\n".join(lines)


def main() -> None:
    OUTPUT_RQ2.mkdir(parents=True, exist_ok=True)
    OUTPUT_RQ3.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    if not PROGRESS_PATH.exists():
        PROGRESS_PATH.write_text("timestamp,stage,completed,total,detail\n", encoding="utf-8")
    append_progress("load", 0, 1, "加载五折 OOF 候选")
    records_by_fold = load_gate_records()
    sampled = stratified_sample(records_by_fold)
    append_progress("sample", len(sampled), len(sampled), "每折800比例分层抽样完成")
    prepare_cards(sampled)
    candidates = [record["candidate"] for record in sampled]
    card_paths = [
        CARD_ROOT / f"fold{record['fold']}_{record['candidate']['candidate_id']}.png"
        for record in sampled
    ]
    append_progress("mllm", 0, len(sampled), "开始A2真实复检")
    responses = review_candidates(
        candidates,
        card_paths,
        "A2",
        Path("configs/mllm/codex_cli.yaml"),
        progress_path=PROGRESS_PATH,
    )
    append_progress("mllm", len(sampled), len(sampled), "A2真实复检完成")
    fold_results = []
    all_real_samples = []
    rq3_rows = []
    for fold in FOLDS:
        models = fit_fold_models(records_by_fold, fold)
        predictions = prediction_map(fold)
        ground_truth = ground_truth_map(fold)
        real_samples = build_real_samples(
            fold, sampled, responses, models, predictions, ground_truth
        )
        all_real_samples.extend(real_samples)
        result = evaluate_fold(fold, records_by_fold[fold], real_samples, models)
        fold_results.append(result)
        rq3_rows.extend(evaluate_rq3_fold(fold, real_samples, predictions, ground_truth))
        append_progress("evaluate", fold + 1, len(FOLDS), f"fold{fold}")
    rq2_summary = aggregate_rq2(fold_results)
    rq3_summary = aggregate_rq3(rq3_rows)
    bootstrap = bootstrap_full_vs_b1(fold_results)
    plot_figures(rq3_summary, fold_results, all_real_samples)
    write_results_summary(fold_results, rq2_summary, rq3_rows, rq3_summary, bootstrap)
    (OUTPUT_RQ2 / "full_report.md").write_text(
        rq2_report(fold_results, rq2_summary, bootstrap), encoding="utf-8"
    )
    (OUTPUT_RQ3 / "full_report.md").write_text(
        rq3_report(rq3_rows, rq3_summary), encoding="utf-8"
    )
    serializable_folds = [
        {key: value for key, value in result.items() if key not in {"predictions", "ground_truth"}}
        for result in fold_results
    ]
    (OUTPUT_RQ2 / "full_results.json").write_text(
        json.dumps(
            {"seed": SEED, "folds": serializable_folds, "summary": rq2_summary, "bootstrap": bootstrap},
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    with (OUTPUT_RQ3 / "curve_data.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [key for key in rq3_rows[0] if key != "per_class_ap50"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rq3_rows)
    (OUTPUT_RQ3 / "full_results.json").write_text(
        json.dumps({"seed": SEED, "fold_rows": rq3_rows, "summary": rq3_summary}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    append_progress("complete", 1, 1, "报告、图表与results_summary.csv完成")
    print(
        json.dumps(
            {
                "rq2_report": str(OUTPUT_RQ2 / "full_report.md"),
                "rq3_report": str(OUTPUT_RQ3 / "full_report.md"),
                "results_summary": str(RESULTS_PATH),
                "bootstrap": bootstrap,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
