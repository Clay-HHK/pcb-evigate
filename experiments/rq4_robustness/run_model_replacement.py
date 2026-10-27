"""RQ4 fold0 200-candidate Logistic and LightGBM gate replacement."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.rq2_correction.run_full import (
    acceptance_probability,
    best_f1_threshold,
    fit_fold_models,
    ground_truth_map,
    load_gate_records,
    prediction_map,
)
from src.evaluation.apply import apply_actions
from src.evaluation.metrics import correction_metrics, detection_metrics
from src.gates.features import review_features
from src.gates.safety import accept_action


SEED = 20260811
OUTPUT = Path("experiments/rq4_robustness/model_replacement.json")


def sample_200(records: list[dict]) -> list[dict]:
    strata: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        strata[f"{record['error_type']}|{record['candidate']['source']}"] .append(record)
    exact = {name: 200 * len(items) / len(records) for name, items in strata.items()}
    counts = {name: int(value) for name, value in exact.items()}
    for name in sorted(strata, key=lambda item: (-(exact[item] - counts[item]), item))[: 200 - sum(counts.values())]:
        counts[name] += 1
    rng = np.random.default_rng(SEED)
    output = []
    for name in sorted(strata):
        indices = rng.permutation(len(strata[name]))[: counts[name]]
        output.extend(strata[name][int(index)] for index in indices)
    return output


def evaluate(
    name: str,
    threshold: float,
    records: list[dict],
    models: dict,
    predictions: dict[str, dict],
    ground_truth: dict[str, list[dict]],
) -> dict:
    features = np.vstack([review_features(record["candidate"]) for record in records])
    gate = models[name]
    scores = gate.predict_proba(features)[:, 1]
    grouped_candidates: dict[str, list[dict]] = defaultdict(list)
    grouped_decisions: dict[str, list[dict]] = defaultdict(list)
    for record, score in zip(records, scores):
        candidate = record["candidate"]
        action = record["action"]
        probability = acceptance_probability(models, action, candidate, record["conclusion"])
        called = float(score) >= threshold
        accepted = called and action in {"remove", "relabel", "add"} and probability >= models["acceptance_thresholds"].get(action, 1.0)
        if accepted:
            accepted = accept_action(action, candidate, record["conclusion"], probability)
        grouped_candidates[record["pair_id"]].append(candidate)
        grouped_decisions[record["pair_id"]].append(
            {
                "action": action,
                "conclusion": record["conclusion"],
                "accept_prob": probability,
                "positive_utility": record["positive_utility"],
                "accepted": accepted,
                "mllm_called": called,
                "error_type": record["error_type"],
                "source": candidate["source"],
                "review_label": record["review_label"],
            }
        )
    corrected = []
    audits = []
    for pair_id, prediction in predictions.items():
        final, pair_audits = apply_actions(prediction, grouped_candidates[pair_id], grouped_decisions[pair_id])
        corrected.append(final)
        audits.extend(pair_audits)
    counts = Counter(record["error_type"] for record in records)
    metrics = detection_metrics(corrected, ground_truth)
    metrics.update(correction_metrics(audits, dict(counts)))
    return {
        "threshold": threshold,
        "calls": sum(item["mllm_called"] for item in audits),
        "accepted_actions": sum(item["accepted"] for item in audits),
        **{key: value for key, value in metrics.items() if not key.startswith("per_class")},
    }


def main() -> None:
    records_by_fold = load_gate_records()
    models = fit_fold_models(records_by_fold, 0)
    all_sampled = [
        item for item in map(json.loads, Path("experiments/rq2_correction/sampled_candidates.jsonl").read_text(encoding="utf-8").splitlines())
        if int(item["fold"]) == 0
    ]
    sampled = sample_200(all_sampled)
    validation = records_by_fold[1]
    validation_features = np.vstack([review_features(item["candidate"]) for item in validation])
    validation_labels = np.asarray([item["review_label"] for item in validation], dtype=int)
    lightgbm_threshold = best_f1_threshold(models["lightgbm"].predict_proba(validation_features)[:, 1], validation_labels)
    predictions = prediction_map(0)
    ground_truth = ground_truth_map(0)
    logistic = evaluate("logistic", models["logistic_threshold"], sampled, models, predictions, ground_truth)
    lightgbm = evaluate("lightgbm", lightgbm_threshold, sampled, models, predictions, ground_truth)
    rows = []
    for name, metrics in (("Logistic", logistic), ("LightGBM", lightgbm)):
        for metric, value in metrics.items():
            rows.append({"model": name, "metric": metric, "value": value})
    report = (
        f"门控替换严格限制在 fold0 的 200 个比例分层候选上。Logistic 的验证阈值为 {logistic['threshold']:.6f}，"
        f"调用 {logistic['calls']} 次，最终 F1={logistic['f1']:.4f}，HCR={logistic['hcr']:.4f}。"
        f"LightGBM 的验证阈值为 {lightgbm['threshold']:.6f}，调用 {lightgbm['calls']} 次，"
        f"最终 F1={lightgbm['f1']:.4f}，HCR={lightgbm['hcr']:.4f}。"
    )
    OUTPUT.write_text(
        json.dumps(
            {"seed": SEED, "fold": 0, "candidates": len(sampled), "logistic": logistic, "lightgbm": lightgbm, "rows": rows, "report": report},
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(report)


if __name__ == "__main__":
    main()
