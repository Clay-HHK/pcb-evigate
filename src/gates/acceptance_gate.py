"""Action-specific positive-utility acceptance models."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.gates.features import acceptance_features


class AcceptanceGate:
    def __init__(self, minimum_samples: int = 30) -> None:
        self.minimum_samples = minimum_samples
        self.models: dict[str, LogisticRegression | None] = {}
        self.training_counts: dict[str, int] = {}
        self.label_counts: dict[str, dict[str, int]] = {}

    def fit(self, samples: list[dict]) -> "AcceptanceGate":
        for action in ("keep", "remove", "relabel", "add"):
            subset = [sample for sample in samples if sample["action"] == action]
            self.training_counts[action] = len(subset)
            labels = [int(sample["positive_utility"]) for sample in subset]
            self.label_counts[action] = {
                "positive": sum(labels),
                "negative": len(labels) - sum(labels),
            }
            if len(subset) < self.minimum_samples:
                warnings.warn(
                    f"{action} 接受门训练样本 {len(subset)} 个，少于 {self.minimum_samples} 个；启用保守兜底并拒绝该动作",
                    stacklevel=2,
                )
                self.models[action] = None
                continue
            if len(set(labels)) < 2:
                warnings.warn(
                    f"{action} 接受门训练样本 {len(subset)} 个，但标签单一；启用保守兜底并拒绝该动作",
                    stacklevel=2,
                )
                self.models[action] = None
                continue
            features = np.vstack(
                [acceptance_features(sample["candidate"], sample["conclusion"], sample.get("json_valid", True)) for sample in subset]
            )
            model = LogisticRegression(C=1.0, l1_ratio=0.0, max_iter=1000, random_state=20260811)
            model.fit(features, np.asarray(labels, dtype=np.int64))
            self.models[action] = model
        return self

    def predict_probability(self, action: str, candidate: dict, conclusion: dict, json_valid: bool = True) -> float:
        model = self.models[action]
        if model is None:
            return 0.0
        features = acceptance_features(candidate, conclusion, json_valid).reshape(1, -1)
        return float(model.predict_proba(features)[0, 1])


def fit_oof_acceptance_gate(records: list[dict], evaluation_fold: int, minimum_samples: int = 30) -> AcceptanceGate:
    training = [record for record in records if int(record["fold"]) != evaluation_fold]
    return AcceptanceGate(minimum_samples).fit(training)
