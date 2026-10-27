"""Learn whether a candidate contains a repairable detector error."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.gates.features import review_features


class ReviewGate:
    def __init__(self, model_type: str = "logistic") -> None:
        self.model_type = model_type
        self.model = None
        self.training_count = 0

    def fit(self, candidates: list[dict], labels: list[int]) -> "ReviewGate":
        self.training_count = len(candidates)
        features = np.vstack([review_features(candidate) for candidate in candidates])
        if self.model_type == "logistic":
            self.model = LogisticRegression(C=1.0, l1_ratio=0.0, max_iter=1000, random_state=20260811)
        else:
            from lightgbm import LGBMClassifier

            self.model = LGBMClassifier(reg_lambda=1.0, random_state=20260811, verbosity=-1)
        self.model.fit(features, np.asarray(labels, dtype=np.int64))
        return self

    def predict_risk(self, candidates: list[dict]) -> np.ndarray:
        features = np.vstack([review_features(candidate) for candidate in candidates])
        return self.model.predict_proba(features)[:, 1]


def fit_oof_review_gate(records: list[dict], evaluation_fold: int, model_type: str = "logistic") -> ReviewGate:
    training = [record for record in records if int(record["fold"]) != evaluation_fold]
    return ReviewGate(model_type).fit([record["candidate"] for record in training], [int(record["review_label"]) for record in training])
