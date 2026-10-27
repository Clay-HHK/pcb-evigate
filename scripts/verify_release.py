"""Level-1 reproducibility check: recompute RQ2 Full metrics from released per-image predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detector.calibrate import read_gt  # noqa: E402
from src.detector.dataset import rows_for_role  # noqa: E402
from src.evaluation.metrics import detection_metrics  # noqa: E402

TOLERANCE = 1e-9


def check_manifest(manifest: Path) -> int:
    """Return the number of files whose SHA256 does not match MANIFEST.sha256."""
    failures = 0
    for line in manifest.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = ROOT / name.lstrip("*")
        if not path.exists():
            print(f"[missing] {name}")
            failures += 1
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            print(f"[mismatch] {name}")
            failures += 1
    return failures


def ground_truth_map(fold: int) -> dict[str, list[dict]]:
    return {
        row["pair_id"]: [{"box": box, "class": int(class_id)} for box, class_id in read_gt(Path(row["anno_path"]))]
        for row in rows_for_role(fold, "test")
    }


def recompute_rq2_full() -> bool:
    """Recompute five-fold Full metrics and compare with experiments/rq2_correction/full_results.json."""
    reported = json.loads(Path("experiments/rq2_correction/full_results.json").read_text())
    outputs = Path("experiments/rq2_correction/fold_outputs")
    ok = True
    f1_values: list[float] = []
    for fold_result in reported["folds"]:
        fold = int(fold_result["fold"])
        predictions = [
            json.loads(line) for line in (outputs / f"fold{fold}_full_predictions.jsonl").read_text().splitlines() if line
        ]
        metrics = detection_metrics(predictions, ground_truth_map(fold))
        expected = fold_result["metrics"]["Full"]
        for key in ("precision", "recall", "f1", "map50", "map50_95"):
            match = abs(metrics[key] - expected[key]) <= TOLERANCE
            ok &= match
            print(f"fold{fold} {key:<9} recomputed={metrics[key]:.6f} reported={expected[key]:.6f} {'OK' if match else 'DIFF'}")
        f1_values.append(metrics["f1"])
    summary = reported["summary"]["Full"]["f1"]
    mean, std = statistics.mean(f1_values), statistics.stdev(f1_values)
    print(f"Full F1 five-fold: {mean:.4f} ± {std:.4f} (reported {summary['mean']:.4f} ± {summary['std']:.4f})")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-hash", action="store_true", help="skip MANIFEST.sha256 check")
    args = parser.parse_args()
    passed = True
    if not args.skip_hash:
        failures = check_manifest(ROOT / "MANIFEST.sha256")
        print(f"manifest: {failures} failure(s)")
        passed &= failures == 0
    passed &= recompute_rq2_full()
    print("PASS" if passed else "FAIL")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
