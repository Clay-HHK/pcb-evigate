"""Drive five-fold detector training and build the final summary."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from src.detector.dataset import build_fold_dataset


PYTHON = sys.executable
STAGE_DIR = Path("experiments/stage2")
LOG_DIR = STAGE_DIR / "logs"
PROGRESS = STAGE_DIR / "train_progress.csv"
CONFIG = "configs/detector/yolov8s.yaml"
ERROR_NAMES = ["TP", "错分类", "FP", "漏检", "纯定位错误"]


def append_progress(fold: int, stage: str, status: str, note: str = "") -> None:
    with PROGRESS.open("a", newline="", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        csv.writer(handle).writerow(
            [fold, stage, status, datetime.now().astimezone().isoformat(timespec="seconds"), note]
        )
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)


def run_stage(fold: int, stage: str, command: list[str]) -> None:
    append_progress(fold, stage, "running")
    result = subprocess.run(command)
    status = "completed" if result.returncode == 0 else "failed"
    append_progress(fold, stage, status, f"returncode={result.returncode}")
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def run_worker(fold: int, log_path: Path) -> None:
    started = time.monotonic()
    train_command = [PYTHON, "-m", "src.detector.train", "--fold", str(fold), "--config", CONFIG]
    append_progress(fold, "train", "running", "batch=-1")
    result = subprocess.run(train_command)
    batch_note = "batch=-1"
    if result.returncode != 0 and "out of memory" in log_path.read_text(encoding="utf-8", errors="ignore").lower():
        append_progress(fold, "train", "retrying", "自动 batch OOM，固定 batch=16")
        result = subprocess.run(train_command + ["--batch", "16"])
        batch_note = "batch=16（自动 batch OOM 后重跑）"
    status = "completed" if result.returncode == 0 else "failed"
    append_progress(fold, "train", status, batch_note)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    training_seconds = round(time.monotonic() - started, 3)

    run_stage(fold, "predict", [PYTHON, "-m", "src.detector.predict", "--fold", str(fold), "--conf", "0.001"])
    run_stage(fold, "calibrate", [PYTHON, "-m", "src.detector.calibrate", "--fold", str(fold)])
    run_stage(fold, "error_decompose", [PYTHON, "-m", "src.detector.error_decompose", "--fold", str(fold)])
    run_stage(fold, "evaluate", [PYTHON, "-m", "src.detector.evaluate", "--fold", str(fold)])

    results_path = Path(f"cache/detector_runs/fold{fold}/results.csv")
    epochs = len(list(csv.DictReader(results_path.open(encoding="utf-8"))))
    (STAGE_DIR / f"training_meta_fold{fold}.json").write_text(
        json.dumps(
            {
                "fold": fold,
                "duration_seconds": training_seconds,
                "epochs": epochs,
                "batch_note": batch_note,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    append_progress(fold, "fold", "completed", f"epochs={epochs}; {batch_note}")


def load_fold_summary(fold: int) -> dict[str, float | int | str]:
    metrics = json.loads((STAGE_DIR / f"metrics_fold{fold}.json").read_text(encoding="utf-8"))
    meta = json.loads((STAGE_DIR / f"training_meta_fold{fold}.json").read_text(encoding="utf-8"))
    with (STAGE_DIR / f"error_decomposition_fold{fold}.csv").open(encoding="utf-8-sig") as handle:
        error_rows = list(csv.DictReader(handle))
    counts = {name: sum(row["error_type"] == name for row in error_rows) for name in ERROR_NAMES}
    predicted = counts["TP"] + counts["错分类"] + counts["FP"] + counts["纯定位错误"]
    actual = counts["TP"] + counts["错分类"] + counts["漏检"] + counts["纯定位错误"]
    precision = counts["TP"] / predicted if predicted else 0.0
    recall = counts["TP"] / actual if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "fold": fold,
        "map50": metrics["map50"],
        "map50_95": metrics["map50_95"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **counts,
        "correctable": counts["错分类"] + counts["FP"] + counts["漏检"],
        "duration_seconds": meta["duration_seconds"],
        "epochs": meta["epochs"],
        "batch_note": meta["batch_note"],
    }


def mean_std(rows: list[dict[str, float | int | str]], key: str) -> str:
    values = [float(row[key]) for row in rows]
    return f"{statistics.mean(values):.4f} ± {statistics.pstdev(values):.4f}"


def write_summary() -> None:
    rows = [load_fold_summary(fold) for fold in range(5)]
    lines = [
        "# 5 折检测器全量训练汇总",
        "",
        "mAP 由低阈值 conf=0.001 的测试集评估得到。工作点由各折验证集最优 F1 选择。标准差采用总体标准差（ddof=0）。",
        "",
        "## 检测指标",
        "",
        "| 折 | mAP@0.5 | mAP@0.5:0.95 | P | R | F1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['fold']} | {row['map50']:.4f} | {row['map50_95']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |"
        )
    lines.extend(
        [
            f"| 均值 ± 标准差 | {mean_std(rows, 'map50')} | {mean_std(rows, 'map50_95')} | "
            f"{mean_std(rows, 'precision')} | {mean_std(rows, 'recall')} | {mean_std(rows, 'f1')} |",
            "",
            "## 五类错误",
            "",
            "| 折 | TP | 错分类 | FP | 漏检 | 纯定位错误 | 可纠错空间 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['fold']} | {row['TP']} | {row['错分类']} | {row['FP']} | "
            f"{row['漏检']} | {row['纯定位错误']} | {row['correctable']} |"
        )
    lines.append(
        "| 合计 | " + " | ".join(str(sum(int(row[key]) for row in rows)) for key in [*ERROR_NAMES, "correctable"]) + " |"
    )
    lines.extend(
        [
            "",
            f"可纠错空间总量：**{sum(int(row['correctable']) for row in rows)}**（错分类 + FP + 漏检）。",
            "",
            "## 训练信息",
            "",
            "| 折 | 耗时 | 早停轮次 | batch |",
            "| ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        duration = float(row["duration_seconds"])
        duration_text = f"{int(duration // 3600)}:{int(duration % 3600 // 60):02d}:{int(duration % 60):02d}"
        early_stop = str(row["epochs"]) if int(row["epochs"]) < 100 else "未早停（100）"
        lines.append(f"| {row['fold']} | {duration_text} | {early_stop} | {row['batch_note']} |")
    (STAGE_DIR / "five_fold_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    append_progress(-1, "summary", "completed", "experiments/stage2/five_fold_summary.md")


def launch_worker(fold: int, gpu: int) -> tuple[subprocess.Popen[bytes], object]:
    log_path = LOG_DIR / f"fold{fold}.log"
    log_handle = log_path.open("wb")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    process = subprocess.Popen(
        [PYTHON, "-m", "scripts.run_full_training", "--worker", str(fold), "--log-path", str(log_path)],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=environment,
    )
    return process, log_handle


def run_all() -> None:
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text("fold,stage,status,timestamp,note\n", encoding="utf-8")
    for fold in range(5):
        append_progress(fold, "dataset", "running")
        data_yaml = build_fold_dataset(fold)
        append_progress(fold, "dataset", "completed", str(data_yaml))

    workers = {fold: launch_worker(fold, fold) for fold in range(4)}
    fold0_code = workers[0][0].wait()
    workers[0][1].close()
    if fold0_code != 0:
        for fold in [1, 2, 3]:
            workers[fold][0].wait()
            workers[fold][1].close()
        append_progress(-1, "run", "failed", f"fold0 returncode={fold0_code}")
        raise SystemExit(1)
    workers[4] = launch_worker(4, 0)
    return_codes = [fold0_code]
    for fold in [1, 2, 3, 4]:
        process, log_handle = workers[fold]
        return_codes.append(process.wait())
        log_handle.close()
    if any(return_codes):
        append_progress(-1, "run", "failed", f"returncodes={return_codes}")
        raise SystemExit(1)
    write_summary()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int)
    parser.add_argument("--log-path", type=Path)
    args = parser.parse_args()
    if args.worker is None:
        run_all()
    else:
        run_worker(args.worker, args.log_path)


if __name__ == "__main__":
    main()
