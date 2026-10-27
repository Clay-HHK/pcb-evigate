#!/usr/bin/env python3
"""按模板组生成 DeepPCB GroupKFold 五折划分。"""

import csv
import json
import random
from pathlib import Path

from sklearn.model_selection import GroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/deeppcb.csv"
OUTPUT_PATH = PROJECT_ROOT / "data/splits/groupkfold5.json"
SEED = 20260811
CLASS_FIELDS = ("n_open", "n_short", "n_mousebite", "n_spur", "n_copper", "n_pinhole")


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as manifest_file:
        records = list(csv.DictReader(manifest_file))
    return sorted(records, key=lambda record: record["pair_id"])


def group_names(records: list[dict[str, str]], indices: list[int]) -> list[str]:
    return sorted({records[index]["group"] for index in indices})


def class_totals(records: list[dict[str, str]], indices: list[int]) -> dict[str, int]:
    return {
        field: sum(int(records[index][field]) for index in indices)
        for field in CLASS_FIELDS
    }


def print_fold_report(
    fold_index: int,
    test_indices: list[int],
    val_indices: list[int],
    train_indices: list[int],
    records: list[dict[str, str]],
) -> None:
    total_pairs = len(records)
    totals = class_totals(records, test_indices)
    print(f"第 {fold_index} 折")
    print(f"测试组：{', '.join(group_names(records, test_indices))}")
    print(f"验证组：{', '.join(group_names(records, val_indices))}")
    print(f"训练组：{', '.join(group_names(records, train_indices))}")
    print(f"测试图像对数：{len(test_indices)}，占比：{len(test_indices) / total_pairs:.2%}")
    print(
        "测试六类缺陷框数："
        + "，".join(f"{field[2:]} {totals[field]}" for field in CLASS_FIELDS)
    )


def main() -> None:
    random.seed(SEED)
    records = load_manifest()
    groups = [record["group"] for record in records]
    splitter = GroupKFold(n_splits=5)
    fold_indices = [test_indices.tolist() for _, test_indices in splitter.split(records, groups=groups)]
    fold_indices = [sorted(indices) for indices in fold_indices]

    split_data = {}
    all_indices = set(range(len(records)))
    for fold_index, test_indices in enumerate(fold_indices):
        val_indices = fold_indices[(fold_index + 1) % 5]
        train_indices = sorted(all_indices - set(test_indices) - set(val_indices))
        split_data[f"fold_{fold_index}"] = {
            "test": group_names(records, test_indices),
            "val": group_names(records, val_indices),
            "train": group_names(records, train_indices),
        }

    OUTPUT_PATH.write_text(
        json.dumps(split_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"已写入划分：{OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    for fold_index, test_indices in enumerate(fold_indices):
        val_indices = fold_indices[(fold_index + 1) % 5]
        train_indices = sorted(all_indices - set(test_indices) - set(val_indices))
        print_fold_report(fold_index, test_indices, val_indices, train_indices, records)


if __name__ == "__main__":
    main()
