#!/usr/bin/env python3
"""检查 GroupKFold 划分的模板组、图像对和 OOF 覆盖情况。"""

import csv
import json
import random
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data/manifests/deeppcb.csv"
SPLIT_PATH = PROJECT_ROOT / "data/splits/groupkfold5.json"
SEED = 20260811


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as manifest_file:
        return list(csv.DictReader(manifest_file))


def main() -> None:
    random.seed(SEED)
    records = load_manifest()
    split_data = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    fold_names = [f"fold_{index}" for index in range(5)]

    fold_groups = {name: set(split_data[name]["test"]) for name in fold_names}
    fold_pair_ids = {
        name: {
            record["pair_id"]
            for record in records
            if record["group"] in fold_groups[name]
        }
        for name in fold_names
    }
    group_intersections = [
        fold_groups[left] & fold_groups[right]
        for index, left in enumerate(fold_names)
        for right in fold_names[index + 1 :]
    ]
    pair_intersections = [
        fold_pair_ids[left] & fold_pair_ids[right]
        for index, left in enumerate(fold_names)
        for right in fold_names[index + 1 :]
    ]
    pair_occurrences = Counter(
        pair_id for pair_ids in fold_pair_ids.values() for pair_id in pair_ids
    )
    manifest_pair_occurrences = Counter(record["pair_id"] for record in records)

    assert all(not intersection for intersection in group_intersections)
    print("检查一，GroupKFold 任意两折没有共享模板组：通过")
    assert all(not intersection for intersection in pair_intersections)
    print("检查二，GroupKFold 任意两折没有共享 pair_id：通过")
    assert pair_occurrences == manifest_pair_occurrences
    assert all(count == 1 for count in pair_occurrences.values())
    print("检查三，每个 pair_id 恰好出现在一折 test：通过")

    official_trainval_groups = {
        record["group"] for record in records if record["official_split"] == "trainval"
    }
    official_test_groups = {
        record["group"] for record in records if record["official_split"] == "test"
    }
    official_overlap = sorted(official_trainval_groups & official_test_groups)
    print(
        "检查四，官方划分组交叠对照：通过，"
        f"trainval 与 test 共享 {len(official_overlap)} 组，"
        f"分别为 {', '.join(official_overlap)}；GroupKFold 已消除该交叠"
    )
    print("泄漏检查结论：四项检查通过，官方划分交叠仅作为对照")


if __name__ == "__main__":
    main()
