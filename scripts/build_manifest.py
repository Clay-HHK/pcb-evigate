#!/usr/bin/env python3
"""生成 DeepPCB 图像对 manifest。"""

import csv
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data/raw/DeepPCB/PCBData"
OUTPUT_PATH = PROJECT_ROOT / "data/manifests/deeppcb.csv"
SEED = 20260811
CLASS_NAMES = (
    (1, "open"),
    (2, "short"),
    (3, "mousebite"),
    (4, "spur"),
    (5, "copper"),
    (6, "pinhole"),
)


def relative_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def read_records() -> list[dict[str, str | int]]:
    records = []
    for official_split in ("trainval", "test"):
        split_path = DATA_ROOT / f"{official_split}.txt"
        for line in split_path.read_text().splitlines():
            image_rel, annotation_rel = line.split()
            listed_image_path = DATA_ROOT / image_rel
            pair_id = listed_image_path.stem
            group = listed_image_path.parts[-3]
            template_id = listed_image_path.parts[-2]
            test_path = listed_image_path.with_name(f"{pair_id}_test.jpg")
            temp_path = listed_image_path.with_name(f"{pair_id}_temp.jpg")
            anno_path = DATA_ROOT / annotation_rel
            class_counts = {name: 0 for _, name in CLASS_NAMES}
            n_defects = 0
            for annotation_line in anno_path.read_text().splitlines():
                if not annotation_line.strip():
                    continue
                _, _, _, _, class_id = map(int, annotation_line.split())
                class_counts[dict(CLASS_NAMES)[class_id]] += 1
                n_defects += 1
            records.append(
                {
                    "pair_id": pair_id,
                    "group": group,
                    "template_id": template_id,
                    "test_path": relative_path(test_path),
                    "temp_path": relative_path(temp_path),
                    "anno_path": relative_path(anno_path),
                    "n_defects": n_defects,
                    "official_split": official_split,
                    **{f"n_{name}": count for name, count in class_counts.items()},
                }
            )
    return sorted(records, key=lambda record: record["pair_id"])


def main() -> None:
    random.seed(SEED)
    records = read_records()
    manifest_fields = [
        "pair_id",
        "group",
        "template_id",
        "test_path",
        "temp_path",
        "anno_path",
        "n_defects",
        "official_split",
        "n_open",
        "n_short",
        "n_mousebite",
        "n_spur",
        "n_copper",
        "n_pinhole",
    ]
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(records)

    listed_temp_paths = {PROJECT_ROOT / record["temp_path"] for record in records}
    all_temp_paths = set(DATA_ROOT.rglob("*_temp.jpg"))
    excluded_temp_paths = sorted(all_temp_paths - listed_temp_paths)
    print(f"已写入 manifest：{relative_path(OUTPUT_PATH)}")
    print(f"图像对数量：{len(records)}")
    print(f"模板组数量：{len({record['group'] for record in records})}")
    print(f"已排除未配对模板图数量：{len(excluded_temp_paths)}")
    for path in excluded_temp_paths:
        print(f"已排除：{relative_path(path)}")


if __name__ == "__main__":
    main()
