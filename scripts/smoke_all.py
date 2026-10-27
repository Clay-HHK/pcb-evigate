"""Run all four smoke experiments sequentially and print a Chinese summary table."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = [
    ("RQ1 证据消融", ROOT / "experiments/rq1_evidence/run.py", ROOT / "experiments/rq1_evidence/smoke_results.json"),
    ("RQ2 端到端纠错", ROOT / "experiments/rq2_correction/run.py", ROOT / "experiments/rq2_correction/smoke_results.json"),
    ("RQ3 双门控", ROOT / "experiments/rq3_gating/run.py", ROOT / "experiments/rq3_gating/smoke_results.json"),
    ("RQ4 鲁棒性", ROOT / "experiments/rq4_robustness/run.py", ROOT / "experiments/rq4_robustness/smoke_results.json"),
]


def main() -> None:
    rows = []
    for name, entry, output in EXPERIMENTS:
        started = time.perf_counter()
        print(f"\n开始 {name}: {entry.relative_to(ROOT)}", flush=True)
        subprocess.run([sys.executable, str(entry), "--smoke"], cwd=ROOT, check=True)
        elapsed = time.perf_counter() - started
        payload = json.loads(output.read_text(encoding="utf-8"))
        passed = bool(payload["passed"])
        rows.append((name, elapsed, output.relative_to(ROOT), passed))
        print(f"完成 {name}，耗时 {elapsed:.2f} 秒，产出 {output.relative_to(ROOT)}，通过 {passed}", flush=True)
    print("\n烟雾实验总表")
    print("| 实验 | 耗时（秒） | 产出文件 | 状态 |")
    print("| --- | ---: | --- | --- |")
    for name, elapsed, output, passed in rows:
        print(f"| {name} | {elapsed:.2f} | {output} | {'通过' if passed else '失败'} |")
    rq2 = json.loads((ROOT / "experiments/rq2_correction/smoke_results.json").read_text(encoding="utf-8"))
    print("\n动作路径总表")
    print("| keep | remove | relabel | add | reject | defer | 接受动作 | 正效用接受动作 |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    counts = rq2["action_counts"]
    print(
        f"| {counts['keep']} | {counts['remove']} | {counts['relabel']} | "
        f"{counts['add']} | {counts['reject']} | {counts['defer']} | "
        f"{rq2['accepted_actions']} | {rq2['accepted_positive_utility']} |"
    )
    for reason in rq2["unreachable_action_reasons"].values():
        print(f"不可达说明：{reason}")
    print(f"\n总结果：{'全部通过' if all(row[3] for row in rows) else '存在失败'}")


if __name__ == "__main__":
    main()
