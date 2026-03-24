"""一键训练所有场景的DQN模型。"""
from __future__ import annotations

import subprocess
import sys
import time


TASKS = [
    {
        "name": "丁字路口 (t_intersection)",
        "cmd": [
            sys.executable, "-m", "training.train_dqn",
            "--scenario", "t_intersection",
            "--episodes", "200",
            "--double-dqn", "--dueling",
        ],
    },
    {
        "name": "十字路口 (x_intersection)",
        "cmd": [
            sys.executable, "-m", "training.train_dqn",
            "--scenario", "x_intersection",
            "--episodes", "200",
            "--double-dqn", "--dueling",
        ],
    },
    {
        "name": "级联双路口 (cascaded_intersection)",
        "cmd": [
            sys.executable, "-m", "training.train_cascaded",
            "--scenario", "cascaded_intersection",
            "--episodes", "300",
        ],
    },
]


def main() -> None:
    print(f"\n{'='*60}")
    print(f"  一键训练所有场景 (共 {len(TASKS)} 个)")
    print(f"{'='*60}\n")

    results: list[tuple[str, bool, float]] = []

    for i, task in enumerate(TASKS, 1):
        print(f"[{i}/{len(TASKS)}] 开始训练: {task['name']}")
        print(f"  命令: {' '.join(task['cmd'])}")
        print("-" * 60)

        start = time.time()
        ret = subprocess.run(task["cmd"])
        elapsed = time.time() - start
        ok = ret.returncode == 0
        results.append((task["name"], ok, elapsed))

        status = "完成" if ok else "失败"
        print(f"\n  [{status}] {task['name']}  耗时: {elapsed/60:.1f} 分钟\n")

    # 汇总
    print(f"\n{'='*60}")
    print(f"  训练汇总")
    print(f"{'='*60}")
    total_time = 0.0
    for name, ok, elapsed in results:
        status = "OK" if ok else "FAIL"
        print(f"  [{status:>4s}] {name:<40s} {elapsed/60:>6.1f} 分钟")
        total_time += elapsed
    print(f"  {'':->56s}")
    print(f"  {'总耗时':<44s} {total_time/60:>6.1f} 分钟")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
