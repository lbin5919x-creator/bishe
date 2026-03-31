"""一键训练所有场景的DQN模型（统一随机种子与设备，便于论文实验可复现）。"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

from config.settings import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="顺序训练丁字、十字、级联三场景")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Python 与 SUMO 共用随机种子（论文实验请固定并记录）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="计算设备，默认自动选择（有 CUDA 则用 GPU）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device if args.device else get_device()
    seed_s = str(args.seed)

    print(f"统一配置: seed={args.seed}, device={device}")

    tasks: list[dict] = [
        {
            "name": "丁字路口 (t_intersection)",
            "cmd": [
                sys.executable,
                "-m",
                "training.train_dqn",
                "--scenario",
                "t_intersection",
                "--episodes",
                "200",
                "--double-dqn",
                "--dueling",
                "--seed",
                seed_s,
                "--device",
                device,
            ],
        },
        {
            "name": "十字路口 (x_intersection)",
            "cmd": [
                sys.executable,
                "-m",
                "training.train_dqn",
                "--scenario",
                "x_intersection",
                "--episodes",
                "200",
                "--double-dqn",
                "--dueling",
                "--seed",
                seed_s,
                "--device",
                device,
            ],
        },
        {
            "name": "级联双路口 (cascaded_intersection)",
            "cmd": [
                sys.executable,
                "-m",
                "training.train_cascaded",
                "--scenario",
                "cascaded_intersection",
                "--episodes",
                "300",
                "--seed",
                seed_s,
                "--device",
                device,
            ],
        },
    ]

    print(f"\n{'=' * 60}")
    print(f"  一键训练所有场景 (共 {len(tasks)} 个)")
    print(f"{'=' * 60}\n")

    results: list[tuple[str, bool, float]] = []

    for i, task in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] 开始训练: {task['name']}")
        print(f"  命令: {' '.join(task['cmd'])}")
        print("-" * 60)

        start = time.time()
        ret = subprocess.run(task["cmd"])
        elapsed = time.time() - start
        ok = ret.returncode == 0
        results.append((task["name"], ok, elapsed))

        status = "完成" if ok else "失败"
        print(f"\n  [{status}] {task['name']}  耗时: {elapsed / 60:.1f} 分钟\n")

    print(f"\n{'=' * 60}")
    print("  训练汇总")
    print(f"{'=' * 60}")
    total_time = 0.0
    for name, ok, elapsed in results:
        status = "OK" if ok else "FAIL"
        print(f"  [{status:>4s}] {name:<40s} {elapsed / 60:>6.1f} 分钟")
        total_time += elapsed
    print(f"  {'':->56s}")
    print(f"  {'总耗时':<44s} {total_time / 60:>6.1f} 分钟")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
    sys.exit(0)
