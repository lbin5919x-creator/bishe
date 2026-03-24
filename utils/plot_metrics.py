"""训练指标可视化脚本。"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# 滑动平均窗口大小（默认10回合）
_SMOOTH_WINDOW = 10


def _smooth(series: pd.Series, window: int = _SMOOTH_WINDOW) -> pd.Series:
    """对序列做滑动平均平滑处理。"""
    return series.rolling(window=window, min_periods=1).mean()


def _plot_with_smooth(ax, x, y, *, label: str, color: str = None, window: int = _SMOOTH_WINDOW):
    """绘制原始数据（半透明）+ 平滑曲线（实线）。"""
    kwargs = {}
    if color:
        kwargs["color"] = color
    ax.plot(x, y, alpha=0.25, **kwargs)
    ax.plot(x, _smooth(y, window), label=label, linewidth=2, **kwargs)


def plot_training_metrics(csv_path: str, output_dir: str = None, window: int = _SMOOTH_WINDOW) -> None:
    """
    绘制训练指标曲线（含滑动平均平滑）。

    参数:
        csv_path: CSV文件路径
        output_dir: 输出目录（默认与CSV同目录）
        window: 滑动平均窗口大小
    """
    df = pd.read_csv(csv_path)

    if output_dir is None:
        output_dir = Path(csv_path).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 奖励曲线
    ax1 = axes[0, 0]
    _plot_with_smooth(ax1, df['episode'], df['reward'], label='总奖励', window=window)
    if 'coordination_bonus' in df.columns:
        _plot_with_smooth(ax1, df['episode'], df['coordination_bonus'],
                          label='协同奖励', color='orange', window=window)
    ax1.set_xlabel('回合')
    ax1.set_ylabel('奖励')
    ax1.set_title('奖励曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 等待时间
    ax2 = axes[0, 1]
    _plot_with_smooth(ax2, df['episode'], df['waiting_time'],
                      label='总等待时间', color='red', window=window)
    ax2.set_xlabel('回合')
    ax2.set_ylabel('等待时间 (秒)')
    ax2.set_title('总等待时间')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 排队长度
    ax3 = axes[1, 0]
    _plot_with_smooth(ax3, df['episode'], df['avg_queue'], label='平均排队', window=window)
    if 'link_queue' in df.columns:
        _plot_with_smooth(ax3, df['episode'], df['link_queue'],
                          label='连接路排队', color='orange', window=window)
    ax3.set_xlabel('回合')
    ax3.set_ylabel('排队长度')
    ax3.set_title('排队长度')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 通行量
    ax4 = axes[1, 1]
    _plot_with_smooth(ax4, df['episode'], df['throughput'],
                      label='通行量', color='green', window=window)
    ax4.set_xlabel('回合')
    ax4.set_ylabel('通行量 (车辆)')
    ax4.set_title('通行量')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    output_path = output_dir / 'training_curves.png'
    plt.savefig(output_path, dpi=150)
    print(f"图表已保存至: {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="绘制训练指标曲线")
    parser.add_argument("--csv", type=str, required=True, help="CSV文件路径")
    parser.add_argument("--output", type=str, default=None, help="输出目录")
    parser.add_argument("--window", type=int, default=_SMOOTH_WINDOW,
                        help=f"滑动平均窗口大小（默认{_SMOOTH_WINDOW}）")
    args = parser.parse_args()

    plot_training_metrics(args.csv, args.output, args.window)


if __name__ == "__main__":
    main()
