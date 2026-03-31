"""训练运行元数据写入，便于论文实验可复现与追溯。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_training_run_meta(path: Path, fields: dict[str, Any]) -> None:
    """
    将本次训练的关键超参与环境信息写入 JSON。

    参数:
        path: 输出文件路径，如 outputs/t_intersection/dqn/training_run_meta.json
        fields: 需记录的键值对（由调用方传入）
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        **fields,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

