"""评估指标跟踪和持久化工具。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import pandas as pd


@dataclass
class MetricsRecorder:
    """
    指标记录器。
    
    用于收集训练/评估过程中的指标并保存到CSV文件。
    """
    output_dir: Path
    records: List[Dict[str, float]] = field(default_factory=list)

    def log(self, metrics: Dict[str, float]) -> None:
        """
        记录一条指标。
        
        参数:
            metrics: 指标字典
        """
        self.records.append(metrics)

    def flush(self, filename: str) -> Path:
        """
        将记录的指标保存到CSV文件。
        
        参数:
            filename: 文件名
            
        返回:
            保存的文件路径
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.records)
        file_path = self.output_dir / filename
        df.to_csv(file_path, index=False)
        return file_path

    def reset(self) -> None:
        """清空记录。"""
        self.records.clear()
