"""EpisodeLogger: CSV-backed result logger. Canonical location: vla_manipulation/evaluation/logger.py"""
from __future__ import annotations
import csv
import os
from vla_manipulation.common.types import EpisodeResult

CSV_FIELDS = [
    'condition', 'seed', 'success', 'episode_length',
    'ee_x_initial', 'failure_type', 'policy', 'timestamp',
]

class EpisodeLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        write_header = not os.path.exists(path)
        self._f = open(path, 'a', newline='', buffering=1)
        self._writer = csv.DictWriter(self._f, fieldnames=CSV_FIELDS)
        if write_header:
            self._writer.writeheader()

    def log(self, result: EpisodeResult) -> None:
        self._writer.writerow({
            'condition':      result.condition,
            'seed':           result.seed,
            'success':        result.success,
            'episode_length': result.episode_length,
            'ee_x_initial':   round(result.ee_x_initial, 4),
            'failure_type':   result.failure_type,
            'policy':         result.policy,
            'timestamp':      result.timestamp,
        })
        self._f.flush()

    def already_done(self, condition: str, seed: int, policy: str) -> bool:
        if not os.path.exists(self.path):
            return False
        try:
            import pandas as pd
            df = pd.read_csv(self.path)
            mask = (
                (df['condition'] == condition) &
                (df['seed'] == seed) &
                (df['policy'] == policy) &
                df['success'].notna()
            )
            return bool(mask.any())
        except Exception:
            return False

    def __del__(self):
        try:
            self._f.close()
        except Exception:
            pass
