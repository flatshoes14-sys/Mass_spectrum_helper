from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class WarningMessage:
    level: str
    message: str


def infer_bin_width(x: np.ndarray) -> float:
    """Infer bin width using the median positive spacing (robust to slight irregularity)."""
    if len(x) < 2:
        return 0.0
    diffs = np.diff(np.sort(np.asarray(x, dtype=float)))
    positive = diffs[diffs > 0]
    if len(positive) == 0:
        return 0.0
    return float(np.median(positive))


def robust_local_baseline(y: np.ndarray, window: int = 31) -> np.ndarray:
    """Simple transparent baseline: rolling quantile with edge handling."""
    s = pd.Series(y.astype(float))
    window = max(5, int(window) | 1)  # odd and >=5
    base = s.rolling(window=window, center=True, min_periods=1).quantile(0.2)
    return base.to_numpy()


def safe_normalize(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    ymax = np.nanmax(y) if len(y) else 0.0
    if not np.isfinite(ymax) or ymax == 0:
        return np.zeros_like(y)
    return y / ymax


def nearest_index(x: np.ndarray, target: float) -> int:
    if len(x) == 0:
        raise ValueError("Empty axis")
    return int(np.argmin(np.abs(x - target)))


def integrate_window(x: np.ndarray, y: np.ndarray, left: float, right: float) -> float:
    """Integrate corrected counts in [left, right] by trapezoidal rule."""
    if right <= left:
        return 0.0
    mask = (x >= left) & (x <= right)
    if not np.any(mask):
        return 0.0
    xw = x[mask]
    yw = y[mask]
    if len(xw) == 1:
        return float(yw[0])
    return float(np.trapz(yw, xw))


def parse_formula_elements(formula: str) -> dict[str, int]:
    """Very simple parser for formulas like Fe, Fe2O, C2H3.

    Keeps this tool transparent and conservative.
    """
    import re

    pattern = r"([A-Z][a-z]?)(\d*)"
    out: dict[str, int] = {}
    for elem, n in re.findall(pattern, formula):
        out[elem] = out.get(elem, 0) + (int(n) if n else 1)
    return out


def composition_from_ranges(assignments: Iterable[dict], x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    rows = []
    for item in assignments:
        area = integrate_window(x, y, float(item["start_da"]), float(item["end_da"]))
        rows.append(
            {
                "species": item.get("species", "unknown"),
                "formula": item.get("formula", ""),
                "charge": int(item.get("charge", 1)),
                "area": area,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["abs_area"] = df["area"].abs()
    total = df["abs_area"].sum()
    df["at_frac"] = (df["abs_area"] / total) if total > 0 else 0.0
    return df


def check_overlaps(ranges_df: pd.DataFrame) -> list[Tuple[int, int]]:
    overlaps: list[Tuple[int, int]] = []
    if ranges_df.empty:
        return overlaps
    sorted_df = ranges_df.sort_values("start_da").reset_index()
    for i in range(len(sorted_df) - 1):
        a = sorted_df.iloc[i]
        b = sorted_df.iloc[i + 1]
        if a["end_da"] > b["start_da"]:
            overlaps.append((int(a["index"]), int(b["index"])))
    return overlaps
