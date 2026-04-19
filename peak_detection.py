from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from utils import nearest_index, robust_local_baseline


@dataclass
class Peak:
    index: int
    mz: float
    height: float
    prominence: float


def detect_peaks(
    x: np.ndarray,
    y: np.ndarray,
    prominence: float | None = None,
    distance_bins: int = 3,
) -> pd.DataFrame:
    baseline = robust_local_baseline(y)
    y_corr = y - baseline
    if prominence is None:
        scale = np.nanstd(y_corr) if len(y_corr) else 1.0
        prominence = max(scale * 1.5, 1e-9)

    idx, props = find_peaks(y_corr, prominence=prominence, distance=distance_bins)
    out = pd.DataFrame(
        {
            "index": idx,
            "mz": x[idx],
            "height": y_corr[idx],
            "raw_height": y[idx],
            "prominence": props.get("prominences", np.zeros(len(idx))),
        }
    )
    return out.sort_values("mz").reset_index(drop=True)


def estimate_asymmetric_width(
    x: np.ndarray,
    y: np.ndarray,
    peak_mz: float,
    fraction: float = 0.5,
    max_search_da: float = 3.0,
) -> dict:
    """Estimate separate left/right width at chosen fraction of local peak max."""
    i0 = nearest_index(x, peak_mz)
    y0 = y[i0]
    thresh = y0 * fraction

    left_i = i0
    while left_i > 0 and x[i0] - x[left_i] <= max_search_da and y[left_i] > thresh:
        left_i -= 1

    right_i = i0
    while right_i < len(x) - 1 and x[right_i] - x[i0] <= max_search_da and y[right_i] > thresh:
        right_i += 1

    left_da = float(max(0.0, x[i0] - x[left_i]))
    right_da = float(max(0.0, x[right_i] - x[i0]))
    return {
        "peak_mz": float(x[i0]),
        "fraction": fraction,
        "left_da": left_da,
        "right_da": right_da,
        "start_da": float(x[i0] - left_da),
        "end_da": float(x[i0] + right_da),
    }


def boundary_from_mode(x: np.ndarray, y: np.ndarray, peak_mz: float, mode: str, custom_fraction: float = 0.3) -> dict:
    mode_map = {
        "FWHM": 0.5,
        "FW0.2M": 0.2,
        "FW0.1M": 0.1,
        "CUSTOM": custom_fraction,
    }
    frac = mode_map.get(mode.upper(), 0.5)
    return estimate_asymmetric_width(x, y, peak_mz, fraction=frac)
