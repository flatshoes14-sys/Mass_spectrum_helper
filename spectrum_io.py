from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from utils import infer_bin_width


@dataclass
class SpectrumData:
    df: pd.DataFrame
    x_col: str
    y_col: str
    bin_width: float

    @property
    def x(self) -> np.ndarray:
        return self.df[self.x_col].to_numpy(dtype=float)

    @property
    def y(self) -> np.ndarray:
        return self.df[self.y_col].to_numpy(dtype=float)


def detect_columns(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    """Detect likely mass/charge and count columns from names + coercible numeric checks."""
    exact_x = "Mass to Charge State Ratio (Da)"
    exact_y = "Corrected Count"
    if exact_x in df.columns and exact_y in df.columns:
        return exact_x, exact_y

    # Allow columns that are currently object/string but mostly numeric after coercion.
    numeric_like_cols: list[str] = []
    for c in df.columns:
        coerced = pd.to_numeric(df[c], errors="coerce")
        if coerced.notna().mean() >= 0.8:
            numeric_like_cols.append(c)
    if len(numeric_like_cols) < 2:
        return None, None

    lowered = {c: c.lower() for c in numeric_like_cols}
    x_candidates = [
        c
        for c in numeric_like_cols
        if any(k in lowered[c] for k in ["mass", "m/z", "mz", "charge state ratio", "da"])
    ]
    y_candidates = [
        c
        for c in numeric_like_cols
        if any(k in lowered[c] for k in ["count", "intensity", "signal"])
    ]

    x_col = x_candidates[0] if x_candidates else numeric_like_cols[0]
    y_col = y_candidates[0] if y_candidates else (
        numeric_like_cols[1] if numeric_like_cols[1] != x_col else numeric_like_cols[0]
    )
    if y_col == x_col and len(numeric_like_cols) > 1:
        y_col = numeric_like_cols[1]
    return x_col, y_col


def load_spectrum_csv(path_or_buf, x_col: Optional[str] = None, y_col: Optional[str] = None) -> SpectrumData:
    df = pd.read_csv(path_or_buf)
    df.columns = [str(c).strip() for c in df.columns]

    auto_x, auto_y = detect_columns(df)
    x_col = x_col or auto_x
    y_col = y_col or auto_y

    if x_col is None or y_col is None:
        raise ValueError("Could not detect x/y columns. Please select columns manually.")

    for c in [x_col, y_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    clean = df[[x_col, y_col]].dropna().sort_values(x_col).reset_index(drop=True)
    bw = infer_bin_width(clean[x_col].to_numpy(dtype=float))
    return SpectrumData(df=clean, x_col=x_col, y_col=y_col, bin_width=bw)
