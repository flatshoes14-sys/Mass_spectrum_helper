from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from spectrum_io import load_spectrum_csv
from utils import integrate_window, safe_normalize


@dataclass
class BatchSpectrum:
    name: str
    x: np.ndarray
    y: np.ndarray


def load_batch(files) -> list[BatchSpectrum]:
    out = []
    for f in files:
        spec = load_spectrum_csv(f)
        out.append(BatchSpectrum(name=getattr(f, "name", "spectrum"), x=spec.x, y=spec.y))
    return out


def compare_ranges(batch: list[BatchSpectrum], ranges_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b in batch:
        for _, r in ranges_df.iterrows():
            area = integrate_window(b.x, b.y, float(r["start_da"]), float(r["end_da"]))
            rows.append(
                {
                    "file": b.name,
                    "species": r["species"],
                    "charge": r.get("charge", 1),
                    "start_da": r["start_da"],
                    "end_da": r["end_da"],
                    "area": area,
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["abs_area"] = df["area"].abs()
    return df


def overlay_table(batch: list[BatchSpectrum]) -> pd.DataFrame:
    rows = []
    for b in batch:
        rows.append(
            {
                "file": b.name,
                "mz_min": float(np.min(b.x)),
                "mz_max": float(np.max(b.x)),
                "n_points": len(b.x),
                "peak_norm_max": float(np.max(safe_normalize(b.y))),
            }
        )
    return pd.DataFrame(rows)
