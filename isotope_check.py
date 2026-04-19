from __future__ import annotations

import numpy as np
import pandas as pd

from candidates import expected_isotope_spacings
from utils import nearest_index


def check_isotopic_pattern(
    x: np.ndarray,
    y: np.ndarray,
    peak_mz: float,
    formula: str,
    charge: int,
    search_tol_da: float = 0.08,
) -> dict:
    i0 = nearest_index(x, peak_mz)
    base_int = max(float(y[i0]), 1e-12)
    companions = expected_isotope_spacings(formula, charge)

    rows = []
    agreement = []
    for delta, rel_ab in companions:
        target = peak_mz + delta
        j = nearest_index(x, target)
        if abs(x[j] - target) > search_tol_da:
            rows.append({"target_mz": target, "observed_mz": np.nan, "expected_rel": rel_ab, "observed_rel": np.nan, "status": "not found"})
            agreement.append(0.0)
            continue

        obs_rel = float(max(y[j], 0.0) / base_int)
        ratio_diff = abs(obs_rel - rel_ab)
        score = max(0.0, 1.0 - ratio_diff / max(rel_ab, 0.1))
        agreement.append(score)
        rows.append({
            "target_mz": target,
            "observed_mz": float(x[j]),
            "expected_rel": float(rel_ab),
            "observed_rel": obs_rel,
            "status": "matched" if score > 0.6 else "weak",
        })

    mean_score = float(np.mean(agreement)) if agreement else 0.3
    if mean_score >= 0.7:
        label = "good agreement"
    elif mean_score >= 0.4:
        label = "partial agreement"
    else:
        label = "poor agreement"

    return {
        "score": mean_score,
        "label": label,
        "table": pd.DataFrame(rows),
        "notes": "Conservative heuristic check; interpret with operator judgment.",
    }
