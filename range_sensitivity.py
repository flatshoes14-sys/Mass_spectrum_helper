from __future__ import annotations

import pandas as pd

from peak_detection import boundary_from_mode
from utils import integrate_window


MODES = ["FWHM", "FW0.2M", "FW0.1M", "CUSTOM"]


def sensitivity_table(x, y, peak_mz: float, custom_fraction: float = 0.3) -> pd.DataFrame:
    rows = []
    for mode in MODES:
        b = boundary_from_mode(x, y, peak_mz, mode, custom_fraction=custom_fraction)
        area = integrate_window(x, y, b["start_da"], b["end_da"])
        rows.append(
            {
                "mode": mode,
                "start_da": b["start_da"],
                "end_da": b["end_da"],
                "left_da": b["left_da"],
                "right_da": b["right_da"],
                "integrated_count": area,
            }
        )
    out = pd.DataFrame(rows)
    ref = out.loc[out["mode"] == "FWHM", "integrated_count"].iloc[0] if not out.empty else 0.0
    out["delta_vs_fwhm"] = out["integrated_count"] - ref
    return out


def overlap_warning(table: pd.DataFrame, neighbor_distance_da: float = 0.3) -> list[str]:
    warnings = []
    for _, r in table.iterrows():
        if (r["end_da"] - r["start_da"]) > neighbor_distance_da:
            warnings.append(f"{r['mode']} window is wide; may overlap nearby peaks.")
    return warnings


def composition_sensitivity_for_peak(
    x,
    y,
    peak_mz: float,
    assignments_df: pd.DataFrame,
    selected_species: str,
    custom_fraction: float = 0.3,
) -> pd.DataFrame:
    """Recompute composition while varying the selected species window by mode."""
    if assignments_df.empty or selected_species not in assignments_df["species"].values:
        return pd.DataFrame()

    rows: list[dict] = []
    for mode in MODES:
        b = boundary_from_mode(x, y, peak_mz, mode, custom_fraction=custom_fraction)
        work = assignments_df.copy()
        sel_mask = work["species"] == selected_species
        work.loc[sel_mask, "start_da"] = b["start_da"]
        work.loc[sel_mask, "end_da"] = b["end_da"]
        work["area"] = work.apply(
            lambda r: integrate_window(x, y, float(r["start_da"]), float(r["end_da"])),
            axis=1,
        )
        total = work["area"].abs().sum()
        work["at_frac"] = work["area"].abs() / total if total > 0 else 0.0
        for _, r in work.iterrows():
            rows.append(
                {
                    "mode": mode,
                    "species": r["species"],
                    "area": r["area"],
                    "at_frac": r["at_frac"],
                }
            )
    return pd.DataFrame(rows)
