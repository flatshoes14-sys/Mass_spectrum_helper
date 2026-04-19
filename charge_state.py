from __future__ import annotations

import pandas as pd


def charge_state_ratios(assignments_df: pd.DataFrame, composition_df: pd.DataFrame, element: str) -> pd.DataFrame:
    if assignments_df.empty or composition_df.empty:
        return pd.DataFrame()

    merged = assignments_df.merge(
        composition_df[["species", "area"]], how="left", left_on="species", right_on="species"
    )
    subset = merged[merged["formula"].str.contains(element, na=False)].copy()
    if subset.empty:
        return pd.DataFrame()

    grp = subset.groupby("charge", as_index=False)["area"].sum()
    total = grp["area"].abs().sum()
    grp["ratio"] = grp["area"].abs() / total if total > 0 else 0.0
    grp["element"] = element
    return grp.sort_values("charge")
