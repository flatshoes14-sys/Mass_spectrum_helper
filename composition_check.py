from __future__ import annotations

import pandas as pd

from utils import composition_from_ranges, parse_formula_elements


def run_composition_checks(assignments: list[dict], x, y) -> dict:
    comp = composition_from_ranges(assignments, x, y)
    warnings: list[str] = []

    if comp.empty:
        return {"composition": comp, "warnings": ["No assignments loaded."]}

    total = comp["at_frac"].sum()
    if abs(total - 1.0) > 0.02:
        warnings.append(f"Composition sum is {total:.3f} (expected ~1.0).")

    if (comp["area"] < 0).any():
        warnings.append("Some integrated corrected counts are negative; inspect baseline/range selection.")

    # Basic isotope imbalance check (formula-level crude check)
    for _, row in comp.iterrows():
        elems = parse_formula_elements(row["formula"])
        if any(v > 3 for v in elems.values()):
            warnings.append(f"Complex formula {row['formula']} may need manual isotopic verification.")

    major = comp.sort_values("at_frac", ascending=False).head(3)
    if major["at_frac"].max() < 0.15:
        warnings.append("No clearly dominant species; assignments may be diffuse or incomplete.")

    return {"composition": comp.sort_values("at_frac", ascending=False), "warnings": warnings}
