from __future__ import annotations

import pandas as pd

from isotope_check import check_isotopic_pattern


def resolve_overlaps(x, y, peak_mz: float, candidate_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df

    ranked = []
    for _, row in candidate_df.iterrows():
        iso = check_isotopic_pattern(x, y, peak_mz, row["formula"], int(row["charge"]))
        consistency = iso["score"]
        final_score = 0.7 * float(row["plausibility"]) + 0.3 * consistency
        ranked.append(
            {
                **row.to_dict(),
                "isotope_consistency": consistency,
                "overlap_rank_score": final_score,
                "uncertainty_note": "Ambiguous region: use neighboring peaks and domain knowledge.",
            }
        )

    return pd.DataFrame(ranked).sort_values("overlap_rank_score", ascending=False).reset_index(drop=True)
