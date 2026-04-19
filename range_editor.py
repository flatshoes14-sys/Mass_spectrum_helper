from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = ["species", "formula", "charge", "start_da", "end_da", "notes"]


def empty_ranges() -> pd.DataFrame:
    return pd.DataFrame(columns=REQUIRED_COLUMNS)


def validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in REQUIRED_COLUMNS:
        if c not in out.columns:
            out[c] = "" if c in ["species", "formula", "notes"] else 0
    out["charge"] = pd.to_numeric(out["charge"], errors="coerce").fillna(1).astype(int)
    out["start_da"] = pd.to_numeric(out["start_da"], errors="coerce")
    out["end_da"] = pd.to_numeric(out["end_da"], errors="coerce")
    out = out.dropna(subset=["start_da", "end_da"]).reset_index(drop=True)
    return out[REQUIRED_COLUMNS]


def save_ranges_json(df: pd.DataFrame, path: str | Path) -> None:
    valid = validate_ranges(df)
    Path(path).write_text(json.dumps(valid.to_dict(orient="records"), indent=2), encoding="utf-8")


def load_ranges_json(path: str | Path) -> pd.DataFrame:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_ranges(pd.DataFrame(data))


def save_ranges_csv(df: pd.DataFrame, path: str | Path) -> None:
    validate_ranges(df).to_csv(path, index=False)


def load_ranges_csv(path_or_buf) -> pd.DataFrame:
    return validate_ranges(pd.read_csv(path_or_buf))


def parse_rrng_like(text: str) -> pd.DataFrame:
    """Very simple optional rrng-like import: start,end,species,formula,charge,notes"""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        start, end, species, formula, charge = parts[:5]
        notes = parts[5] if len(parts) > 5 else ""
        rows.append(
            {
                "species": species,
                "formula": formula,
                "charge": int(charge),
                "start_da": float(start),
                "end_da": float(end),
                "notes": notes,
            }
        )
    return validate_ranges(pd.DataFrame(rows))


def export_rrng_like(df: pd.DataFrame) -> str:
    valid = validate_ranges(df)
    lines = ["# start_da,end_da,species,formula,charge,notes"]
    for _, r in valid.iterrows():
        lines.append(f"{r['start_da']},{r['end_da']},{r['species']},{r['formula']},{r['charge']},{r['notes']}")
    return "\n".join(lines)
