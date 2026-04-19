import numpy as np
import pandas as pd

from isotope_check import check_isotopic_pattern
from range_editor import load_ranges_csv, save_ranges_csv


def test_simple_isotope_logic():
    x = np.array([55.9, 56.9, 57.9])
    y = np.array([100.0, 20.0, 5.0])
    out = check_isotopic_pattern(x, y, 55.9, "Fe", 1, search_tol_da=0.2)
    assert out["label"] in {"good agreement", "partial agreement", "poor agreement"}
    assert "table" in out


def test_range_file_save_load(tmp_path):
    df = pd.DataFrame(
        [
            {"species": "Fe+", "formula": "Fe", "charge": 1, "start_da": 55.8, "end_da": 56.1, "notes": "main"}
        ]
    )
    p = tmp_path / "ranges.csv"
    save_ranges_csv(df, p)
    loaded = load_ranges_csv(p)
    assert list(loaded.columns) == ["species", "formula", "charge", "start_da", "end_da", "notes"]
    assert loaded.iloc[0]["species"] == "Fe+"
