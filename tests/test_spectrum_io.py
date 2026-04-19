import io

from spectrum_io import detect_columns, load_spectrum_csv


def test_csv_loading_and_column_detection():
    text = "Mass to Charge State Ratio (Da),Corrected Count\n0,-1\n0.1,2\n"
    dfobj = io.StringIO(text)
    spec = load_spectrum_csv(dfobj)
    assert spec.x_col == "Mass to Charge State Ratio (Da)"
    assert spec.y_col == "Corrected Count"
    assert len(spec.x) == 2


def test_detect_columns_fallback_numeric():
    import pandas as pd

    df = pd.DataFrame({"a": [0, 1], "b": [2, 3]})
    x, y = detect_columns(df)
    assert x == "a"
    assert y == "b"


def test_exact_named_columns_with_string_values():
    text = (
        "Mass to Charge State Ratio (Da),Corrected Count,Other\n"
        "0.00,-1.0,a\n"
        "0.10,2.5,b\n"
    )
    spec = load_spectrum_csv(io.StringIO(text))
    assert spec.x_col == "Mass to Charge State Ratio (Da)"
    assert spec.y_col == "Corrected Count"
    assert spec.bin_width == 0.1
