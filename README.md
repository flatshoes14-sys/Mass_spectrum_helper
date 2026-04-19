# APT Mass Spectrum Ranging Assistant (Local Streamlit App)

A practical local Python tool to assist **atom probe tomography (APT)** mass spectrum ranging.

> This software is an assistance tool for consistency and efficiency. It does **not** replace expert scientific judgment.

## Features implemented

- Spectrum CSV loader/viewer with automatic x/y column detection and optional manual override.
- Peak candidate generator (elements, isotopes, charge states, simple molecular ions).
- Isotopic pattern checker with qualitative agreement labels.
- Overlapping peak resolver with ranked candidate list + uncertainty notes.
- Range boundary suggester and sensitivity table (FWHM, FW0.2M, FW0.1M, custom).
- Composition sanity checker.
- Charge-state-ratio helper.
- Batch spectrum comparison.
- Range file parser/editor (CSV + JSON internal format, optional simple rrng-like text import/export).

## Input CSV format

Expected at minimum two numeric columns:
- **x-axis**: mass-to-charge in Da (m/z)
- **y-axis**: corrected count

Example:

```csv
Mass to Charge State Ratio (Da), Corrected Count
0,-265.2145
0.12,-142.12167
0.24,-17.607536
0.35999998,21.758339
```

Notes:
- Corrected counts may be floating-point values.
- Corrected counts may be negative in some regions; this is supported.
- Bin width is inferred from data using median positive spacing and handles slightly irregular spacing.

## Miniforge (Windows) setup

In **Miniforge Prompt**:

```bash
cd path\to\Mass_spectrum_helper
conda env create -f environment.yml
conda activate apt-ranging-assistant
streamlit run app.py
```

Alternative using pip inside an active conda env:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

- `app.py`
- `spectrum_io.py`
- `peak_detection.py`
- `candidates.py`
- `isotope_check.py`
- `overlap_resolver.py`
- `range_sensitivity.py`
- `composition_check.py`
- `charge_state.py`
- `batch_compare.py`
- `range_editor.py`
- `utils.py`
- `requirements.txt`
- `environment.yml`
- `sample_data/`
- `tests/`

## Testing

Run:

```bash
pytest -q
```

Current tests cover:
- CSV loading
- column detection
- bin-width inference
- peak width calculation
- range integration
- simple isotope check logic
- range file save/load

## Scientific transparency and caveats

- Heuristic scores are intentionally conservative.
- Peak shapes are not assumed to be perfectly Gaussian.
- Background handling is a simple transparent local baseline estimate.
- Overlap resolution is suggestive, not definitive.
- Always validate final ranges with domain expertise and full experiment context.

## Troubleshooting

- If you see `AttributeError: module 'numpy' has no attribute 'trapz'`, update to the latest code in this branch.
  The integration helper is NumPy 2.x compatible and uses `np.trapezoid` with an older-version fallback.
