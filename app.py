from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from batch_compare import compare_ranges, load_batch, overlay_table
from candidates import generate_candidates
from charge_state import charge_state_ratios
from composition_check import run_composition_checks
from isotope_check import check_isotopic_pattern
from overlap_resolver import resolve_overlaps
from peak_detection import detect_peaks
from range_editor import (
    empty_ranges,
    export_rrng_like,
    load_ranges_csv,
    parse_rrng_like,
    validate_ranges,
)
from range_sensitivity import composition_sensitivity_for_peak, overlap_warning, sensitivity_table
from spectrum_io import detect_columns, load_spectrum_csv

st.set_page_config(page_title="APT Mass Spectrum Ranging Assistant", layout="wide")
st.title("APT Mass Spectrum Ranging Assistant")
st.caption("Conservative, transparent helper for operator-guided APT ranging (not an automated final assignment engine).")

if "ranges" not in st.session_state:
    st.session_state["ranges"] = empty_ranges()

uploaded = st.sidebar.file_uploader("Load spectrum CSV", type=["csv"])
manual_cols = st.sidebar.checkbox("Manual column selection", value=False)

spec = None
if uploaded is not None:
    raw_df = pd.read_csv(uploaded)
    auto_x, auto_y = detect_columns(raw_df)
    x_col = auto_x
    y_col = auto_y
    if manual_cols:
        cols = list(raw_df.columns)
        x_col = st.sidebar.selectbox("x column (m/z, Da)", options=cols, index=cols.index(auto_x) if auto_x in cols else 0)
        y_col = st.sidebar.selectbox("y column (corrected count)", options=cols, index=cols.index(auto_y) if auto_y in cols else min(1, len(cols)-1))

    uploaded.seek(0)
    spec = load_spectrum_csv(uploaded, x_col=x_col, y_col=y_col)


def plot_spectrum(x, y, title="Spectrum"):
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=x, y=y, mode="lines", name="corrected count"))
    fig.update_layout(title=title, xaxis_title="m/z (Da)", yaxis_title="Corrected count", height=400)
    return fig


tabs = st.tabs([
    "1. Spectrum Loader / Viewer",
    "2. Peak Candidates",
    "3. Isotope Check",
    "4. Range Sensitivity",
    "5. Composition Check",
    "6. Charge-State Ratio",
    "7. Batch Comparison",
    "8. Range Editor",
])

with tabs[0]:
    st.subheader("Spectrum Loader / Viewer")
    if spec is None:
        st.info("Upload a CSV spectrum to begin.")
    else:
        top_l, top_r = st.columns([2, 1])
        with top_l:
            st.plotly_chart(plot_spectrum(spec.x, spec.y), use_container_width=True)
        with top_r:
            st.markdown(
                f"**x:** `{spec.x_col}`  \n**y:** `{spec.y_col}`  \n**Inferred bin width:** `{spec.bin_width:.6g} Da`"
            )
            peaks = detect_peaks(spec.x, spec.y)
            st.dataframe(peaks.head(20), use_container_width=True, height=320)
            st.caption("Conservative peak candidates only.")

with tabs[1]:
    st.subheader("Peak candidate generator")
    if spec is None:
        st.info("Load a spectrum first.")
    else:
        mz = st.number_input("Target peak m/z (Da)", min_value=float(spec.x.min()), max_value=float(spec.x.max()), value=float(spec.x.mean()))
        tol = st.slider("Candidate tolerance (Da)", 0.01, 1.0, 0.25, 0.01)
        cand = generate_candidates(mz, tolerance_da=tol)
        if cand.empty:
            st.warning("No candidates found in tolerance window.")
        else:
            st.dataframe(cand.head(200), use_container_width=True)
            resolved = resolve_overlaps(spec.x, spec.y, mz, cand.head(20))
            st.markdown("**Overlap-aware ranking**")
            st.dataframe(resolved.head(10), use_container_width=True)

with tabs[2]:
    st.subheader("Isotopic pattern checker")
    if spec is None:
        st.info("Load a spectrum first.")
    else:
        peak_mz = st.number_input("Primary peak m/z", value=float(spec.x.mean()), key="iso_mz")
        formula = st.text_input("Formula", value="Fe")
        charge = st.number_input("Charge state", min_value=1, max_value=5, value=1)
        out = check_isotopic_pattern(spec.x, spec.y, peak_mz, formula, int(charge))
        st.metric("Agreement", out["label"], f"score={out['score']:.2f}")
        st.dataframe(out["table"], use_container_width=True)
        st.info(out["notes"])

with tabs[3]:
    st.subheader("Range boundary suggester + sensitivity")
    if spec is None:
        st.info("Load a spectrum first.")
    else:
        peak_mz = st.number_input("Peak m/z for boundary analysis", value=float(spec.x.mean()), key="rng_mz")
        custom = st.slider("Custom fraction of peak max", 0.05, 0.9, 0.3, 0.05)
        table = sensitivity_table(spec.x, spec.y, peak_mz, custom_fraction=custom)
        mode_cols = st.columns(4)
        for i, mode in enumerate(["FWHM", "FW0.2M", "FW0.1M", "CUSTOM"]):
            row = table[table["mode"] == mode].iloc[0]
            with mode_cols[i]:
                st.markdown(f"**{mode}**")
                st.metric("Window (Da)", f"{row['start_da']:.4f}–{row['end_da']:.4f}")
                st.metric("Integrated", f"{row['integrated_count']:.3g}")
        st.dataframe(table, use_container_width=True, height=220)
        for w in overlap_warning(table):
            st.warning(w)

        fig = plot_spectrum(spec.x, spec.y, "Sensitivity windows")
        for _, r in table.iterrows():
            fig.add_vrect(x0=r["start_da"], x1=r["end_da"], opacity=0.12, annotation_text=r["mode"])
        st.plotly_chart(fig, use_container_width=True)

        ranges = validate_ranges(st.session_state["ranges"])
        if not ranges.empty and "species" in ranges.columns:
            species = st.selectbox("Species for per-peak window sensitivity", options=ranges["species"].tolist())
            comp_cmp = composition_sensitivity_for_peak(
                spec.x, spec.y, peak_mz, ranges, selected_species=species, custom_fraction=custom
            )
            if not comp_cmp.empty:
                pivot = comp_cmp.pivot_table(index="species", columns="mode", values="at_frac", aggfunc="mean")
                st.markdown("**Composition sensitivity (atomic fraction by mode)**")
                st.dataframe(pivot, use_container_width=True, height=240)
            else:
                st.info("No compatible range assignments available for composition sensitivity.")
        else:
            st.info("Define assignments in Range Editor to compare composition sensitivity across modes.")

with tabs[4]:
    st.subheader("Composition sanity checker")
    if spec is None:
        st.info("Load spectrum and define ranges first.")
    else:
        ranges = validate_ranges(st.session_state["ranges"])
        res = run_composition_checks(ranges.to_dict(orient="records"), spec.x, spec.y)
        if res["composition"].empty:
            st.warning("No ranges available.")
        else:
            st.dataframe(res["composition"], use_container_width=True)
        for w in res["warnings"]:
            st.warning(w)

with tabs[5]:
    st.subheader("Charge-state-ratio helper")
    ranges = validate_ranges(st.session_state["ranges"])
    if spec is None or ranges.empty:
        st.info("Need loaded spectrum and range assignments.")
    else:
        comp = run_composition_checks(ranges.to_dict(orient="records"), spec.x, spec.y)["composition"]
        element = st.text_input("Element symbol", value="Fe")
        ratios = charge_state_ratios(ranges, comp, element)
        if ratios.empty:
            st.warning("No charge-state entries found for this element.")
        else:
            st.dataframe(ratios, use_container_width=True)
            st.bar_chart(ratios.set_index("charge")["ratio"])

with tabs[6]:
    st.subheader("Batch spectrum comparison")
    batch_files = st.file_uploader("Load multiple spectra", type=["csv"], accept_multiple_files=True)
    if batch_files:
        batch = load_batch(batch_files)
        st.dataframe(overlay_table(batch), use_container_width=True)

        fig = go.Figure()
        for b in batch:
            y = b.y
            ymax = float(abs(y).max()) if len(y) else 1.0
            yn = y / ymax if ymax > 0 else y
            fig.add_trace(go.Scattergl(x=b.x, y=yn, mode="lines", name=b.name))
        fig.update_layout(title="Normalized overlay", xaxis_title="m/z (Da)", yaxis_title="normalized corrected count")
        st.plotly_chart(fig, use_container_width=True)

        ranges = validate_ranges(st.session_state["ranges"])
        if not ranges.empty:
            cmp_df = compare_ranges(batch, ranges)
            st.dataframe(cmp_df, use_container_width=True)
        else:
            st.info("Add ranges in Range Editor to compare integrated areas.")

with tabs[7]:
    st.subheader("Range file parser/editor")
    current = validate_ranges(st.session_state["ranges"])
    edited = st.data_editor(current, num_rows="dynamic", use_container_width=True)
    st.session_state["ranges"] = validate_ranges(edited)

    c1, c2, c3 = st.columns(3)
    with c1:
        up = st.file_uploader("Load range CSV", type=["csv"], key="rangecsv")
        if up is not None:
            st.session_state["ranges"] = load_ranges_csv(up)
            st.success("Loaded range CSV")
    with c2:
        rrng_text = st.text_area("Optional rrng-like import")
        if st.button("Parse rrng-like text"):
            st.session_state["ranges"] = parse_rrng_like(rrng_text)
    with c3:
        st.download_button(
            "Export ranges CSV",
            data=validate_ranges(st.session_state["ranges"]).to_csv(index=False).encode("utf-8"),
            file_name="ranges.csv",
            mime="text/csv",
        )
        st.download_button(
            "Export rrng-like text",
            data=export_rrng_like(validate_ranges(st.session_state["ranges"])).encode("utf-8"),
            file_name="ranges.rrng.txt",
            mime="text/plain",
        )

st.sidebar.markdown("---")
st.sidebar.warning("Transparency: all scores are heuristic. Always confirm with expert APT judgment.")
