from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Iterable

import numpy as np
import pandas as pd

ELEMENTS = {
    "H": {"mass": 1.00784, "isotopes": [(1.0078, 0.9999), (2.0141, 0.0001)]},
    "C": {"mass": 12.0, "isotopes": [(12.0, 0.989), (13.0034, 0.011)]},
    "N": {"mass": 14.0031, "isotopes": [(14.0031, 0.996), (15.0001, 0.004)]},
    "O": {"mass": 15.9949, "isotopes": [(15.9949, 0.998), (17.9992, 0.002)]},
    "Al": {"mass": 26.9815, "isotopes": [(26.9815, 1.0)]},
    "Si": {"mass": 27.9769, "isotopes": [(27.9769, 0.922), (28.9765, 0.047), (29.9738, 0.031)]},
    "Fe": {"mass": 55.9349, "isotopes": [(53.9396, 0.058), (55.9349, 0.917), (56.9354, 0.022)]},
    "Cr": {"mass": 51.9405, "isotopes": [(49.9460, 0.043), (51.9405, 0.838), (52.9407, 0.095)]},
    "Ni": {"mass": 57.9353, "isotopes": [(57.9353, 0.681), (59.9308, 0.262)]},
    "Cu": {"mass": 62.9296, "isotopes": [(62.9296, 0.691), (64.9278, 0.309)]},
}


@dataclass
class Candidate:
    name: str
    formula: str
    charge: int
    mz_theory: float
    mass_error: float
    isotope_note: str
    plausibility: float


def _molecular_formulas(elements: list[str], max_atoms: int = 2) -> list[str]:
    formulas = []
    for n in range(2, max_atoms + 1):
        for combo in combinations_with_replacement(elements, n):
            counts: dict[str, int] = {}
            for e in combo:
                counts[e] = counts.get(e, 0) + 1
            formula = "".join([f"{k}{v if v > 1 else ''}" for k, v in sorted(counts.items())])
            formulas.append(formula)
    return sorted(set(formulas))


def formula_mass(formula: str) -> float:
    import re

    parts = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    mass = 0.0
    for el, n in parts:
        if el not in ELEMENTS:
            return np.nan
        mass += ELEMENTS[el]["mass"] * (int(n) if n else 1)
    return mass


def generate_candidates(target_mz: float, tolerance_da: float = 0.3) -> pd.DataFrame:
    rows = []
    elems = list(ELEMENTS.keys())
    formulas = elems + _molecular_formulas(elems, max_atoms=2)

    for formula in formulas:
        m = formula_mass(formula)
        if not np.isfinite(m):
            continue
        for z in (1, 2, 3):
            mz = m / z
            err = target_mz - mz
            if abs(err) <= tolerance_da:
                isotope_note = "Has isotopic companions" if any(len(ELEMENTS[e]["isotopes"]) > 1 for e in ELEMENTS if e in formula) else "Single dominant isotope"
                plaus = max(0.0, 1.0 - abs(err) / tolerance_da)
                if z > 1:
                    plaus *= 0.95
                rows.append(
                    {
                        "candidate": formula + f"^{z}+",
                        "name": formula,
                        "formula": formula,
                        "charge": z,
                        "mz_theory": mz,
                        "mass_error": err,
                        "expected_isotopes": isotope_note,
                        "plausibility": plaus,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["plausibility", "mass_error"], ascending=[False, True]).reset_index(drop=True)


def expected_isotope_spacings(formula: str, charge: int) -> list[tuple[float, float]]:
    """Return (delta_mz, relative_abundance) for first-order isotopic companions."""
    out = []
    for elem, data in ELEMENTS.items():
        if elem in formula and len(data["isotopes"]) > 1:
            base_mass, base_ab = data["isotopes"][0]
            for iso_mass, ab in data["isotopes"][1:]:
                out.append(((iso_mass - base_mass) / charge, ab / max(base_ab, 1e-12)))
    return out
