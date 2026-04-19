import numpy as np

from peak_detection import estimate_asymmetric_width
from utils import infer_bin_width, integrate_window


def test_bin_width_inference_irregular():
    x = np.array([0.0, 0.1, 0.2, 0.31, 0.41])
    bw = infer_bin_width(x)
    assert 0.09 <= bw <= 0.11


def test_peak_width_calculation():
    x = np.linspace(0, 10, 101)
    y = np.exp(-0.5 * ((x - 5) / 0.5) ** 2)
    out = estimate_asymmetric_width(x, y, 5.0, fraction=0.5)
    assert out["start_da"] < 5.0 < out["end_da"]
    assert out["left_da"] > 0


def test_range_integration():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 0.0])
    area = integrate_window(x, y, 0.0, 2.0)
    assert abs(area - 1.0) < 1e-6
