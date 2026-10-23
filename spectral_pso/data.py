"""Loaders for the measured spectra shipped in data/ and shared task definitions."""

import os

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# (name, metric key in ColourModel.metrics, maximise?)
TASKS = [
    ("DeltaE_min", "delta_e", False),
    ("DeltaE_max", "delta_e", True),
    ("DeltaRGB_min", "delta_rgb", False),
    ("DeltaRGB_max", "delta_rgb", True),
    ("Contrast_min", "michelson", False),
    ("Contrast_max", "michelson", True),
]
N_LEDS = 10


def _load_two_column(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected spectrum file not found: {path}")
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{path} should have two columns (wavelength, value); got shape {data.shape}")
    return data[:, 0], data[:, 1]


def load_led_spectra(wavelengths, led_dir=None):
    """The 10 Ledmotive channels, measured with the PMA-12, resampled to `wavelengths`.

    Values stay in raw spectrometer counts (85 ms, mean of 10 spectra). Only the
    relative mix matters for colour, so no normalisation is applied, matching the
    2025 run. Returns shape (10, N).
    """
    led_dir = led_dir or os.path.join(DATA_DIR, "led_spectra")
    rows = []
    for i in range(1, N_LEDS + 1):
        wl, counts = _load_two_column(os.path.join(led_dir, f"LED{i}.txt"))
        rows.append(np.interp(wavelengths, wl, counts))
    return np.array(rows)


def load_weights(task, folder=None):
    """LED weights for one task, shape (10,)."""
    folder = folder or os.path.join(DATA_DIR, "optimal")
    path = os.path.join(folder, f"{task}_weights.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weights for {task} not found: {path}")
    w = np.loadtxt(path, comments="#").ravel()
    if w.size != N_LEDS:
        raise ValueError(f"{path} has {w.size} weights, expected {N_LEDS}")
    return w


def load_optimal_spectrum(task, folder=None):
    """PSO-predicted emission for one task: (wavelengths, counts)."""
    folder = folder or os.path.join(DATA_DIR, "optimal")
    return _load_two_column(os.path.join(folder, f"{task}_spectrum.txt"))


def load_measured_spectrum(task, folder=None):
    """Spectrum of the light the Ledmotive produced for one task: (wavelengths, counts)."""
    folder = folder or os.path.join(DATA_DIR, "measured")
    return _load_two_column(os.path.join(folder, f"{task}.txt"))


def load_reflectance(path=None):
    """(wavelengths, r1, r2) written by scripts/extract_reflectance.py."""
    path = path or os.path.join(DATA_DIR, "reflectance", "R1_R2.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run scripts/extract_reflectance.py on the Specim cube first "
            "(the cube is not in the repository; see README, 'Data')."
        )
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1], data[:, 2]
