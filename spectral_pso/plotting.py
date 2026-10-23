"""Shared figure style so the plots match the LaTeX report (Libertinus)."""

import glob
import os
import shutil
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# One hue per metric; min is the lighter tone, max the darker one.
TASK_COLOURS = {
    "DeltaE_min": "#f59e0b", "DeltaE_max": "#b45309",
    "DeltaRGB_min": "#60a5fa", "DeltaRGB_max": "#1d4ed8",
    "Contrast_min": "#34d399", "Contrast_max": "#047857",
}
TASK_LABELS = {
    "DeltaE_min": r"$\Delta E$ min", "DeltaE_max": r"$\Delta E$ max",
    "DeltaRGB_min": r"$\Delta RGB$ min", "DeltaRGB_max": r"$\Delta RGB$ max",
    "Contrast_min": "Contrast min", "Contrast_max": "Contrast max",
}
INK = "#1f2937"
MUTED = "#6b7280"
R1_COLOUR = "#dc2626"   # numeral (pinkish red)
R2_COLOUR = "#6b7280"   # background (grey)


def _find_libertinus():
    """Locate Libertinus OTFs in a TeX installation; return [] if absent."""
    roots = []
    kpse = shutil.which("kpsewhich")
    if kpse:
        try:
            out = subprocess.run([kpse, "LibertinusSans-Regular.otf"], capture_output=True,
                                 text=True, timeout=10).stdout.strip()
            if out:
                roots.append(os.path.dirname(out))
        except (OSError, subprocess.SubprocessError):
            pass
    files = []
    for root in roots:
        files += glob.glob(os.path.join(root, "LibertinusSans-*.otf"))
        files += glob.glob(os.path.join(root, "LibertinusSerif-*.otf"))
    return files


def apply_style():
    """Set rcParams; falls back to matplotlib's default font when Libertinus is missing."""
    family = "DejaVu Sans"
    fonts = _find_libertinus()
    for f in fonts:
        try:
            font_manager.fontManager.addfont(f)
        except (OSError, RuntimeError) as e:
            print(f"[plotting] could not load font {f}: {e}")
    if any("LibertinusSans" in f for f in fonts):
        family = "Libertinus Sans"
    else:
        print("[plotting] Libertinus not found (needs a TeX install); using DejaVu Sans")

    plt.rcParams.update({
        "font.family": family,
        "mathtext.fontset": "custom",
        "mathtext.rm": family,
        "mathtext.it": f"{family}:italic",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e5e7eb",
        "grid.linewidth": 0.5,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")
