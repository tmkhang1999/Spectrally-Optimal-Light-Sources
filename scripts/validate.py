"""Step 3: compare each PSO-predicted spectrum with what the Ledmotive emitted.

Runs from files in the repository alone (data/optimal and data/measured).
The two spectra come from different spectrometer settings, so absolute counts
differ. The comparison is therefore scale-free: Pearson r, and the relative
RMSE after the best least-squares scaling of the measured curve.
"""

import argparse
import csv
import os
import sys

import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spectral_pso import plotting  # noqa: E402
from spectral_pso.data import (REPO_ROOT, RESULTS_DIR, TASKS, load_measured_spectrum,  # noqa: E402
                               load_optimal_spectrum)


def compare(predicted, measured):
    """Return (pearson_r, relative_rmse, scale) with measured scaled onto predicted."""
    scale = float(measured @ predicted) / float(measured @ measured)
    residual = predicted - scale * measured
    rel_rmse = float(np.linalg.norm(residual) / np.linalg.norm(predicted))
    r = float(np.corrcoef(predicted, measured)[0, 1])
    return r, rel_rmse, scale


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--optimal", default=None, help="Folder of *_spectrum.txt (default: data/optimal)")
    p.add_argument("--fig-dir", default=os.path.join(REPO_ROOT, "report", "figures"))
    args = p.parse_args()

    plotting.apply_style()
    fig, axes = plotting.plt.subplots(2, 3, figsize=(7.0, 3.9), sharex=True, sharey=True)
    rows = []
    for ax, (task, _, _) in zip(axes.ravel(), TASKS):
        wl, predicted = load_optimal_spectrum(task, args.optimal)
        mwl, mcounts = load_measured_spectrum(task)
        measured = np.interp(wl, mwl, mcounts)
        r, rel, scale = compare(predicted, measured)
        peak_p, peak_m = wl[np.argmax(predicted)], wl[np.argmax(measured)]
        rows.append([task, f"{r:.4f}", f"{rel:.4f}", f"{scale:.4f}", f"{peak_p:.1f}", f"{peak_m:.1f}"])

        top = predicted.max()
        colour = plotting.TASK_COLOURS[task]
        ax.fill_between(wl, predicted / top, color=colour, alpha=0.18, linewidth=0)
        ax.plot(wl, predicted / top, color=colour)
        ax.plot(wl, scale * measured / top, color=plotting.INK, linestyle=(0, (3, 2)), linewidth=1.0)
        ax.set_title(plotting.TASK_LABELS[task])
        # Peaks sit near 430/520 or 505/660 nm, so the middle of the top band is free.
        ax.text(0.6, 0.97, f"r = {r:.3f}   rel. RMSE = {100 * rel:.1f}%", transform=ax.transAxes,
                ha="center", va="top", fontsize=7.5, color=plotting.INK)
        ax.set_xlim(400, 700)
        ax.set_ylim(0, 1.25)
    for ax in axes[1]:
        ax.set_xlabel("Wavelength (nm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Relative power")
    # Proxy lines: the real curves are task-coloured, the legend should not be.
    proxies = [Line2D([], [], color=plotting.MUTED), Line2D([], [], color=plotting.INK, linestyle=(0, (3, 2)))]
    fig.legend(proxies, ["PSO prediction", "Measured (PMA-12)"], loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    plotting.save(fig, os.path.join(args.fig_dir, "validation.pdf"))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "validation.csv")
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows([["task", "pearson_r", "relative_rmse", "scale", "peak_pred_nm", "peak_meas_nm"]] + rows)
    print(f"  wrote {os.path.relpath(out)}")
    for row in rows:
        print("  " + "  ".join(row))


if __name__ == "__main__":
    main()
