"""Report figures that need only the files in data/ (no Specim cube).

leds.pdf     - the ten LED channels on the 400-700 nm grid
weights.pdf  - the LED mix PSO chose for each objective (2025 run)
metrics.pdf  - each metric under D65 versus the optimised lights
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spectral_pso import plotting  # noqa: E402
from spectral_pso.data import DATA_DIR, REPO_ROOT, TASKS, load_led_spectra, load_optimal_spectrum, load_weights  # noqa: E402

plt = plotting.plt


def wavelength_to_rgb(nm):
    """Approximate display colour of a monochromatic line (Bruton's piecewise fit).

    Used only to colour plot lines; it is not colorimetry.
    """
    if nm < 440:
        r, g, b = -(nm - 440) / 60, 0.0, 1.0
    elif nm < 490:
        r, g, b = 0.0, (nm - 440) / 50, 1.0
    elif nm < 510:
        r, g, b = 0.0, 1.0, -(nm - 510) / 20
    elif nm < 580:
        r, g, b = (nm - 510) / 70, 1.0, 0.0
    elif nm < 645:
        r, g, b = 1.0, -(nm - 645) / 65, 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0
    # Darken slightly so thin lines stay readable on white.
    return tuple(0.85 * np.clip([r, g, b], 0, 1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fig-dir", default=os.path.join(REPO_ROOT, "report", "figures"))
    args = p.parse_args()
    plotting.apply_style()

    wl, _ = load_optimal_spectrum("DeltaE_min")            # the 400-700 nm grid of the 2025 run
    leds = load_led_spectra(wl)
    peaks = wl[np.argmax(leds, axis=1)]
    led_colours = [wavelength_to_rgb(pk) for pk in peaks]
    led_names = [f"LED{i + 1} ({pk:.0f} nm)" for i, pk in enumerate(peaks)]
    print("LED peaks (nm): " + ", ".join(f"{pk:.0f}" for pk in peaks))

    # 1. LED channels
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    for i in range(len(leds)):
        ax.fill_between(wl, leds[i] / 1e3, color=led_colours[i], alpha=0.12, linewidth=0)
        ax.plot(wl, leds[i] / 1e3, color=led_colours[i], linewidth=1.1)
        ax.annotate(f"{i + 1}", (peaks[i], leds[i].max() / 1e3), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=7.5, color=plotting.INK)
    ax.set_xlim(400, 700)
    ax.set_ylim(0, leds.max() / 1e3 * 1.15)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"Counts ($\times 10^3$)")
    plotting.save(fig, os.path.join(args.fig_dir, "leds.pdf"))

    # 2. LED mix per objective
    tasks = [t for t, _, _ in TASKS]
    weights = np.array([load_weights(t) for t in tasks])
    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    left = np.zeros(len(tasks))
    y = np.arange(len(tasks))[::-1]
    for i in range(weights.shape[1]):
        if weights[:, i].max() < 1e-4:
            continue                                        # LED never used; keep it out of the legend
        ax.barh(y, weights[:, i], left=left, color=led_colours[i], height=0.62, label=led_names[i],
                edgecolor="white", linewidth=0.6)
        for yy, w, l0 in zip(y, weights[:, i], left):
            if w > 0.08:
                ax.text(l0 + w / 2, yy, f"{w:.2f}", ha="center", va="center", fontsize=7, color="white")
        left += weights[:, i]
    ax.set_yticks(y)
    ax.set_yticklabels([plotting.TASK_LABELS[t] for t in tasks])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of total LED drive")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plotting.save(fig, os.path.join(args.fig_dir, "weights.pdf"))

    # 3. Metrics: D65 vs optimised (values recorded by the 2025 run)
    with open(os.path.join(DATA_DIR, "optimal", "summary_2025.csv"), newline="") as f:
        summary = {row["task"]: row for row in csv.DictReader(f)}
    groups = [("DeltaE", r"$\Delta E^*_{ab}$"), ("DeltaRGB", r"$\Delta RGB$"), ("Contrast", "Michelson contrast")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    for ax, (key, title) in zip(axes, groups):
        lo, hi = summary[f"{key}_min"], summary[f"{key}_max"]
        vals = [float(lo["optimal_value"]), float(lo["d65_value"]), float(hi["optimal_value"])]
        cols = [plotting.TASK_COLOURS[f"{key}_min"], "#d1d5db", plotting.TASK_COLOURS[f"{key}_max"]]
        bars = ax.bar(["min", "D65", "max"], vals, color=cols, width=0.62)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3g}", ha="center", va="bottom", fontsize=7.5,
                    color=plotting.INK)
        ax.set_title(title)
        ax.set_ylim(0, max(vals) * 1.18)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    plotting.save(fig, os.path.join(args.fig_dir, "metrics.pdf"))


if __name__ == "__main__":
    main()
