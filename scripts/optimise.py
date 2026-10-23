"""Step 2: PSO search for the six LED mixes, compared against D65.

Needs data/reflectance/R1_R2.csv from step 1. Writes per-task weights and
emission spectra to results/optimal/, a summary table, and a convergence plot.

With --evaluate DIR it skips the search and scores existing weights instead,
for example the 2025 run in data/optimal/.
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spectral_pso import plotting  # noqa: E402
from spectral_pso.colorimetry import ColourModel  # noqa: E402
from spectral_pso.data import RESULTS_DIR, TASKS, load_led_spectra, load_reflectance, load_weights  # noqa: E402
from spectral_pso.pso import optimise  # noqa: E402


def write_outputs(out_dir, task, weights, wl, emission):
    np.savetxt(os.path.join(out_dir, f"{task}_weights.txt"), weights[None, :], fmt="%.6f", delimiter="\t",
               header=f"Optimal LED weights for {task}\n" + "\t".join(f"LED{i}" for i in range(1, 11)))
    np.savetxt(os.path.join(out_dir, f"{task}_spectrum.txt"), np.column_stack([wl, emission]), fmt="%.6f",
               header="wavelength_nm counts")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reflectance", default=None, help="CSV from extract_reflectance.py")
    p.add_argument("--evaluate", default=None, metavar="DIR", help="Score weights in DIR instead of optimising")
    p.add_argument("--particles", type=int, default=200)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(RESULTS_DIR, "optimal"))
    args = p.parse_args()

    wl, r1, r2 = load_reflectance(args.reflectance)
    model = ColourModel(wl)
    leds = load_led_spectra(model.wavelengths)               # (10, N)
    d65 = model.metrics(r1, r2)
    print("D65 reference: " + ", ".join(f"{k} = {float(v):.6f}" for k, v in d65.items()))

    os.makedirs(args.out, exist_ok=True)
    rows, histories = [], {}
    for task, metric, maximise in TASKS:
        if args.evaluate:
            weights = load_weights(task, args.evaluate)
        else:
            print(f"\n{task}: {'maximise' if maximise else 'minimise'} {metric}")

            def cost(w, metric=metric):
                return model.metrics(r1, r2, w @ leds)[metric]

            weights, _, histories[task] = optimise(
                cost, leds.shape[0], maximise=maximise, n_particles=args.particles,
                n_iterations=args.iterations, seed=args.seed)
            write_outputs(args.out, task, weights, model.wavelengths, weights @ leds)

        value = float(model.metrics(r1, r2, weights @ leds)[metric])
        ref = float(d65[metric])
        change = 100.0 * (value - ref) / ref
        rows.append([task, metric, "max" if maximise else "min", f"{ref:.6f}", f"{value:.6f}", f"{change:+.1f}"])
        print(f"  {task:13s} D65 {ref:10.4f} -> {value:10.4f} ({change:+.1f}%)")

    summary = os.path.join(args.out, "summary.csv")
    with open(summary, "w", newline="") as f:
        csv.writer(f).writerows([["task", "metric", "objective", "d65_value", "optimal_value", "change_pct"]] + rows)
    print(f"\nwrote {os.path.relpath(summary)}")

    if histories:
        plotting.apply_style()
        fig, axes = plotting.plt.subplots(2, 3, figsize=(7.0, 3.8), sharex=True)
        for ax, (task, _, _) in zip(axes.ravel(), TASKS):
            ax.plot(histories[task], color=plotting.TASK_COLOURS[task])
            ax.set_title(plotting.TASK_LABELS[task])
        for ax in axes[1]:
            ax.set_xlabel("Iteration")
        fig.tight_layout()
        plotting.save(fig, os.path.join(args.out, "convergence.pdf"))


if __name__ == "__main__":
    main()
