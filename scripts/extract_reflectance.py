"""Step 1: Specim V10E cube -> reflectance spectra of the two Ishihara regions.

Writes data/reflectance/R1_R2.csv (400-700 nm) and, for rendering, the cropped
reflectance cube to results/reflectance_roi.npy.

The cube is not in the repository (430 MB of references alone). Point
--capture-dir at the Lumo "capture" folder that holds sample/WHITEREF/DARKREF.
Defaults reproduce the 2025 run: global reference averaging and single pixels
at the coordinates used then.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spectral_pso.colorimetry import ColourModel  # noqa: E402
from spectral_pso.data import DATA_DIR, RESULTS_DIR  # noqa: E402
from spectral_pso.envi import read_cube, reflectance  # noqa: E402


def patch_mean(cube, y, x, half):
    """Mean spectrum of a (2*half+1)^2 window, clipped to the image."""
    y0, y1 = max(0, y - half), min(cube.shape[0], y + half + 1)
    x0, x1 = max(0, x - half), min(cube.shape[1], x + half + 1)
    return np.nanmean(cube[y0:y1, x0:x1].reshape(-1, cube.shape[2]), axis=0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture-dir", required=True, help="Folder with <scan>.hdr/.raw and WHITEREF_/DARKREF_ files")
    p.add_argument("--scan", default="sample_scan_0063", help="Scan basename (default: %(default)s)")
    p.add_argument("--mode", choices=["global", "column"], default="global",
                   help="Reference averaging; 'global' reproduces 2025, 'column' is standard push-broom")
    # Full-frame (y, x) pixel coordinates used in 2025: ROI origin (y=0, x=950) + ROI offsets.
    p.add_argument("--r1", type=int, nargs=2, default=[315, 1105], metavar=("Y", "X"), help="Numeral pixel")
    p.add_argument("--r2", type=int, nargs=2, default=[285, 1095], metavar=("Y", "X"), help="Background pixel")
    p.add_argument("--half-window", type=int, default=0,
                   help="Average a (2h+1)^2 patch instead of one pixel (2025 used 0)")
    p.add_argument("--roi", type=int, nargs=4, default=[0, 585, 950, 1610], metavar=("Y0", "Y1", "X0", "X1"),
                   help="Crop saved for rendering")
    args = p.parse_args()

    def path(prefix):
        return os.path.join(args.capture_dir, f"{prefix}{args.scan}.hdr")

    print("Loading cube and references...")
    sample, wl = read_cube(path(""))
    white, _ = read_cube(path("WHITEREF_"))
    dark, _ = read_cube(path("DARKREF_"))
    print(f"  sample {sample.shape}, white {white.shape}, dark {dark.shape}, {wl.min():.1f}-{wl.max():.1f} nm")

    vis = (wl >= 400) & (wl <= 700)
    if vis.sum() < 10:
        raise ValueError(f"Only {int(vis.sum())} bands fall in 400-700 nm; check the header wavelengths")
    wl_vis = wl[vis]

    # Calibrate only the bands and rows we need, so the full cube never sits in memory as float64.
    y0, y1, x0, x1 = args.roi
    rows = sorted({args.r1[0], args.r2[0]} | set(range(y0, y1)))
    refl = reflectance(np.asarray(sample[rows][:, :, vis]), white[:, :, vis], dark[:, :, vis], mode=args.mode)
    row_index = {r: i for i, r in enumerate(rows)}

    h = args.half_window
    r1 = patch_mean(refl, row_index[args.r1[0]], args.r1[1], h)
    r2 = patch_mean(refl, row_index[args.r2[0]], args.r2[1], h)
    if np.any(~np.isfinite(r1)) or np.any(~np.isfinite(r2)):
        raise ValueError("Reflectance at the chosen pixels contains NaN; pick other coordinates")

    out_dir = os.path.join(DATA_DIR, "reflectance")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "R1_R2.csv")
    np.savetxt(out_csv, np.column_stack([wl_vis, r1, r2]), delimiter=",", fmt="%.6f",
               header="wavelength_nm,r1_numeral,r2_background", comments="")
    print(f"  wrote {os.path.relpath(out_csv)} ({len(wl_vis)} bands)")

    roi = np.clip(refl[[row_index[r] for r in range(y0, y1)], x0:x1], 0, 1).astype(np.float32)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(os.path.join(RESULTS_DIR, "reflectance_roi.npy"), roi)
    np.save(os.path.join(RESULTS_DIR, "reflectance_roi_wavelengths.npy"), wl_vis)
    print(f"  wrote results/reflectance_roi.npy {roi.shape}")

    d65 = ColourModel(wl_vis).metrics(r1, r2)
    print("Under D65: " + ", ".join(f"{k} = {float(v):.4f}" for k, v in d65.items()))


if __name__ == "__main__":
    main()
