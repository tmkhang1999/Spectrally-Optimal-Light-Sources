"""Reflectance + illuminant -> CIE XYZ, CIELAB, sRGB, and the three cost metrics.

Conventions follow the UEF Color Science Laboratory 2025 course so the numbers
in the report can be reproduced:
- CIE 1931 2-degree observer, sums on the measurement grid (no step factor;
  it cancels in the normalisation), spectra cropped to 360-830 nm.
- XYZ is scaled so a perfect white under the *test* illuminant has Y = 100.
- CIELAB always uses the D65 white point, also under the LED illuminants.
  There is no chromatic adaptation to the test light, so Delta E here is the
  difference an observer adapted to D65 would see. See the report's limitations.
- sRGB uses the IEC 61966-2-1 matrix. Out-of-gamut channels can be clipped
  (for display) or set to NaN (inside the optimiser, so such lights are rejected).
"""

import numpy as np

try:
    import colour
except ImportError as e:  # pragma: no cover - import guard
    raise ImportError(
        "colour-science is needed for the CIE tables: pip install colour-science"
    ) from e

_CMFS = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
_D65 = colour.SDS_ILLUMINANTS["D65"]

_M_XYZ_TO_SRGB = np.array([
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.2040, 1.0570],
])
# skimage rgb2gray weights, as used in the course for Michelson contrast
_GRAY_WEIGHTS = np.array([0.2125, 0.7154, 0.0721])


def _interp_table(sd_like, wavelengths):
    wl = np.asarray(sd_like.wavelengths, dtype=float)
    values = np.asarray(sd_like.values, dtype=float)
    if values.ndim == 1:
        return np.interp(wavelengths, wl, values)
    return np.stack([np.interp(wavelengths, wl, values[:, i]) for i in range(values.shape[1])], axis=1)


def cmfs_at(wavelengths):
    """CIE 1931 colour-matching functions on the given grid, shape (N, 3)."""
    return _interp_table(_CMFS, wavelengths)


def d65_at(wavelengths):
    """CIE D65 relative spectral power on the given grid, shape (N,)."""
    return _interp_table(_D65, wavelengths)


def d65_white_xyz():
    """XYZ of a perfect reflector under D65 (Y = 100), on a 1 nm 380-780 grid."""
    wl = np.arange(380.0, 781.0, 1.0)
    cmf = cmfs_at(wl)
    light = d65_at(wl)
    return 100.0 * (light @ cmf) / (light @ cmf[:, 1])


class ColourModel:
    """Pre-computes the CIE tables on one wavelength grid.

    The optimiser evaluates tens of thousands of candidate lights on the same
    grid, so the tables are interpolated once here instead of per call.
    """

    def __init__(self, wavelengths):
        wavelengths = np.asarray(wavelengths, dtype=float)
        self.keep = (wavelengths >= 360) & (wavelengths <= 830)
        if not np.any(self.keep):
            raise ValueError("No wavelengths inside the CIE range 360-830 nm")
        self.wavelengths = wavelengths[self.keep]
        self.cmfs = cmfs_at(self.wavelengths)
        self.d65 = d65_at(self.wavelengths)
        self.white_xyz = d65_white_xyz()

    def _crop(self, spectra):
        return np.asarray(spectra, dtype=float)[..., self.keep]

    def xyz(self, reflectance, lights=None):
        """XYZ of one or more reflectances under one or more lights.

        reflectance: (..., N) on the model grid. lights: (N,) or (P, N); None = D65.
        Returns (P, ..., 3) when lights is 2-D, else (..., 3).
        """
        r = self._crop(reflectance)
        l = self.d65 if lights is None else self._crop(lights)
        single = l.ndim == 1
        l2 = np.atleast_2d(l)                                   # (P, N)
        k = 100.0 / (l2 @ self.cmfs[:, 1])                      # (P,)
        weighted = r[..., :, None] * self.cmfs                  # (..., N, 3)
        out = np.einsum("pn,...nc->p...c", l2, weighted)
        out *= k.reshape((-1,) + (1,) * (out.ndim - 1))
        return out[0] if single else out

    def lab(self, xyz):
        """CIELAB relative to the D65 white (CIE 1976 constants)."""
        t = np.asarray(xyz, dtype=float) / self.white_xyz
        f = np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16.0 / 116.0)
        L = 116.0 * f[..., 1] - 16.0
        a = 500.0 * (f[..., 0] - f[..., 1])
        b = 200.0 * (f[..., 1] - f[..., 2])
        return np.stack([L, a, b], axis=-1)

    @staticmethod
    def srgb(xyz, out_of_gamut="clip"):
        """Gamma-encoded sRGB in [0, 1].

        out_of_gamut="clip" clamps to [0, 1]; "nan" marks any channel outside
        [0, 1] as NaN so callers can reject that light.
        """
        rgb = np.asarray(xyz, dtype=float) @ _M_XYZ_TO_SRGB.T / 100.0
        if out_of_gamut == "clip":
            rgb = np.clip(rgb, 0.0, 1.0)
        elif out_of_gamut == "nan":
            rgb = np.where((rgb < 0.0) | (rgb > 1.0), np.nan, rgb)
        else:
            raise ValueError(f"out_of_gamut must be 'clip' or 'nan', got {out_of_gamut!r}")
        with np.errstate(invalid="ignore"):
            return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(rgb, 1.0 / 2.4) - 0.055)

    def metrics(self, r1, r2, lights=None):
        """Delta E (CIE76), Delta RGB and Michelson contrast between two reflectances.

        Any light that pushes either colour outside the sRGB gamut, or gives a
        non-positive XYZ, returns NaN for all three metrics. The optimiser treats
        NaN as infeasible, which keeps it to lights a display could show.
        Returns a dict of arrays shaped (P,) (or scalars for a single light).
        """
        xyz = self.xyz(np.stack([r1, r2]), lights)              # (P, 2, 3) or (2, 3)
        xyz1, xyz2 = xyz[..., 0, :], xyz[..., 1, :]
        rgb1 = self.srgb(xyz1, out_of_gamut="nan")
        rgb2 = self.srgb(xyz2, out_of_gamut="nan")

        invalid = (
            np.any(xyz1 <= 0, axis=-1) | np.any(xyz2 <= 0, axis=-1)
            | np.any(np.isnan(rgb1), axis=-1) | np.any(np.isnan(rgb2), axis=-1)
        )
        delta_e = np.linalg.norm(self.lab(xyz1) - self.lab(xyz2), axis=-1)
        delta_rgb = np.linalg.norm(rgb1 - rgb2, axis=-1)
        g1, g2 = rgb1 @ _GRAY_WEIGHTS, rgb2 @ _GRAY_WEIGHTS
        with np.errstate(invalid="ignore", divide="ignore"):
            michelson = np.abs(g1 - g2) / (g1 + g2)

        out = {}
        for name, value in (("delta_e", delta_e), ("delta_rgb", delta_rgb), ("michelson", michelson)):
            out[name] = np.where(invalid, np.nan, value)
        return out

    def render(self, cube, light=None):
        """Display sRGB image of a reflectance cube (H, W, N) under one light."""
        return self.srgb(self.xyz(cube, light), out_of_gamut="clip")
