"""Minimal ENVI reader and white/dark calibration for push-broom cubes."""

import os
import re

import numpy as np

# ENVI "data type" codes -> numpy dtypes
_ENVI_DTYPES = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    12: np.uint16,
    13: np.uint32,
    14: np.int64,
    15: np.uint64,
}


def read_header(hdr_path):
    """Parse an ENVI .hdr file into a dict of lower-case keys and string values."""
    with open(hdr_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    if not text.lstrip().startswith("ENVI"):
        raise ValueError(f"{hdr_path} does not start with 'ENVI'; not an ENVI header")

    header = {}
    # Values in braces may span several lines, so match them as one block.
    for key, value in re.findall(r"^\s*([^=\n]+?)\s*=\s*(\{[^}]*\}|[^\n]*)", text, re.M):
        header[key.strip().lower()] = value.strip()
    return header


def _brace_list(value, cast=float):
    return np.array([cast(v) for v in value.strip("{}").split(",") if v.strip()])


def read_cube(hdr_path, raw_path=None, memmap=True):
    """Return (cube, wavelengths) with cube shaped (lines, samples, bands).

    raw_path defaults to the header path with a .raw extension. With memmap=True
    the data stays on disk until sliced, which matters for 300+ MB white references.
    """
    header = read_header(hdr_path)
    if raw_path is None:
        raw_path = os.path.splitext(hdr_path)[0] + ".raw"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data for {hdr_path} not found at {raw_path}")

    try:
        lines = int(header["lines"])
        samples = int(header["samples"])
        bands = int(header["bands"])
        dtype_code = int(header["data type"])
    except KeyError as e:
        raise ValueError(f"{hdr_path} is missing required key {e}") from e
    if dtype_code not in _ENVI_DTYPES:
        raise ValueError(f"Unsupported ENVI data type {dtype_code} in {hdr_path}")

    dtype = np.dtype(_ENVI_DTYPES[dtype_code])
    if header.get("byte order", "0") == "1":
        dtype = dtype.newbyteorder(">")
    offset = int(header.get("header offset", "0"))
    interleave = header.get("interleave", "bsq").lower()

    shapes = {
        "bil": (lines, bands, samples),
        "bip": (lines, samples, bands),
        "bsq": (bands, lines, samples),
    }
    if interleave not in shapes:
        raise ValueError(f"Unknown interleave '{interleave}' in {hdr_path}")

    expected = offset + int(np.prod(shapes[interleave])) * dtype.itemsize
    actual = os.path.getsize(raw_path)
    if actual < expected:
        raise ValueError(
            f"{raw_path} has {actual} bytes but the header implies {expected}; "
            "check lines/samples/bands/data type"
        )

    if memmap:
        data = np.memmap(raw_path, dtype=dtype, mode="r", offset=offset, shape=shapes[interleave])
    else:
        data = np.fromfile(raw_path, dtype=dtype, offset=offset).reshape(shapes[interleave])

    if interleave == "bil":
        cube = data.transpose(0, 2, 1)
    elif interleave == "bsq":
        cube = data.transpose(1, 2, 0)
    else:
        cube = data

    wavelengths = _brace_list(header["wavelength"]) if "wavelength" in header else np.arange(bands, dtype=float)
    return cube, wavelengths


def reflectance(sample, white, dark, mode="column"):
    """Convert raw counts to reflectance, R = (S - D) / (W - D).

    mode="column" averages each reference over its scan lines, keeping one value
    per (sample column, band). That is the usual push-broom calibration because
    every column is its own detector pixel.
    mode="global" averages each reference over all pixels, keeping one value per
    band. The 2025 lab run used this; it is kept to reproduce those numbers.
    """
    if mode == "column":
        axes = (0,)
    elif mode == "global":
        axes = (0, 1)
    else:
        raise ValueError(f"mode must be 'column' or 'global', got {mode!r}")

    # Accumulate in float64 without first copying the (large) references to float64.
    w = np.asarray(white).mean(axis=axes, dtype=np.float64, keepdims=True)
    d = np.asarray(dark).mean(axis=axes, dtype=np.float64, keepdims=True)
    span = w - d
    bad = span <= 0
    if np.any(bad):
        # A dead or saturated reference pixel would divide by zero or flip sign.
        print(f"[reflectance] {int(bad.sum())} reference entries have white <= dark; set to NaN")
        span = np.where(bad, np.nan, span)
    return (np.asarray(sample, dtype=np.float64) - d) / span
