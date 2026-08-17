#!/usr/bin/env python3
"""Download and compare repeated HARPS spectra from the ESO shortlist.

The analysis is intentionally conservative: it searches for narrow features
that change between two epochs, then labels common instrumental/atmospheric
explanations. Local significance values are ranking statistics rather than
source-origin probabilities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.time import Time
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks, peak_widths
from scipy.stats import binned_statistic, norm


C_KMS = 299_792.458
HARPS_RESOLVING_POWER = 115_000.0
USER_AGENT = "eso-underexplored-harps/0.1"
DEFAULT_PROGRAM = "106.21TJ.001"
TOI2458_FEATURE_DP_ID = "ADP.2022-02-20T01:06:53.624"
TOI2458_CANDIDATE_WAVELENGTHS = (5761.26, 6432.95)
OBSERVER_FRAME_LINES_ANGSTROM = {
    "OI_sky_5577": 5577.338,
    "OI_sky_6300": 6300.304,
    "OI_sky_6363": 6363.776,
}
STELLAR_FRAME_LINES_ANGSTROM = {
    "Na_D2_5889": 5889.950,
    "Na_D1_5895": 5895.924,
    "H_alpha_6563": 6562.800,
}


@dataclass
class Spectrum:
    dp_id: str
    target: str
    observed_at: str
    path: Path
    wave: np.ndarray
    flux: np.ndarray
    normalized: np.ndarray
    berv_kms: float
    header_snr: float


@dataclass
class PairResult:
    target: str
    first: Spectrum
    second: Spectrum
    wave: np.ndarray
    first_norm: np.ndarray
    second_norm: np.ndarray
    residual: np.ndarray
    sigma: np.ndarray
    zscore: np.ndarray
    valid: np.ndarray
    shift_kms: float
    noise_median: float
    candidates: list[dict[str, object]]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_filename(dp_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", dp_id) + ".fits"


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value).strip("._") or "unnamed"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_fits(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(6) == b"SIMPLE"
    except OSError:
        return False


def download_file(url: str, destination: Path, timeout: float) -> None:
    if destination.exists() and is_fits(destination):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with temporary.open("wb") as handle:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    handle.write(chunk)
        if not is_fits(temporary):
            raise RuntimeError(f"download was not a FITS file: {url}")
        temporary.replace(destination)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        temporary.unlink(missing_ok=True)
        raise


def download_command(args: argparse.Namespace) -> int:
    selected = [
        row
        for row in read_csv(args.review_csv)
        if row.get("program_id") == args.program
    ][: args.limit]
    if not selected:
        raise SystemExit(f"no {args.program} rows found in {args.review_csv}")

    manifest: list[dict[str, object]] = []
    for index, row in enumerate(selected, 1):
        dp_id = row["dp_id"]
        path = args.data_dir / safe_filename(dp_id)
        url = f"https://dataportal.eso.org/dataPortal/file/{dp_id}"
        print(f"[{index}/{len(selected)}] {dp_id}", flush=True)
        download_file(url, path, args.timeout)
        manifest.append(
            {
                "dp_id": dp_id,
                "program_id": args.program,
                "target_name": row["target_name"],
                "observed_at_utc": row["observed_at_utc"],
                "archive_snr": row["snr"],
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "source_url": url,
            }
        )
    write_csv(
        args.manifest,
        manifest,
        [
            "dp_id",
            "program_id",
            "target_name",
            "observed_at_utc",
            "archive_snr",
            "path",
            "bytes",
            "sha256",
            "source_url",
        ],
    )
    print(f"wrote {len(manifest)} rows to {args.manifest}")
    return 0


def continuum_normalize(
    wave: np.ndarray, flux: np.ndarray, *, bin_width_angstrom: float = 5.0
) -> np.ndarray:
    valid = np.isfinite(wave) & np.isfinite(flux) & (flux > 0)
    if valid.sum() < 100:
        raise ValueError("spectrum has too few positive finite samples")
    low = math.floor(float(np.nanmin(wave[valid])) / bin_width_angstrom) * bin_width_angstrom
    high = math.ceil(float(np.nanmax(wave[valid])) / bin_width_angstrom) * bin_width_angstrom
    edges = np.arange(low, high + bin_width_angstrom, bin_width_angstrom)
    medians, _, _ = binned_statistic(
        wave[valid], flux[valid], statistic="median", bins=edges
    )
    centers = (edges[:-1] + edges[1:]) / 2
    good = np.isfinite(medians) & (medians > 0)
    if good.sum() < 3:
        raise ValueError("could not estimate a spectral continuum")
    continuum = np.interp(wave, centers[good], medians[good])
    normalized = np.full_like(flux, np.nan, dtype=float)
    normalized[valid] = flux[valid] / continuum[valid]
    return normalized


def load_spectrum(row: dict[str, str]) -> Spectrum:
    path = Path(row["path"])
    with fits.open(path, memmap=True) as hdus:
        header = hdus[0].header
        table = hdus[1].data
        wave = np.array(table["WAVE"][0], dtype=float)
        flux = np.array(table["FLUX"][0], dtype=float)
        fits_target = str(header.get("OBJECT", "")).strip()
        fits_program = str(
            header.get("PROG_ID", header.get("HIERARCH ESO OBS PROG ID", ""))
        ).strip()
        if fits_target and fits_target.replace(" ", "") != row["target_name"].replace(" ", ""):
            raise ValueError(
                f"target mismatch for {row['dp_id']}: manifest={row['target_name']} FITS={fits_target}"
            )
        if fits_program and fits_program != row["program_id"]:
            raise ValueError(
                f"program mismatch for {row['dp_id']}: manifest={row['program_id']} FITS={fits_program}"
            )
        berv = float(header.get("ESO DRS BERV", np.nan))
        header_snr = float(header.get("SNR", np.nan))
    normalized = continuum_normalize(wave, flux)
    return Spectrum(
        dp_id=row["dp_id"],
        target=row["target_name"],
        observed_at=row["observed_at_utc"],
        path=path,
        wave=wave,
        flux=flux,
        normalized=normalized,
        berv_kms=berv,
        header_snr=header_snr,
    )


def _overlap(reference: Spectrum, moving: Spectrum) -> tuple[np.ndarray, np.ndarray]:
    low = max(float(reference.wave[0]), float(moving.wave[0])) + 2.0
    high = min(float(reference.wave[-1]), float(moving.wave[-1])) - 2.0
    mask = (reference.wave >= low) & (reference.wave <= high)
    return reference.wave[mask], reference.normalized[mask]


def align_velocity(reference: Spectrum, moving: Spectrum) -> float:
    wave, ref = _overlap(reference, moving)
    sample = np.arange(0, len(wave), 20)
    wave_sample = wave[sample]
    ref_sample = ref[sample]

    def objective(velocity_kms: float) -> float:
        query = wave_sample * (1.0 + velocity_kms / C_KMS)
        mov = np.interp(query, moving.wave, moving.normalized, left=np.nan, right=np.nan)
        valid = (
            np.isfinite(ref_sample)
            & np.isfinite(mov)
            & (ref_sample > 0.15)
            & (ref_sample < 1.8)
            & (mov > 0.15)
            & (mov < 1.8)
        )
        if valid.sum() < 1000:
            return float("inf")
        difference = mov[valid] - ref_sample[valid]
        return float(np.median(np.abs(difference - np.median(difference))))

    result = minimize_scalar(
        objective, bounds=(-2.0, 2.0), method="bounded", options={"xatol": 0.002}
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"could not align {moving.dp_id} to {reference.dp_id}")
    return float(result.x)


def local_robust_sigma(
    wave: np.ndarray, residual: np.ndarray, valid: np.ndarray, *, bin_width: float = 20.0
) -> np.ndarray:
    low = math.floor(float(wave[0]) / bin_width) * bin_width
    high = math.ceil(float(wave[-1]) / bin_width) * bin_width
    edges = np.arange(low, high + bin_width, bin_width)
    centers = (edges[:-1] + edges[1:]) / 2
    sigma = np.full(len(centers), np.nan)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        section = valid & (wave >= left) & (wave < right)
        values = residual[section]
        if len(values) < 100:
            continue
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        sigma[index] = 1.4826 * mad
    good = np.isfinite(sigma) & (sigma > 1e-8)
    if good.sum() < 3:
        raise RuntimeError("could not estimate local residual noise")
    floor = float(np.nanpercentile(sigma[good], 10)) * 0.5
    return np.maximum(np.interp(wave, centers[good], sigma[good]), floor)


def nearest_known_line(
    barycentric_wavelength: float,
    observer_wavelength: float,
    tolerance: float = 0.20,
) -> str:
    matches = [
        name
        for name, known_wavelength in STELLAR_FRAME_LINES_ANGSTROM.items()
        if abs(barycentric_wavelength - known_wavelength) <= tolerance
    ]
    matches.extend(
        name
        for name, known_wavelength in OBSERVER_FRAME_LINES_ANGSTROM.items()
        if abs(observer_wavelength - known_wavelength) <= tolerance
    )
    return ";".join(matches)


def detect_candidates(
    *,
    target: str,
    first: Spectrum,
    second: Spectrum,
    wave: np.ndarray,
    first_norm: np.ndarray,
    second_norm: np.ndarray,
    residual: np.ndarray,
    sigma: np.ndarray,
    zscore: np.ndarray,
    valid: np.ndarray,
    min_peak_z: float,
) -> list[dict[str, object]]:
    search = np.where(valid, np.abs(zscore), 0.0)
    peaks, properties = find_peaks(
        search,
        height=min_peak_z,
        prominence=max(5.0, min_peak_z * 0.65),
        distance=3,
    )
    widths = peak_widths(search, peaks, rel_height=0.5)[0] if len(peaks) else []
    candidates: list[dict[str, object]] = []
    pixel_scale = float(np.nanmedian(np.diff(wave)))
    for peak, width, height, prominence in zip(
        peaks,
        widths,
        properties.get("peak_heights", []),
        properties.get("prominences", []),
        strict=True,
    ):
        if width > 14:
            continue
        wavelength = float(wave[peak])
        instrumental_fwhm_pixels = wavelength / HARPS_RESOLVING_POWER / pixel_scale
        width_ratio = float(width) / instrumental_fwhm_pixels
        sign = "brighter_later" if residual[peak] > 0 else "brighter_earlier"
        brighter = second_norm if residual[peak] > 0 else first_norm
        half_context = max(8, int(math.ceil(3.5 * instrumental_fwhm_pixels)))
        center_exclusion = max(2, int(math.ceil(0.75 * instrumental_fwhm_pixels)))
        left = max(0, peak - half_context)
        right = min(len(wave), peak + half_context + 1)
        side_indices = np.r_[left : max(left, peak - center_exclusion), min(right, peak + center_exclusion + 1) : right]
        side_indices = side_indices[valid[side_indices]]
        local_baseline = (
            float(np.nanmedian(brighter[side_indices])) if len(side_indices) else float("nan")
        )
        emission_excess = float(brighter[peak] - local_baseline)

        neighbor_radius = max(5, int(math.ceil(3 * instrumental_fwhm_pixels)))
        nearby = residual[max(0, peak - neighbor_radius) : min(len(residual), peak + neighbor_radius + 1)]
        opposite = nearby * residual[peak] < 0
        opposite_abs_z = np.abs(
            zscore[max(0, peak - neighbor_radius) : min(len(zscore), peak + neighbor_radius + 1)]
        )
        opposite_values = opposite_abs_z[opposite]
        opposite_values = opposite_values[np.isfinite(opposite_values)]
        strongest_opposite_z = (
            float(np.max(opposite_values)) if len(opposite_values) else 0.0
        )
        bipolar = strongest_opposite_z >= max(6.0, 0.35 * float(height))

        first_topocentric = wavelength / (1.0 + first.berv_kms / C_KMS)
        second_topocentric = wavelength / (1.0 + second.berv_kms / C_KMS)
        feature_topocentric = (
            second_topocentric if residual[peak] > 0 else first_topocentric
        )
        known_line = nearest_known_line(wavelength, feature_topocentric)
        status = "follow_up_required"
        reason = "instrument-width bright feature in one epoch; cosmic ray remains unexcluded"
        if wavelength < 4000.0 or wavelength > 6800.0:
            status = "low_sensitivity_spectral_edge"
            reason = "feature is in the low-sensitivity edge of the extracted spectrum"
        elif width_ratio < 0.75:
            status = "likely_single_pixel_or_cosmic_ray"
            reason = "feature is narrower than the instrumental line-spread function"
        elif width_ratio > 1.8:
            status = "likely_stellar_line_profile_change"
            reason = "feature is substantially broader than an unresolved HARPS line"
        elif known_line:
            status = "known_line_proximity"
            reason = "feature is close to a common sky/stellar reference line"
        elif not np.isfinite(local_baseline) or emission_excess <= max(
            0.01, 3.0 * float(sigma[peak])
        ):
            status = "likely_stellar_absorption_variability"
            reason = "difference does not form a bright line above the local spectrum"
        elif bipolar:
            status = "likely_alignment_or_line_profile_change"
            reason = "nearby opposite-sign residual is characteristic of a shifted or changing line"
        candidates.append(
            {
                "target_name": target,
                "first_dp_id": first.dp_id,
                "second_dp_id": second.dp_id,
                "wavelength_barycentric_angstrom": round(wavelength, 5),
                "first_topocentric_angstrom": round(
                    first_topocentric, 5
                ),
                "second_topocentric_angstrom": round(
                    second_topocentric, 5
                ),
                "feature_topocentric_angstrom": round(feature_topocentric, 5),
                "direction": sign,
                "peak_abs_z": round(float(height), 3),
                "prominence_z": round(float(prominence), 3),
                "fwhm_pixels": round(float(width), 3),
                "instrumental_fwhm_pixels": round(instrumental_fwhm_pixels, 3),
                "fwhm_to_instrument_ratio": round(width_ratio, 3),
                "residual_normalized_flux": round(float(residual[peak]), 6),
                "brighter_epoch_excess_above_local_spectrum": round(emission_excess, 6),
                "strongest_nearby_opposite_sign_z": round(strongest_opposite_z, 3),
                "known_line": known_line,
                "status": status,
                "reason": reason,
            }
        )
    return candidates


def compare_pair(
    first: Spectrum, second: Spectrum, *, min_peak_z: float = 8.0
) -> PairResult:
    shift = align_velocity(first, second)
    wave, first_norm = _overlap(first, second)
    query = wave * (1.0 + shift / C_KMS)
    second_norm = np.interp(
        query, second.wave, second.normalized, left=np.nan, right=np.nan
    )
    valid = (
        np.isfinite(first_norm)
        & np.isfinite(second_norm)
        & (first_norm > 0.15)
        & (second_norm > 0.15)
        & (first_norm < 3.0)
        & (second_norm < 3.0)
    )
    raw_residual = second_norm - first_norm
    filled = np.where(valid, raw_residual, 0.0)
    broad = gaussian_filter1d(filled, sigma=40.0, mode="nearest")
    residual = raw_residual - broad
    sigma = local_robust_sigma(wave, residual, valid)
    zscore = np.where(valid, residual / sigma, np.nan)
    candidates = detect_candidates(
        target=first.target,
        first=first,
        second=second,
        wave=wave,
        first_norm=first_norm,
        second_norm=second_norm,
        residual=residual,
        sigma=sigma,
        zscore=zscore,
        valid=valid,
        min_peak_z=min_peak_z,
    )
    return PairResult(
        target=first.target,
        first=first,
        second=second,
        wave=wave,
        first_norm=first_norm,
        second_norm=second_norm,
        residual=residual,
        sigma=sigma,
        zscore=zscore,
        valid=valid,
        shift_kms=shift,
        noise_median=float(np.nanmedian(sigma[valid])),
        candidates=candidates,
    )


def label_cross_target_artifacts(results: Sequence[PairResult]) -> None:
    all_candidates = [candidate for result in results for candidate in result.candidates]
    for candidate in all_candidates:
        if candidate["status"] != "follow_up_required":
            continue
        matches: set[str] = set()
        topo = float(candidate["feature_topocentric_angstrom"])
        for other in all_candidates:
            if other is candidate or other["target_name"] == candidate["target_name"]:
                continue
            other_topo = float(other["feature_topocentric_angstrom"])
            if abs(topo - other_topo) <= 0.05:
                matches.add(str(other["target_name"]))
        if matches:
            candidate["status"] = "likely_observer_frame_artifact"
            candidate["reason"] = "topocentric wavelength recurs in unrelated targets"
            candidate["cross_target_matches"] = ";".join(sorted(matches))
        else:
            candidate["cross_target_matches"] = ""


def score_candidate(candidate: dict[str, object], independent_trials: int) -> dict[str, object]:
    """Return idealized single-trial and global noise-ranking statistics."""

    peak_z = float(candidate["peak_abs_z"])
    single_log10_p = (math.log(2.0) + norm.logsf(peak_z)) / math.log(10.0)
    global_log10_fap = min(
        0.0, single_log10_p + math.log10(max(1, independent_trials))
    )

    return {
        "gaussian_single_trial_log10_p": round(single_log10_p, 3),
        "gaussian_global_log10_fap": round(global_log10_fap, 3),
    }


def injection_trial(
    result: PairResult, *, amplitudes: Sequence[float], trials: int, seed: int
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    valid_indices = np.flatnonzero(
        result.valid
        & (result.first_norm > 0.5)
        & (result.first_norm < 1.5)
        & (result.wave > result.wave[0] + 20)
        & (result.wave < result.wave[-1] - 20)
    )
    if len(valid_indices) < trials:
        return []
    centers = rng.choice(valid_indices, size=trials, replace=False)
    rows: list[dict[str, object]] = []
    for amplitude in amplitudes:
        recovered = 0
        for center in centers:
            instrumental_fwhm = (
                result.wave[center] / HARPS_RESOLVING_POWER
                / float(np.nanmedian(np.diff(result.wave)))
            )
            sigma_pixels = instrumental_fwhm / 2.355
            local_pixel = np.arange(len(result.wave)) - center
            gaussian = amplitude * np.exp(-0.5 * (local_pixel / sigma_pixels) ** 2)
            injected = result.residual + gaussian
            zscore = np.where(result.valid, injected / result.sigma, np.nan)
            radius = max(5, int(math.ceil(instrumental_fwhm)))
            window = zscore[max(0, center - radius) : center + radius + 1]
            if np.nanmax(window) >= 8.0 and np.sum(window >= 5.0) >= 2:
                recovered += 1
        rows.append(
            {
                "target_name": result.target,
                "amplitude_fraction_of_continuum": amplitude,
                "profile": "unresolved_at_harps_R115000",
                "trials": trials,
                "recovered": recovered,
                "recovery_fraction": round(recovered / trials, 3),
            }
        )
    return rows


def plot_pair(result: PairResult, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    follow_up = [
        item for item in result.candidates if item["status"] == "follow_up_required"
    ]
    candidates = sorted(
        follow_up or result.candidates,
        key=lambda item: float(item["peak_abs_z"]),
        reverse=True,
    )[:8]
    figure, axes = plt.subplots(2 + len(candidates), 1, figsize=(12, 3 * (2 + len(candidates))))
    sample = slice(None, None, 50)
    axes[0].plot(result.wave[sample], result.first_norm[sample], lw=0.5, label="first")
    axes[0].plot(result.wave[sample], result.second_norm[sample], lw=0.5, alpha=0.7, label="second")
    axes[0].set_ylabel("normalized flux")
    axes[0].legend(loc="upper right")
    axes[0].set_title(
        f"{result.target}: {result.first.dp_id} vs {result.second.dp_id}"
    )
    axes[1].plot(result.wave[sample], result.zscore[sample], lw=0.5, color="black")
    axes[1].axhline(8, color="red", ls="--", lw=0.7)
    axes[1].axhline(-8, color="red", ls="--", lw=0.7)
    axes[1].set_ylabel("local residual z")
    for axis, candidate in zip(axes[2:], candidates, strict=True):
        center = float(candidate["wavelength_barycentric_angstrom"])
        region = (result.wave >= center - 0.4) & (result.wave <= center + 0.4)
        axis.plot(result.wave[region], result.first_norm[region], label="first")
        axis.plot(result.wave[region], result.second_norm[region], label="second")
        axis.set_title(
            f"{center:.3f} Å; |z|={candidate['peak_abs_z']}; {candidate['status']}"
        )
        axis.set_ylabel("normalized flux")
    axes[-1].set_xlabel("barycentric wavelength [Å]")
    figure.tight_layout()
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def spectrum_row_from_fits(path: Path) -> dict[str, str]:
    """Build the manifest fields needed by ``load_spectrum`` from a local FITS file."""

    header = fits.getheader(path)
    arcfile = str(header.get("ARCFILE", path.stem)).removesuffix(".fits")
    program = str(
        header.get("PROG_ID", header.get("HIERARCH ESO OBS PROG ID", ""))
    ).strip()
    return {
        "path": str(path),
        "dp_id": arcfile,
        "target_name": str(header.get("OBJECT", "")).strip(),
        "program_id": program,
        "observed_at_utc": str(header.get("DATE-OBS", "")),
    }


def narrow_line_excess(spectrum: Spectrum, wavelength: float) -> tuple[float, float]:
    """Measure the highest local excess within two instrumental FWHM."""

    fwhm = wavelength / HARPS_RESOLVING_POWER
    core = np.abs(spectrum.wave - wavelength) <= 2.0 * fwhm
    side = (
        (np.abs(spectrum.wave - wavelength) >= 3.0 * fwhm)
        & (np.abs(spectrum.wave - wavelength) <= 8.0 * fwhm)
    )
    valid_core = core & np.isfinite(spectrum.normalized)
    valid_side = side & np.isfinite(spectrum.normalized)
    if not np.any(valid_core) or valid_side.sum() < 5:
        return float("nan"), float("nan")
    baseline = float(np.median(spectrum.normalized[valid_side]))
    indices = np.flatnonzero(valid_core)
    peak = int(indices[np.argmax(spectrum.normalized[indices])])
    return float(spectrum.normalized[peak] - baseline), float(spectrum.wave[peak])


def narrow_line_metrics(spectrum: Spectrum, wavelength: float) -> dict[str, float]:
    """Measure a narrow peak and robust local scatter around one wavelength."""

    fwhm = wavelength / HARPS_RESOLVING_POWER
    distance = np.abs(spectrum.wave - wavelength)
    core = distance <= 2.0 * fwhm
    side = (distance >= 3.0 * fwhm) & (distance <= 8.0 * fwhm)
    valid_core = core & np.isfinite(spectrum.normalized)
    valid_side = side & np.isfinite(spectrum.normalized)
    if not np.any(valid_core) or valid_side.sum() < 5:
        return {
            "query_wavelength": wavelength,
            "peak_wavelength": float("nan"),
            "peak_normalized_flux": float("nan"),
            "local_baseline": float("nan"),
            "local_sigma": float("nan"),
            "excess": float("nan"),
            "local_z": float("nan"),
        }
    side_values = spectrum.normalized[valid_side]
    baseline = float(np.median(side_values))
    mad = float(np.median(np.abs(side_values - baseline)))
    local_sigma = max(1.4826 * mad, 1e-8)
    indices = np.flatnonzero(valid_core)
    peak = int(indices[np.argmax(spectrum.normalized[indices])])
    peak_flux = float(spectrum.normalized[peak])
    excess = peak_flux - baseline
    return {
        "query_wavelength": wavelength,
        "peak_wavelength": float(spectrum.wave[peak]),
        "peak_normalized_flux": peak_flux,
        "local_baseline": baseline,
        "local_sigma": local_sigma,
        "excess": excess,
        "local_z": excess / local_sigma,
    }


def observer_frame_barycentric_wavelength(
    topocentric_wavelength: float, berv_kms: float
) -> float:
    """Locate a fixed observer-frame wavelength on a barycentric wavelength grid."""

    return topocentric_wavelength * (1.0 + berv_kms / C_KMS)


def toi2458_archive_query() -> str:
    """Return the fixed ADQL query used for the complete public epoch list."""

    return """SELECT dp_id,obs_id,proposal_id,target_name,instrument_name,
t_min,t_exptime,em_res_power,snr,access_estsize,access_url
FROM ivoa.ObsCore
WHERE dataproduct_type='spectrum'
  AND calib_level>=2
  AND target_name='TOI-2458'
  AND instrument_name LIKE 'HARPS%'
  AND em_res_power>=100000
ORDER BY t_min"""


def fetch_toi2458_archive_rows(timeout: float) -> list[dict[str, str]]:
    """Query ESO TAP lazily so the analysis module retains a small import surface."""

    from find_underexplored import tap_csv

    return tap_csv(toi2458_archive_query(), timeout=timeout)


def archive_manifest_row(
    row: dict[str, str], data_dir: Path
) -> dict[str, object]:
    dp_id = row["dp_id"].strip()
    observed_at = Time(float(row["t_min"]), format="mjd").isot + "+00:00"
    path = data_dir / safe_filename(dp_id)
    return {
        "dp_id": dp_id,
        "obs_id": row.get("obs_id", ""),
        "program_id": row.get("proposal_id", "").strip(),
        "target_name": row.get("target_name", "").strip(),
        "instrument_name": row.get("instrument_name", "").strip(),
        "observed_at_utc": observed_at,
        "archive_snr": row.get("snr", ""),
        "path": str(path),
        "bytes": "",
        "sha256": "",
        "source_url": f"https://dataportal.eso.org/dataPortal/file/{dp_id}",
    }


def toi2458_download_command(args: argparse.Namespace) -> int:
    """Query, download, validate, and checksum all public TOI-2458 spectra."""

    if args.archive_csv.exists() and not args.refresh_query:
        rows = read_csv(args.archive_csv)
    else:
        rows = fetch_toi2458_archive_rows(args.timeout)
        write_csv(
            args.archive_csv,
            rows,
            [
                "dp_id",
                "obs_id",
                "proposal_id",
                "target_name",
                "instrument_name",
                "t_min",
                "t_exptime",
                "em_res_power",
                "snr",
                "access_estsize",
                "access_url",
            ],
        )
    unique = {row["dp_id"].strip(): row for row in rows if row.get("dp_id", "").strip()}
    manifest_rows = [archive_manifest_row(row, args.data_dir) for row in unique.values()]
    manifest_rows.sort(key=lambda row: str(row["observed_at_utc"]))

    def retrieve(manifest_row: dict[str, object]) -> dict[str, object]:
        path = Path(str(manifest_row["path"]))
        download_file(str(manifest_row["source_url"]), path, args.timeout)
        with fits.open(path, memmap=True) as hdus:
            hdus.verify("exception")
            target = str(hdus[0].header.get("OBJECT", "")).replace(" ", "")
            if target != "TOI-2458":
                raise ValueError(f"unexpected target {target!r} in {path}")
        manifest_row["bytes"] = path.stat().st_size
        manifest_row["sha256"] = sha256_file(path)
        return manifest_row

    completed: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(retrieve, row): row for row in manifest_rows}
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            completed.append(row)
            print(f"[{index}/{len(manifest_rows)}] {row['dp_id']}", flush=True)
    completed.sort(key=lambda row: str(row["observed_at_utc"]))
    write_csv(
        args.manifest,
        completed,
        [
            "dp_id",
            "obs_id",
            "program_id",
            "target_name",
            "instrument_name",
            "observed_at_utc",
            "archive_snr",
            "path",
            "bytes",
            "sha256",
            "source_url",
        ],
    )
    print(f"wrote {len(completed)} verified spectra to {args.manifest}")
    return 0


def _plot_toi2458_line_scan(
    rows: Sequence[dict[str, object]], wavelengths: Sequence[float], destination: Path
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(len(wavelengths), 2, figsize=(13, 4 * len(wavelengths)))
    axes = np.atleast_2d(axes)
    for index, wavelength in enumerate(wavelengths):
        selected = [row for row in rows if float(row["candidate_wavelength_angstrom"]) == wavelength]
        dates = [datetime.fromisoformat(str(row["observed_at_utc"])) for row in selected]
        excess = np.array([float(row["star_frame_excess"]) for row in selected])
        feature = np.array([bool(row["is_feature_exposure"]) for row in selected])
        axes[index, 0].plot(dates, excess, ".", color="tab:blue", label="other epochs")
        axes[index, 0].plot(
            np.array(dates, dtype=object)[feature], excess[feature], "*", ms=14,
            color="tab:red", label="candidate exposure"
        )
        axes[index, 0].axhline(0.05, color="gray", ls="--", lw=0.8)
        axes[index, 0].set_ylabel("local normalized excess")
        axes[index, 0].set_title(f"{wavelength:.2f} Å across all epochs")
        axes[index, 0].legend(loc="best")
        other = excess[~feature & np.isfinite(excess)]
        axes[index, 1].hist(other, bins=20, color="tab:blue", alpha=0.75)
        if np.any(feature):
            axes[index, 1].axvline(excess[feature][0], color="tab:red", lw=2)
        axes[index, 1].set_xlabel("local normalized excess")
        axes[index, 1].set_ylabel("exposures")
        axes[index, 1].set_title("comparison distribution")
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def toi2458_scan_command(args: argparse.Namespace) -> int:
    """Measure pre-registered candidate wavelengths in every downloaded epoch."""

    spectra = [load_spectrum(row) for row in read_csv(args.manifest)]
    feature = next((item for item in spectra if item.dp_id == args.feature_dp_id), None)
    if feature is None:
        raise ValueError(f"feature exposure {args.feature_dp_id} is absent from the manifest")

    output_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for wavelength in args.wavelengths:
        feature_star = narrow_line_metrics(feature, wavelength)
        topocentric = wavelength / (1.0 + feature.berv_kms / C_KMS)
        threshold = max(0.05, 0.5 * feature_star["excess"])
        recurrence_min_local_z = 5.0
        line_rows: list[dict[str, object]] = []
        for spectrum in spectra:
            star = narrow_line_metrics(spectrum, wavelength)
            observer_query = observer_frame_barycentric_wavelength(
                topocentric, spectrum.berv_kms
            )
            observer = narrow_line_metrics(spectrum, observer_query)
            row = {
                "candidate_wavelength_angstrom": wavelength,
                "candidate_topocentric_angstrom": round(topocentric, 6),
                "dp_id": spectrum.dp_id,
                "program_id": spectrum_row_from_fits(spectrum.path)["program_id"],
                "observed_at_utc": spectrum.observed_at,
                "berv_kms": round(spectrum.berv_kms, 6),
                "header_snr": spectrum.header_snr,
                "is_feature_exposure": spectrum.dp_id == args.feature_dp_id,
                "star_frame_peak_angstrom": round(star["peak_wavelength"], 6),
                "star_frame_excess": round(star["excess"], 7),
                "star_frame_local_sigma": round(star["local_sigma"], 7),
                "star_frame_local_z": round(star["local_z"], 3),
                "observer_frame_query_barycentric_angstrom": round(observer_query, 6),
                "observer_frame_peak_angstrom": round(observer["peak_wavelength"], 6),
                "observer_frame_excess": round(observer["excess"], 7),
                "observer_frame_local_z": round(observer["local_z"], 3),
                "recurrence_threshold": round(threshold, 7),
                "recurrence_min_local_z": recurrence_min_local_z,
            }
            line_rows.append(row)
            output_rows.append(row)
        comparisons = [row for row in line_rows if not row["is_feature_exposure"]]
        star_recurrences = [
            row
            for row in comparisons
            if float(row["star_frame_excess"]) >= threshold
            and float(row["star_frame_local_z"]) >= recurrence_min_local_z
        ]
        observer_recurrences = [
            row
            for row in comparisons
            if float(row["observer_frame_excess"]) >= threshold
            and float(row["observer_frame_local_z"]) >= recurrence_min_local_z
        ]
        ranked = sorted(
            line_rows, key=lambda row: float(row["star_frame_excess"]), reverse=True
        )
        feature_rank = next(
            index + 1 for index, row in enumerate(ranked) if row["is_feature_exposure"]
        )
        summaries.append(
            {
                "candidate_wavelength_angstrom": wavelength,
                "feature_dp_id": args.feature_dp_id,
                "feature_excess": round(feature_star["excess"], 7),
                "comparison_exposures": len(comparisons),
                "feature_rank_by_excess": feature_rank,
                "recurrence_threshold": round(threshold, 7),
                "recurrence_min_local_z": recurrence_min_local_z,
                "star_frame_recurrence_count": len(star_recurrences),
                "observer_frame_recurrence_count": len(observer_recurrences),
                "star_frame_recurrence_dp_ids": [
                    row["dp_id"] for row in star_recurrences
                ],
                "observer_frame_recurrence_dp_ids": [
                    row["dp_id"] for row in observer_recurrences
                ],
                "classification": (
                    "significantly_recurrent"
                    if star_recurrences or observer_recurrences
                    else "no_significant_recurrence_in_complete_public_epoch_set"
                ),
            }
        )

    fields = [
        "candidate_wavelength_angstrom",
        "candidate_topocentric_angstrom",
        "dp_id",
        "program_id",
        "observed_at_utc",
        "berv_kms",
        "header_snr",
        "is_feature_exposure",
        "star_frame_peak_angstrom",
        "star_frame_excess",
        "star_frame_local_sigma",
        "star_frame_local_z",
        "observer_frame_query_barycentric_angstrom",
        "observer_frame_peak_angstrom",
        "observer_frame_excess",
        "observer_frame_local_z",
        "recurrence_threshold",
        "recurrence_min_local_z",
    ]
    write_csv(args.output, output_rows, fields)
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "archive_spectra": len(spectra),
        "unique_dp_ids": len({spectrum.dp_id for spectrum in spectra}),
        "paper_reported_spectra": 87,
        "archive_count_note": (
            "The public archive query returns 88 distinct products. All are retained; "
            "this is one more than the paper's reported acquired sample and may reflect "
            "different archive-selection or provenance rules."
        ),
        "pre_registered_wavelengths_angstrom": list(args.wavelengths),
        "lines": summaries,
        "scope": "fixed star-frame and observer-frame recurrence checks",
        "statistical_warning": (
            "Local significance values are diagnostics, not calibrated source-origin probabilities."
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _plot_toi2458_line_scan(output_rows, args.wavelengths, args.plot)
    print(json.dumps(summary, indent=2))
    return 0


def recurrence_command(args: argparse.Namespace) -> int:
    """Check shortlisted features in every downloaded same-target spectrum."""

    candidate_rows = [
        row
        for row in read_csv(args.candidates)
        if row.get("status") == "follow_up_required"
    ]
    if not candidate_rows:
        raise SystemExit(f"no follow-up candidates found in {args.candidates}")

    paths = [Path(row["path"]) for row in read_csv(args.manifest)]
    paths.extend(sorted(args.followup_dir.glob("*.fits")))
    unique_paths = list(dict.fromkeys(path.resolve() for path in paths))
    spectra = [load_spectrum(spectrum_row_from_fits(path)) for path in unique_paths]
    by_target: dict[str, list[Spectrum]] = defaultdict(list)
    for spectrum in spectra:
        by_target[spectrum.target].append(spectrum)

    output_rows: list[dict[str, object]] = []
    for candidate in candidate_rows:
        target = str(candidate["target_name"])
        wavelength = float(candidate["wavelength_barycentric_angstrom"])
        feature_dp_id = str(
            candidate["second_dp_id"]
            if candidate["direction"] == "brighter_later"
            else candidate["first_dp_id"]
        )
        target_spectra = by_target.get(target, [])
        feature_spectrum = next(
            (spectrum for spectrum in target_spectra if spectrum.dp_id == feature_dp_id),
            None,
        )
        if feature_spectrum is None:
            continue
        feature_excess, feature_peak = narrow_line_excess(feature_spectrum, wavelength)
        comparison_values: list[tuple[Spectrum, float, float]] = []
        for spectrum in target_spectra:
            if spectrum.dp_id == feature_dp_id:
                continue
            excess, peak = narrow_line_excess(spectrum, wavelength)
            if np.isfinite(excess):
                comparison_values.append((spectrum, excess, peak))
        strongest = max(comparison_values, key=lambda item: item[1], default=None)
        recurrence_threshold = max(0.05, 0.5 * feature_excess)
        recurring = [
            item
            for item in comparison_values
            if item[1] >= recurrence_threshold
            and abs(item[2] - feature_peak) <= wavelength / HARPS_RESOLVING_POWER
        ]
        output_rows.append(
            {
                "target_name": target,
                "wavelength_barycentric_angstrom": round(wavelength, 5),
                "feature_dp_id": feature_dp_id,
                "feature_excess_above_local_spectrum": round(feature_excess, 6),
                "feature_peak_wavelength": round(feature_peak, 5),
                "comparison_exposures": len(comparison_values),
                "strongest_comparison_dp_id": strongest[0].dp_id if strongest else "",
                "strongest_comparison_excess": round(strongest[1], 6) if strongest else "",
                "strongest_comparison_peak_wavelength": round(strongest[2], 5) if strongest else "",
                "recurrence_threshold": round(recurrence_threshold, 6),
                "recurrence_count": len(recurring),
                "recurrence_status": (
                    "recurrent" if recurring else "not_recurrent_in_downloaded_comparisons"
                ),
            }
        )

    fields = [
        "target_name",
        "wavelength_barycentric_angstrom",
        "feature_dp_id",
        "feature_excess_above_local_spectrum",
        "feature_peak_wavelength",
        "comparison_exposures",
        "strongest_comparison_dp_id",
        "strongest_comparison_excess",
        "strongest_comparison_peak_wavelength",
        "recurrence_threshold",
        "recurrence_count",
        "recurrence_status",
    ]
    write_csv(args.output, output_rows, fields)

    followup_manifest: list[dict[str, object]] = []
    for path in sorted(args.followup_dir.glob("*.fits")):
        row = spectrum_row_from_fits(path)
        followup_manifest.append(
            {
                **row,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "source_url": f"https://dataportal.eso.org/dataPortal/file/{row['dp_id']}",
            }
        )
    write_csv(
        args.followup_manifest,
        followup_manifest,
        [
            "dp_id",
            "program_id",
            "target_name",
            "observed_at_utc",
            "path",
            "bytes",
            "sha256",
            "source_url",
        ],
    )
    print(f"wrote {len(output_rows)} recurrence checks to {args.output}")
    print(f"wrote {len(followup_manifest)} spectra to {args.followup_manifest}")
    return 0


def analyze_command(args: argparse.Namespace) -> int:
    rows = read_csv(args.manifest)
    spectra = [load_spectrum(row) for row in rows]
    grouped: dict[str, list[Spectrum]] = defaultdict(list)
    for spectrum in spectra:
        grouped[spectrum.target].append(spectrum)
    pairs: list[tuple[Spectrum, Spectrum]] = []
    for target, target_spectra in sorted(grouped.items()):
        target_spectra.sort(key=lambda spectrum: spectrum.observed_at)
        if len(target_spectra) < 2:
            continue
        pairs.append((target_spectra[0], target_spectra[-1]))
    if not pairs:
        raise SystemExit("manifest contains no repeated targets")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[PairResult] = []
    for index, (first, second) in enumerate(pairs, 1):
        print(f"[{index}/{len(pairs)}] comparing {first.target}", flush=True)
        results.append(compare_pair(first, second, min_peak_z=args.min_peak_z))
    label_cross_target_artifacts(results)
    for result in results:
        independent_trials = max(1, int(result.valid.sum() / 3.3))
        for candidate in result.candidates:
            candidate.update(score_candidate(candidate, independent_trials))

    candidate_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    injection_rows: list[dict[str, object]] = []
    for index, result in enumerate(results):
        candidate_rows.extend(result.candidates)
        status_counts: dict[str, int] = defaultdict(int)
        for candidate in result.candidates:
            status_counts[str(candidate["status"])] += 1
        pair_rows.append(
            {
                "target_name": result.target,
                "first_dp_id": result.first.dp_id,
                "second_dp_id": result.second.dp_id,
                "first_observed_at": result.first.observed_at,
                "second_observed_at": result.second.observed_at,
                "first_berv_kms": round(result.first.berv_kms, 6),
                "second_berv_kms": round(result.second.berv_kms, 6),
                "fitted_relative_shift_kms": round(result.shift_kms, 6),
                "valid_pixels": int(result.valid.sum()),
                "median_normalized_noise": round(result.noise_median, 7),
                "candidate_count": len(result.candidates),
                "follow_up_count": status_counts.get("follow_up_required", 0),
                "max_abs_z": round(
                    max((float(row["peak_abs_z"]) for row in result.candidates), default=0.0),
                    3,
                ),
            }
        )
        injection_rows.extend(
            injection_trial(
                result,
                amplitudes=(0.02, 0.05, 0.10),
                trials=args.injection_trials,
                seed=20260815 + index,
            )
        )
        plot_pair(result, args.output_dir / "plots" / f"{safe_slug(result.target)}.png")

    candidate_fields = [
        "target_name",
        "first_dp_id",
        "second_dp_id",
        "wavelength_barycentric_angstrom",
        "first_topocentric_angstrom",
        "second_topocentric_angstrom",
        "feature_topocentric_angstrom",
        "direction",
        "peak_abs_z",
        "prominence_z",
        "fwhm_pixels",
        "instrumental_fwhm_pixels",
        "fwhm_to_instrument_ratio",
        "residual_normalized_flux",
        "brighter_epoch_excess_above_local_spectrum",
        "strongest_nearby_opposite_sign_z",
        "known_line",
        "status",
        "reason",
        "cross_target_matches",
        "gaussian_single_trial_log10_p",
        "gaussian_global_log10_fap",
    ]
    pair_fields = [
        "target_name",
        "first_dp_id",
        "second_dp_id",
        "first_observed_at",
        "second_observed_at",
        "first_berv_kms",
        "second_berv_kms",
        "fitted_relative_shift_kms",
        "valid_pixels",
        "median_normalized_noise",
        "candidate_count",
        "follow_up_count",
        "max_abs_z",
    ]
    injection_fields = [
        "target_name",
        "amplitude_fraction_of_continuum",
        "profile",
        "trials",
        "recovered",
        "recovery_fraction",
    ]
    write_csv(args.output_dir / "candidates.csv", candidate_rows, candidate_fields)
    write_csv(args.output_dir / "pair_summary.csv", pair_rows, pair_fields)
    write_csv(args.output_dir / "injection_results.csv", injection_rows, injection_fields)

    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "program_id": rows[0]["program_id"],
        "spectra": len(spectra),
        "target_pairs": len(results),
        "detection_threshold_abs_z": args.min_peak_z,
        "candidates_total": len(candidate_rows),
        "candidates_still_requiring_follow_up": sum(
            row["status"] == "follow_up_required" for row in candidate_rows
        ),
        "status_counts": dict(sorted(
            (status, sum(row["status"] == status for row in candidate_rows))
            for status in {str(row["status"]) for row in candidate_rows}
        )),
        "method": (
            "5-Angstrom median-bin continuum normalization; robust relative velocity "
            "alignment; broad-residual subtraction; local 20-Angstrom MAD noise; "
            "narrow peak detection; HARPS line-spread width, local bright-line, bipolar "
            "residual, known-line, and cross-target topocentric checks"
        ),
        "scope_warning": (
            "This detects narrow features that change between the selected epochs. "
            "A perfectly constant narrow signal can subtract away."
        ),
        "statistical_warning": (
            "The Gaussian false-alarm calculation estimates idealized noise odds. "
            "Non-Gaussian spectral and instrumental structure makes it a ranking "
            "statistic rather than a calibrated global false-alarm probability."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    download_parser = subparsers.add_parser("download", help="download FITS spectra")
    download_parser.add_argument(
        "--review-csv", type=Path, default=Path("results/review_sample.csv")
    )
    download_parser.add_argument("--program", default=DEFAULT_PROGRAM)
    download_parser.add_argument("--limit", type=int, default=20)
    download_parser.add_argument(
        "--data-dir", type=Path, default=Path("data/harps_106_21tj")
    )
    download_parser.add_argument(
        "--manifest", type=Path, default=Path("data/harps_106_21tj/manifest.csv")
    )
    download_parser.add_argument("--timeout", type=float, default=180.0)
    download_parser.set_defaults(func=download_command)

    analyze_parser = subparsers.add_parser("analyze", help="compare repeated spectra")
    analyze_parser.add_argument(
        "--manifest", type=Path, default=Path("data/harps_106_21tj/manifest.csv")
    )
    analyze_parser.add_argument(
        "--output-dir", type=Path, default=Path("results/harps_residual_search")
    )
    analyze_parser.add_argument("--min-peak-z", type=float, default=8.0)
    analyze_parser.add_argument("--injection-trials", type=int, default=12)
    analyze_parser.set_defaults(func=analyze_command)

    recurrence_parser = subparsers.add_parser(
        "recurrence", help="check candidates in downloaded comparison spectra"
    )
    recurrence_parser.add_argument(
        "--manifest", type=Path, default=Path("data/harps_106_21tj/manifest.csv")
    )
    recurrence_parser.add_argument(
        "--candidates", type=Path, default=Path("results/harps_residual_search/candidates.csv")
    )
    recurrence_parser.add_argument(
        "--followup-dir", type=Path, default=Path("data/harps_followup")
    )
    recurrence_parser.add_argument(
        "--output", type=Path, default=Path("results/harps_residual_search/recurrence_results.csv")
    )
    recurrence_parser.add_argument(
        "--followup-manifest",
        type=Path,
        default=Path("data/harps_followup/manifest.csv"),
    )
    recurrence_parser.set_defaults(func=recurrence_command)

    toi_download_parser = subparsers.add_parser(
        "toi2458-download", help="download every public processed TOI-2458 HARPS spectrum"
    )
    toi_download_parser.add_argument(
        "--archive-csv",
        type=Path,
        default=Path("results/harps_residual_search/toi2458_archive_epochs.csv"),
    )
    toi_download_parser.add_argument(
        "--data-dir", type=Path, default=Path("data/toi2458_all_epochs")
    )
    toi_download_parser.add_argument(
        "--manifest", type=Path, default=Path("data/toi2458_all_epochs/manifest.csv")
    )
    toi_download_parser.add_argument("--workers", type=int, default=4)
    toi_download_parser.add_argument("--timeout", type=float, default=240.0)
    toi_download_parser.add_argument("--refresh-query", action="store_true")
    toi_download_parser.set_defaults(func=toi2458_download_command)

    toi_scan_parser = subparsers.add_parser(
        "toi2458-scan", help="scan all TOI-2458 epochs at the two pre-registered wavelengths"
    )
    toi_scan_parser.add_argument(
        "--manifest", type=Path, default=Path("data/toi2458_all_epochs/manifest.csv")
    )
    toi_scan_parser.add_argument(
        "--feature-dp-id", default=TOI2458_FEATURE_DP_ID
    )
    toi_scan_parser.add_argument(
        "--wavelengths",
        nargs="+",
        type=float,
        default=list(TOI2458_CANDIDATE_WAVELENGTHS),
    )
    toi_scan_parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/harps_residual_search/toi2458_all_epoch_line_scan.csv"),
    )
    toi_scan_parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/harps_residual_search/toi2458_all_epoch_summary.json"),
    )
    toi_scan_parser.add_argument(
        "--plot",
        type=Path,
        default=Path("results/harps_residual_search/plots/toi2458_all_epochs_lines.png"),
    )
    toi_scan_parser.set_defaults(func=toi2458_scan_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
