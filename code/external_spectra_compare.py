#!/usr/bin/env python3
"""Compare the TOI-2458 HARPS transient features with public CHIRON/TRES data.

The independent spectra predate the HARPS event by about one year. They can
test persistence or recurrence, but a non-detection cannot exclude a one-off
astrophysical transient.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
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
from numpy.polynomial.chebyshev import chebval
from scipy.ndimage import gaussian_filter1d

from harps_residual_search import (
    C_KMS,
    HARPS_RESOLVING_POWER,
    continuum_normalize,
    download_file,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "toi2458_external_spectra"
RESULT_DIR = PROJECT_ROOT / "results" / "harps_residual_search"
HARPS_DIR = PROJECT_ROOT / "data" / "toi2458_all_epochs"
HARPS_CANDIDATE = HARPS_DIR / "ADP.2022-02-20T01_06_53.624.fits"
HARPS_CONTROL = HARPS_DIR / "ADP.2022-02-20T01_06_53.630.fits"
TARGETS_AIR_ANGSTROM = (5761.26, 6432.95)
PIPELINE_VACUUM_ANGSTROM = {5761.26: 5762.863443, 6432.95: 6434.724733}
TRES_RELEVANT_SPEC_NUMBER = {5761.26: 30, 6432.95: 36}


@dataclass(frozen=True)
class ExternalFile:
    file_id: int
    filename: str
    instrument: str
    telescope: str
    resolving_power: float
    exofop_snr: float
    source_observation_date: str

    @property
    def url(self) -> str:
        return f"https://exofop.ipac.caltech.edu/tess/get_file.php?id={self.file_id}"

    @property
    def path(self) -> Path:
        return DATA_DIR / self.filename


EXTERNAL_FILES = (
    ExternalFile(
        1041566,
        "TIC449197831S-ct20210219_1127.fits",
        "CHIRON",
        "SMARTS 1.5-m, CTIO",
        80_000.0,
        59.8,
        "2021-02-19",
    ),
    ExternalFile(
        1041608,
        "TIC449197831S-ct20210304_1117.fits",
        "CHIRON",
        "SMARTS 1.5-m, CTIO",
        80_000.0,
        58.9,
        "2021-03-04",
    ),
    ExternalFile(
        197463,
        "TIC0449197831S-ab20210209.fits",
        "TRES",
        "FLWO 1.5-m",
        44_000.0,
        32.4,
        "2021-02-09",
    ),
    ExternalFile(
        200545,
        "TIC0449197831S-ab20210226.fits",
        "TRES",
        "FLWO 1.5-m",
        44_000.0,
        62.1,
        "2021-02-26",
    ),
)


@dataclass
class HarpsSpectrum:
    wave: np.ndarray
    flux: np.ndarray
    normalized: np.ndarray
    date_obs: str
    wavelength_ucd: str
    spectral_frame: str


@dataclass
class ExternalOrder:
    source: ExternalFile
    target_air: float
    wave: np.ndarray
    flux: np.ndarray
    normalized: np.ndarray
    resolving_power: float
    order_index_zero_based: int
    date_obs: str
    fits_object: str


def _write_csv(path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def air_to_vacuum(air_angstrom: float | np.ndarray) -> float | np.ndarray:
    """Convert standard dry-air wavelength to vacuum using the Ciddor relation."""

    air = np.asarray(air_angstrom, dtype=float)
    sigma = 10_000.0 / air
    refractive_index = (
        1.0
        + 0.00008336624212083
        + 0.02408926869968 / (130.1065924522 - sigma**2)
        + 0.0001599740894897 / (38.92568793293 - sigma**2)
    )
    result = air * refractive_index
    if result.ndim == 0:
        return float(result)
    return result


def load_harps(path: Path) -> HarpsSpectrum:
    with fits.open(path, memmap=False) as hdus:
        header = hdus[0].header
        table_header = hdus[1].header
        table = hdus[1].data
        wave = np.asarray(table["WAVE"][0], dtype=float)
        flux = np.asarray(table["FLUX"][0], dtype=float)
        date_obs = str(header.get("DATE-OBS", ""))
        spectral_frame = str(header.get("SPECSYS", ""))
        wavelength_ucd = str(table_header.get("TUCD1", ""))
    return HarpsSpectrum(
        wave=wave,
        flux=flux,
        normalized=continuum_normalize(wave, flux),
        date_obs=date_obs,
        wavelength_ucd=wavelength_ucd,
        spectral_frame=spectral_frame,
    )


def _wat2_text(header: fits.Header) -> str:
    return "".join(str(header[key]) for key in header if key.startswith("WAT2_"))


def tres_multispec_wave(header: fits.Header, spec_number: int) -> np.ndarray:
    """Evaluate one TRES IRAF multispec Chebyshev wavelength solution."""

    match = re.search(
        rf"spec{spec_number}\s*=\s*\"([^\"]+)\"", _wat2_text(header)
    )
    if match is None:
        raise ValueError(f"TRES multispec solution {spec_number} is missing")
    parts = match.group(1).split(maxsplit=15)
    if len(parts) != 16:
        raise ValueError(f"malformed TRES multispec solution {spec_number}")
    npix = int(parts[5])
    function_type = int(parts[11])
    polynomial_order = int(parts[12])
    pixel_min = float(parts[13])
    pixel_max = float(parts[14])
    # The FITS writer can place a card boundary between adjacent coefficients.
    # Scientific notation in these files always has a two-digit exponent, so
    # this pattern recovers coefficients even when the separating blank falls
    # exactly at a card boundary.
    coefficients = [
        float(value)
        for value in re.findall(
            r"[+-]?\d+\.\d+[eE][+-]\d{2}", parts[15]
        )
    ]
    if function_type != 2 or len(coefficients) != polynomial_order:
        raise ValueError(
            f"unsupported TRES wavelength function for spec {spec_number}: "
            f"type={function_type}, order={polynomial_order}, "
            f"coefficients={len(coefficients)}"
        )
    pixels = np.arange(1, npix + 1, dtype=float)
    normalized_pixels = (
        2.0 * pixels - (pixel_max + pixel_min)
    ) / (pixel_max - pixel_min)
    wave = chebval(normalized_pixels, coefficients)
    approximate_start = float(parts[3])
    if not np.isclose(wave[0], approximate_start, atol=0.002):
        raise ValueError(
            f"TRES wavelength validation failed for spec {spec_number}: "
            f"computed={wave[0]:.6f}, header={approximate_start:.6f}"
        )
    return wave


def load_external_order(source: ExternalFile, target_air: float) -> ExternalOrder:
    with fits.open(source.path, memmap=False) as hdus:
        header = hdus[0].header
        data = np.asarray(hdus[0].data, dtype=float)
        date_obs = str(header.get("DATE-OBS", header.get("UTSHUT", "")))
        fits_object = str(header.get("OBJECT", "")).strip()
        if fits_object != "T0449197831":
            raise ValueError(
                f"target mismatch in {source.filename}: OBJECT={fits_object!r}"
            )
        if source.instrument == "CHIRON":
            candidates = [
                index
                for index in range(data.shape[0])
                if data[index, 0, 0] <= target_air <= data[index, -1, 0]
            ]
            if not candidates:
                raise ValueError(f"CHIRON does not cover {target_air:.2f} A")
            order_index = max(
                candidates,
                key=lambda index: min(
                    target_air - data[index, 0, 0],
                    data[index, -1, 0] - target_air,
                ),
            )
            wave = data[order_index, :, 0]
            flux = data[order_index, :, 1]
            resolving_power = float(header.get("RESOLUTN", source.resolving_power))
        else:
            spec_number = TRES_RELEVANT_SPEC_NUMBER[target_air]
            order_index = spec_number - 1
            wave = tres_multispec_wave(header, spec_number)
            flux = data[order_index]
            resolving_power = source.resolving_power
    if not (wave[0] <= target_air <= wave[-1]):
        raise ValueError(
            f"{source.instrument} order {order_index} does not cover {target_air:.2f} A"
        )
    return ExternalOrder(
        source=source,
        target_air=target_air,
        wave=np.asarray(wave, dtype=float),
        flux=np.asarray(flux, dtype=float),
        normalized=continuum_normalize(np.asarray(wave), np.asarray(flux)),
        resolving_power=resolving_power,
        order_index_zero_based=order_index,
        date_obs=date_obs,
        fits_object=fits_object,
    )


def _degrade_harps(
    flux: np.ndarray,
    harps_wave: np.ndarray,
    target_air: float,
    target_resolution: float,
) -> np.ndarray:
    if target_resolution >= HARPS_RESOLVING_POWER:
        return flux.copy()
    pixel_velocity = C_KMS * float(np.nanmedian(np.diff(harps_wave))) / target_air
    added_fwhm_velocity = math.sqrt(
        (C_KMS / target_resolution) ** 2
        - (C_KMS / HARPS_RESOLVING_POWER) ** 2
    )
    sigma_pixels = added_fwhm_velocity / 2.354820045 / pixel_velocity
    return gaussian_filter1d(flux, sigma_pixels)


def _highpass_log_flux(
    wave: np.ndarray, flux: np.ndarray, resolving_power: float
) -> np.ndarray:
    valid = np.isfinite(wave) & np.isfinite(flux) & (flux > 0)
    if valid.sum() < 100:
        raise ValueError("too few valid pixels for alignment")
    pixel = float(np.nanmedian(np.diff(wave)))
    indices = np.arange(len(wave))
    log_flux = np.interp(indices, indices[valid], np.log(flux[valid]))
    broad = gaussian_filter1d(log_flux, 2.0 / pixel)
    highpass = log_flux - broad
    half_lsf_sigma_pixels = (
        float(np.nanmedian(wave)) / resolving_power / 2.354820045 * 0.5 / pixel
    )
    return gaussian_filter1d(highpass, half_lsf_sigma_pixels)


def align_external_order(
    harps: HarpsSpectrum, external: ExternalOrder
) -> tuple[float, float]:
    """Return external/reference velocity mapping and peak correlation."""

    degraded = _degrade_harps(
        harps.flux,
        harps.wave,
        external.target_air,
        external.resolving_power,
    )
    harps_highpass = _highpass_log_flux(
        harps.wave, degraded, external.resolving_power
    )
    external_highpass = _highpass_log_flux(
        external.wave, external.flux, external.resolving_power
    )
    use = (
        (external.wave > external.wave[0] + 4.0)
        & (external.wave < external.wave[-1] - 4.0)
        & (np.abs(external.wave - external.target_air) > 1.0)
    )
    wave = external.wave[use]
    observed = external_highpass[use]
    best_correlation = -np.inf
    best_velocity = np.nan
    for velocity in np.arange(-120.0, 120.0001, 0.05):
        reference_wave = wave / (1.0 + velocity / C_KMS)
        reference = np.interp(
            reference_wave,
            harps.wave,
            harps_highpass,
            left=np.nan,
            right=np.nan,
        )
        valid = np.isfinite(reference) & np.isfinite(observed)
        if valid.sum() < 500:
            continue
        first = reference[valid] - np.mean(reference[valid])
        second = observed[valid] - np.mean(observed[valid])
        denominator = math.sqrt(float(first @ first) * float(second @ second))
        if denominator <= 0:
            continue
        correlation = float(first @ second) / denominator
        if correlation > best_correlation:
            best_correlation = correlation
            best_velocity = float(velocity)
    if not np.isfinite(best_velocity):
        raise RuntimeError(f"could not align {external.source.filename}")
    return best_velocity, best_correlation


def _fit_gaussian_amplitude(
    wave: np.ndarray,
    residual: np.ndarray,
    center: float,
    fwhm: float,
    *,
    span: float = 0.8,
) -> float:
    use = np.isfinite(residual) & (np.abs(wave - center) <= span)
    x = wave[use] - center
    y = residual[use]
    if len(x) < 8:
        return float("nan")
    gaussian = np.exp(-0.5 * (x / (fwhm / 2.354820045)) ** 2)
    design = np.column_stack((np.ones(len(x)), x, gaussian))
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return float(coefficients[2])


def measure_external_feature(
    harps_candidate: HarpsSpectrum,
    harps_control: HarpsSpectrum,
    external: ExternalOrder,
    alignment_velocity_kms: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    target = external.target_air
    resolution = external.resolving_power
    candidate_degraded = _degrade_harps(
        harps_candidate.normalized,
        harps_candidate.wave,
        target,
        resolution,
    )
    control_degraded = _degrade_harps(
        harps_control.normalized,
        harps_control.wave,
        target,
        resolution,
    )
    reference_wave = external.wave / (1.0 + alignment_velocity_kms / C_KMS)
    control_on_external = np.interp(
        reference_wave,
        harps_control.wave,
        control_degraded,
        left=np.nan,
        right=np.nan,
    )
    residual = external.normalized - control_on_external

    sideband = (
        np.isfinite(residual)
        & (np.abs(reference_wave - target) < 8.0)
        & (np.abs(reference_wave - target) > 0.5)
    )
    x = reference_wave[sideband] - target
    y = residual[sideband]
    keep = np.ones(len(x), dtype=bool)
    polynomial = np.zeros(3)
    for _ in range(5):
        polynomial = np.polyfit(x[keep], y[keep], 2)
        difference = y - np.polyval(polynomial, x)
        median = float(np.median(difference[keep]))
        sigma = 1.4826 * float(np.median(np.abs(difference[keep] - median)))
        if sigma <= 0:
            break
        keep = np.abs(difference - median) < 3.0 * sigma
    detrended = residual - np.polyval(polynomial, reference_wave - target)
    fwhm = target / resolution
    observed_amplitude = _fit_gaussian_amplitude(
        reference_wave, detrended, target, fwhm
    )

    placebo_centers = np.concatenate(
        (
            np.arange(target - 7.5, target - 1.0, 0.2),
            np.arange(target + 1.0, target + 7.5, 0.2),
        )
    )
    placebo_amplitudes = np.asarray(
        [
            _fit_gaussian_amplitude(
                reference_wave,
                detrended,
                center,
                center / resolution,
            )
            for center in placebo_centers
        ]
    )
    placebo_median = float(np.nanmedian(placebo_amplitudes))
    amplitude_sigma = 1.4826 * float(
        np.nanmedian(np.abs(placebo_amplitudes - placebo_median))
    )
    observed_z = (observed_amplitude - placebo_median) / amplitude_sigma

    expected_difference = candidate_degraded - control_degraded
    expected_amplitude = _fit_gaussian_amplitude(
        harps_candidate.wave,
        expected_difference,
        target,
        fwhm,
    )
    expected_detectability_z = expected_amplitude / amplitude_sigma
    recovered_fraction = (
        (observed_amplitude - placebo_median) / expected_amplitude
        if expected_amplitude > 0
        else float("nan")
    )
    if abs(observed_z) >= 3.0:
        result = "significant_matched_excess_requires_manual_review"
    elif expected_detectability_z >= 3.0:
        result = "no_match_with_moderate_sensitivity_to_full_harps_strength"
    else:
        result = "no_match_but_insufficient_sensitivity_to_full_harps_strength"

    row: dict[str, object] = {
        "instrument": external.source.instrument,
        "telescope": external.source.telescope,
        "filename": external.source.filename,
        "file_id": external.source.file_id,
        "date_obs_utc": external.date_obs,
        "target_air_angstrom": target,
        "target_vacuum_angstrom": air_to_vacuum(target),
        "resolving_power": round(resolution, 3),
        "instrument_fwhm_angstrom": round(fwhm, 6),
        "order_index_zero_based": external.order_index_zero_based,
        "alignment_velocity_kms": round(alignment_velocity_kms, 3),
        "expected_external_grid_wavelength_angstrom": round(
            target * (1.0 + alignment_velocity_kms / C_KMS), 6
        ),
        "observed_matched_amplitude": round(observed_amplitude, 8),
        "placebo_amplitude_median": round(placebo_median, 8),
        "empirical_amplitude_sigma": round(amplitude_sigma, 8),
        "observed_empirical_z": round(observed_z, 4),
        "expected_full_harps_strength_amplitude": round(expected_amplitude, 8),
        "expected_full_strength_detectability_z": round(
            expected_detectability_z, 4
        ),
        "recovered_fraction_of_full_harps_strength": round(recovered_fraction, 4),
        "placebo_amplitude_2p5": round(
            float(np.nanpercentile(placebo_amplitudes, 2.5)), 8
        ),
        "placebo_amplitude_97p5": round(
            float(np.nanpercentile(placebo_amplitudes, 97.5)), 8
        ),
        "result": result,
        "temporal_limitation": "spectrum_predates_harps_event_by_about_one_year",
    }
    curves = {
        "reference_wave": reference_wave,
        "detrended_residual": detrended,
        "harps_wave": harps_candidate.wave,
        "expected_difference": expected_difference,
    }
    return row, curves


def validate_and_manifest() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in EXTERNAL_FILES:
        with fits.open(source.path, memmap=False) as hdus:
            header = hdus[0].header
            shape = "x".join(str(value) for value in hdus[0].data.shape)
            fits_object = str(header.get("OBJECT", "")).strip()
            date_obs = str(header.get("DATE-OBS", header.get("UTSHUT", "")))
            if fits_object != "T0449197831":
                raise ValueError(
                    f"target mismatch in {source.filename}: {fits_object!r}"
                )
        coverage = []
        for target in TARGETS_AIR_ANGSTROM:
            order = load_external_order(source, target)
            coverage.append(
                f"{target:.2f}:{order.wave[0]:.3f}-{order.wave[-1]:.3f}"
            )
        rows.append(
            {
                "file_id": source.file_id,
                "filename": source.filename,
                "instrument": source.instrument,
                "telescope": source.telescope,
                "source_observation_date": source.source_observation_date,
                "fits_date_obs_utc": date_obs,
                "fits_object": fits_object,
                "shape": shape,
                "resolving_power": source.resolving_power,
                "exofop_snr": source.exofop_snr,
                "candidate_order_coverage_angstrom": ";".join(coverage),
                "bytes": source.path.stat().st_size,
                "sha256": sha256_file(source.path),
                "source_url": source.url,
                "validation": "valid_fits_target_and_candidate_orders",
            }
        )
    return rows


def download_command(args: argparse.Namespace) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(EXTERNAL_FILES, 1):
        print(f"[{index}/{len(EXTERNAL_FILES)}] {source.instrument} {source.filename}")
        download_file(source.url, source.path, args.timeout)
    rows = validate_and_manifest()
    _write_csv(
        args.manifest,
        rows,
        [
            "file_id",
            "filename",
            "instrument",
            "telescope",
            "source_observation_date",
            "fits_date_obs_utc",
            "fits_object",
            "shape",
            "resolving_power",
            "exofop_snr",
            "candidate_order_coverage_angstrom",
            "bytes",
            "sha256",
            "source_url",
            "validation",
        ],
    )
    print(f"wrote {len(rows)} validated rows to {args.manifest}")
    return 0


def _plot_comparison(
    rows_and_curves: list[tuple[dict[str, object], dict[str, np.ndarray]]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(13, 13), sharex="col")
    for axis, (row, curves) in zip(axes.flat, rows_and_curves, strict=True):
        target = float(row["target_air_angstrom"])
        reference_wave = curves["reference_wave"]
        residual = curves["detrended_residual"]
        harps_wave = curves["harps_wave"]
        expected = curves["expected_difference"]
        local = np.abs(reference_wave - target) <= 1.0
        expected_local = np.abs(harps_wave - target) <= 1.0
        axis.plot(
            reference_wave[local] - target,
            residual[local],
            color="#1769aa",
            linewidth=1.0,
            label="external - normal HARPS template",
        )
        axis.plot(
            harps_wave[expected_local] - target,
            expected[expected_local],
            color="#d95f02",
            linewidth=1.5,
            label="full HARPS event at this resolution",
        )
        axis.axhline(0, color="0.55", linewidth=0.8)
        axis.axvline(0, color="black", linestyle="--", linewidth=0.9)
        axis.set_title(
            f"{row['instrument']} {str(row['date_obs_utc'])[:10]} — "
            f"{target:.2f} A\n"
            f"observed z={float(row['observed_empirical_z']):.2f}; "
            f"full-strength expectation="
            f"{float(row['expected_full_strength_detectability_z']):.2f} sigma"
        )
        axis.set_ylabel("continuum-normalized residual")
        axis.grid(alpha=0.18)
    for axis in axes[-1]:
        axis.set_xlabel("star-aligned air wavelength offset (A)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "TOI-2458 independent-spectrum comparison\n"
        "Orange shows how the HARPS event would look at each instrument's resolution",
        y=0.997,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def stack_measurements(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Combine independent epochs as a transparent inverse-variance diagnostic."""

    stacked: list[dict[str, object]] = []
    for instrument in ("CHIRON", "TRES"):
        for target in TARGETS_AIR_ANGSTROM:
            selected = [
                row
                for row in rows
                if row["instrument"] == instrument
                and float(row["target_air_angstrom"]) == target
            ]
            signals = np.asarray(
                [
                    float(row["observed_matched_amplitude"])
                    - float(row["placebo_amplitude_median"])
                    for row in selected
                ]
            )
            sigmas = np.asarray(
                [float(row["empirical_amplitude_sigma"]) for row in selected]
            )
            expected = np.asarray(
                [
                    float(row["expected_full_harps_strength_amplitude"])
                    for row in selected
                ]
            )
            weights = 1.0 / sigmas**2
            combined_signal = float(np.sum(weights * signals) / np.sum(weights))
            combined_sigma = float(math.sqrt(1.0 / np.sum(weights)))
            combined_expected = float(np.sum(weights * expected) / np.sum(weights))
            observed_z = combined_signal / combined_sigma
            expected_z = combined_expected / combined_sigma
            difference_from_full_strength_z = (
                combined_signal - combined_expected
            ) / combined_sigma
            if abs(observed_z) >= 3.0:
                interpretation = "stacked_matched_excess_requires_manual_review"
            elif expected_z >= 3.0:
                interpretation = "no_stacked_match_despite_full_strength_sensitivity"
            else:
                interpretation = "no_stacked_match_and_full_strength_sensitivity_is_limited"
            stacked.append(
                {
                    "instrument": instrument,
                    "target_air_angstrom": target,
                    "exposures": len(selected),
                    "combined_signal_amplitude": round(combined_signal, 8),
                    "combined_empirical_sigma": round(combined_sigma, 8),
                    "combined_observed_z": round(observed_z, 4),
                    "combined_expected_full_strength_amplitude": round(
                        combined_expected, 8
                    ),
                    "combined_expected_full_strength_z": round(expected_z, 4),
                    "difference_from_full_strength_z": round(
                        difference_from_full_strength_z, 4
                    ),
                    "recovered_fraction_of_full_harps_strength": round(
                        combined_signal / combined_expected, 4
                    ),
                    "interpretation": interpretation,
                    "caveat": (
                        "inverse_variance_diagnostic_using_local_placebo_scatter; "
                        "not_a_calibrated_global_false_alarm_probability"
                    ),
                }
            )
    return stacked


def analyze_command(args: argparse.Namespace) -> int:
    manifest_rows = validate_and_manifest()
    _write_csv(
        args.manifest,
        manifest_rows,
        [
            "file_id",
            "filename",
            "instrument",
            "telescope",
            "source_observation_date",
            "fits_date_obs_utc",
            "fits_object",
            "shape",
            "resolving_power",
            "exofop_snr",
            "candidate_order_coverage_angstrom",
            "bytes",
            "sha256",
            "source_url",
            "validation",
        ],
    )
    harps_candidate = load_harps(args.harps_candidate)
    harps_control = load_harps(args.harps_control)
    if "obs.atmos" not in harps_candidate.wavelength_ucd:
        raise ValueError(
            "HARPS archive product does not identify its wavelength axis as air"
        )

    rows_and_curves: list[tuple[dict[str, object], dict[str, np.ndarray]]] = []
    for source in EXTERNAL_FILES:
        for target in TARGETS_AIR_ANGSTROM:
            external = load_external_order(source, target)
            velocity, correlation = align_external_order(harps_control, external)
            row, curves = measure_external_feature(
                harps_candidate,
                harps_control,
                external,
                velocity,
            )
            row["alignment_correlation"] = round(correlation, 5)
            rows_and_curves.append((row, curves))
            print(
                f"{source.instrument} {external.date_obs[:10]} {target:.2f} A: "
                f"observed z={row['observed_empirical_z']}, "
                f"full-strength expectation="
                f"{row['expected_full_strength_detectability_z']} sigma"
            )

    rows = [row for row, _ in rows_and_curves]
    fields = [
        "instrument",
        "telescope",
        "filename",
        "file_id",
        "date_obs_utc",
        "target_air_angstrom",
        "target_vacuum_angstrom",
        "resolving_power",
        "instrument_fwhm_angstrom",
        "order_index_zero_based",
        "alignment_velocity_kms",
        "alignment_correlation",
        "expected_external_grid_wavelength_angstrom",
        "observed_matched_amplitude",
        "placebo_amplitude_median",
        "empirical_amplitude_sigma",
        "observed_empirical_z",
        "expected_full_harps_strength_amplitude",
        "expected_full_strength_detectability_z",
        "recovered_fraction_of_full_harps_strength",
        "placebo_amplitude_2p5",
        "placebo_amplitude_97p5",
        "result",
        "temporal_limitation",
    ]
    _write_csv(args.output_csv, rows, fields)
    stacked = stack_measurements(rows)
    _write_csv(
        args.output_stacked_csv,
        stacked,
        [
            "instrument",
            "target_air_angstrom",
            "exposures",
            "combined_signal_amplitude",
            "combined_empirical_sigma",
            "combined_observed_z",
            "combined_expected_full_strength_amplitude",
            "combined_expected_full_strength_z",
            "difference_from_full_strength_z",
            "recovered_fraction_of_full_harps_strength",
            "interpretation",
            "caveat",
        ],
    )
    _plot_comparison(rows_and_curves, args.output_plot)

    conversions = []
    for air in TARGETS_AIR_ANGSTROM:
        predicted_vacuum = air_to_vacuum(air)
        pipeline_vacuum = PIPELINE_VACUUM_ANGSTROM[air]
        conversions.append(
            {
                "archive_air_angstrom": air,
                "ciddor_vacuum_angstrom": round(predicted_vacuum, 6),
                "pipeline_vacuum_angstrom_at_same_pixel": pipeline_vacuum,
                "pipeline_minus_conversion_angstrom": round(
                    pipeline_vacuum - predicted_vacuum, 6
                ),
                "pipeline_minus_conversion_kms": round(
                    (pipeline_vacuum / predicted_vacuum - 1.0) * C_KMS, 4
                ),
            }
        )
    significant = [row for row in rows if abs(float(row["observed_empirical_z"])) >= 3]
    chiron = [row for row in rows if row["instrument"] == "CHIRON"]
    tres = [row for row in rows if row["instrument"] == "TRES"]
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "target": {
            "name": "TOI-2458",
            "aliases": [
                "HD 34562",
                "TIC 449197831",
                "TYC 0100-00856-1",
                "2MASS J05182917+0115139",
                "Gaia DR3 3233785128901216128",
            ],
        },
        "harps_candidate": str(args.harps_candidate),
        "harps_control": str(args.harps_control),
        "external_spectra": manifest_rows,
        "wavelength_convention": {
            "archive_evidence": {
                "TUCD1": harps_candidate.wavelength_ucd,
                "SPECSYS": harps_candidate.spectral_frame,
            },
            "current_pipeline_array": "WAVEDATA_VAC_BARY",
            "conversions": conversions,
            "conclusion": (
                "The approximately 83 km/s same-pixel label offset is explained "
                "by air versus vacuum wavelength convention to within 0.3 km/s."
            ),
        },
        "method": {
            "alignment": (
                "Per-order high-pass cross-correlation against the feature-free "
                "HARPS exposure 1.7 hours later; candidate +/-1 A excluded."
            ),
            "resolution_matching": (
                "HARPS candidate and control were Gaussian-convolved from "
                "R=115000 to each external spectrum's resolving power."
            ),
            "measurement": (
                "Gaussian line-spread matched amplitude after subtracting the "
                "resolution-matched normal HARPS template and a clipped quadratic "
                "local trend. Noise is the robust spread of matched amplitudes at "
                "nearby placebo wavelengths."
            ),
            "threshold": "absolute empirical z >= 3 requires manual review",
        },
        "measurements": rows,
        "stacked_diagnostics": stacked,
        "aggregate": {
            "significant_external_matches": len(significant),
            "chiron_observed_abs_z_range": [
                round(min(abs(float(row["observed_empirical_z"])) for row in chiron), 4),
                round(max(abs(float(row["observed_empirical_z"])) for row in chiron), 4),
            ],
            "chiron_full_strength_expected_z_range": [
                round(
                    min(
                        float(row["expected_full_strength_detectability_z"])
                        for row in chiron
                    ),
                    4,
                ),
                round(
                    max(
                        float(row["expected_full_strength_detectability_z"])
                        for row in chiron
                    ),
                    4,
                ),
            ],
            "tres_observed_abs_z_range": [
                round(min(abs(float(row["observed_empirical_z"])) for row in tres), 4),
                round(max(abs(float(row["observed_empirical_z"])) for row in tres), 4),
            ],
            "tres_full_strength_expected_z_range": [
                round(
                    min(
                        float(row["expected_full_strength_detectability_z"])
                        for row in tres
                    ),
                    4,
                ),
                round(
                    max(
                        float(row["expected_full_strength_detectability_z"])
                        for row in tres
                    ),
                    4,
                ),
            ],
        },
        "conclusion": (
            "No CHIRON or TRES spectrum contains a >=3-sigma matched excess at "
            "either candidate wavelength. CHIRON had moderate sensitivity to a "
            "feature persisting at the full HARPS-event strength and did not see "
            "one. TRES is mostly too noisy for a decisive full-strength test. "
            "Because every external spectrum predates the event by about one year, "
            "this rejects persistence, not a one-off transient. The overall evidence "
            "continues to favor a HARPS detector/reduction event without proving it."
        ),
    }
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} comparisons to {args.output_csv}")
    print(f"wrote {len(stacked)} stacked diagnostics to {args.output_stacked_csv}")
    print(f"wrote summary to {args.output_json}")
    print(f"wrote plot to {args.output_plot}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download", help="download and validate spectra")
    download.add_argument("--timeout", type=float, default=120.0)
    download.add_argument(
        "--manifest", type=Path, default=DATA_DIR / "manifest.csv"
    )
    download.set_defaults(function=download_command)

    analyze = subparsers.add_parser("analyze", help="run the comparison")
    analyze.add_argument("--harps-candidate", type=Path, default=HARPS_CANDIDATE)
    analyze.add_argument("--harps-control", type=Path, default=HARPS_CONTROL)
    analyze.add_argument(
        "--manifest", type=Path, default=DATA_DIR / "manifest.csv"
    )
    analyze.add_argument(
        "--output-csv",
        type=Path,
        default=RESULT_DIR / "external_spectra_line_comparison.csv",
    )
    analyze.add_argument(
        "--output-json",
        type=Path,
        default=RESULT_DIR / "external_spectra_summary.json",
    )
    analyze.add_argument(
        "--output-stacked-csv",
        type=Path,
        default=RESULT_DIR / "external_spectra_stacked_summary.csv",
    )
    analyze.add_argument(
        "--output-plot",
        type=Path,
        default=RESULT_DIR / "plots" / "external_spectra_comparison.png",
    )
    analyze.set_defaults(function=analyze_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
