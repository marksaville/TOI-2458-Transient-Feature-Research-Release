#!/usr/bin/env python3
"""Small, reproducible helpers for the focused HARPS candidate reprocessing."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.io.votable import parse_single_table

from harps_residual_search import (
    C_KMS,
    Spectrum,
    continuum_normalize,
    download_file,
    load_spectrum,
    narrow_line_metrics,
    sha256_file,
)


FOCUSED_RAW_COUNTS = {
    "ORDERDEF_A": 1,
    "ORDERDEF_B": 1,
    "FLAT": 5,
    "THAR_THAR": 1,
    "THAR_FP": 1,
}


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def selected_association_rows(path: Path) -> list[dict[str, object]]:
    """Select the first applicable static and night calibration branches.

    ESO's association response contains nested branches for the flux standard and
    older fallbacks.  For the focused candidate reduction we keep every first
    static category plus the nearest order-definition, flat, and ThAr set.
    """

    table = parse_single_table(path).to_table(use_names_over_ids=True)
    seen_static: set[str] = set()
    raw_counts: dict[str, int] = {}
    selected: list[dict[str, object]] = []
    for index, row in enumerate(table):
        semantics = _text(row["semantics"])
        category = _text(row["eso_category"])
        url = _text(row["access_url"])
        if semantics != "#calibration" or not url.startswith("https://"):
            continue
        dataset_id = unquote(urlparse(url).path.rsplit("/", 1)[-1])
        is_static = dataset_id.startswith("M.HARPS.")
        if is_static:
            if category in seen_static:
                continue
            seen_static.add(category)
        else:
            limit = FOCUSED_RAW_COUNTS.get(category, 0)
            count = raw_counts.get(category, 0)
            if count >= limit:
                continue
            raw_counts[category] = count + 1
        selected.append(
            {
                "association_index": index,
                "dataset_id": dataset_id,
                "category": category,
                "kind": "static" if is_static else "raw",
                "expected_bytes": int(row["content_length"]),
                "source_url": url,
            }
        )
    return selected


def download_calibrations(args: argparse.Namespace) -> int:
    rows = selected_association_rows(args.association)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        row["path"] = str(args.data_dir / f"{row['dataset_id']}.fits")

    def verify_complete_fits(path: Path) -> None:
        if path.stat().st_size % 2880:
            raise OSError(f"incomplete FITS block in {path}")
        with fits.open(path, memmap=False) as hdus:
            hdus.verify("exception")
            for hdu in hdus:
                _ = hdu.data

    def expand_unix_compress(part: Path, destination: Path) -> bool:
        if not part.exists():
            return False
        with part.open("rb") as handle:
            if handle.read(2) != b"\x1f\x9d":
                return False
        expanded = destination.with_suffix(destination.suffix + ".expanded")
        with expanded.open("wb") as handle:
            subprocess.run(["uncompress", "-c", str(part)], stdout=handle, check=True)
        verify_complete_fits(expanded)
        expanded.replace(destination)
        part.unlink(missing_ok=True)
        return True

    def retrieve(row: dict[str, object]) -> dict[str, object]:
        path = Path(str(row["path"]))
        part = path.with_suffix(path.suffix + ".part")
        try:
            verify_complete_fits(path)
        except (OSError, ValueError):
            path.unlink(missing_ok=True)
        if not path.exists() and not expand_unix_compress(part, path):
            last_error: Exception | None = None
            for _ in range(3):
                try:
                    download_file(str(row["source_url"]), path, args.timeout)
                    if not path.exists() and expand_unix_compress(part, path):
                        break
                    verify_complete_fits(path)
                    break
                except (OSError, RuntimeError, ValueError) as exc:
                    last_error = exc
                    path.unlink(missing_ok=True)
                    if part.exists() and part.stat().st_size > 1:
                        if expand_unix_compress(part, path):
                            break
                        part.unlink(missing_ok=True)
            else:
                raise RuntimeError(f"failed to retrieve {row['dataset_id']}") from last_error
        verify_complete_fits(path)
        row["bytes"] = path.stat().st_size
        row["sha256"] = sha256_file(path)
        return row

    completed: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(retrieve, row): row for row in rows}
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            completed.append(row)
            print(f"[{index}/{len(rows)}] {row['category']} {row['dataset_id']}", flush=True)
    completed.sort(key=lambda row: int(row["association_index"]))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "association_index",
            "dataset_id",
            "category",
            "kind",
            "expected_bytes",
            "bytes",
            "sha256",
            "path",
            "source_url",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(completed)
    total = sum(int(row["bytes"]) for row in completed)
    print(f"wrote {len(completed)} verified calibrations ({total} bytes) to {args.manifest}")
    return 0


def _manifest_by_category(path: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["category"], []).append(Path(row["path"]).resolve())
    return grouped


def _read_sof(path: Path) -> list[tuple[Path, str]]:
    entries: list[tuple[Path, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            filename, category = stripped.rsplit(maxsplit=1)
            product = Path(filename)
            if not product.is_absolute():
                product = path.parent / product
            entries.append((product.resolve(), category))
    return entries


def _only(
    entries: list[tuple[Path, str]], categories: set[str]
) -> list[tuple[Path, str]]:
    return [(path, category) for path, category in entries if category in categories]


def _run_recipe(
    *,
    esorex: Path,
    plugin_dir: Path,
    install_prefix: Path,
    work_root: Path,
    stage: str,
    recipe: str,
    entries: list[tuple[Path, str]],
    parameters: list[str] | None = None,
) -> list[tuple[Path, str]]:
    stage_dir = (work_root / stage).resolve()
    stage_dir.mkdir(parents=True, exist_ok=True)
    sof = stage_dir / "inputs.sof"
    products = stage_dir / "products.sof"
    with sof.open("w", encoding="utf-8") as handle:
        for path, category in entries:
            handle.write(f"{path.resolve()} {category}\n")
    command = [
        str(esorex),
        f"--recipe-dir={plugin_dir}",
        f"--output-dir={stage_dir}",
        f"--link-dir={stage_dir}",
        f"--log-dir={stage_dir}",
        f"--log-file={recipe}.log",
        f"--products-sof={products}",
        "--msg-level=warning",
        "--suppress-link=TRUE",
        "--suppress-prefix=TRUE",
        recipe,
        *(parameters or []),
        str(sof),
    ]
    environment = os.environ.copy()
    environment["CPLDIR"] = str(install_prefix)
    existing = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = str(install_prefix / "lib") + (
        f":{existing}" if existing else ""
    )
    print("running", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=stage_dir,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    (stage_dir / "console.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        tail = "\n".join(completed.stdout.splitlines()[-60:])
        print(tail)
        raise subprocess.CalledProcessError(completed.returncode, command)
    print(f"completed {recipe}; log: {stage_dir / 'console.log'}", flush=True)
    if not products.exists():
        raise RuntimeError(f"{recipe} did not create {products}")
    result = _read_sof(products)
    if not result:
        raise RuntimeError(f"{recipe} produced an empty products SOF")
    return result


def reduce_candidate(args: argparse.Namespace) -> int:
    """Run the minimal official calibration cascade and LA Cosmic science step."""

    install = args.install_prefix.resolve()
    esorex = install / "bin" / "esorex"
    plugin = install / "lib" / "esopipes-plugins" / "harps-3.6.0"
    if not esorex.exists() or not plugin.exists():
        raise FileNotFoundError("the project-local HARPS 3.6.0 installation is incomplete")
    grouped = _manifest_by_category(args.calibration_manifest)

    def one(category: str) -> Path:
        values = grouped.get(category, [])
        if len(values) != 1:
            raise ValueError(f"expected one {category}, found {len(values)}")
        return values[0]

    static = args.static_dir.resolve()
    static_names = {
        "CCD_GEOM": "HARPS_FAST_CCD_geom_config_2003-01-02.fits",
        "INST_CONFIG": "HARPS15_reduced_inst_config_2015-05-25.fits",
        "HOT_PIXEL_MASK": "HARPS_2021-12-19_hot_pixels.fits",
        "BAD_PIXEL_MASK": "HARPS_2003-01-01_bad_pixels.fits",
        "STATIC_WAVE_MATRIX_A": "HARPS_HARPS15_STATIC_WAVE_MATRIX_A_2015-05-25.fits",
        "STATIC_WAVE_MATRIX_B": "HARPS_HARPS15_STATIC_WAVE_MATRIX_B_2015-05-25.fits",
        "STATIC_DLL_MATRIX_A": "HARPS_HARPS15_STATIC_DLL_MATRIX_A_2015-05-25.fits",
        "STATIC_DLL_MATRIX_B": "HARPS_HARPS15_STATIC_DLL_MATRIX_B_2015-05-25.fits",
        "STATIC_LINE_TABLE_A": "HARPS_HARPS15_STATIC_TH_LINE_TABLE_A_2015-05-25.fits",
        "STATIC_LINE_TABLE_B": "HARPS_HARPS15_STATIC_TH_LINE_TABLE_B_2015-05-25.fits",
        "REF_LINE_TABLE_A": "HARPS_HARPS15_REF_LINE_TABLE_A_2015-05-25.fits",
        "REF_LINE_TABLE_B": "HARPS_HARPS15_REF_LINE_TABLE_B_2015-05-25.fits",
        "PIXEL_GEOM_A": "HARPS_PIXEL_GEOM_A.fits",
        "PIXEL_GEOM_B": "HARPS_PIXEL_GEOM_B.fits",
        "PIXEL_SIZE_A": "HARPS_PIXEL_SIZE_A.fits",
        "PIXEL_SIZE_B": "HARPS_PIXEL_SIZE_B.fits",
        "REL_EFF_B": "HARPS_HARPS15_REL_EFF_B.fits",
        "EXT_TABLE": "HARPS_EXTINCTION_TABLE.fits",
        "STD_TABLE": "HARPS_STD_TABLE.fits",
        "MASK_LUT": "HARPS_mask_lut.fits",
        "FLUX_TEMPLATE": "HARPS_FLUX_TEMPLATE.fits",
        "MASK_TABLE": "HARPS_G2.fits",
    }
    static_paths = {category: static / name for category, name in static_names.items()}
    for path in [*static_paths.values(), args.raw.resolve()]:
        if not path.exists():
            raise FileNotFoundError(path)

    common = [
        (static_paths["CCD_GEOM"], "CCD_GEOM"),
        (static_paths["INST_CONFIG"], "INST_CONFIG"),
        (static_paths["HOT_PIXEL_MASK"], "HOT_PIXEL_MASK"),
        (static_paths["BAD_PIXEL_MASK"], "BAD_PIXEL_MASK"),
    ]
    order_entries = [
        (one("ORDERDEF_A"), "ORDERDEF_A"),
        (one("ORDERDEF_B"), "ORDERDEF_B"),
        *common,
    ]
    order_products = _run_recipe(
        esorex=esorex,
        plugin_dir=plugin,
        install_prefix=install,
        work_root=args.work_dir,
        stage="01_orderdef",
        recipe="espdr_orderdef",
        entries=order_entries,
    )

    flat_entries = [
        *((path, "FLAT") for path in grouped.get("FLAT", [])),
        *common,
        *order_products,
        (static_paths["STATIC_WAVE_MATRIX_A"], "STATIC_WAVE_MATRIX_A"),
        (static_paths["STATIC_WAVE_MATRIX_B"], "STATIC_WAVE_MATRIX_B"),
        (static_paths["STATIC_DLL_MATRIX_A"], "STATIC_DLL_MATRIX_A"),
        (static_paths["STATIC_DLL_MATRIX_B"], "STATIC_DLL_MATRIX_B"),
    ]
    flat_products = _run_recipe(
        esorex=esorex,
        plugin_dir=plugin,
        install_prefix=install,
        work_root=args.work_dir,
        stage="02_mflat",
        recipe="espdr_mflat",
        entries=flat_entries,
    )
    flat_science_products = _only(
        flat_products,
        {
            "ORDER_PROFILE_A",
            "ORDER_PROFILE_B",
            "FSPECTRUM_A",
            "FSPECTRUM_B",
            "BLAZE_A",
            "BLAZE_B",
        },
    )

    wave_calibration_entries = [
        *common,
        *order_products,
        *flat_science_products,
        *(
            (static_paths[category], category)
            for category in [
                "STATIC_WAVE_MATRIX_A",
                "STATIC_WAVE_MATRIX_B",
                "STATIC_DLL_MATRIX_A",
                "STATIC_DLL_MATRIX_B",
                "STATIC_LINE_TABLE_A",
                "STATIC_LINE_TABLE_B",
                "REF_LINE_TABLE_A",
                "REF_LINE_TABLE_B",
                "PIXEL_GEOM_A",
                "PIXEL_GEOM_B",
            ]
        ),
    ]
    wave_thar_thar_products = _run_recipe(
        esorex=esorex,
        plugin_dir=plugin,
        install_prefix=install,
        work_root=args.work_dir,
        stage="03a_wave_thar_thar",
        recipe="espdr_wave_TH_drift",
        entries=[(one("THAR_THAR"), "THAR_THAR"), *wave_calibration_entries],
    )
    wave_thar_fp_products = _run_recipe(
        esorex=esorex,
        plugin_dir=plugin,
        install_prefix=install,
        work_root=args.work_dir,
        stage="03b_wave_thar_fp",
        recipe="espdr_wave_TH_drift",
        entries=[(one("THAR_FP"), "THAR_FP"), *wave_calibration_entries],
    )
    wave_science_products = _only(
        [*wave_thar_thar_products, *wave_thar_fp_products],
        {
            "S2D_BLAZE_THAR_FP_A",
            "S2D_BLAZE_THAR_FP_B",
            "S2D_BLAZE_THAR_THAR_A",
            "S2D_BLAZE_THAR_THAR_B",
            "DLL_MATRIX_DRIFT_THAR_FP_A",
            "DLL_MATRIX_DRIFT_THAR_THAR_A",
            "DLL_MATRIX_DRIFT_THAR_THAR_B",
            "WAVE_MATRIX_DRIFT_THAR_FP_A",
            "WAVE_MATRIX_DRIFT_THAR_THAR_A",
            "WAVE_MATRIX_DRIFT_THAR_THAR_B",
        },
    )

    science_entries = [
        (args.raw.resolve(), "OBJ_FP"),
        *common,
        *order_products,
        *flat_science_products,
        *wave_science_products,
        *((static_paths[category], category) for category in [
            "EXT_TABLE",
            "STD_TABLE",
            "REL_EFF_B",
            "MASK_LUT",
            "FLUX_TEMPLATE",
            "MASK_TABLE",
        ]),
    ]
    shared_science_parameters = [
        "--bias_res_removal_sw=off",
        "--flux_correction_type=NONE",
        "--mask_table_id=G2",
    ]
    no_rejection_products = _run_recipe(
        esorex=esorex,
        plugin_dir=plugin,
        install_prefix=install,
        work_root=args.work_dir,
        stage="04_science_no_cosmic_rejection",
        recipe="espdr_sci_red",
        entries=science_entries,
        parameters=[
            "--cosmic_detection_sw=0",
            "--extra_products_sw=FALSE",
            "--ksigma_cosmic=-1",
            *shared_science_parameters,
        ],
    )
    control_products = _run_recipe(
        esorex=esorex,
        plugin_dir=plugin,
        install_prefix=install,
        work_root=args.work_dir,
        stage="04_science_no_lacosmic",
        recipe="espdr_sci_red",
        entries=science_entries,
        parameters=[
            "--cosmic_detection_sw=0",
            "--extra_products_sw=FALSE",
            *shared_science_parameters,
        ],
    )
    science_products = _run_recipe(
        esorex=esorex,
        plugin_dir=plugin,
        install_prefix=install,
        work_root=args.work_dir,
        stage="04_science_lacosmic",
        recipe="espdr_sci_red",
        entries=science_entries,
        parameters=[
            "--cosmic_detection_sw=1",
            "--extra_products_sw=TRUE",
            "--lacosmic.sigma_lim=4.0",
            "--lacosmic.f_lim=4.0",
            "--lacosmic.max_iter=5",
            *shared_science_parameters,
        ],
    )
    categories = {category: path for path, category in science_products}
    if "CRH_MAP" not in categories:
        raise RuntimeError("LA Cosmic was enabled, but the science recipe produced no CRH_MAP")
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "pipeline": "ESO HARPS 3.6.0",
        "raw": str(args.raw.resolve()),
        "static_calibration_selection": {
            category: str(path) for category, path in static_paths.items()
        },
        "parameters": {
            "cosmic_detection_sw": 1,
            "extra_products_sw": True,
            "lacosmic_sigma_lim": 4.0,
            "lacosmic_f_lim": 4.0,
            "lacosmic_max_iter": 5,
            "ksigma_cosmic": 3.5,
        },
        "products": {
            category: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path, category in science_products
        },
        "no_lacosmic_control_products": {
            category: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path, category in control_products
        },
        "no_cosmic_rejection_control_products": {
            category: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path, category in no_rejection_products
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def _load_current_s1d(path: Path) -> Spectrum:
    with fits.open(path, memmap=False) as hdus:
        row = hdus[1].data[0]
        wave = np.asarray(row["WAVE"], dtype=float)
        flux = np.asarray(row["FLUX_EL"], dtype=float)
        berv = float(hdus[0].header.get("HIERARCH ESO QC BERV", np.nan))
    valid = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
    wave = wave[valid]
    flux = flux[valid]
    order = np.argsort(wave)
    wave = wave[order]
    flux = flux[order]
    return Spectrum(
        dp_id=path.parent.name,
        target="TOI-2458",
        observed_at="2022-02-19T00:15:10.071+00:00",
        path=path,
        wave=wave,
        flux=flux,
        normalized=continuum_normalize(wave, flux),
        berv_kms=berv,
        header_snr=float("nan"),
    )


def _half_max_width(spectrum: Spectrum, wavelength: float) -> dict[str, float | int]:
    window = np.abs(spectrum.wave - wavelength) <= 0.18
    indices = np.flatnonzero(window)
    peak = int(indices[np.nanargmax(spectrum.normalized[indices])])
    side = window & (np.abs(spectrum.wave - wavelength) >= 0.12)
    baseline = float(np.nanmedian(spectrum.normalized[side]))
    half = baseline + (float(spectrum.normalized[peak]) - baseline) / 2.0
    low = peak
    high = peak
    while low > indices[0] and spectrum.normalized[low - 1] >= half:
        low -= 1
    while high < indices[-1] and spectrum.normalized[high + 1] >= half:
        high += 1
    return {
        "half_max_sample_count": high - low + 1,
        "half_max_width_angstrom": round(float(spectrum.wave[high] - spectrum.wave[low]), 6),
    }


def inspect_candidates(args: argparse.Namespace) -> int:
    archive_rows: dict[str, dict[str, str]] = {}
    with args.archive_manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            archive_rows[row["dp_id"]] = row
    archived = load_spectrum(archive_rows["ADP.2022-02-20T01:06:53.624"])
    stages = {
        "no_cosmic_rejection": args.work_dir / "04_science_no_cosmic_rejection",
        "ksigma_only": args.work_dir / "04_science_no_lacosmic",
        "ksigma_plus_lacosmic": args.work_dir / "04_science_lacosmic",
    }
    current = {
        name: _load_current_s1d(path / "HARPS_S1D_FINAL_A.fits")
        for name, path in stages.items()
    }
    with fits.open(stages["no_cosmic_rejection"] / "HARPS_S2D_A.fits", memmap=False) as hdus:
        s2d_wave = np.array(hdus["WAVEDATA_VAC_BARY"].data, dtype=float)
    with fits.open(args.order_table, memmap=False) as hdus:
        order_map = np.array(hdus[2].data)
    with fits.open(args.crh_map, memmap=False) as hdus:
        crh_map = np.array(hdus[2].data, dtype=bool)
    recurrence = json.loads(args.recurrence_summary.read_text(encoding="utf-8"))
    recurrence_by_wave = {
        float(line["candidate_wavelength_angstrom"]): line for line in recurrence["lines"]
    }

    mappings = [
        {"archive_wavelength": 5761.26, "s2d_row_zero_based": 53, "detector_x_zero_based": 1245, "order_label": 9},
        {"archive_wavelength": 6432.95, "s2d_row_zero_based": 64, "detector_x_zero_based": 1506, "order_label": 20},
    ]
    lines: list[dict[str, object]] = []
    for mapping in mappings:
        wavelength = float(mapping["archive_wavelength"])
        row = int(mapping["s2d_row_zero_based"])
        x = int(mapping["detector_x_zero_based"])
        label = int(mapping["order_label"])
        official_wavelength = float(s2d_wave[row, x])
        per_x = np.array(
            [
                np.count_nonzero(crh_map[:, column] & (order_map[:, column] == label))
                for column in range(order_map.shape[1])
            ]
        )
        window_counts = np.convolve(per_x, np.ones(31, dtype=int), mode="same")
        interior = window_counts[15:-15]
        hits: list[dict[str, int]] = []
        for column in range(x - 15, x + 16):
            ys = np.flatnonzero((order_map[:, column] == label) & crh_map[:, column])
            hits.extend({"x": column, "y": int(y)} for y in ys)
        settings = {
            name: narrow_line_metrics(spectrum, official_wavelength)
            for name, spectrum in current.items()
        }
        archived_metrics = narrow_line_metrics(archived, wavelength)
        repeat = recurrence_by_wave[wavelength]
        comparison_shape: dict[str, object] | None = None
        repeat_ids = repeat["star_frame_recurrence_dp_ids"]
        if repeat_ids:
            repeat_spectrum = load_spectrum(archive_rows[repeat_ids[0]])
            comparison_shape = {
                "dp_id": repeat_ids[0],
                **_half_max_width(repeat_spectrum, wavelength),
                "instrumental_fwhm_angstrom": round(wavelength / 115_000.0, 6),
            }
        line = {
            **mapping,
            "official_pipeline_wavelength_at_same_detector_pixel": round(official_wavelength, 6),
            "archive_to_official_frame_offset_kms": round(
                C_KMS * (official_wavelength / wavelength - 1.0), 3
            ),
            "archive_processed_metrics": archived_metrics,
            "official_pipeline_metrics_at_same_detector_pixel": settings,
            "crh_exact_pixel_flag_count_in_order_aperture": int(per_x[x]),
            "crh_order_aperture_flags_within_plus_minus_15_x": len(hits),
            "crh_flag_coordinates": hits,
            "crh_window_percentile_within_same_order": round(
                100.0 * float(np.mean(interior <= window_counts[x])), 3
            ),
            "all_epoch_recurrence": repeat,
            "comparison_event_shape": comparison_shape,
            "interpretation": (
                "detector_or_reduction_artifact_favored_but_not_proven; "
                "the feature survives all three current-pipeline extraction settings"
            ),
        }
        lines.append(line)

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "pipeline": "ESO HARPS 3.6.0",
        "candidate_exposure": "HARPS.2022-02-19T00:15:10.071",
        "result": (
            "Both detector features survive reprocessing, including with all explicit "
            "cosmic rejection disabled. Detector/reduction artifacts remain favored by "
            "the CRH-map neighborhoods and artifact-like comparison event, but are not proven."
        ),
        "lines": lines,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    args.plot.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    colors = {
        "no_cosmic_rejection": "tab:blue",
        "ksigma_only": "tab:orange",
        "ksigma_plus_lacosmic": "tab:green",
    }
    for axis, line in zip(axes, lines, strict=True):
        archive_wave = float(line["archive_wavelength"])
        official_wave = float(line["official_pipeline_wavelength_at_same_detector_pixel"])
        mask = np.abs(archived.wave - archive_wave) <= 0.25
        velocity = C_KMS * (archived.wave[mask] / archive_wave - 1.0)
        axis.plot(velocity, archived.normalized[mask], color="black", lw=1.8, label="archive product")
        for name, spectrum in current.items():
            mask = np.abs(spectrum.wave - official_wave) <= 0.25
            velocity = C_KMS * (spectrum.wave[mask] / official_wave - 1.0)
            axis.plot(
                velocity,
                spectrum.normalized[mask],
                marker=".",
                lw=1.0,
                ms=3,
                color=colors[name],
                label=name.replace("_", " "),
            )
        axis.axvline(0, color="gray", ls="--", lw=0.8)
        axis.set_ylabel("continuum-normalized flux")
        axis.set_title(
            f"archive {archive_wave:.2f} Å / pipeline detector-matched {official_wave:.3f} Å"
        )
        axis.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("velocity relative to detector-matched feature [km/s]")
    figure.tight_layout()
    figure.savefig(args.plot, dpi=160)
    plt.close(figure)
    print(json.dumps(report, indent=2))
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser(
        "download-calibrations",
        help="download the focused Raw2Raw association subset for the candidate",
    )
    download.add_argument(
        "--association",
        type=Path,
        default=Path("data/harps_calibrations/association_raw2master.xml"),
    )
    download.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/harps_calibrations/night_2022-02-18"),
    )
    download.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/harps_calibrations/night_2022-02-18/manifest.csv"),
    )
    download.add_argument("--workers", type=int, default=4)
    download.add_argument("--timeout", type=float, default=240.0)
    download.set_defaults(func=download_calibrations)

    reduce = subparsers.add_parser(
        "reduce-candidate",
        help="reprocess the candidate raw exposure and create an official CRH_MAP",
    )
    reduce.add_argument(
        "--raw",
        type=Path,
        default=Path("data/harps_candidate_raw/HARPS.2022-02-19T00_15_10.071.fits"),
    )
    reduce.add_argument(
        "--calibration-manifest",
        type=Path,
        default=Path("data/harps_calibrations/night_2022-02-18/manifest.csv"),
    )
    reduce.add_argument(
        "--static-dir",
        type=Path,
        default=Path(
            "data/harps_calibrations/static/share/esopipes/datastatic/harps-3.6.0"
        ),
    )
    reduce.add_argument(
        "--install-prefix",
        type=Path,
        default=Path("data/harps_pipeline/install"),
    )
    reduce.add_argument(
        "--work-dir", type=Path, default=Path("work/harps_candidate")
    )
    reduce.add_argument(
        "--report",
        type=Path,
        default=Path("results/harps_residual_search/candidate_pipeline_report.json"),
    )
    reduce.set_defaults(func=reduce_candidate)

    inspect = subparsers.add_parser(
        "inspect-candidates",
        help="compare archive and pipeline controls and inspect the CRH-map neighborhoods",
    )
    inspect.add_argument(
        "--archive-manifest", type=Path, default=Path("data/toi2458_all_epochs/manifest.csv")
    )
    inspect.add_argument("--work-dir", type=Path, default=Path("work/harps_candidate"))
    inspect.add_argument(
        "--order-table",
        type=Path,
        default=Path("work/harps_candidate/01_orderdef/HARPS_ORDER_TABLE_A.fits"),
    )
    inspect.add_argument(
        "--crh-map",
        type=Path,
        default=Path("work/harps_candidate/04_science_lacosmic/HARPS_crh_map.fits"),
    )
    inspect.add_argument(
        "--recurrence-summary",
        type=Path,
        default=Path("results/harps_residual_search/toi2458_all_epoch_summary.json"),
    )
    inspect.add_argument(
        "--output",
        type=Path,
        default=Path("results/harps_residual_search/candidate_cosmic_check.json"),
    )
    inspect.add_argument(
        "--plot",
        type=Path,
        default=Path("results/harps_residual_search/plots/candidate_pipeline_controls.png"),
    )
    inspect.set_defaults(func=inspect_candidates)
    return parser


def main() -> int:
    args = make_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
