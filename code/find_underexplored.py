#!/usr/bin/env python3
"""Find older ESO spectral programs with no telbib-linked publications.

This intentionally small CLI does two things:

* ``discover`` ranks public, processed, high-resolution spectral programs.
* ``sample`` creates a human-review CSV for the best-ranked programs.

It uses only the Python standard library.  Remote data come from ESO's TAP
service and the ESO Telescope Bibliography (telbib) API.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence


TAP_SYNC_URL = "https://archive.eso.org/tap_obs/sync"
TELBIB_API_URL = "https://telbib.eso.org/api_v2.php"
USER_AGENT = "eso-underexplored/0.1 (+https://www.eso.org/sci/libraries/telbib_info.html)"
NUM_FOUND_RE = re.compile(rb"<numFound>\s*(\d+)\s*</numFound>")
RUN_SUFFIX_RE = re.compile(r"\s*\([A-Za-z0-9]+\)\s*$")


PROGRAM_FIELDS = [
    "rank",
    "score",
    "program_id",
    "archive_proposal_value",
    "product_count",
    "target_count",
    "repeat_products",
    "repeat_ratio",
    "size_gb",
    "max_resolving_power",
    "mean_snr",
    "preview_fraction",
    "publication_count",
    "telbib_status",
    "cutoff_date",
    "telbib_url",
]

CHECKED_FIELDS = [
    "program_id",
    "archive_proposal_value",
    "product_count",
    "target_count",
    "size_kb",
    "max_resolving_power",
    "mean_snr",
    "preview_count",
    "publication_count",
    "telbib_status",
]

OBSERVATION_FIELDS = [
    "program_rank",
    "program_score",
    "program_id",
    "target_pool_count",
    "dp_id",
    "obs_id",
    "target_name",
    "instrument_name",
    "obs_title",
    "observed_at_utc",
    "t_exptime_seconds",
    "wavelength_min_nm",
    "wavelength_max_nm",
    "resolving_power",
    "snr",
    "size_mb",
    "preview_url",
    "data_url",
    "review_status",
    "repeat_confirmed",
    "conventional_explanations_checked",
    "notes",
]


class ExternalServiceError(RuntimeError):
    """Raised when ESO returns an unavailable or malformed response."""


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(str(value).strip())
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def normalize_program_id(value: str) -> str:
    """Convert an archive run ID such as 072.C-0488(E) to 072.C-0488."""

    return RUN_SUFFIX_RE.sub("", value.strip())


def split_program_ids(value: str) -> list[str]:
    """Return normalized IDs from the archive's occasionally combined field."""

    ids: list[str] = []
    for item in re.split(r"[,;]", value or ""):
        normalized = normalize_program_id(item)
        if normalized and normalized.upper() not in {"N/A", "NONE", "NULL"}:
            ids.append(normalized)
    return list(dict.fromkeys(ids))


def _request(
    url: str, *, params: dict[str, str], timeout: float, method: str = "POST"
):
    encoded = urllib.parse.urlencode(params)
    if method == "GET":
        request_url = url + "?" + encoded
        body = None
    else:
        request_url = url
        body = encoded.encode("ascii")
    request = urllib.request.Request(
        request_url,
        data=body,
        headers={"User-Agent": USER_AGENT},
        method=method,
    )
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        raise ExternalServiceError(f"request failed for {url}: {exc}") from exc


def tap_csv(query: str, *, timeout: float = 90.0) -> list[dict[str, str]]:
    """Run an ADQL query and return ESO's CSV rows."""

    params = {
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": "csv",
        "QUERY": query,
    }
    with _request(TAP_SYNC_URL, params=params, timeout=timeout) as response:
        raw = response.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ExternalServiceError("ESO TAP response was not UTF-8 CSV") from exc
    if not text.lstrip().startswith(("proposal_id,", "dp_id,")):
        excerpt = " ".join(text[:300].split())
        raise ExternalServiceError(f"ESO TAP returned an unexpected response: {excerpt}")
    return list(csv.DictReader(text.splitlines()))


def build_archive_query(
    *, cutoff: date, limit: int, min_resolving_power: float
) -> str:
    return f"""SELECT TOP {int(limit)}
proposal_id,
COUNT(*) AS product_count,
COUNT(DISTINCT target_name) AS target_count,
SUM(access_estsize) AS size_kb,
MAX(em_res_power) AS max_resolving_power,
AVG(snr) AS mean_snr,
COUNT(preview_html) AS preview_count
FROM ivoa.ObsCore
WHERE dataproduct_type='spectrum'
  AND proposal_id IS NOT NULL
  AND obs_release_date < '{cutoff.isoformat()}'
  AND calib_level >= 2
  AND em_res_power >= {float(min_resolving_power):.1f}
  AND (is_solar = 0 OR is_solar IS NULL)
GROUP BY proposal_id
ORDER BY product_count DESC"""


def fetch_archive_programs(
    *, cutoff: date, limit: int, min_resolving_power: float, timeout: float
) -> list[dict[str, str]]:
    return tap_csv(
        build_archive_query(
            cutoff=cutoff,
            limit=limit,
            min_resolving_power=min_resolving_power,
        ),
        timeout=timeout,
    )


def parse_telbib_count_prefix(chunks: Iterable[bytes]) -> int:
    """Parse numFound without downloading telbib's full abstracts."""

    prefix = bytearray()
    for chunk in chunks:
        prefix.extend(chunk)
        match = NUM_FOUND_RE.search(prefix)
        if match:
            return int(match.group(1))
        if len(prefix) > 131_072:
            break
    raise ExternalServiceError("telbib response did not contain numFound near its start")


def fetch_telbib_count(program_id: str, *, timeout: float = 45.0) -> int:
    params = {"programid": program_id}
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with _request(
                TELBIB_API_URL, params=params, timeout=timeout, method="GET"
            ) as response:
                return parse_telbib_count_prefix(iter(lambda: response.read(4096), b""))
        except ExternalServiceError as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.5 * (2**attempt))
    assert last_error is not None
    raise last_error


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_telbib_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in read_csv(path):
        status = row.get("telbib_status", "")
        if status and status not in {"ok", "fixture", "cache"}:
            continue
        if not str(row.get("publication_count", "")).strip():
            continue
        counts[normalize_program_id(row["program_id"])] = _safe_int(
            row["publication_count"]
        )
    return counts


def consolidate_archive_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, object]]:
    """Normalize run suffixes and combine rows belonging to the same program."""

    combined: dict[str, dict[str, object]] = {}
    for row in rows:
        archive_value = (row.get("proposal_id") or "").strip()
        for program_id in split_program_ids(archive_value):
            record = combined.setdefault(
                program_id,
                {
                    "program_id": program_id,
                    "archive_values": set(),
                    "product_count": 0,
                    "target_count": 0,
                    "size_kb": 0.0,
                    "max_resolving_power": 0.0,
                    "snr_weighted_sum": 0.0,
                    "snr_weight": 0,
                    "preview_count": 0,
                },
            )
            record["archive_values"].add(archive_value)  # type: ignore[union-attr]
            products = _safe_int(row.get("product_count"))
            record["product_count"] = int(record["product_count"]) + products
            record["target_count"] = int(record["target_count"]) + _safe_int(
                row.get("target_count")
            )
            record["size_kb"] = float(record["size_kb"]) + _safe_float(
                row.get("size_kb")
            )
            record["max_resolving_power"] = max(
                float(record["max_resolving_power"]),
                _safe_float(row.get("max_resolving_power")),
            )
            mean_snr = _safe_float(row.get("mean_snr"))
            if mean_snr > 0 and products > 0:
                record["snr_weighted_sum"] = (
                    float(record["snr_weighted_sum"]) + mean_snr * products
                )
                record["snr_weight"] = int(record["snr_weight"]) + products
            record["preview_count"] = int(record["preview_count"]) + _safe_int(
                row.get("preview_count")
            )

    result: list[dict[str, object]] = []
    for record in combined.values():
        weight = int(record.pop("snr_weight"))
        weighted_sum = float(record.pop("snr_weighted_sum"))
        record["mean_snr"] = weighted_sum / weight if weight else 0.0
        archive_values = record.pop("archive_values")
        record["archive_proposal_value"] = ";".join(sorted(archive_values))
        result.append(record)
    return result


def attach_publication_counts(
    records: list[dict[str, object]],
    *,
    offline_counts: dict[str, int] | None,
    cached_counts: dict[str, int] | None,
    max_lookups: int,
    workers: int,
    timeout: float,
) -> None:
    candidates = sorted(
        records, key=lambda row: int(row["product_count"]), reverse=True
    )[:max_lookups]

    if offline_counts is not None:
        for record in candidates:
            program_id = str(record["program_id"])
            if program_id in offline_counts:
                record["publication_count"] = offline_counts[program_id]
                record["telbib_status"] = "fixture"
            else:
                record["publication_count"] = ""
                record["telbib_status"] = "missing_fixture"
        return

    pending: list[dict[str, object]] = []
    for record in candidates:
        program_id = str(record["program_id"])
        if cached_counts is not None and program_id in cached_counts:
            record["publication_count"] = cached_counts[program_id]
            record["telbib_status"] = "cache"
        else:
            pending.append(record)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_to_record = {
            executor.submit(
                fetch_telbib_count, str(record["program_id"]), timeout=timeout
            ): record
            for record in pending
        }
        for future in as_completed(future_to_record):
            record = future_to_record[future]
            try:
                record["publication_count"] = future.result()
                record["telbib_status"] = "ok"
            except ExternalServiceError as exc:
                record["publication_count"] = ""
                record["telbib_status"] = f"error: {exc}"


def _normalized_logs(values: Sequence[float]) -> list[float]:
    logs = [math.log1p(max(0.0, value)) for value in values]
    low, high = min(logs, default=0.0), max(logs, default=0.0)
    if high <= low:
        return [1.0 if logs else 0.0 for _ in logs]
    return [(value - low) / (high - low) for value in logs]


def rank_unpublished(
    records: Sequence[dict[str, object]], *, cutoff: date
) -> list[dict[str, object]]:
    eligible = [
        dict(record)
        for record in records
        if record.get("publication_count") == 0
        and str(record.get("telbib_status")) in {"ok", "fixture", "cache"}
    ]
    corpus_scores = _normalized_logs(
        [_safe_float(record["product_count"]) for record in eligible]
    )
    resolution_scores = _normalized_logs(
        [_safe_float(record["max_resolving_power"]) for record in eligible]
    )

    for record, corpus_score, resolution_score in zip(
        eligible, corpus_scores, resolution_scores, strict=True
    ):
        products = max(1, _safe_int(record["product_count"]))
        targets = min(products, _safe_int(record["target_count"]))
        repeats = max(0, products - targets)
        repeat_ratio = repeats / products
        preview_fraction = min(
            1.0, _safe_int(record["preview_count"]) / products
        )
        snr_score = min(1.0, max(0.0, _safe_float(record["mean_snr"]) / 200.0))
        score = (
            0.45 * corpus_score
            + 0.25 * repeat_ratio
            + 0.15 * resolution_score
            + 0.10 * snr_score
            + 0.05 * preview_fraction
        )
        record.update(
            {
                "score": round(score, 6),
                "repeat_products": repeats,
                "repeat_ratio": round(repeat_ratio, 6),
                "size_gb": round(_safe_float(record["size_kb"]) / 1_000_000, 3),
                "max_resolving_power": round(
                    _safe_float(record["max_resolving_power"]), 1
                ),
                "mean_snr": round(_safe_float(record["mean_snr"]), 2),
                "preview_fraction": round(preview_fraction, 6),
                "cutoff_date": cutoff.isoformat(),
                "telbib_url": TELBIB_API_URL
                + "?"
                + urllib.parse.urlencode({"programid": record["program_id"]}),
            }
        )

    eligible.sort(
        key=lambda row: (float(row["score"]), int(row["product_count"])),
        reverse=True,
    )
    for rank, record in enumerate(eligible, 1):
        record["rank"] = rank
    return eligible


def discover(args: argparse.Namespace) -> Path:
    cutoff = date.today() - timedelta(days=round(args.older_than_years * 365.2425))
    if args.archive_csv:
        archive_rows = read_csv(args.archive_csv)
        archive_source = str(args.archive_csv)
    else:
        archive_rows = fetch_archive_programs(
            cutoff=cutoff,
            limit=args.archive_limit,
            min_resolving_power=args.min_resolving_power,
            timeout=args.timeout,
        )
        archive_source = TAP_SYNC_URL

    records = consolidate_archive_rows(archive_rows)
    offline_counts = (
        load_telbib_counts(args.telbib_counts_csv)
        if args.telbib_counts_csv
        else None
    )
    cache_path = args.output_dir / "checked_programs.csv"
    cached_counts = (
        load_telbib_counts(cache_path)
        if args.reuse_checked and cache_path.exists() and offline_counts is None
        else None
    )
    attach_publication_counts(
        records,
        offline_counts=offline_counts,
        cached_counts=cached_counts,
        max_lookups=args.max_telbib_lookups,
        workers=args.workers,
        timeout=args.timeout,
    )
    ranked = rank_unpublished(records, cutoff=cutoff)[: args.target_count]

    output_dir: Path = args.output_dir
    checked = sorted(
        [record for record in records if "telbib_status" in record],
        key=lambda row: int(row["product_count"]),
        reverse=True,
    )
    write_csv(output_dir / "checked_programs.csv", checked, CHECKED_FIELDS)
    output_path = output_dir / "underexplored_programs.csv"
    write_csv(output_path, ranked, PROGRAM_FIELDS)
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "archive_source": archive_source,
        "telbib_source": str(args.telbib_counts_csv or TELBIB_API_URL),
        "reused_checked_counts": len(cached_counts or {}),
        "cutoff_date": cutoff.isoformat(),
        "archive_groups_received": len(archive_rows),
        "normalized_programs": len(records),
        "telbib_lookup_limit": args.max_telbib_lookups,
        "telbib_programs_checked": len(checked),
        "telbib_lookup_errors": sum(
            str(record.get("telbib_status", "")) not in {"ok", "fixture", "cache"}
            for record in checked
        ),
        "unpublished_programs_written": len(ranked),
        "minimum_resolving_power": args.min_resolving_power,
        "score_formula": (
            "0.45*log_product_count + 0.25*repeat_ratio + "
            "0.15*log_resolving_power + 0.10*capped_mean_snr + "
            "0.05*preview_fraction"
        ),
        "warning": (
            "Zero linked telbib papers is a discovery lead, not proof that the data "
            "have never been examined or mentioned elsewhere."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(ranked)} programs to {output_path}")
    return output_path


def _adql_quote(value: str) -> str:
    return value.replace("'", "''")


def build_observation_query(
    *, program_id: str, cutoff: str, limit: int, min_resolving_power: float
) -> str:
    program = _adql_quote(program_id)
    return f"""SELECT TOP {int(limit)}
dp_id, obs_id, proposal_id, target_name, instrument_name, obs_title,
t_min, t_exptime, em_min, em_max, em_res_power, snr, access_estsize,
preview_html, access_url
FROM ivoa.ObsCore
WHERE dataproduct_type='spectrum'
  AND proposal_id LIKE '{program}%'
  AND obs_release_date < '{_adql_quote(cutoff)}'
  AND calib_level >= 2
  AND em_res_power >= {float(min_resolving_power):.1f}
  AND (is_solar = 0 OR is_solar IS NULL)
ORDER BY snr DESC"""


def fetch_observations(
    *,
    program_id: str,
    cutoff: str,
    limit: int,
    min_resolving_power: float,
    timeout: float,
) -> list[dict[str, str]]:
    return tap_csv(
        build_observation_query(
            program_id=program_id,
            cutoff=cutoff,
            limit=limit,
            min_resolving_power=min_resolving_power,
        ),
        timeout=timeout,
    )


def select_repeat_friendly_sample(
    rows: Sequence[dict[str, str]], limit: int
) -> list[dict[str, object]]:
    """Favor targets represented by multiple, temporally separated products."""

    deduped: dict[str, dict[str, str]] = {}
    for row in rows:
        key = (row.get("dp_id") or row.get("obs_id") or "").strip()
        if key:
            deduped.setdefault(key, dict(row))

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in deduped.values():
        target = (row.get("target_name") or "(unnamed target)").strip()
        groups[target].append(row)
    ordered_groups = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    for _, group in ordered_groups:
        group.sort(key=lambda row: _safe_float(row.get("t_min")))

    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()

    def add(group: list[dict[str, str]], index: int) -> None:
        row = group[index]
        key = (row.get("dp_id") or row.get("obs_id") or "").strip()
        if not key or key in selected_ids or len(selected) >= limit:
            return
        formatted: dict[str, object] = dict(row)
        formatted["target_pool_count"] = len(group)
        selected.append(formatted)
        selected_ids.add(key)

    # Guarantee useful same-target comparisons first: oldest and newest
    # products for as many repeated targets as the requested limit permits.
    for _, group in ordered_groups:
        if len(group) < 2 or len(selected) >= limit:
            continue
        add(group, 0)
        add(group, len(group) - 1)

    # Fill any remaining space with intermediate epochs, then singletons.
    offset = 1
    while len(selected) < limit:
        before = len(selected)
        for _, group in ordered_groups:
            if len(group) > 2 and offset < len(group) - 1:
                add(group, offset)
            if len(selected) >= limit:
                break
        if len(selected) == before:
            break
        offset += 1
    for _, group in ordered_groups:
        if len(selected) >= limit:
            break
        if len(group) == 1:
            add(group, 0)
    return selected


def mjd_to_iso(value: object) -> str:
    mjd = _safe_float(value, default=-1.0)
    if mjd < 0:
        return ""
    epoch = datetime(1858, 11, 17, tzinfo=UTC)
    return (epoch + timedelta(days=mjd)).isoformat()


def format_observation(
    row: dict[str, object], *, program: dict[str, str]
) -> dict[str, object]:
    return {
        "program_rank": program.get("rank", ""),
        "program_score": program.get("score", ""),
        "program_id": program["program_id"],
        "target_pool_count": row.get("target_pool_count", ""),
        "dp_id": row.get("dp_id", ""),
        "obs_id": row.get("obs_id", ""),
        "target_name": row.get("target_name", ""),
        "instrument_name": row.get("instrument_name", ""),
        "obs_title": row.get("obs_title", ""),
        "observed_at_utc": mjd_to_iso(row.get("t_min")),
        "t_exptime_seconds": row.get("t_exptime", ""),
        "wavelength_min_nm": round(_safe_float(row.get("em_min")) * 1e9, 5),
        "wavelength_max_nm": round(_safe_float(row.get("em_max")) * 1e9, 5),
        "resolving_power": row.get("em_res_power", ""),
        "snr": row.get("snr", ""),
        "size_mb": round(_safe_float(row.get("access_estsize")) / 1000, 3),
        "preview_url": row.get("preview_html", ""),
        "data_url": row.get("access_url", ""),
        "review_status": "unreviewed",
        "repeat_confirmed": "",
        "conventional_explanations_checked": "",
        "notes": "",
    }


def sample(args: argparse.Namespace) -> Path:
    programs = read_csv(args.programs_csv)[: args.programs]
    if not programs:
        raise SystemExit(f"no programs found in {args.programs_csv}")

    fixture_rows = read_csv(args.observations_csv) if args.observations_csv else None
    output_rows: list[dict[str, object]] = []
    for program in programs:
        program_id = program["program_id"]
        if fixture_rows is not None:
            pool = [
                row
                for row in fixture_rows
                if normalize_program_id(row.get("proposal_id", "")) == program_id
            ]
        else:
            pool = fetch_observations(
                program_id=program_id,
                cutoff=program["cutoff_date"],
                limit=args.pool_per_program,
                min_resolving_power=args.min_resolving_power,
                timeout=args.timeout,
            )
        selected = select_repeat_friendly_sample(pool, args.per_program)
        output_rows.extend(
            format_observation(row, program=program) for row in selected
        )

    write_csv(args.output, output_rows, OBSERVATION_FIELDS)
    print(f"wrote {len(output_rows)} observations to {args.output}")
    return args.output


def add_discover_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--older-than-years", type=float, default=3.0)
    parser.add_argument("--min-resolving-power", type=float, default=10_000.0)
    parser.add_argument("--archive-limit", type=int, default=500)
    parser.add_argument("--max-telbib-lookups", type=int, default=120)
    parser.add_argument("--target-count", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--archive-csv", type=Path)
    parser.add_argument("--telbib-counts-csv", type=Path)
    parser.add_argument(
        "--reuse-checked",
        action="store_true",
        help="reuse successful counts already present in OUTPUT_DIR/checked_programs.csv",
    )


def add_sample_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--programs-csv", type=Path, default=Path("results/underexplored_programs.csv")
    )
    parser.add_argument("--programs", type=int, default=5)
    parser.add_argument("--per-program", type=int, default=20)
    parser.add_argument("--pool-per-program", type=int, default=300)
    parser.add_argument("--min-resolving-power", type=float, default=10_000.0)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--observations-csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/review_sample.csv"))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find old ESO spectral programs with no telbib-linked papers."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    discover_parser = subparsers.add_parser("discover", help="rank unpublished programs")
    add_discover_arguments(discover_parser)
    discover_parser.set_defaults(func=discover)
    sample_parser = subparsers.add_parser("sample", help="make a manual-review CSV")
    add_sample_arguments(sample_parser)
    sample_parser.set_defaults(func=sample)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        args.func(args)
        return 0
    except ExternalServiceError as exc:
        print(f"external service error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
