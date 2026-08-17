import csv
import tempfile
import unittest
from unittest.mock import patch
from argparse import Namespace
from datetime import date
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import find_underexplored as finder


FIXTURES = Path(__file__).parent / "fixtures"


class ProgramIdTests(unittest.TestCase):
    def test_normalizes_run_suffix(self):
        self.assertEqual(finder.normalize_program_id("072.C-0488(E)"), "072.C-0488")

    def test_splits_combined_ids(self):
        self.assertEqual(
            finder.split_program_ids("099.C-0001(A), 100.D-0002"),
            ["099.C-0001", "100.D-0002"],
        )


class TelbibTests(unittest.TestCase):
    def test_reads_count_from_stream_prefix(self):
        chunks = [b'<?xml version="1.0"?><xml><num', b"Found>17</numFound><item>"]
        self.assertEqual(finder.parse_telbib_count_prefix(chunks), 17)

    def test_telbib_uses_get(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def read(self, _size):
                if hasattr(self, "read_once"):
                    return b""
                self.read_once = True
                return b"<xml><numFound>0</numFound></xml>"

        with patch.object(finder, "_request", return_value=FakeResponse()) as request:
            self.assertEqual(finder.fetch_telbib_count("099.C-0001"), 0)
        self.assertEqual(request.call_args.kwargs["method"], "GET")


class RankingTests(unittest.TestCase):
    def test_only_zero_publication_records_are_ranked(self):
        rows = finder.read_csv(FIXTURES / "archive_programs.csv")
        records = finder.consolidate_archive_rows(rows)
        finder.attach_publication_counts(
            records,
            offline_counts=finder.load_telbib_counts(FIXTURES / "telbib_counts.csv"),
            cached_counts=None,
            max_lookups=20,
            workers=1,
            timeout=1,
        )
        ranked = finder.rank_unpublished(records, cutoff=date(2023, 8, 15))
        self.assertEqual([row["program_id"] for row in ranked], ["099.C-0001", "100.D-0002"])
        self.assertGreater(float(ranked[0]["score"]), float(ranked[1]["score"]))

    def test_sample_favors_repeat_targets_and_temporal_edges(self):
        rows = finder.read_csv(FIXTURES / "observations.csv")
        pool = [row for row in rows if row["proposal_id"].startswith("099.C-0001")]
        selected = finder.select_repeat_friendly_sample(pool, 3)
        targets = [row["target_name"] for row in selected]
        self.assertEqual(targets, ["HD 123", "HD 123", "HD 456"])
        self.assertEqual([row["dp_id"] for row in selected[:2]], ["ADP.1", "ADP.3"])

    def test_error_rows_are_not_loaded_as_zero_count_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "counts.csv"
            path.write_text(
                "program_id,publication_count,telbib_status\n"
                "099.C-0001,,error: timeout\n"
                "100.D-0002,0,ok\n",
                encoding="utf-8",
            )
            self.assertEqual(finder.load_telbib_counts(path), {"100.D-0002": 0})


class OfflineWorkflowTests(unittest.TestCase):
    def test_discover_and_sample_write_expected_csvs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            discover_args = Namespace(
                older_than_years=3.0,
                min_resolving_power=10_000.0,
                archive_limit=50,
                max_telbib_lookups=50,
                target_count=20,
                workers=1,
                timeout=1.0,
                archive_csv=FIXTURES / "archive_programs.csv",
                telbib_counts_csv=FIXTURES / "telbib_counts.csv",
                reuse_checked=False,
                output_dir=output_dir,
            )
            programs_path = finder.discover(discover_args)
            self.assertTrue(programs_path.exists())
            self.assertTrue((output_dir / "checked_programs.csv").exists())

            sample_args = Namespace(
                programs_csv=programs_path,
                programs=1,
                per_program=3,
                pool_per_program=20,
                min_resolving_power=10_000.0,
                timeout=1.0,
                observations_csv=FIXTURES / "observations.csv",
                output=output_dir / "review_sample.csv",
            )
            review_path = finder.sample(sample_args)
            with review_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["review_status"] == "unreviewed" for row in rows))


if __name__ == "__main__":
    unittest.main()
