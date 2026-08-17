import tempfile
import unittest
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import harps_residual_search as search


class HarpsAnalysisTests(unittest.TestCase):
    def test_continuum_normalize_flattens_a_sloped_spectrum(self):
        wave = np.arange(4000.0, 4100.0, 0.01)
        continuum = 1000 + 2 * (wave - 4000)
        line = 1 - 0.3 * np.exp(-0.5 * ((wave - 4050) / 0.08) ** 2)
        normalized = search.continuum_normalize(wave, continuum * line)
        outside = np.abs(wave - 4050) > 1
        self.assertAlmostEqual(float(np.nanmedian(normalized[outside])), 1.0, places=2)
        self.assertLess(float(np.nanmin(normalized)), 0.75)

    def test_candidate_detector_finds_a_resolved_injected_line(self):
        wave = np.arange(5000.0, 5020.0, 0.01)
        residual = np.zeros_like(wave)
        center = np.argmin(np.abs(wave - 5010.0))
        pixels = np.arange(len(wave))
        residual += 0.1 * np.exp(-0.5 * ((pixels - center) / 1.5) ** 2)
        zscore = residual / 0.005
        valid = np.ones_like(wave, dtype=bool)
        dummy = search.Spectrum(
            "ADP.1", "TEST", "2020-01-01", Path("test.fits"), wave, wave, wave, 10, 100
        )
        candidates = search.detect_candidates(
            target="TEST",
            first=dummy,
            second=dummy,
            wave=wave,
            first_norm=np.ones_like(wave),
            second_norm=np.ones_like(wave) + residual,
            residual=residual,
            sigma=np.full_like(wave, 0.005),
            zscore=zscore,
            valid=valid,
            min_peak_z=8,
        )
        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(
            float(candidates[0]["wavelength_barycentric_angstrom"]), 5010.0, places=2
        )
        self.assertEqual(candidates[0]["status"], "follow_up_required")

    def test_candidate_detector_rejects_a_too_narrow_spike(self):
        wave = np.arange(5000.0, 5020.0, 0.01)
        residual = np.zeros_like(wave)
        center = np.argmin(np.abs(wave - 5010.0))
        residual[center] = 0.1
        zscore = residual / 0.005
        valid = np.ones_like(wave, dtype=bool)
        dummy = search.Spectrum(
            "ADP.1", "TEST", "2020-01-01", Path("test.fits"), wave, wave, wave, 10, 100
        )
        candidates = search.detect_candidates(
            target="TEST",
            first=dummy,
            second=dummy,
            wave=wave,
            first_norm=np.ones_like(wave),
            second_norm=np.ones_like(wave) + residual,
            residual=residual,
            sigma=np.full_like(wave, 0.005),
            zscore=zscore,
            valid=valid,
            min_peak_z=8,
        )
        self.assertEqual(candidates[0]["status"], "likely_single_pixel_or_cosmic_ray")

    def test_safe_filename_replaces_colons(self):
        self.assertEqual(
            search.safe_filename("ADP.2020-01-01T01:02:03"),
            "ADP.2020-01-01T01_02_03.fits",
        )

    def test_narrow_line_excess_measures_an_injected_peak(self):
        wave = np.arange(5000.0, 5020.0, 0.01)
        normalized = np.ones_like(wave)
        center = np.argmin(np.abs(wave - 5010.0))
        pixels = np.arange(len(wave))
        normalized += 0.08 * np.exp(-0.5 * ((pixels - center) / 1.8) ** 2)
        spectrum = search.Spectrum(
            "ADP.1",
            "TEST",
            "2020-01-01",
            Path("test.fits"),
            wave,
            normalized,
            normalized,
            10,
            100,
        )
        excess, peak = search.narrow_line_excess(spectrum, 5010.0)
        self.assertGreater(excess, 0.07)
        self.assertAlmostEqual(peak, 5010.0, places=2)

    def test_narrow_line_metrics_reports_local_significance(self):
        wave = np.arange(5759.0, 5763.0, 0.005)
        normalized = np.ones_like(wave)
        center = np.argmin(np.abs(wave - 5761.26))
        pixels = np.arange(len(wave))
        normalized += 0.12 * np.exp(-0.5 * ((pixels - center) / 3.0) ** 2)
        normalized += 0.001 * np.sin(pixels / 3)
        spectrum = search.Spectrum(
            "ADP.1", "TOI-2458", "2022-02-19", Path("test.fits"),
            wave, normalized, normalized, -26.4, 100
        )
        metrics = search.narrow_line_metrics(spectrum, 5761.26)
        self.assertGreater(metrics["excess"], 0.11)
        self.assertGreater(metrics["local_z"], 20)
        self.assertAlmostEqual(metrics["peak_wavelength"], 5761.26, places=2)

    def test_observer_frame_conversion_round_trips(self):
        barycentric = 5761.26
        berv = -26.419257
        topocentric = barycentric / (1.0 + berv / search.C_KMS)
        recovered = search.observer_frame_barycentric_wavelength(topocentric, berv)
        self.assertAlmostEqual(recovered, barycentric, places=9)

    def test_toi2458_query_is_fixed_to_public_harps_spectra(self):
        query = search.toi2458_archive_query()
        self.assertIn("target_name='TOI-2458'", query)
        self.assertIn("instrument_name LIKE 'HARPS%'", query)
        self.assertIn("calib_level>=2", query)

    def test_noise_ranking_statistics_are_returned(self):
        candidate = {
            "peak_abs_z": 12.0,
            "fwhm_pixels": 3.5,
            "fwhm_to_instrument_ratio": 0.8,
            "known_line": "",
            "cross_target_matches": "",
            "status": "follow_up_required",
        }
        scored = search.score_candidate(candidate, independent_trials=100_000)
        self.assertLess(scored["gaussian_single_trial_log10_p"], 0)
        self.assertLessEqual(scored["gaussian_global_log10_fap"], 0)


if __name__ == "__main__":
    unittest.main()
