import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from external_spectra_compare import (
    _fit_gaussian_amplitude,
    air_to_vacuum,
    stack_measurements,
)


class ExternalSpectraCompareTests(unittest.TestCase):
    def test_air_to_vacuum_matches_candidate_pipeline_labels(self) -> None:
        self.assertAlmostEqual(air_to_vacuum(5761.26), 5762.858, places=3)
        self.assertAlmostEqual(air_to_vacuum(6432.95), 6434.728, places=3)

    def test_gaussian_amplitude_recovers_injected_line(self) -> None:
        center = 6000.0
        fwhm = 0.1
        wave = np.linspace(center - 1.0, center + 1.0, 401)
        expected = 0.075
        gaussian = np.exp(
            -0.5 * ((wave - center) / (fwhm / 2.354820045)) ** 2
        )
        residual = 0.004 + 0.002 * (wave - center) + expected * gaussian
        measured = _fit_gaussian_amplitude(wave, residual, center, fwhm)
        self.assertAlmostEqual(measured, expected, places=8)

    def test_stack_measurements_inverse_variance_combination(self) -> None:
        rows = []
        for instrument in ("CHIRON", "TRES"):
            for target in (5761.26, 6432.95):
                for signal, sigma in ((0.02, 0.02), (0.04, 0.04)):
                    rows.append(
                        {
                            "instrument": instrument,
                            "target_air_angstrom": target,
                            "observed_matched_amplitude": signal,
                            "placebo_amplitude_median": 0.0,
                            "empirical_amplitude_sigma": sigma,
                            "expected_full_harps_strength_amplitude": 0.1,
                        }
                    )
        result = stack_measurements(rows)[0]
        self.assertAlmostEqual(result["combined_signal_amplitude"], 0.024, places=8)
        self.assertAlmostEqual(result["combined_empirical_sigma"], 0.01788854, places=8)


if __name__ == "__main__":
    unittest.main()
