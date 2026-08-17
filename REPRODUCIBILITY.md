# Reproducibility protocol for the TOI-2458 transient-feature study

This document defines a code-independent route from the public source spectra to the claims in `manuscript/Mark_Saville_TOI2458_Public_Research_Report.pdf`. It contains identifiers, decision thresholds, intermediate checks, and expected end points. An independent analyst may use any FITS-capable astronomical software that preserves the native wavelength samples.

## 1. Scope and frozen inputs

Use the data and archive state available on 15 August 2026.

The initial screen consists of the 20 ESO HARPS products listed in Table A.1 of the manuscript and in `data/harps_106_21tj/manifest.csv`. Verify that the manifest SHA-256 digest is:

`5ab71598ad25c527e8ad89c39799c8afddfeb0444e424c64ea9b60b65c6c593c`

The complete TOI-2458 comparison consists of the 88 products in `data/toi2458_all_epochs/manifest.csv`. Verify that its digest is:

`a38050c4bcc2b736211531331afb8c788a8ed3f211aa285dc4858163c9179a4e`

The independent comparison consists of the four ExoFOP-TESS products listed in Table B.1 and `data/toi2458_external_spectra/manifest.csv`. Verify that its digest is:

`f90ec77e3493df45d35e4b37f2bf6b2a50c4e1269f1c0ab699f1ceef0c27a7ab`

Each manifest provides a checksum for every source spectrum. Confirm the target and programme FITS headers before measurement. A checksum mismatch means the input is not the analyzed version.

## 2. Reproduce the initial HARPS screen

For each of the ten target pairs:

1. Read the wavelength and flux arrays from the public calibrated spectrum.
2. Retain finite, positive flux samples.
3. Divide the spectrum into 5 Å wavelength bins, calculate the median flux in each bin, interpolate those medians to the native samples, and divide flux by the interpolated continuum.
4. Align the second epoch to the first by minimizing the median absolute deviation of their normalized difference. Use every twentieth valid sample and allow a velocity shift from −2 to +2 km s−1.
5. Interpolate the moving spectrum to the reference grid and subtract first epoch from second epoch.
6. Smooth that difference with a 40-pixel Gaussian kernel and subtract the smoothed component.
7. Estimate the local scale as 1.4826 times the median absolute deviation in 20 Å bins. Interpolate the scale across wavelength and impose a floor equal to half the tenth percentile of valid bin scales.
8. Find absolute residual peaks at or above 8 local scale units, with prominence at least 5.2, separation of at least three samples, and measured width no greater than 14 samples.

The output should contain 1,534 peaks. Their automated counts must be:

| Category | Expected count |
|---|---:|
| Low-sensitivity edge | 782 |
| Absorption variability | 472 |
| Too narrow | 226 |
| Alignment/profile residual | 18 |
| Broad profile change | 14 |
| Common-line proximity | 6 |
| Observer-frame artifact | 3 |
| Manual review | 13 |

If these counts differ, first check wavelength convention, velocity interpolation direction, median-absolute-deviation scaling, and whether the 40-pixel value was treated as a Gaussian sigma rather than a FWHM.

## 3. Apply the fixed screening rules

For a resolving power of 115,000, calculate the instrumental FWHM as wavelength divided by resolving power. Apply the rules in this order:

1. label wavelengths below 4000 Å or above 6800 Å as low-sensitivity edges;
2. label measured widths below 0.75 instrumental FWHM as too narrow;
3. label widths above 1.8 instrumental FWHM as broad profile changes;
4. label positions within 0.20 Å of Na I D1, Na I D2, Hα, or the [O I] 5577, 6300, and 6364 Å night-sky lines as common-line proximity;
5. require the brighter epoch to exceed its local baseline by more than both 1% and three local scale units;
6. reject a candidate as bipolar when the strongest nearby opposite-sign residual exceeds both 6 local scale units and 35% of the candidate height; and
7. label a remaining candidate as observer-frame recurrent when a feature occurs within 0.05 Å in an unrelated target after barycentric-to-observer-frame conversion.

Manual inspection of the 13 retained features should leave the five entries in Table 2 of the manuscript. The machine-readable dispositions are in `results/harps_residual_search/manual_review_notes.csv`.

## 4. Reproduce the injection check

At 12 randomly fixed valid continuum locations in each target-pair residual, inject Gaussian profiles with FWHM equal to wavelength divided by 115,000. Test peak amplitudes of 2%, 5%, and 10% of the continuum. A recovery requires at least one sample at 8 local scale units and at least two samples at 5 units within the nearby instrumental window.

Across 120 trials per amplitude, the expected totals are:

| Injected peak | Recovered |
|---:|---:|
| 2% | 0/120 |
| 5% | 0/120 |
| 10% | 59/120 |

The exact target-level counts are in `results/harps_residual_search/injection_results.csv`.

## 5. Reproduce the focused TOI-2458 measurements

Use event product `ADP.2022-02-20T01:06:53.624` and fixed archive air wavelengths 5761.26 and 6432.95 Å.

For each wavelength, define the core as samples within two HARPS FWHM and the sidebands as samples three to eight FWHM away. The local baseline is the median sideband flux and the scale is 1.4826 times the sideband median absolute deviation. Choose the brightest core sample.

Expected event measurements are:

| Quantity | 5761.26 Å | 6432.95 Å |
|---|---:|---:|
| Peak normalized flux | 1.194201 | 1.112560 |
| Local baseline | 0.996170 | 1.003122 |
| Excess | 0.198032 | 0.109438 |
| Local significance | 6.2790 | 9.9877 |

For each of the other 87 products, make this measurement once at the fixed stellar-frame wavelength and once at the fixed observer-frame wavelength. A recurrence must exceed both 0.05 and half the event excess and must have local significance of at least 5.

Expected results:

- 5761.26 Å: zero recurrences in both frames.
- 6432.95 Å: one recurrence in both frames, product `ADP.2022-03-09T01:01:43.431`.
- The latter event has a half-maximum width of 0.020 Å; the HARPS FWHM is 0.055939 Å. It must therefore be rejected as sub-resolution.

## 6. Reproduce the detector and pipeline checks

Locate the events in the extracted object-fibre orders at zero-based order-row and detector-x coordinates (53, 1245) and (64, 1506). Confirm both features in the `e2ds_A` orders before the final one-dimensional merge.

At the first coordinate, compare the raw cross-dispersion event profile with the normal object trace. The expected Pearson correlation is 0.98255. Verify that neither coordinate coincides with a maximum of the simultaneous reference-fibre comb.

Inspect the `CRH_MAP` within the object-order aperture and ±15 detector-x samples:

| Coordinate | Exact-coordinate flags | Window flags | Same-order percentile |
|---|---:|---:|---:|
| 5761 Å event | 0 | 2 | 99.803 |
| 6433 Å event | 1 | 19 | 97.196 |

Reduce raw exposure `HARPS.2022-02-19T00:15:10.071` with ESO HARPS pipeline 3.6.0 and the same-night order, flat, ThAr/ThAr, and ThAr/Fabry--Pérot calibrations. Use the three rejection configurations listed in Sect. 2.3 of the manuscript. The resulting excesses should agree with Table 3 within the rounding shown. The persistence of the features across these reductions is a required result; their cosmic-mask neighborhoods are a separate diagnostic.

## 7. Verify wavelength convention

Read the archive table metadata `TUCD1=em.wl;obs.atmos` and `SPECSYS=BARYCENT`. Treat archive labels as barycentric air wavelengths. Treat the pipeline array `WAVEDATA_VAC_BARY` as barycentric vacuum wavelength.

Using the Ciddor standard-air relation, reproduce:

| Air wavelength | Converted vacuum wavelength | Pipeline same-pixel label | Velocity residual |
|---:|---:|---:|---:|
| 5761.260000 Å | 5762.857943 Å | 5762.863443 Å | +0.286 km s−1 |
| 6432.950000 Å | 6434.728081 Å | 6434.724733 Å | −0.156 km s−1 |

This check must be performed at fixed detector pixels. The approximately 83 km s−1 difference between the unconverted labels is not a source velocity.

## 8. Reproduce the CHIRON and TRES comparison

For each external spectrum and wavelength:

1. Select the echelle order that covers the fixed air wavelength.
2. Continuum-normalize the order by the same 5 Å median-bin procedure.
3. Align the order to feature-free HARPS product `ADP.2022-02-20T01:06:53.630` by high-pass cross-correlation. Exclude ±1 Å around the candidate and search −120 to +120 km s−1 in 0.05 km s−1 increments.
4. Convolve both HARPS spectra from resolving power 115,000 to the external resolving power, adding Gaussian widths in quadrature.
5. Interpolate the degraded feature-free HARPS spectrum to the aligned external grid and subtract it from the external spectrum.
6. Fit a quadratic trend using samples 0.5--8 Å from the candidate, with five iterations of 3σ clipping, and subtract the trend.
7. Fit the amplitude of a Gaussian with FWHM equal to one external-instrument resolution element.
8. Repeat the same amplitude fit every 0.2 Å at placebo centers 1.0--7.5 Å from the candidate on both sides.
9. Subtract the median placebo amplitude from the candidate amplitude and divide by 1.4826 times the placebo median absolute deviation.
10. Apply the same matched filter to the resolution-degraded HARPS event-minus-control profile to obtain the full-strength expectation.

The eight individual observed and expected values must match Table 4 after rounding. No observed absolute empirical value may reach 3.

For each instrument and wavelength, combine the two median-subtracted amplitudes using inverse empirical-variance weights. The expected stacks are:

| Instrument | Wavelength | Observed z | Full-strength expected z |
|---|---:|---:|---:|
| CHIRON | 5761.26 Å | 1.5189 | 5.6061 |
| CHIRON | 6432.95 Å | 0.4756 | 4.9650 |
| TRES | 5761.26 Å | −1.0305 | 1.3989 |
| TRES | 6432.95 Å | 1.5518 | 2.7690 |

## 9. Decision logic

The paper’s conclusion follows only if all of these checks hold:

1. both features are present in the event exposure and extracted orders;
2. both survive the controlled reductions;
3. neither has a credible recurrence among 87 other public HARPS products;
4. the mask neighborhoods provide independent instrumental warnings;
5. no individual external measurement reaches the review threshold;
6. CHIRON was capable of detecting a persistent full-strength counterpart; and
7. the external spectra predate the event, so their non-detections constrain persistence but not a short-lived event.

These statements support the conclusion that an exposure-specific detector or reduction origin is favored but not uniquely proven. Any replication that changes one of these seven statements should report the changed input checksum, wavelength convention, threshold, or calculation before revising the interpretation.

## 10. Analysis environment used for the archived result

The archived calculations were produced with CPython 3.11.15, Astropy 7.2.2, NumPy 2.4.6, SciPy 1.17.1, and Matplotlib 3.11.1. These versions are provenance information, not a requirement for an independent implementation. The full scientific result is encoded in the public identifiers, checksums, parameter values, formulas, and expected intermediate values above.
