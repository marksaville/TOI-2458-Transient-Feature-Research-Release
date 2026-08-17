# Derived results

These are the compact, machine-readable publication results. They can be
inspected without downloading the source FITS files.

## Screening and sensitivity

- `harps_residual_search/pair_summary.csv`: ten target-pair summaries.
- `harps_residual_search/candidates.csv`: all 1,534 threshold crossings and
  their automated classifications.
- `harps_residual_search/manual_review_notes.csv`: dispositions for the 13
  features sent to manual review.
- `harps_residual_search/injection_results.csv`: 30 target/amplitude summaries
  covering 360 injection trials.
- `harps_residual_search/recurrence_results.csv`: initial same-target recurrence
  checks for the 13 reviewed cases.

## Focused TOI-2458 checks

- `harps_residual_search/toi2458_archive_epochs.csv`: the frozen 88-product
  archive-query result.
- `harps_residual_search/toi2458_all_epoch_line_scan.csv`: two fixed-wavelength
  measurements in all 88 products, for 176 rows total.
- `harps_residual_search/order_level_checks.csv`: extracted-order and raw-frame
  diagnostics at both event coordinates.
- `harps_residual_search/candidate_cosmic_check.json`: detector-coordinate,
  cosmic-mask, controlled-reduction, and recurrence summary.
- `harps_residual_search/candidate_pipeline_report.json`: exact parameters and
  checksums for controlled pipeline products. Its paths document the original
  analysis workspace; the large products themselves are intentionally omitted.

## Independent spectra and plots

- `harps_residual_search/external_spectra_line_comparison.csv`: eight individual
  CHIRON/TRES matched-filter measurements.
- `harps_residual_search/external_spectra_stacked_summary.csv`: four
  instrument/wavelength stacks.
- `harps_residual_search/plots/`: per-target screening plots plus the three
  figures used in the paper.

Table C.1 in the manuscript gives SHA-256 values for the principal result
files. The repository-wide `SHA256SUMS` file covers every release artifact.

