# TOI-2458 transient-feature research release

This is the complete public research release for Mark Saville's report,
*A search for transient unresolved optical features in public HARPS spectra:
The single-exposure TOI-2458 case* (version 1.0, 15 August 2026).

The principal conclusion is conservative: the two recorded features are real
features of one HARPS exposure, but their simultaneous appearance, unusual
cosmic-mask neighborhoods, lack of credible recurrence in 87 comparison HARPS
products, and absence of persistent full-strength counterparts in CHIRON data
favor an exposure-specific detector or reduction origin. The available data do
not identify one unique mechanism.

This is an author-released public research report, not a peer-reviewed journal
article. The manuscript carries no journal affiliation or submission claim.

## Start here

- [`manuscript/Mark_Saville_TOI2458_Public_Research_Report.pdf`](manuscript/Mark_Saville_TOI2458_Public_Research_Report.pdf) is the final paper.
- `REPRODUCIBILITY.md` is a code-independent replication protocol.
- `data/` contains compact manifests with public download URLs, file sizes, and
  SHA-256 checksums. Large telescope FITS files are intentionally not copied
  into this repository.
- [`results/harps_residual_search/`](results/harps_residual_search/) contains the machine-readable measurements
  and diagnostic figures supporting the paper.
- `code/` contains the analysis source, frozen dependency lock, and tests.
- `SHA256SUMS` is the integrity record for the entire release.

From the repository root, verify every archived file with:

```bash
sha256sum -c SHA256SUMS
```

## Reproducing the calculations

Python 3.11 or newer is required. The archived environment used CPython
3.11.15, Astropy 7.2.2, NumPy 2.4.6, SciPy 1.17.1, and Matplotlib 3.11.1.

```bash
cd code
uv sync --extra analysis
cd ..
code/.venv/bin/python -m unittest discover -s code/tests -v
```

The analysis commands are:

```bash
code/.venv/bin/python code/harps_residual_search.py download
code/.venv/bin/python code/harps_residual_search.py analyze
code/.venv/bin/python code/harps_residual_search.py recurrence
code/.venv/bin/python code/harps_residual_search.py toi2458-download
code/.venv/bin/python code/harps_residual_search.py toi2458-scan
code/.venv/bin/python code/external_spectra_compare.py download
code/.venv/bin/python code/external_spectra_compare.py analyze
```

The detector-level controlled reductions additionally require ESO HARPS
pipeline 3.6.0 and the public calibration inputs listed in `data/`. See
`data/README.md` before attempting that resource-intensive stage.

The supplied result files are the frozen publication artifacts. Re-running a
network download at a later date may encounter archive updates; compare every
download with its recorded checksum before interpreting differences.

## Authorship and contact

Mark Saville — <mark.saville17@gmail.com>

No institutional affiliation is asserted.
