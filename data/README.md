# Source-data record

This repository stores manifests and derived measurements, not duplicate
copies of the large public telescope products. The manifests give stable
dataset identifiers, direct or discovery URLs, byte counts, and SHA-256
checksums for the exact files analyzed.

## Included manifests

- `harps_106_21tj/manifest.csv`: the 20 HARPS products in the initial screen.
- `toi2458_all_epochs/manifest.csv`: all 88 public TOI-2458 HARPS products in
  the fixed archive query as of 15 August 2026.
- `harps_followup/manifest.csv`: the smaller same-target comparison download.
- `toi2458_external_spectra/manifest.csv`: four CHIRON and TRES spectra from
  ExoFOP-TESS.
- `event_raw_and_ancillary_manifest.csv`: the event product, raw detector
  exposure, and ancillary extracted-order package.
- `harps_calibrations/night_2022-02-18/manifest.csv`: the selected same-night
  raw and master calibration inputs.
- `harps_calibrations/association_raw2master.xml`: ESO's archived calibration
  association response.
- `pipeline_kit_manifest.csv`: provenance for the HARPS 3.6.0 software kit and
  hashes of its principal embedded payloads.

Downloading every listed item can require roughly 1.1 GB before unpacking and
pipeline installation. The complete installed pipeline and temporary
reduction products require substantially more space; they are not GitHub
supplementary data.

## Pipeline-kit version note

The analysis used a local ESO `harps-kit-3.6.0.tar.gz` container of 296,681,727
bytes with SHA-256
`a2d29b47c6e87d9b9ab1571fad899b830bea4ef55c672450fc667a34db9cd7db`.
ESO's development-kit URL served a 296,679,593-byte container when checked on
15 August 2026. Because the outer containers are not byte-identical, the
manifest also records hashes for the embedded HARPS recipe, calibration, and
configuration archives used locally. A byte-for-byte audit should compare
those inner payloads before treating the current outer kit as identical.

The exact local kit is 283 MB and should not be committed as an ordinary Git
object. If long-term preservation of that outer container is desired, attach
it separately to the tagged GitHub release or deposit it in a repository that
accepts large research artifacts, subject to ESO's redistribution terms.

## Data ownership

HARPS source products remain subject to ESO archive terms. CHIRON and TRES
source products remain subject to ExoFOP-TESS and originating-observatory
terms. This release's license applies to the author's manuscript, analysis
source, derived tables, and original figures; it does not relicense the source
telescope observations.

