# Shifts in staple-food sourcing after supplier disruptions are short-lived

Minimal computational companion to the manuscript of the same title. The code
constructs annual supplier-disruption and calm-year comparisons, estimates
fixed-effect event-study paths, and performs clustered bootstrap inference for
six staple crops. The estimates describe sourcing responses; they do not
identify changes in risk beliefs or a psychological memory half-life.

## Files

```text
code/
  rm2.py                 Core data, event, fixed-effect and numerical routines
  run_v2.py              Estimates, bootstrap inference and sensitivity analyses
data/
  panel_dyad.parquet
  fao_prod.parquet
  cru_monthly.parquet
  country_names.csv
  cru_crosswalk.csv
  policy_years_prohibition.csv
  policy_years_prohib_quota.csv
run_analysis.py          Portable numerical-analysis entry point
requirements.txt
LICENSE
README.md
```

## Environment and installation

The project specifies Python 3.11 with NumPy 1.26, pandas 2.1, SciPy 1.15 and
PyArrow 14 or later. Use Python 3.11 in a dedicated environment, then run:

```sh
python -m pip install -r requirements.txt
```

The release was also checked on Python 3.13.3, NumPy 2.2.6, pandas 3.0.5,
SciPy 1.17.1 and PyArrow 24.0.0. These checks comprise input loading, a reduced
bootstrap trial and main point-estimate comparison, not a complete rerun of
the manuscript inference. The declared Python 3.11 environment was not
installed during release preparation. A grouping-column compatibility fix in
`run_v2.py` preserves the original weather-skill calculation on pandas 3.

## Run

Run from this folder. No downloads or external local files are required.

```sh
python run_analysis.py --check
python run_analysis.py --smoke
python run_analysis.py --nproc 1
```

`--check` parses the scripts, imports the dependencies and reads all seven
inputs without estimation. `--smoke` prepares the supplied data and runs the
existing two-year clean-history sensitivity with two bootstrap draws. This
diagnostic does not produce manuscript-valid confidence intervals or replace
the main analysis. The last command runs the full numerical pipeline, followed
by the original one- and two-year clean-history sensitivities; it can take a
long time. Windows requires one worker. On systems supporting fork, multiple
workers may be selected with `--nproc`.

Outputs go to a new `runs/<timestamp>/results/` directory. To choose a location,
use `--output <new-or-empty-directory>`. A nonempty output directory is rejected.

## Parameters and methods

The supplied code retains the research model, thresholds and bootstrap seed
rule. Trade observations cover 1993–2024; eligible disruption and comparison
years are 1996–2023. The annual event-time range is −3 through 8, with −1 as
reference. Pre-event dependence uses the three preceding years. The main
outcome is `min(s / d, 3) - 1`, where `s` is the current supplier share and
`d` is its pre-disruption mean. Dependence-bin-by-event-time and
supplier-crop-by-event-time fixed effects are absorbed numerically.

Main production events require log deviations at or below `log(0.85)` for
production and `log(0.90)` for yield, using a backward ten-year log trend with
at least seven observations. Major-exporter thresholds are a 2% share and
200,000 tonnes; importer thresholds are 20,000 tonnes and 2% dependence.
The default clean-history length is three years. Weather validation uses
leave-one-year-out ridge prediction with penalty 20. The code also performs
the manuscript's sensitivity, margin, supplier-path and recurrence analyses.

Bootstrap defaults are `B_MAIN=1999`, `B_SUB=999` and `B_ROB=499`. Each draw
uses `numpy.random.default_rng(b)` for `b=1,...,B`; the main resampling unit
is the supplier-crop pair, with an onset-year alternative. Environment
variables of those names can change the draw counts for diagnostic runs;
retain the defaults for the manuscript analysis. `SKIP_ROBUST`, if set, omits
the main robustness grid. Invalid bootstrap draws can be discarded; formal
outputs report the number retained. Annual observations, rather than a
time-stepped stochastic dynamical simulation, determine the time resolution.

## Necessary inputs and provenance

The seven files are fixed, derived analysis inputs, not fitted outcomes or
precomputed event-study results. Every file is read by the retained numerical
pipeline; including these snapshots avoids rebuilding inputs from changing
upstream releases.

| Input | Source and computational role |
| --- | --- |
| `panel_dyad.parquet` | Reconciled FAOSTAT bilateral quantities and values for wheat, maize, rice, soybeans, barley and sorghum; 293,357 dyad-years. Used to compute sourcing shares and trade margins. |
| `fao_prod.parquet` | FAOSTAT production, harvested area and yield observations; used to fit trends and identify production events. |
| `cru_monthly.parquet` | CRU-CY v4.10 country-average monthly maximum temperature, precipitation and potential evapotranspiration; necessary for the weather-validation calculations performed during preparation. |
| `country_names.csv` | FAO country codes and names for event labels. |
| `cru_crosswalk.csv` | FAO-to-CRU country mapping for weather features. |
| `policy_years_prohibition.csv` | Audited annual export-prohibition indicators derived from the AMIS Policy Database; main restriction-event input. |
| `policy_years_prohib_quota.csv` | Annual prohibitions and quotas from the same source; required by the quota sensitivity and loaded by the pipeline. |

The project records the FAOSTAT release of December 2025 and AMIS update of
September 2025. Quantities are in tonnes; rice quantities use milled
equivalents. Where importer reports are unavailable, mirror export reports
are used, with the project's value reconciliation retained.

Data attribution: FAO, FAOSTAT; Agricultural Market Information System (AMIS),
Policy Database; and **Climatic Research Unit, University of East Anglia**,
CRU-CY v4.10. For the climate dataset, also cite Harris et al. (2020),
*Scientific Data* 7, 109, [doi:10.1038/s41597-020-0453-3](https://doi.org/10.1038/s41597-020-0453-3).
See [FAO database terms](https://www.fao.org/contact-us/terms/db-terms-of-use/en)
and [CRU v4.10 data and licensing](https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.10/).
FAO's applicable database terms include CC BY 4.0, subject to stated exceptions.
CRU data retain the Open Database License and Database Contents License
attribution/share-alike terms. AMIS-derived records retain their source terms.
The MIT software license does not replace third-party data rights.

## Generated outputs and exclusions

Runs generate CSV estimates, event and balance audits, weather predictions,
Parquet analysis panels, JSON descriptive summaries and, with sufficient
bootstrap draws, NumPy bootstrap arrays. No generated output is bundled.
The package excludes figures, plotting and manuscript-layout scripts, papers,
submission materials, previously computed results, raw bulk downloads,
preprocessing/audit intermediates, caches and historical copies. It starts
from the provided derived inputs; rebuilding them from raw sources is outside
this minimal package.

## License, contact and citation

The code and documentation are licensed under MIT; see `LICENSE`. Input data
retain the source rights described above. Contact the maintainers through the
GitHub repository's Issues page. No personal contact details are included.

Until bibliographic details are finalized, cite the associated manuscript by
its exact title: *Shifts in staple-food sourcing after supplier disruptions
are short-lived* (unpublished manuscript), together with the repository URL
and release identifier. When available, use the final author list, publication
year and DOI. This release supplies no fabricated author identities or DOI.
