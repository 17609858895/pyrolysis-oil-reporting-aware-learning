# Reporting-aware learning of pyrolysis-oil product slates

Core research code for **Learning compound-level pyrolysis-oil product slates from censored and presence-only literature records**.

The implementation joins exact GC–MS peak areas, non-listed compounds and presence-only lists. RL-CureBoost combines boosted abundance and mixture components, nested study-grouped configuration selection and cross-fitted listing calibration. The code also implements the 22 positive-only regressors, their non-negative stack, censoring-aware comparators, component ablations, within-study catalyst contrasts, sensitivity analyses and model-response diagnostics.

## Inputs and installation

Use Python 3.12. The dependency versions in `requirements.txt` record the environment used for this release.

```bash
python -m venv .venv
# Activate .venv with the command appropriate to your operating system.
python -m pip install -r requirements.txt
python run_pipeline.py setup
```

Download **version 1.0** of `compound_records.xlsx` from the original dataset, [doi:10.23642/USN.30630182](https://doi.org/10.23642/USN.30630182), and place it in `data/`. The input workbook SHA-256 is:

```text
a026dd38587f80ebc3051f61547f58969342c394992ac51d0666bbf41c882c0b
```

The workbook contains sheets `Extracted data (8738)` and `References`. The original dataset is maintained by its authors at DataverseNO; consult its documentation and usage terms and cite that dataset when reusing it. This repository distributes code only. Input data, computed tables, lookup caches and model outputs are created locally and excluded from Git.

`PYROLYSIS_DATA` can point to a workbook outside the repository. Structure resolution queries the public PubChem PUG REST service and resumes from `cache/pubchem_names.csv`. The optional `PYROLYSIS_PUBCHEM_CACHE` variable lets descriptor generation read an existing lookup cache. Because PubChem is updated, a fresh lookup can produce different resolutions from the study-time cache.

## Running the analyses

Run commands from the repository root. The stages below execute sequentially; `--jobs` controls the main cross-validation workers. Nested boosting, 22-learner stacking, seed checks and all ablations are computationally intensive. Start with one worker, then increase it to match available memory.

```bash
# Fast synthetic check: no research data or network required.
python smoke_check.py

# Cleaning, PubChem resolution, descriptors and the main/sensitivity grids.
python run_pipeline.py prepare

# Conventional/censoring-aware benchmarks, nested RL-CureBoost and paired tests.
python run_pipeline.py benchmark --jobs 1

# Component/feature/data-definition ablations and leave-family-out validation.
python run_pipeline.py ablation --jobs 1

# Catalyst contrasts, counterfactual fidelity, SHAP, response curves and diagnostics.
python run_pipeline.py analysis --jobs 1
```

`python run_pipeline.py all --jobs 1` performs all stages. The fidelity script schedules one worker per eligible held-out study; run it on a machine with enough memory. Each script can also be invoked separately. The main benchmark entry points provide `--help`.

## Code map

| Script | Purpose |
|---|---|
| `s00_resolve_pubchem.py` | Name-to-structure queries with resumable cache |
| `s01_build_dataset.py` | Clean records and define study-linked oils |
| `s02_compound_descriptors.py` | Formula-guarded identities, RDKit descriptors and classes |
| `s03_build_grid.py` | Candidate vocabulary and censoring intervals |
| `s04_benchmark.py` | Learners, mixture-cure EM, finite contribution scores, study folds and metrics |
| `s05_within_study_causal.py` | Matched observational catalyst contrasts and pooled/within-study DML |
| `s07_counterfactual_fidelity.py` | Held-out agreement of model responses with observed contrasts |
| `s08_interpret.py` | Channel-specific SHAP and catalyst-associated response curves |
| `s09_paired_tests.py` | Study-level signed-rank comparisons |
| `s10_final.py` | Nested configuration selection, calibration, ablations and stability |
| `s11_primary_tests.py` | Prespecified comparisons with Holm adjustment |
| `s12_ablation_summary.py` | Paired ablation and sensitivity tables |
| `s13_figure_data.py` | Exceedance maps, family responses and distance diagnostics |
| `s15_si_analyses.py` | Stack weights, catalyst sensitivity and temperature ALE/PD |

Scripts retain the study's numbering so inputs and outputs can be traced between stages. Scientific plotting assets and document-production tools are outside this core-code release.

## Definitions needed to interpret outputs

- An oil is the study/feedstock/catalyst/technology/temperature/measurement combination. The main analysis has 152 quantified oils and 60 presence-only oils from 70 unique primary studies, with 171 candidate structures and 36,252 oil–compound records.
- Candidate vocabulary and presence-only bounds are defined using the complete compilation. Study-held-out validation therefore measures new-study prediction for this fixed vocabulary and bounds.
- Listing AUC concerns reported versus non-listed candidates. The latent mixture component is a statistical zero/positive-abundance decomposition and is not an independently observed formation mechanism.
- Contribution scores integrate a normalized error density over the standardized range −6 to 6 on 241 points. Their normalized class shares are relative scores, not measured mass fractions or an unrestricted abundance expectation.
- The legacy internal class key `Anhydrosugars` includes sugars and anhydrosugars. Class labels identify operational structure groups.
- Catalyst contrasts are observational. Within-study matches use feedstock family and nearest temperature; model-response fidelity and prediction accuracy address different questions.

The synthetic check verifies fitting and numerical behavior. It does not reproduce the paper's cross-validation results. A full rerun requires the original input workbook, structure queries and the computational stages above.
