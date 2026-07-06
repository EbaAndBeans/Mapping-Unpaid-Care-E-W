# Mapping Unpaid Care and Economic Activity Across England and Wales, 2011–2021

A multi-scale visual analytics study examining how unpaid-care prevalence, intensity and economic inactivity changed across Local Authorities and neighbourhoods in England and Wales between the 2011 and 2021 Censuses.

Built as MSc Visual Analytics coursework at the University of Bristol.

## Interactive workbook

**[Explore the dashboard on Tableau Public →](https://public.tableau.com/views/ru22097_Tableau_Workbook/Dashboard1-LocalAuthoritychangeoverview)**

The four-dashboard Tableau workbook lets you move from national Local Authority trends down to individual neighbourhood (MSOA) patterns, neighbourhood clustering via PCA / UMAP / t-SNE, and Bayesian expected-vs-observed care residuals.

## Report

A summary PDF covering the abstract, introduction and conclusion is included in this repository. The full report and methodology are also in this repository. The interactive Tableau workbook is hosted on [Tableau Public](https://public.tableau.com/views/ru22097_Tableau_Workbook/Dashboard1-LocalAuthoritychangeoverview).

## Folder structure

```text
Visual_analytics/
│
├── Report Summary.pdf   
│
├── Full Report.pdf  
│
├── Data/
│   ├── raw/
│   │   ├── 2011/          ← 2011 Local Authority Census tables
│   │   ├── 2021/          ← 2021 Local Authority and MSOA Census tables
│   │   └── lookups/       ← geography lookups (2011→2021 LA, MSOA→LA)
│   │
│   ├── processed/         ← intermediate analytical outputs
│   │
│   └── tableau/           ← final CSV files used in the Tableau workbook
│       ├── tableau_la_overview.csv
│       ├── tableau_msoa_2021.csv
│       ├── tableau_cluster_summary.csv
│       ├── tableau_model_comparison_summary.csv
│       └── tableau_pca_loadings.csv
│
└── scripts/
    ├── prepare_lookups.py
    ├── clean_2011_la.py
    ├── clean_2021_la.py
    ├── clean_2021_msoa.py
    ├── final_clean_dataset.py
    ├── build_projections.py
    ├── chosen_clusters_projection.py
    ├── cluster_stability_check.py
    ├── expected_care_model.py
    ├── projection_quality_metrics.py
    └── join_files_tableau.py
```

## Data guide

### `Data/raw/`

Original Census CSV extracts downloaded from ONS/Nomis covering four themes: demographic structure, ethnic composition, unpaid care and economic activity.

### `Data/processed/`

Intermediate outputs from the Python pipeline: cleaned and harmonised tables, projection coordinates, cluster labels, stability checks and Bayesian residual fields.

### `Data/tableau/`

The five CSV files that feed the Tableau workbook directly:

- **tableau_la_overview.csv** — harmonised Local Authority rates, 2011–2021 changes and MSOA bridge summaries.
- **tableau_msoa_2021.csv** — 2021 MSOA variables, projection coordinates, cluster labels and residual fields.
- **tableau_cluster_summary.csv** — mean profiles for the five socio-care neighbourhood clusters.
- **tableau_model_comparison_summary.csv** — Bayesian Ridge and Gradient Boosting performance metrics.
- **tableau_pca_loadings.csv** — PCA variable loadings for projection interpretation.

## Scripts

The Python scripts follow this pipeline:

```text
lookup preparation
→ 2011 and 2021 data cleaning
→ Local Authority boundary harmonisation
→ dimensionality reduction (PCA, UMAP, t-SNE)
→ KMeans clustering and stability checks
→ Bayesian Ridge residual modelling
→ projection quality metrics
→ Tableau CSV export
```

## Boundary files

Boundary shapefiles are not included in this repository due to file size. If you want to reproduce the maps, download them from the ONS Open Geography Portal:

- **Local Authority boundaries (December 2021):** [LAD_Dec_2021_GB_BFC](https://geoportal.statistics.gov.uk/datasets/ons::local-authority-districts-december-2021-boundaries-gb-bfc/about)
- **MSOA boundaries (December 2021):** [MSOA_Dec_2021_EW_BGC_V3](https://geoportal.statistics.gov.uk/datasets/ons::middle-layer-super-output-areas-december-2021-boundaries-ew-bgc-v3/about)

Place the downloaded shapefiles in `boundaries/LA_2021/` and `boundaries/MSOA_2021/` respectively.

## Key findings

- Every one of the twelve Local Authorities with the largest rises in 20+ hours unpaid care also saw economic inactivity increase over 2011–2021.
- Newcastle upon Tyne sat close to the national mean but had the widest neighbourhood spread of any Local Authority (3.63 percentage points).
- The highest intensive-care rate (6.5%) appeared in a health-limited cluster driven by long-term sickness, not in the oldest retirement-heavy areas.
- The most ethnically diverse neighbourhoods consistently recorded the lowest care rates, challenging assumptions about informal family-based care.
- In East Lindsey, three neighbourhoods exceeded model expectations by 2.4–3.1 pp even after accounting for their already old and inactive populations.

## Licence

Census data is published by the Office for National Statistics under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
