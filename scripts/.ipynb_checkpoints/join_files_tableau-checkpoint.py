from pathlib import Path
import pandas as pd

base_path = Path("/Users/eduvieojoboh/Downloads/Visual_analytics")
processed_path = base_path / "Data" / "processed"
tableau_path = base_path / "Data" / "tableau"

tableau_path.mkdir(parents=True, exist_ok=True)

# read files
msoa_base = pd.read_csv(processed_path / "msoa_2021_profile.csv")
msoa_model = pd.read_csv(processed_path / "model_outputs_msoa_2021.csv")

la_change = pd.read_csv(processed_path / "la_2011_2021_change.csv")
la_bridge = pd.read_csv(processed_path / "la_msoa_bridge_2021.csv")
la_model = pd.read_csv(processed_path / "model_la_summary_2021.csv")

cluster_summary = pd.read_csv(processed_path / "cluster_summary_msoa_2021.csv")
context_cluster_summary = pd.read_csv(processed_path / "context_cluster_summary_msoa_2021.csv")
model_summary = pd.read_csv(processed_path / "model_comparison_summary.csv")
pca_loadings = pd.read_csv(processed_path / "pca_loadings_msoa_2021.csv")

# add projection, cluster and model columns to full msoa profile
extra_cols = [
    col for col in msoa_model.columns
    if col not in msoa_base.columns or col == "msoa_code"
]

msoa = msoa_base.merge(
    msoa_model[extra_cols],
    on="msoa_code",
    how="left"
)

# MSOA Tableau file
msoa_cols = [
    "msoa_code",
    "msoa_name",
    "la_code_2021",
    "la_name_2021",

    "population_2021",
    "pct_0_15_2021",
    "pct_16_64_2021",
    "pct_65_plus_2021",
    "old_age_dependency_ratio_2021",
    "youth_dependency_ratio_2021",
    "pct_female_2021",

    "pct_unpaid_care_any_2021",
    "pct_care_20_plus_2021",
    "pct_care_50_plus_2021",

    "pct_economically_active_2021",
    "pct_in_employment_2021",
    "pct_unemployed_2021",
    "pct_economically_inactive_2021",
    "pct_retired_2021",
    "pct_student_2021",
    "pct_looking_after_home_family_2021",
    "pct_long_term_sick_disabled_2021",

    "pct_white_2021",
    "pct_mixed_2021",
    "pct_asian_2021",
    "pct_black_2021",
    "pct_other_ethnic_group_2021",
    "ethnic_diversity_index_2021",

    "pca_1",
    "pca_2",
    "umap_1",
    "umap_2",
    "tsne_1",
    "tsne_2",

    "cluster",
    "cluster_name",
    "cluster_distance",
    "cluster_outlier",

    "context_cluster",
    "context_cluster_name",
    "context_cluster_distance",
    "context_cluster_outlier",

    "care_20_plus_bayes_expected",
    "care_20_plus_bayes_lower_95",
    "care_20_plus_bayes_upper_95",
    "care_20_plus_bayes_residual",
    "care_20_plus_bayes_residual_z",
    "care_20_plus_residual_group",
    "care_20_plus_gb_expected",
    "care_20_plus_gb_residual",

    "any_unpaid_care_bayes_expected",
    "any_unpaid_care_bayes_residual",
    "any_unpaid_care_residual_group",
    "any_unpaid_care_gb_expected",
    "any_unpaid_care_gb_residual"
]

# check columns before selecting
missing_cols = [col for col in msoa_cols if col not in msoa.columns]

if len(missing_cols) > 0:
    print("missing columns:")
    for col in missing_cols:
        print("-", col)
    raise ValueError("Some MSOA columns are missing.")

msoa_tableau = msoa[msoa_cols].copy()

msoa_tableau.to_csv(
    tableau_path / "tableau_msoa_2021.csv",
    index=False
)

# LA overview file
la = la_change.copy()
la_bridge_merge = la_bridge.drop(columns=["la_name_2021"])
la_model_merge = la_model.drop(columns=["la_name_2021"])

if "number_of_msoas" in la_model_merge.columns:
    la_model_merge = la_model_merge.drop(columns=["number_of_msoas"])

la = la.merge(
    la_bridge_merge,
    on="la_code_2021",
    how="left"
)

la = la.merge(
    la_model_merge,
    on="la_code_2021",
    how="left"
)

la.to_csv(
    tableau_path / "tableau_la_overview.csv",
    index=False
)

# summary files
cluster_summary.to_csv(
    tableau_path / "tableau_cluster_summary.csv",
    index=False
)

context_cluster_summary.to_csv(
    tableau_path / "tableau_context_cluster_summary.csv",
    index=False
)

model_summary.to_csv(
    tableau_path / "tableau_model_comparison_summary.csv",
    index=False
)

pca_loadings.to_csv(
    tableau_path / "tableau_pca_loadings.csv",
    index=False
)

# checks
print("tableau files saved")
print("MSOA rows:", len(msoa_tableau))
print("LA rows:", len(la))
print("cluster summary rows:", len(cluster_summary))
print("context cluster summary rows:", len(context_cluster_summary))
print("model summary rows:", len(model_summary))

print("\nmissing values")
print("MSOA:", msoa_tableau.isna().sum().sum())
print("LA:", la.isna().sum().sum())

print("\nfiles:")
print(tableau_path / "tableau_msoa_2021.csv")
print(tableau_path / "tableau_la_overview.csv")
print(tableau_path / "tableau_cluster_summary.csv")
print(tableau_path / "tableau_context_cluster_summary.csv")
print(tableau_path / "tableau_model_comparison_summary.csv")
print(tableau_path / "tableau_pca_loadings.csv")
