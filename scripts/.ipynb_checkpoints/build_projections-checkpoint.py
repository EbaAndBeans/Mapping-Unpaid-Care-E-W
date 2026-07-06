from pathlib import Path
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import umap


base_path = Path("/Users/eduvieojoboh/Downloads/Visual_analytics")
processed_path = base_path / "Data" / "processed"

data = pd.read_csv(processed_path / "projection_input_msoa_2021.csv")

id_cols = [
    "msoa_code",
    "msoa_name",
    "la_code_2021",
    "la_name_2021"
]

projection_vars = [
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
    "ethnic_diversity_index_2021"
]

x = data[projection_vars].copy()

print("projection input")
print("rows:", len(x))
print("variables:", len(projection_vars))
print("missing values:", x.isna().sum().sum())

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


# PCA
pca = PCA(n_components=2, random_state=7)
pca_result = pca.fit_transform(x_scaled)

pca_output = data[id_cols].copy()
pca_output["pca_1"] = pca_result[:, 0]
pca_output["pca_2"] = pca_result[:, 1]

pca_output.to_csv(processed_path / "pca_msoa_2021.csv", index=False)

pca_loadings = pd.DataFrame({
    "variable": projection_vars,
    "pca_1_loading": pca.components_[0],
    "pca_2_loading": pca.components_[1]
})

pca_loadings.to_csv(processed_path / "pca_loadings_msoa_2021.csv", index=False)

print("\nPCA done")
print("PCA explained variance:", pca.explained_variance_ratio_)


# UMAP
umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=30,
    min_dist=0.1,
    metric="euclidean",
    random_state=7
)

umap_result = umap_model.fit_transform(x_scaled)

umap_output = data[id_cols].copy()
umap_output["umap_1"] = umap_result[:, 0]
umap_output["umap_2"] = umap_result[:, 1]

umap_output.to_csv(processed_path / "umap_msoa_2021.csv", index=False)

print("UMAP done")


# t-SNE

tsne = TSNE(
    n_components=2,
    perplexity=40,
    learning_rate="auto",
    init="pca",
    max_iter=1000,
    random_state=7
)

tsne_result = tsne.fit_transform(x_scaled)


tsne_output = data[id_cols].copy()
tsne_output["tsne_1"] = tsne_result[:, 0]
tsne_output["tsne_2"] = tsne_result[:, 1]

tsne_output.to_csv(processed_path / "tsne_msoa_2021.csv", index=False)

print("t-SNE done")


# combined file for Tableau
combined = data.copy()

combined["pca_1"] = pca_result[:, 0]
combined["pca_2"] = pca_result[:, 1]


combined["umap_1"] = umap_result[:, 0]
combined["umap_2"] = umap_result[:, 1]

combined["tsne_1"] = tsne_result[:, 0]
combined["tsne_2"] = tsne_result[:, 1]

combined.to_csv(
    processed_path / "projection_outputs_msoa_2021.csv",
    index=False
)

print("\nprojection files saved")
print("pca rows:", len(pca_output))

print("umap rows:", len(umap_output))

print("tsne rows:", len(tsne_output))
print("combined rows:", len(combined))