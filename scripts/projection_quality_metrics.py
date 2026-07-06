from pathlib import Path
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors


base_path = Path("/Users/eduvieojoboh/Downloads/Visual_analytics")
processed_path = base_path / "Data" / "processed"

random_state = 7

data = pd.read_csv(processed_path / "projection_outputs_msoa_2021.csv")

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

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


def knn_overlap(original_data, projected_data, k):
    original_nn = NearestNeighbors(n_neighbors=k + 1).fit(original_data)
    projected_nn = NearestNeighbors(n_neighbors=k + 1).fit(projected_data)

    original_neighbours = original_nn.kneighbors(return_distance=False)[:, 1:]
    projected_neighbours = projected_nn.kneighbors(return_distance=False)[:, 1:]

    overlaps = []

    for i in range(len(original_data)):
        original_set = set(original_neighbours[i])
        projected_set = set(projected_neighbours[i])

        overlap = len(original_set.intersection(projected_set)) / k
        overlaps.append(overlap)

    return sum(overlaps) / len(overlaps)


projection_methods = {
    "PCA": ["pca_1", "pca_2"],
    "UMAP": ["umap_1", "umap_2"],
    "t-SNE": ["tsne_1", "tsne_2"]
}

k_values = [10, 30, 50]

rows = []

for method, cols in projection_methods.items():
    coords = data[cols].copy()

    for k in k_values:
        rows.append({
            "method": method,
            "k": k,
            "trustworthiness": round(trustworthiness(x_scaled, coords, n_neighbors=k), 4),
            "knn_overlap": round(knn_overlap(x_scaled, coords, k), 4)
        })

# add PCA explained variance
pca = PCA(n_components=2, random_state=random_state)
pca.fit(x_scaled)

pca_summary = pd.DataFrame([{
    "explained_variance_pc1": round(pca.explained_variance_ratio_[0], 4),
    "explained_variance_pc2": round(pca.explained_variance_ratio_[1], 4),
    "explained_variance_2d_total": round(pca.explained_variance_ratio_.sum(), 4)
}])

metrics = pd.DataFrame(rows)

metrics.to_csv(processed_path / "projection_quality_metrics.csv", index=False)
pca_summary.to_csv(processed_path / "pca_explained_variance_summary.csv", index=False)

print("projection quality metrics")
print(metrics)

print("\nPCA explained variance")
print(pca_summary)

print("\nfiles saved")
print("projection_quality_metrics.csv")
print("pca_explained_variance_summary.csv")