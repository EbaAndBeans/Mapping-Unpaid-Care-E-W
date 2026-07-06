from pathlib import Path
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


base_path = Path("/Users/eduvieojoboh/Downloads/Visual_analytics")
processed_path = base_path / "Data" / "processed"

random_state = 7
chosen_k = 6

data = pd.read_csv(processed_path / "projection_outputs_msoa_2021_clustered.csv")

# context variables only: no unpaid care variables included here
context_vars = [
    "pct_65_plus_2021",
    "old_age_dependency_ratio_2021",
    "youth_dependency_ratio_2021",
    "pct_female_2021",
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

x = data[context_vars].copy()

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# check a range of possible cluster counts
rows = []

for k in range(2, 13):
    model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    labels = model.fit_predict(x_scaled)

    counts = pd.Series(labels).value_counts()

    rows.append({
        "k": k,
        "silhouette_score": round(silhouette_score(x_scaled, labels), 4),
        "calinski_harabasz_score": round(calinski_harabasz_score(x_scaled, labels), 2),
        "davies_bouldin_score": round(davies_bouldin_score(x_scaled, labels), 4),
        "inertia": round(model.inertia_, 2),
        "smallest_cluster": int(counts.min()),
        "largest_cluster": int(counts.max())
    })

quality = pd.DataFrame(rows)
quality.to_csv(processed_path / "context_cluster_quality_checks.csv", index=False)

print("context-only cluster quality")
print(quality)


# fit chosen k for a comparable context-only cluster file
model = KMeans(n_clusters=chosen_k, random_state=random_state, n_init=20)
data["context_cluster"] = model.fit_predict(x_scaled) + 1
context_cluster_names = {
    1: "older retirement-heavy areas",
    2: "diverse younger urban areas",
    3: "mixed older working-age areas",
    4: "student-heavy low-care areas",
    5: "diverse family-care pressure areas",
    6: "health-limited high-care areas"
}

data["context_cluster_name"] = data["context_cluster"].map(context_cluster_names)

distances = model.transform(x_scaled)
data["context_cluster_distance"] = distances.min(axis=1).round(4)

context_outlier_cutoff = data["context_cluster_distance"].quantile(0.95)
data["context_cluster_outlier"] = data["context_cluster_distance"] >= context_outlier_cutoff

context_summary = data.groupby("context_cluster").agg(
    number_of_msoas=("msoa_code", "count"),
    mean_unpaid_care=("pct_unpaid_care_any_2021", "mean"),
    mean_care_20_plus=("pct_care_20_plus_2021", "mean"),
    mean_care_50_plus=("pct_care_50_plus_2021", "mean"),
    mean_65_plus=("pct_65_plus_2021", "mean"),
    mean_old_age_dependency=("old_age_dependency_ratio_2021", "mean"),
    mean_economically_inactive=("pct_economically_inactive_2021", "mean"),
    mean_retired=("pct_retired_2021", "mean"),
    mean_student=("pct_student_2021", "mean"),
    mean_looking_after_home_family=("pct_looking_after_home_family_2021", "mean"),
    mean_long_term_sick_disabled=("pct_long_term_sick_disabled_2021", "mean"),
    mean_white=("pct_white_2021", "mean"),
    mean_ethnic_diversity=("ethnic_diversity_index_2021", "mean"),
    mean_context_cluster_distance=("context_cluster_distance", "mean")
).reset_index()

context_summary["context_cluster_name"] = (
    context_summary["context_cluster"].map(context_cluster_names)
)

numeric_cols = context_summary.select_dtypes(include="number").columns

for col in numeric_cols:
    if col not in ["context_cluster", "number_of_msoas"]:
        context_summary[col] = context_summary[col].round(3)

# compare original socio-care clusters with context-only clusters
cluster_cross_tab = pd.crosstab(
    data["cluster_name"],
    data["context_cluster"],
    normalize="index"
).round(3)

cluster_cross_tab.to_csv(processed_path / "original_vs_context_cluster_crosstab.csv")

data.to_csv(
    processed_path / "projection_outputs_msoa_2021_context_clusters.csv",
    index=False
)

context_summary.to_csv(
    processed_path / "context_cluster_summary_msoa_2021.csv",
    index=False
)

print("\nchosen context k:", chosen_k)
print("context outlier cutoff:", round(context_outlier_cutoff, 3))

print("\ncontext cluster sizes")
print(data["context_cluster"].value_counts().sort_index())

print("\ncontext cluster summary")
print(context_summary)

print("\nfiles saved")
print("context_cluster_quality_checks.csv")
print("projection_outputs_msoa_2021_context_clusters.csv")
print("context_cluster_summary_msoa_2021.csv")
print("original_vs_context_cluster_crosstab.csv")
