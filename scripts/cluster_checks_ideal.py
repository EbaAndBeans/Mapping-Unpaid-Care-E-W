from pathlib import Path
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


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

checks = pd.DataFrame(rows)
checks.to_csv(processed_path / "cluster_quality_checks.csv", index=False)

print(checks)
print("saved cluster_quality_checks.csv")