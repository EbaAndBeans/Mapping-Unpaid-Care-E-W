from pathlib import Path
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score


base_path = Path("/Users/eduvieojoboh/Downloads/Visual_analytics")
processed_path = base_path / "Data" / "processed"

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

chosen_k = 5
seeds = [7, 21, 42, 100, 123]

labels_by_seed = {}
rows = []

for seed in seeds:
    model = KMeans(n_clusters=chosen_k, random_state=seed, n_init=20)
    labels = model.fit_predict(x_scaled)

    labels_by_seed[seed] = labels

    counts = pd.Series(labels).value_counts()

    rows.append({
        "random_state": seed,
        "silhouette_score": round(silhouette_score(x_scaled, labels), 4),
        "smallest_cluster": int(counts.min()),
        "largest_cluster": int(counts.max()),
        "cluster_sizes": "; ".join(str(x) for x in sorted(counts.tolist()))
    })

stability = pd.DataFrame(rows)

# compare all seeds to the main seed, 7
base_labels = labels_by_seed[7]
ari_rows = []

for seed in seeds:
    ari_rows.append({
        "comparison": f"7 vs {seed}",
        "adjusted_rand_index": round(adjusted_rand_score(base_labels, labels_by_seed[seed]), 4)
    })

ari = pd.DataFrame(ari_rows)

stability.to_csv(processed_path / "cluster_stability_summary.csv", index=False)
ari.to_csv(processed_path / "cluster_stability_ari.csv", index=False)

print("cluster stability summary")
print(stability)

print("\nadjusted rand index compared with random_state 7")
print(ari)

print("\nfiles saved")
print("cluster_stability_summary.csv")
print("cluster_stability_ari.csv")