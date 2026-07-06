from pathlib import Path
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


base_path = Path("/Users/eduvieojoboh/Downloads/Visual_analytics")
processed_path = base_path / "Data" / "processed"

random_state = 7
chosen_k = 5

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

model = KMeans(
    n_clusters=chosen_k,
    random_state=random_state,
    n_init=20
)

data["cluster"] = model.fit_predict(x_scaled) + 1

cluster_names = {
    1: "diverse younger urban care-context areas",
    2: "older retirement-heavy areas",
    3: "high-care health-limited areas",
    4: "mixed working-age areas",
    5: "student-heavy low-care areas"
}

data["cluster_name"] = data["cluster"].map(cluster_names)

# distance from assigned cluster centre
distances = model.transform(x_scaled)
data["cluster_distance"] = distances.min(axis=1).round(4)

# flags for Tableau
care_cutoff = data["pct_unpaid_care_any_2021"].quantile(0.75)
distance_cutoff = data["cluster_distance"].quantile(0.95)

data["high_care_msoa"] = data["pct_unpaid_care_any_2021"] >= care_cutoff
data["cluster_outlier"] = data["cluster_distance"] >= distance_cutoff

# cluster summary
cluster_summary = data.groupby("cluster").agg(
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
    mean_cluster_distance=("cluster_distance", "mean")
).reset_index()

cluster_summary['cluster_name'] = cluster_summary['cluster'].map(cluster_names)

for col in cluster_summary.columns:
    if col not in ["cluster", "number_of_msoas", "cluster_name"]:
        cluster_summary[col] = cluster_summary[col].round(3)



# save
data.to_csv(
    processed_path / "projection_outputs_msoa_2021_clustered.csv",
    index=False
)

cluster_summary.to_csv(
    processed_path / "cluster_summary_msoa_2021.csv",
    index=False
)

print("chosen k:", chosen_k)
print("high care cutoff:", round(care_cutoff, 3))
print("cluster outlier cutoff:", round(distance_cutoff, 3))
print("\ncluster sizes")
print(data["cluster"].value_counts().sort_index())

print("\ncluster summary")
print(cluster_summary)

print("\nfiles saved")
print("projection_outputs_msoa_2021_clustered.csv")
print("cluster_summary_msoa_2021.csv")
