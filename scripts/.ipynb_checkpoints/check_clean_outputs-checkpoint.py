from pathlib import Path
import pandas as pd

base_path = Path("/Users/eduvieojoboh/Downloads/Visual_analytics")
processed_path = base_path / "Data" / "processed"

la_change = pd.read_csv(processed_path / "la_2011_2021_change.csv")
msoa = pd.read_csv(processed_path / "msoa_2021_profile.csv")
bridge = pd.read_csv(processed_path / "la_msoa_bridge_2021.csv")
projection_input = pd.read_csv(processed_path / "projection_input_msoa_2021.csv")

print("row checks")
print("la change:", len(la_change))
print("msoa:", len(msoa))
print("bridge:", len(bridge))
print("projection input:", len(projection_input))

print("\nmissing values")
print("la change:", la_change.isna().sum().sum())
print("msoa:", msoa.isna().sum().sum())
print("bridge:", bridge.isna().sum().sum())
print("projection input:", projection_input.isna().sum().sum())

print("\nkey percentage ranges")

cols_to_check = [
    "pct_unpaid_care_any_2021",
    "pct_care_20_plus_2021",
    "pct_care_50_plus_2021",
    "pct_economically_inactive_2021",
    "pct_65_plus_2021",
    "pct_female_2021",
    "ethnic_diversity_index_2021"
]

for col in cols_to_check:
    print(col, "min:", msoa[col].min(), "max:", msoa[col].max())

print("\nbiggest LA increases in unpaid care")
print(
    la_change[
        ["la_name_2021", "change_pct_unpaid_care_any", "change_pct_economically_inactive"]
    ]
    .sort_values("change_pct_unpaid_care_any", ascending=False)
    .head(10)
)

print("\nbiggest LA decreases in unpaid care")
print(
    la_change[
        ["la_name_2021", "change_pct_unpaid_care_any", "change_pct_economically_inactive"]
    ]
    .sort_values("change_pct_unpaid_care_any")
    .head(10)
)

print("\nLAs with largest hidden MSOA variation")
print(
    bridge[
        ["la_name_2021", "number_of_msoas", "iqr_pct_unpaid_care_any", "share_msoas_high_care"]
    ]
    .sort_values("iqr_pct_unpaid_care_any", ascending=False)
    .head(10)
)

