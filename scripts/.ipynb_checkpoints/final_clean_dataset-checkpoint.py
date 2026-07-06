from pathlib import Path
import pandas as pd

base_path = Path("/Users/eduvieojoboh/Downloads/Visual_analytics")
processed_path = base_path / "Data" / "processed"


def pct(part, total):
    return (part / total * 100).round(3)


# read cleaned files
la_2011 = pd.read_csv(processed_path / "la_2011_profile_raw.csv")
la_2021 = pd.read_csv(processed_path / "la_2021_profile.csv")
msoa_2021 = pd.read_csv(processed_path / "msoa_2021_profile.csv")
la_lookup = pd.read_csv(processed_path / "la_2011_to_2021_clean.csv", dtype=str)


la_2011 = la_2011.merge(
    la_lookup[["la_code_2011", "la_code_2021", "la_name_2021"]],
    on="la_code_2011",
    how="left"
)

# some 2011 files already use the newer LA codes
missing_before = la_2011["la_code_2021"].isna().sum()
print("2011 rows without 2021 LA match before fix:", missing_before)

valid_2021_codes = set(la_2021["la_code_2021"])

self_map = (
    la_2011["la_code_2021"].isna()
    & la_2011["la_code_2011"].isin(valid_2021_codes)
)

name_lookup = la_2021.set_index("la_code_2021")["la_name_2021"]

la_2011.loc[self_map, "la_code_2021"] = la_2011.loc[self_map, "la_code_2011"]
la_2011.loc[self_map, "la_name_2021"] = la_2011.loc[self_map, "la_code_2011"].map(name_lookup)

print("2011 rows self-mapped:", self_map.sum())
print("2011 rows without 2021 LA match after fix:", la_2011["la_code_2021"].isna().sum())

count_cols = [
    "population_2011",
    "age_0_15_2011",
    "age_16_64_2011",
    "age_65_plus_2011",
    "population_sex_2011",
    "female_2011",
    "male_2011",
    "care_population_2011",
    "no_unpaid_care_2011",
    "care_1_19_2011",
    "care_20_49_2011",
    "care_50_plus_2011",
    "unpaid_care_any_2011",
    "care_20_plus_2011",
    "econ_population_16_plus_2011",
    "economically_active_2011",
    "in_employment_2011",
    "unemployed_2011",
    "economically_inactive_2011",
    "ethnicity_population_2011",
    "white_2011",
    "mixed_2011",
    "asian_2011",
    "black_2011",
    "other_ethnic_group_2011"
]

la_2011_new = la_2011.groupby(
    ["la_code_2021", "la_name_2021"],
    as_index=False
)[count_cols].sum()


# recalculate percentages after aggregation
la_2011_new["pct_0_15_2011"] = pct(la_2011_new["age_0_15_2011"], la_2011_new["population_2011"])
la_2011_new["pct_16_64_2011"] = pct(la_2011_new["age_16_64_2011"], la_2011_new["population_2011"])
la_2011_new["pct_65_plus_2011"] = pct(la_2011_new["age_65_plus_2011"], la_2011_new["population_2011"])

la_2011_new["old_age_dependency_ratio_2011"] = (
    la_2011_new["age_65_plus_2011"] / la_2011_new["age_16_64_2011"]
).round(4)

la_2011_new["youth_dependency_ratio_2011"] = (
    la_2011_new["age_0_15_2011"] / la_2011_new["age_16_64_2011"]
).round(4)

la_2011_new["pct_female_2011"] = pct(
    la_2011_new["female_2011"],
    la_2011_new["population_sex_2011"]
)

la_2011_new["pct_unpaid_care_any_2011"] = pct(
    la_2011_new["unpaid_care_any_2011"],
    la_2011_new["care_population_2011"]
)

la_2011_new["pct_care_20_plus_2011"] = pct(
    la_2011_new["care_20_plus_2011"],
    la_2011_new["care_population_2011"]
)

la_2011_new["pct_care_50_plus_2011"] = pct(
    la_2011_new["care_50_plus_2011"],
    la_2011_new["care_population_2011"]
)

la_2011_new["pct_economically_active_2011"] = pct(
    la_2011_new["economically_active_2011"],
    la_2011_new["econ_population_16_plus_2011"]
)

la_2011_new["pct_in_employment_2011"] = pct(
    la_2011_new["in_employment_2011"],
    la_2011_new["econ_population_16_plus_2011"]
)

la_2011_new["pct_unemployed_2011"] = pct(
    la_2011_new["unemployed_2011"],
    la_2011_new["econ_population_16_plus_2011"]
)

la_2011_new["pct_economically_inactive_2011"] = pct(
    la_2011_new["economically_inactive_2011"],
    la_2011_new["econ_population_16_plus_2011"]
)

for group in ["white", "mixed", "asian", "black", "other_ethnic_group"]:
    la_2011_new[f"pct_{group}_2011"] = pct(
        la_2011_new[f"{group}_2011"],
        la_2011_new["ethnicity_population_2011"]
    )

ethnicity_shares_2011 = [
    la_2011_new["white_2011"] / la_2011_new["ethnicity_population_2011"],
    la_2011_new["mixed_2011"] / la_2011_new["ethnicity_population_2011"],
    la_2011_new["asian_2011"] / la_2011_new["ethnicity_population_2011"],
    la_2011_new["black_2011"] / la_2011_new["ethnicity_population_2011"],
    la_2011_new["other_ethnic_group_2011"] / la_2011_new["ethnicity_population_2011"]
]

la_2011_new["ethnic_diversity_index_2011"] = (
    1 - sum(s ** 2 for s in ethnicity_shares_2011)
).round(4)

la_2011_new["year"] = 2011

la_2011_new.to_csv(
    processed_path / "la_2011_profile_on_2021_boundaries.csv",
    index=False
)


# make LA change file
la_change = la_2021.merge(
    la_2011_new,
    on="la_code_2021",
    how="left",
    suffixes=("_2021_file", "_2011_file")
)

# keep the 2021 name from the 2021 census file
if "la_name_2021_2021_file" in la_change.columns:
    la_change = la_change.rename(columns={"la_name_2021_2021_file": "la_name_2021"})

if "la_name_2021_2011_file" in la_change.columns:
    la_change = la_change.drop(columns=["la_name_2021_2011_file"])

# harmonised care denominator for change over time
la_change["pct_unpaid_care_any_2021_comparable"] = pct(
    la_change["unpaid_care_any_2021"],
    la_change["population_2021"]
)

la_change["pct_care_20_plus_2021_comparable"] = pct(
    la_change["care_20_plus_2021"],
    la_change["population_2021"]
)

la_change["pct_care_50_plus_2021_comparable"] = pct(
    la_change["care_50_plus_2021"],
    la_change["population_2021"]
)

la_change["change_pct_unpaid_care_any"] = (
    la_change["pct_unpaid_care_any_2021_comparable"]
    - la_change["pct_unpaid_care_any_2011"]
).round(3)

la_change["change_pct_care_20_plus"] = (
    la_change["pct_care_20_plus_2021_comparable"]
    - la_change["pct_care_20_plus_2011"]
).round(3)

la_change["change_pct_care_50_plus"] = (
    la_change["pct_care_50_plus_2021_comparable"]
    - la_change["pct_care_50_plus_2011"]
).round(3)

la_change["change_pct_economically_active"] = (
    la_change["pct_economically_active_2021"]
    - la_change["pct_economically_active_2011"]
).round(3)

la_change["change_pct_in_employment"] = (
    la_change["pct_in_employment_2021"]
    - la_change["pct_in_employment_2011"]
).round(3)

la_change["change_pct_unemployed"] = (
    la_change["pct_unemployed_2021"]
    - la_change["pct_unemployed_2011"]
).round(3)

la_change["change_pct_economically_inactive"] = (
    la_change["pct_economically_inactive_2021"]
    - la_change["pct_economically_inactive_2011"]
).round(3)

la_change["change_pct_65_plus"] = (
    la_change["pct_65_plus_2021"]
    - la_change["pct_65_plus_2011"]
).round(3)

la_change["change_old_age_dependency_ratio"] = (
    la_change["old_age_dependency_ratio_2021"]
    - la_change["old_age_dependency_ratio_2011"]
).round(4)

la_change["change_ethnic_diversity_index"] = (
    la_change["ethnic_diversity_index_2021"]
    - la_change["ethnic_diversity_index_2011"]
).round(4)

la_change.to_csv(
    processed_path / "la_2011_2021_change.csv",
    index=False
)



# make LA to MSOA bridge file

bridge = msoa_2021.groupby(
    ["la_code_2021", "la_name_2021"],
    as_index=False
).agg(
    number_of_msoas=("msoa_code", "nunique"),
    mean_pct_unpaid_care_any=("pct_unpaid_care_any_2021", "mean"),
    median_pct_unpaid_care_any=("pct_unpaid_care_any_2021", "median"),
    min_pct_unpaid_care_any=("pct_unpaid_care_any_2021", "min"),
    max_pct_unpaid_care_any=("pct_unpaid_care_any_2021", "max"),
    mean_pct_care_20_plus=("pct_care_20_plus_2021", "mean"),
    mean_pct_care_50_plus=("pct_care_50_plus_2021", "mean"),
    mean_pct_65_plus=("pct_65_plus_2021", "mean"),
    mean_pct_economically_inactive=("pct_economically_inactive_2021", "mean"),
    mean_pct_looking_after_home_family=("pct_looking_after_home_family_2021", "mean"),
    mean_pct_long_term_sick_disabled=("pct_long_term_sick_disabled_2021", "mean")
)

q25 = msoa_2021.groupby(
    ["la_code_2021", "la_name_2021"]
)["pct_unpaid_care_any_2021"].quantile(0.25)

q75 = msoa_2021.groupby(
    ["la_code_2021", "la_name_2021"]
)["pct_unpaid_care_any_2021"].quantile(0.75)

iqr = (q75 - q25).reset_index(name="iqr_pct_unpaid_care_any")

bridge = bridge.merge(
    iqr,
    on=["la_code_2021", "la_name_2021"],
    how="left"
)

high_care_cutoff = msoa_2021["pct_unpaid_care_any_2021"].quantile(0.75)

high_care = msoa_2021[[
    "la_code_2021",
    "la_name_2021",
    "msoa_code",
    "pct_unpaid_care_any_2021"
]].copy()

high_care["high_care_msoa"] = (
    high_care["pct_unpaid_care_any_2021"] >= high_care_cutoff
)

high_care_summary = high_care.groupby(
    ["la_code_2021", "la_name_2021"],
    as_index=False
)["high_care_msoa"].mean()

high_care_summary["share_msoas_high_care"] = (
    high_care_summary["high_care_msoa"] * 100
).round(2)

bridge = bridge.merge(
    high_care_summary[[
        "la_code_2021",
        "la_name_2021",
        "share_msoas_high_care"
    ]],
    on=["la_code_2021", "la_name_2021"],
    how="left"
)

for col in bridge.columns:
    if col.startswith(("mean_", "median_", "min_", "max_", "iqr_")):
        bridge[col] = bridge[col].round(3)

bridge.to_csv(
    processed_path / "la_msoa_bridge_2021.csv",
    index=False
)



# make projection input
projection_cols = [
    "msoa_code",
    "msoa_name",
    "la_code_2021",
    "la_name_2021",
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

projection_input = msoa_2021[projection_cols].copy()

projection_input.to_csv(
    processed_path / "projection_input_msoa_2021.csv",
    index=False
)


# final checks
print("final files saved")
print("2011 on 2021 boundaries rows:", len(la_2011_new))
print("LA change rows:", len(la_change))
print("LA-MSOA bridge rows:", len(bridge))
print("projection input rows:", len(projection_input))

print("missing 2011 data in LA change:", la_change["population_2011"].isna().sum())
print("missing values in bridge:", bridge.isna().sum().sum())
print("missing values in projection input:", projection_input.isna().sum().sum())
print("done")