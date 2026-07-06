from pathlib import Path
import pandas as pd

base_path = Path("/Users/eduvieojoboh/Downloads/Visual_analytics")
raw_path = base_path / "Data" / "raw" / "2011"
processed_path = base_path / "Data" / "processed"

processed_path.mkdir(parents=True, exist_ok=True)


def pct(part, total):
    return (part / total * 100).round(3)


# age
age = pd.read_csv(raw_path / "2011_age_LA.csv")

la = age[["geography code", "geography"]].copy()
la = la.rename(columns={
    "geography code": "la_code_2011",
    "geography": "la_name_2011"
})

la["population_2011"] = age["Age: All categories: Age; measures: Value"]

age_0_15_cols = ["Age: Age under 1; measures: Value"] + [
    f"Age: Age {i}; measures: Value" for i in range(1, 16)
]

age_16_64_cols = [
    f"Age: Age {i}; measures: Value" for i in range(16, 65)
]

age_65_plus_cols = [
    f"Age: Age {i}; measures: Value" for i in range(65, 100)
] + ["Age: Age 100 and over; measures: Value"]

la["age_0_15_2011"] = age[age_0_15_cols].sum(axis=1)
la["age_16_64_2011"] = age[age_16_64_cols].sum(axis=1)
la["age_65_plus_2011"] = age[age_65_plus_cols].sum(axis=1)

la["pct_0_15_2011"] = pct(la["age_0_15_2011"], la["population_2011"])
la["pct_16_64_2011"] = pct(la["age_16_64_2011"], la["population_2011"])
la["pct_65_plus_2011"] = pct(la["age_65_plus_2011"], la["population_2011"])

la["old_age_dependency_ratio_2011"] = (
    la["age_65_plus_2011"] / la["age_16_64_2011"]
).round(4)

la["youth_dependency_ratio_2011"] = (
    la["age_0_15_2011"] / la["age_16_64_2011"]
).round(4)




# sex
sex = pd.read_csv(raw_path / "2011_sex_LA.csv")

sex_clean = sex[["geography code"]].copy()
sex_clean = sex_clean.rename(columns={"geography code": "la_code_2011"})

sex_clean["population_sex_2011"] = sex["Sex: All persons; measures: Value"]
sex_clean["female_2011"] = sex["Sex: Females; measures: Value"]
sex_clean["male_2011"] = sex["Sex: Males; measures: Value"]
sex_clean["pct_female_2011"] = pct(
    sex_clean["female_2011"],
    sex_clean["population_sex_2011"]
)

la = la.merge(sex_clean, on="la_code_2011", how="left")


# unpaid care
care = pd.read_csv(raw_path / "2011_unpaid_care_LA.csv")

care_clean = care[["geography code"]].copy()
care_clean = care_clean.rename(columns={"geography code": "la_code_2011"})

care_clean["care_population_2011"] = care[
    "Care Provision: All categories: Provision of unpaid care; measures: Value"
]
care_clean["no_unpaid_care_2011"] = care[
    "Care Provision: Provides no unpaid care; measures: Value"
]
care_clean["care_1_19_2011"] = care[
    "Care Provision: Provides 1 to 19 hours unpaid care a week; measures: Value"
]
care_clean["care_20_49_2011"] = care[
    "Care Provision: Provides 20 to 49 hours unpaid care a week; measures: Value"
]
care_clean["care_50_plus_2011"] = care[
    "Care Provision: Provides 50 or more hours unpaid care a week; measures: Value"
]

care_clean["unpaid_care_any_2011"] = (
    care_clean["care_1_19_2011"]
    + care_clean["care_20_49_2011"]
    + care_clean["care_50_plus_2011"]
)

care_clean["care_20_plus_2011"] = (
    care_clean["care_20_49_2011"]
    + care_clean["care_50_plus_2011"]
)

care_clean["pct_unpaid_care_any_2011"] = pct(
    care_clean["unpaid_care_any_2011"],
    care_clean["care_population_2011"]
)

care_clean["pct_care_20_plus_2011"] = pct(
    care_clean["care_20_plus_2011"],
    care_clean["care_population_2011"]
)

care_clean["pct_care_50_plus_2011"] = pct(
    care_clean["care_50_plus_2011"],
    care_clean["care_population_2011"]
)

la = la.merge(care_clean, on="la_code_2011", how="left")


# economic activity
activity = pd.read_csv(raw_path / "2011_economic_activity_LA.csv")

activity_clean = activity[["geography code"]].copy()
activity_clean = activity_clean.rename(columns={"geography code": "la_code_2011"})

activity_clean["econ_population_16_plus_2011"] = activity[
    "Sex: All persons; Age: All categories: Age 16 and over; Economic Activity: All categories: Economic activity; measures: Value"
]

activity_clean["economically_active_2011"] = activity[
    "Sex: All persons; Age: All categories: Age 16 and over; Economic Activity: Economically active: Total; measures: Value"
]

activity_clean["in_employment_2011"] = activity[
    "Sex: All persons; Age: All categories: Age 16 and over; Economic Activity: Economically active: In employment: Total; measures: Value"
]

activity_clean["unemployed_2011"] = activity[
    "Sex: All persons; Age: All categories: Age 16 and over; Economic Activity: Economically active: Unemployed (including full-time students); measures: Value"
]

activity_clean["economically_inactive_2011"] = activity[
    "Sex: All persons; Age: All categories: Age 16 and over; Economic Activity: Economically inactive; measures: Value"
]

activity_clean["pct_economically_active_2011"] = pct(
    activity_clean["economically_active_2011"],
    activity_clean["econ_population_16_plus_2011"]
)

activity_clean["pct_in_employment_2011"] = pct(
    activity_clean["in_employment_2011"],
    activity_clean["econ_population_16_plus_2011"]
)

activity_clean["pct_unemployed_2011"] = pct(
    activity_clean["unemployed_2011"],
    activity_clean["econ_population_16_plus_2011"]
)

activity_clean["pct_economically_inactive_2011"] = pct(
    activity_clean["economically_inactive_2011"],
    activity_clean["econ_population_16_plus_2011"]
)

la = la.merge(activity_clean, on="la_code_2011", how="left")


# ethnicity
ethnicity = pd.read_csv(raw_path / "2011_ethnicity_LA.csv")

ethnicity_clean = ethnicity[["geography code"]].copy()
ethnicity_clean = ethnicity_clean.rename(columns={"geography code": "la_code_2011"})

ethnicity_clean["ethnicity_population_2011"] = ethnicity[
    "Ethnic Group: All categories: Ethnic group; measures: Value"
]
ethnicity_clean["white_2011"] = ethnicity["Ethnic Group: White; measures: Value"]
ethnicity_clean["mixed_2011"] = ethnicity["Ethnic Group: Mixed; measures: Value"]
ethnicity_clean["asian_2011"] = ethnicity["Ethnic Group: Asian; measures: Value"]
ethnicity_clean["black_2011"] = ethnicity["Ethnic Group: Black; measures: Value"]
ethnicity_clean["other_ethnic_group_2011"] = ethnicity[
    "Ethnic Group: Other; measures: Value"
]

for group in ["white", "mixed", "asian", "black", "other_ethnic_group"]:
    ethnicity_clean[f"pct_{group}_2011"] = pct(
        ethnicity_clean[f"{group}_2011"],
        ethnicity_clean["ethnicity_population_2011"]
    )

shares = [
    ethnicity_clean["white_2011"] / ethnicity_clean["ethnicity_population_2011"],
    ethnicity_clean["mixed_2011"] / ethnicity_clean["ethnicity_population_2011"],
    ethnicity_clean["asian_2011"] / ethnicity_clean["ethnicity_population_2011"],
    ethnicity_clean["black_2011"] / ethnicity_clean["ethnicity_population_2011"],
    ethnicity_clean["other_ethnic_group_2011"] / ethnicity_clean["ethnicity_population_2011"]
]

ethnicity_clean["ethnic_diversity_index_2011"] = (
    1 - sum(s ** 2 for s in shares)
).round(4)

la = la.merge(ethnicity_clean, on="la_code_2011", how="left")


# save
la["year"] = 2011

save_file = processed_path / "la_2011_profile_raw.csv"
la.to_csv(save_file, index=False)

print("2011 file saved")
print("saved here:", save_file)
print("rows:", len(la))
print("columns:", len(la.columns))
print("missing values:", la.isna().sum().sum())
print("duplicate LA codes:", la["la_code_2011"].duplicated().sum())