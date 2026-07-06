from pathlib import Path
import pandas as pd

base_path = Path("/Users/eduvieojoboh/Downloads/Visual_analytics")
raw_path = base_path / "Data" / "raw" / "2021"
processed_path = base_path / "Data" / "processed"

processed_path.mkdir(parents=True, exist_ok=True)


def pct(part, total):
    return (part / total * 100).round(3)


# age
age = pd.read_csv(raw_path / "2021_age_MSOA.csv")

msoa = age[["geography code", "geography"]].copy()
msoa = msoa.rename(columns={
    "geography code": "msoa_code",
    "geography": "msoa_name"
})

msoa["population_2021"] = age["Age: Total; measures: Value"]

msoa["age_0_15_2021"] = (
    age["Age: Aged 4 years and under; measures: Value"]
    + age["Age: Aged 5 to 9 years; measures: Value"]
    + age["Age: Aged 10 to 15 years; measures: Value"]
)

msoa["age_16_64_2021"] = (
    age["Age: Aged 16 to 19 years; measures: Value"]
    + age["Age: Aged 20 to 24 years; measures: Value"]
    + age["Age: Aged 25 to 34 years; measures: Value"]
    + age["Age: Aged 35 to 49 years; measures: Value"]
    + age["Age: Aged 50 to 64 years; measures: Value"]
)

msoa["age_65_plus_2021"] = (
    age["Age: Aged 65 to 74 years; measures: Value"]
    + age["Age: Aged 75 to 84 years; measures: Value"]
    + age["Age: Aged 85 years and over; measures: Value"]
)

msoa["pct_0_15_2021"] = pct(msoa["age_0_15_2021"], msoa["population_2021"])
msoa["pct_16_64_2021"] = pct(msoa["age_16_64_2021"], msoa["population_2021"])
msoa["pct_65_plus_2021"] = pct(msoa["age_65_plus_2021"], msoa["population_2021"])

msoa["old_age_dependency_ratio_2021"] = (
    msoa["age_65_plus_2021"] / msoa["age_16_64_2021"]
).round(4)

msoa["youth_dependency_ratio_2021"] = (
    msoa["age_0_15_2021"] / msoa["age_16_64_2021"]
).round(4)


# sex
sex = pd.read_csv(raw_path / "2021_sex_MSOA.csv")

sex_clean = sex[["geography code"]].copy()
sex_clean = sex_clean.rename(columns={"geography code": "msoa_code"})

sex_clean["population_sex_2021"] = sex["Sex: All persons; measures: Value"]
sex_clean["female_2021"] = sex["Sex: Female; measures: Value"]
sex_clean["male_2021"] = sex["Sex: Male; measures: Value"]

sex_clean["pct_female_2021"] = pct(
    sex_clean["female_2021"],
    sex_clean["population_sex_2021"]
)

sex_clean["pct_male_2021"] = pct(
    sex_clean["male_2021"],
    sex_clean["population_sex_2021"]
)

msoa = msoa.merge(sex_clean, on="msoa_code", how="left")


# unpaid care
care = pd.read_csv(raw_path / "2021_unpaid_care_MSOA.csv")

care_clean = care[["geography code"]].copy()
care_clean = care_clean.rename(columns={"geography code": "msoa_code"})

care_clean["care_population_2021"] = care[
    "Provision of unpaid care: Total: All usual residents aged 5 and over"
]

care_clean["no_unpaid_care_2021"] = care[
    "Provision of unpaid care: Provides no unpaid care"
]

care_clean["care_1_19_2021"] = care[
    "Provision of unpaid care: Provides 19 hours or less unpaid care a week"
]

care_clean["care_20_49_2021"] = care[
    "Provision of unpaid care: Provides 20 to 49 hours unpaid care a week"
]

care_clean["care_50_plus_2021"] = care[
    "Provision of unpaid care: Provides 50 or more hours unpaid care a week"
]

care_clean["unpaid_care_any_2021"] = (
    care_clean["care_1_19_2021"]
    + care_clean["care_20_49_2021"]
    + care_clean["care_50_plus_2021"]
)

care_clean["care_20_plus_2021"] = (
    care_clean["care_20_49_2021"]
    + care_clean["care_50_plus_2021"]
)

care_clean["pct_unpaid_care_any_2021"] = pct(
    care_clean["unpaid_care_any_2021"],
    care_clean["care_population_2021"]
)

care_clean["pct_care_20_plus_2021"] = pct(
    care_clean["care_20_plus_2021"],
    care_clean["care_population_2021"]
)

care_clean["pct_care_50_plus_2021"] = pct(
    care_clean["care_50_plus_2021"],
    care_clean["care_population_2021"]
)

msoa = msoa.merge(care_clean, on="msoa_code", how="left")


# economic activity
activity = pd.read_csv(raw_path / "2021_economic_activity_MSOA.csv")

activity_clean = activity[["geography code"]].copy()
activity_clean = activity_clean.rename(columns={"geography code": "msoa_code"})

activity_clean["econ_population_16_plus_2021"] = activity[
    "Economic activity status: Total: All usual residents aged 16 years and over"
]

activity_clean["economically_active_2021"] = (
    activity["Economic activity status: Economically active (excluding full-time students)"]
    + activity["Economic activity status: Economically active and a full-time student"]
)

activity_clean["in_employment_2021"] = (
    activity["Economic activity status: Economically active (excluding full-time students):In employment"]
    + activity["Economic activity status: Economically active and a full-time student:In employment"]
)

activity_clean["unemployed_2021"] = (
    activity["Economic activity status: Economically active (excluding full-time students): Unemployed"]
    + activity["Economic activity status: Economically active and a full-time student: Unemployed"]
)

activity_clean["economically_inactive_2021"] = activity[
    "Economic activity status: Economically inactive"
]

activity_clean["retired_2021"] = activity[
    "Economic activity status: Economically inactive: Retired"
]

activity_clean["student_2021"] = activity[
    "Economic activity status: Economically inactive: Student"
]

activity_clean["looking_after_home_family_2021"] = activity[
    "Economic activity status: Economically inactive: Looking after home or family"
]

activity_clean["long_term_sick_disabled_2021"] = activity[
    "Economic activity status: Economically inactive: Long-term sick or disabled"
]

activity_clean["pct_economically_active_2021"] = pct(
    activity_clean["economically_active_2021"],
    activity_clean["econ_population_16_plus_2021"]
)

activity_clean["pct_in_employment_2021"] = pct(
    activity_clean["in_employment_2021"],
    activity_clean["econ_population_16_plus_2021"]
)

activity_clean["pct_unemployed_2021"] = pct(
    activity_clean["unemployed_2021"],
    activity_clean["econ_population_16_plus_2021"]
)

activity_clean["pct_economically_inactive_2021"] = pct(
    activity_clean["economically_inactive_2021"],
    activity_clean["econ_population_16_plus_2021"]
)

activity_clean["pct_retired_2021"] = pct(
    activity_clean["retired_2021"],
    activity_clean["econ_population_16_plus_2021"]
)

activity_clean["pct_student_2021"] = pct(
    activity_clean["student_2021"],
    activity_clean["econ_population_16_plus_2021"]
)

activity_clean["pct_looking_after_home_family_2021"] = pct(
    activity_clean["looking_after_home_family_2021"],
    activity_clean["econ_population_16_plus_2021"]
)

activity_clean["pct_long_term_sick_disabled_2021"] = pct(
    activity_clean["long_term_sick_disabled_2021"],
    activity_clean["econ_population_16_plus_2021"]
)

msoa = msoa.merge(activity_clean, on="msoa_code", how="left")


# ethnicity
ethnicity = pd.read_csv(raw_path / "2021_ethnicity_MSOA.csv")

ethnicity_clean = ethnicity[["geography code"]].copy()
ethnicity_clean = ethnicity_clean.rename(columns={"geography code": "msoa_code"})

ethnicity_clean["ethnicity_population_2021"] = ethnicity[
    "Ethnic group: Total: All usual residents"
]

ethnicity_clean["white_2021"] = ethnicity[
    "Ethnic group: White"
]

ethnicity_clean["mixed_2021"] = ethnicity[
    "Ethnic group: Mixed or Multiple ethnic groups"
]

ethnicity_clean["asian_2021"] = ethnicity[
    "Ethnic group: Asian, Asian British or Asian Welsh"
]

ethnicity_clean["black_2021"] = ethnicity[
    "Ethnic group: Black, Black British, Black Welsh, Caribbean or African"
]

ethnicity_clean["other_ethnic_group_2021"] = ethnicity[
    "Ethnic group: Other ethnic group"
]

for group in ["white", "mixed", "asian", "black", "other_ethnic_group"]:
    ethnicity_clean[f"pct_{group}_2021"] = pct(
        ethnicity_clean[f"{group}_2021"],
        ethnicity_clean["ethnicity_population_2021"]
    )

shares = [
    ethnicity_clean["white_2021"] / ethnicity_clean["ethnicity_population_2021"],
    ethnicity_clean["mixed_2021"] / ethnicity_clean["ethnicity_population_2021"],
    ethnicity_clean["asian_2021"] / ethnicity_clean["ethnicity_population_2021"],
    ethnicity_clean["black_2021"] / ethnicity_clean["ethnicity_population_2021"],
    ethnicity_clean["other_ethnic_group_2021"] / ethnicity_clean["ethnicity_population_2021"]
]

ethnicity_clean["ethnic_diversity_index_2021"] = (
    1 - sum(s ** 2 for s in shares)
).round(4)

msoa = msoa.merge(ethnicity_clean, on="msoa_code", how="left")


# add local authority lookup
lookup = pd.read_csv(processed_path / "msoa_to_la_clean.csv", dtype=str)

lookup = lookup[[
    "msoa_code",
    "la_code_2021",
    "la_name_2021"
]].drop_duplicates()

msoa = msoa.merge(lookup, on="msoa_code", how="left")


# save
msoa["year"] = 2021

# move main identifying columns to the front
front_cols = [
    "msoa_code",
    "msoa_name",
    "la_code_2021",
    "la_name_2021",
    "year"
]

other_cols = [col for col in msoa.columns if col not in front_cols]
msoa = msoa[front_cols + other_cols]

save_file = processed_path / "msoa_2021_profile.csv"
msoa.to_csv(save_file, index=False)

print("2021 MSOA file saved")
print("saved here:", save_file)
print("rows:", len(msoa))
print("columns:", len(msoa.columns))
print("missing values:", msoa.isna().sum().sum())
print("duplicate MSOA codes:", msoa["msoa_code"].duplicated().sum())
print("missing LA codes:", msoa["la_code_2021"].isna().sum())
print("unique MSOAs:", msoa["msoa_code"].nunique())
print("unique LAs:", msoa["la_code_2021"].nunique())