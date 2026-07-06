from pathlib import Path
import pandas as pd

base_path = Path("/Users/eduvieojoboh/Downloads/Visual_analytics")
lookup_path = base_path / "Data" / "raw" / "lookups"
processed_path = base_path / "Data" / "processed"

processed_path.mkdir(parents=True, exist_ok=True)

# read lookup files
msoa_lookup = pd.read_csv(lookup_path / "MSOA_to_LA.csv", dtype=str, low_memory=False)
la_lookup = pd.read_csv(lookup_path / "2011_to_2021_LA.csv", dtype=str)

# msoa to local authority lookup
msoa_to_la = msoa_lookup[[
    "MSOA21CD",
    "MSOA21NM",
    "LAD22CD",
    "LAD22NM"]].drop_duplicates()

msoa_to_la = msoa_to_la.rename(columns={
    "MSOA21CD": "msoa_code",
    "MSOA21NM": "msoa_name",
    "LAD22CD": "la_code_2021",
    "LAD22NM": "la_name_2021"})

# 2011 to 2021 local authority lookup
la_2011_to_2021 = la_lookup[[
    "LAD11CD",
    "LAD11NM",
    "LAD21CD",
    "LAD21NM"]].drop_duplicates()

la_2011_to_2021 = la_2011_to_2021.rename(columns={
    "LAD11CD": "la_code_2011",
    "LAD11NM": "la_name_2011",
    "LAD21CD": "la_code_2021",
    "LAD21NM": "la_name_2021"})

# basic checks
print("msoa lookup rows:", len(msoa_to_la))
print("unique msoas:", msoa_to_la["msoa_code"].nunique())
print("unique las:", msoa_to_la["la_code_2021"].nunique())

print("la lookup rows:", len(la_2011_to_2021))
print("2011 las:", la_2011_to_2021["la_code_2011"].nunique())
print("2021 las:", la_2011_to_2021["la_code_2021"].nunique())

# check that each msoa maps to only one local authority
check = msoa_to_la.groupby("msoa_code")["la_code_2021"].nunique()
print("msoas linked to more than one la:", (check > 1).sum())

# save cleaned lookup files
msoa_to_la.to_csv(processed_path / "msoa_to_la_clean.csv", index=False)
la_2011_to_2021.to_csv(processed_path / "la_2011_to_2021_clean.csv", index=False)

print("done")