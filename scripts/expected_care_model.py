from pathlib import Path
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


base_path = Path("/Users/eduvieojoboh/Downloads/Visual_analytics")
processed_path = base_path / "Data" / "processed"

random_state = 7

data = pd.read_csv(processed_path / "projection_outputs_msoa_2021_context_clusters.csv")
settings = pd.read_csv(processed_path / "chosen_model_settings.csv")

model_vars = [
    "pct_65_plus_2021",
    "youth_dependency_ratio_2021",
    "pct_female_2021",
    "pct_unemployed_2021",
    "pct_retired_2021",
    "pct_student_2021",
    "pct_looking_after_home_family_2021",
    "pct_long_term_sick_disabled_2021",
    "pct_mixed_2021",
    "pct_asian_2021",
    "pct_black_2021",
    "pct_other_ethnic_group_2021",
    "ethnic_diversity_index_2021"
]

targets = {
    "care_20_plus": "pct_care_20_plus_2021",
    "any_unpaid_care": "pct_unpaid_care_any_2021"
}

x = data[model_vars].copy()

print("rows:", len(data))
print("model variables:", len(model_vars))
print("missing values in x:", x.isna().sum().sum())

model_summaries = []
bayes_coef_tables = []
gb_importance_tables = []


def get_setting(target_label, model_name):
    row = settings[
        (settings["target_label"] == target_label)
        & (settings["model"] == model_name)
    ]

    if len(row) != 1:
        raise ValueError(f"Could not find one setting for {target_label}, {model_name}")

    return row.iloc[0]


for target_label, target_col in targets.items():
    print("\n------------------------")
    print("target:", target_col)
    print("------------------------")

    y = data[target_col].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=random_state
    )

    # Bayesian Ridge
    bayes_setting = get_setting(target_label, "Bayesian Ridge")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    bayes = BayesianRidge(
        alpha_1=float(bayes_setting["alpha_1"]),
        alpha_2=float(bayes_setting["alpha_2"]),
        lambda_1=float(bayes_setting["lambda_1"]),
        lambda_2=float(bayes_setting["lambda_2"])
    )

    bayes.fit(x_train_scaled, y_train)
    bayes_pred = bayes.predict(x_test_scaled)

    bayes_mae = mean_absolute_error(y_test, bayes_pred)
    bayes_r2 = r2_score(y_test, bayes_pred)

    print("Bayesian Ridge MAE:", round(bayes_mae, 3))
    print("Bayesian Ridge R2:", round(bayes_r2, 3))

    model_summaries.append({
        "target_label": target_label,
        "target_column": target_col,
        "model": "Bayesian Ridge",
        "test_mae": round(bayes_mae, 4),
        "test_r2": round(bayes_r2, 4)
    })

    # fit final Bayesian model on all MSOAs
    final_scaler = StandardScaler()
    x_scaled = final_scaler.fit_transform(x)

    final_bayes = BayesianRidge(
        alpha_1=float(bayes_setting["alpha_1"]),
        alpha_2=float(bayes_setting["alpha_2"]),
        lambda_1=float(bayes_setting["lambda_1"]),
        lambda_2=float(bayes_setting["lambda_2"])
    )

    final_bayes.fit(x_scaled, y)

    bayes_expected, bayes_std = final_bayes.predict(x_scaled, return_std=True)

    data[f"{target_label}_bayes_expected"] = bayes_expected.round(3)
    data[f"{target_label}_bayes_prediction_std"] = bayes_std.round(3)

    data[f"{target_label}_bayes_lower_95"] = (
        data[f"{target_label}_bayes_expected"]
        - 1.96 * data[f"{target_label}_bayes_prediction_std"]
    ).round(3)

    data[f"{target_label}_bayes_upper_95"] = (
        data[f"{target_label}_bayes_expected"]
        + 1.96 * data[f"{target_label}_bayes_prediction_std"]
    ).round(3)

    data[f"{target_label}_bayes_residual"] = (
        data[target_col] - data[f"{target_label}_bayes_expected"]
    ).round(3)

    data[f"{target_label}_bayes_residual_z"] = (
        data[f"{target_label}_bayes_residual"]
        / data[f"{target_label}_bayes_prediction_std"]
    ).round(3)

    data[f"{target_label}_higher_than_expected"] = (
        data[target_col] > data[f"{target_label}_bayes_upper_95"]
    )

    data[f"{target_label}_lower_than_expected"] = (
        data[target_col] < data[f"{target_label}_bayes_lower_95"]
    )

    high_cutoff = data[f"{target_label}_bayes_residual"].quantile(0.95)
    low_cutoff = data[f"{target_label}_bayes_residual"].quantile(0.05)

    data[f"{target_label}_top_5pct_higher"] = (
        data[f"{target_label}_bayes_residual"] >= high_cutoff
    )

    data[f"{target_label}_top_5pct_lower"] = (
        data[f"{target_label}_bayes_residual"] <= low_cutoff
    )

    data[f"{target_label}_residual_group"] = "close to expected"

    data.loc[
        data[f"{target_label}_top_5pct_higher"],
        f"{target_label}_residual_group"
    ] = "much higher than expected"

    data.loc[
        data[f"{target_label}_top_5pct_lower"],
        f"{target_label}_residual_group"
    ] = "much lower than expected"

    coef_table = pd.DataFrame({
        "target_label": target_label,
        "target_column": target_col,
        "variable": model_vars,
        "standardised_coefficient": final_bayes.coef_
    })

    coef_table["abs_coefficient"] = coef_table["standardised_coefficient"].abs()
    coef_table["standardised_coefficient"] = coef_table["standardised_coefficient"].round(4)
    coef_table["abs_coefficient"] = coef_table["abs_coefficient"].round(4)

    bayes_coef_tables.append(coef_table)

    # Gradient Boosting
    gb_setting = get_setting(target_label, "Gradient Boosting")

    gb = GradientBoostingRegressor(
        n_estimators=int(gb_setting["n_estimators"]),
        learning_rate=float(gb_setting["learning_rate"]),
        max_depth=int(gb_setting["max_depth"]),
        min_samples_leaf=int(gb_setting["min_samples_leaf"]),
        random_state=random_state
    )

    gb.fit(x_train, y_train)
    gb_pred = gb.predict(x_test)

    gb_mae = mean_absolute_error(y_test, gb_pred)
    gb_r2 = r2_score(y_test, gb_pred)

    print("Gradient Boosting MAE:", round(gb_mae, 3))
    print("Gradient Boosting R2:", round(gb_r2, 3))

    model_summaries.append({
        "target_label": target_label,
        "target_column": target_col,
        "model": "Gradient Boosting",
        "test_mae": round(gb_mae, 4),
        "test_r2": round(gb_r2, 4)
    })

    final_gb = GradientBoostingRegressor(
        n_estimators=int(gb_setting["n_estimators"]),
        learning_rate=float(gb_setting["learning_rate"]),
        max_depth=int(gb_setting["max_depth"]),
        min_samples_leaf=int(gb_setting["min_samples_leaf"]),
        random_state=random_state
    )

    final_gb.fit(x, y)

    gb_expected = final_gb.predict(x)

    data[f"{target_label}_gb_expected"] = gb_expected.round(3)

    data[f"{target_label}_gb_residual"] = (
        data[target_col] - data[f"{target_label}_gb_expected"]
    ).round(3)

    gb_importance = pd.DataFrame({
        "target_label": target_label,
        "target_column": target_col,
        "variable": model_vars,
        "feature_importance": final_gb.feature_importances_
    })

    gb_importance["feature_importance"] = gb_importance["feature_importance"].round(4)
    gb_importance_tables.append(gb_importance)


model_summary = pd.DataFrame(model_summaries)

bayes_coefficients = pd.concat(bayes_coef_tables, ignore_index=True)
bayes_coefficients = bayes_coefficients.sort_values(
    ["target_label", "abs_coefficient"],
    ascending=[True, False]
)

gb_importances = pd.concat(gb_importance_tables, ignore_index=True)
gb_importances = gb_importances.sort_values(
    ["target_label", "feature_importance"],
    ascending=[True, False]
)


# Local Authority summary for Tableau
la_summary = data.groupby(
    ["la_code_2021", "la_name_2021"],
    as_index=False
).agg(
    number_of_msoas=("msoa_code", "count"),

    mean_care_20_plus_actual=("pct_care_20_plus_2021", "mean"),
    mean_care_20_plus_expected=("care_20_plus_bayes_expected", "mean"),
    mean_care_20_plus_residual=("care_20_plus_bayes_residual", "mean"),
    max_care_20_plus_residual=("care_20_plus_bayes_residual", "max"),
    min_care_20_plus_residual=("care_20_plus_bayes_residual", "min"),
    share_care_20_plus_top_5pct_higher=("care_20_plus_top_5pct_higher", "mean"),
    share_care_20_plus_top_5pct_lower=("care_20_plus_top_5pct_lower", "mean"),

    mean_any_care_actual=("pct_unpaid_care_any_2021", "mean"),
    mean_any_care_expected=("any_unpaid_care_bayes_expected", "mean"),
    mean_any_care_residual=("any_unpaid_care_bayes_residual", "mean"),
    max_any_care_residual=("any_unpaid_care_bayes_residual", "max"),
    min_any_care_residual=("any_unpaid_care_bayes_residual", "min"),
    share_any_care_top_5pct_higher=("any_unpaid_care_top_5pct_higher", "mean"),
    share_any_care_top_5pct_lower=("any_unpaid_care_top_5pct_lower", "mean")
)

for col in [
    "share_care_20_plus_top_5pct_higher",
    "share_care_20_plus_top_5pct_lower",
    "share_any_care_top_5pct_higher",
    "share_any_care_top_5pct_lower"
]:
    la_summary[col] = (la_summary[col] * 100).round(2)

for col in la_summary.columns:
    if col.startswith(("mean_", "max_", "min_")):
        la_summary[col] = la_summary[col].round(3)


# Save files
data.to_csv(processed_path / "model_outputs_msoa_2021.csv", index=False)
model_summary.to_csv(processed_path / "model_comparison_summary.csv", index=False)
bayes_coefficients.to_csv(processed_path / "bayesian_model_coefficients.csv", index=False)
gb_importances.to_csv(processed_path / "gradient_boosting_feature_importance.csv", index=False)
la_summary.to_csv(processed_path / "model_la_summary_2021.csv", index=False)

print("\nfiles saved")
print("model_outputs_msoa_2021.csv")
print("model_comparison_summary.csv")
print("bayesian_model_coefficients.csv")
print("gradient_boosting_feature_importance.csv")
print("model_la_summary_2021.csv")

print("\nmodel comparison")
print(model_summary)

print("\nBayesian coefficients for 20+ care")
print(
    bayes_coefficients[
        bayes_coefficients["target_label"] == "care_20_plus"
    ].head(10)
)

print("\nGradient Boosting importance for 20+ care")
print(
    gb_importances[
        gb_importances["target_label"] == "care_20_plus"
    ].head(10)
)

print("\nHighest positive Bayesian residuals for 20+ care")
print(
    data[
        [
            "msoa_name",
            "la_name_2021",
            "cluster_name",
            "context_cluster_name",
            "pct_care_20_plus_2021",
            "care_20_plus_bayes_expected",
            "care_20_plus_bayes_residual"
        ]
    ]
    .sort_values("care_20_plus_bayes_residual", ascending=False)
    .head(10)
)

print("\nHighest negative Bayesian residuals for 20+ care")
print(
    data[
        [
            "msoa_name",
            "la_name_2021",
            "cluster_name",
            "context_cluster_name",
            "pct_care_20_plus_2021",
            "care_20_plus_bayes_expected",
            "care_20_plus_bayes_residual"
        ]
    ]
    .sort_values("care_20_plus_bayes_residual")
    .head(10)
)