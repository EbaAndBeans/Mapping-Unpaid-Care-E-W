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

results = []

bayesian_settings = [
    {
        "setting_name": "default",
        "alpha_1": 1e-6,
        "alpha_2": 1e-6,
        "lambda_1": 1e-6,
        "lambda_2": 1e-6
    },
    {
        "setting_name": "weaker_prior",
        "alpha_1": 1e-7,
        "alpha_2": 1e-7,
        "lambda_1": 1e-7,
        "lambda_2": 1e-7
    },
    {
        "setting_name": "stronger_prior",
        "alpha_1": 1e-5,
        "alpha_2": 1e-5,
        "lambda_1": 1e-5,
        "lambda_2": 1e-5
    }
]

gb_settings = []

for n_estimators in [100, 200, 300]:
    for learning_rate in [0.03, 0.05, 0.08]:
        for max_depth in [2, 3]:
            for min_samples_leaf in [5, 10, 20]:
                gb_settings.append({
                    "n_estimators": n_estimators,
                    "learning_rate": learning_rate,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf
                })


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

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Bayesian Ridge tuning
    for setting in bayesian_settings:
        model = BayesianRidge(
            alpha_1=setting["alpha_1"],
            alpha_2=setting["alpha_2"],
            lambda_1=setting["lambda_1"],
            lambda_2=setting["lambda_2"]
        )

        model.fit(x_train_scaled, y_train)
        pred = model.predict(x_test_scaled)

        results.append({
            "target_label": target_label,
            "target_column": target_col,
            "model": "Bayesian Ridge",
            "setting_name": setting["setting_name"],
            "test_mae": round(mean_absolute_error(y_test, pred), 4),
            "test_r2": round(r2_score(y_test, pred), 4),
            "alpha_1": setting["alpha_1"],
            "alpha_2": setting["alpha_2"],
            "lambda_1": setting["lambda_1"],
            "lambda_2": setting["lambda_2"],
            "n_estimators": "",
            "learning_rate": "",
            "max_depth": "",
            "min_samples_leaf": ""
        })

    # Gradient Boosting tuning
    for setting in gb_settings:
        model = GradientBoostingRegressor(
            n_estimators=setting["n_estimators"],
            learning_rate=setting["learning_rate"],
            max_depth=setting["max_depth"],
            min_samples_leaf=setting["min_samples_leaf"],
            random_state=random_state
        )

        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        results.append({
            "target_label": target_label,
            "target_column": target_col,
            "model": "Gradient Boosting",
            "setting_name": "gb_sweep",
            "test_mae": round(mean_absolute_error(y_test, pred), 4),
            "test_r2": round(r2_score(y_test, pred), 4),
            "alpha_1": "",
            "alpha_2": "",
            "lambda_1": "",
            "lambda_2": "",
            "n_estimators": setting["n_estimators"],
            "learning_rate": setting["learning_rate"],
            "max_depth": setting["max_depth"],
            "min_samples_leaf": setting["min_samples_leaf"]
        })


results_df = pd.DataFrame(results)

# choose best settings by lowest MAE for each target and model
chosen_rows = []

for target_label in targets.keys():
    for model_name in ["Bayesian Ridge", "Gradient Boosting"]:
        subset = results_df[
            (results_df["target_label"] == target_label)
            & (results_df["model"] == model_name)
        ].copy()

        subset = subset.sort_values(
            ["test_mae", "test_r2"],
            ascending=[True, False]
        )

        chosen_rows.append(subset.iloc[0])

chosen = pd.DataFrame(chosen_rows)

results_df.to_csv(processed_path / "model_tuning_results.csv", index=False)
chosen.to_csv(processed_path / "chosen_model_settings.csv", index=False)

print("\ntuning complete")
print("\nbest settings")
print(chosen[[
    "target_label",
    "model",
    "test_mae",
    "test_r2",
    "setting_name",
    "n_estimators",
    "learning_rate",
    "max_depth",
    "min_samples_leaf",
    "alpha_1",
    "lambda_1"
]])

print("\nfiles saved")
print("model_tuning_results.csv")
print("chosen_model_settings.csv")