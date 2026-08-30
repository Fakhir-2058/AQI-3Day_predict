import hopsworks
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


project = hopsworks.login(
    api_key_value=os.environ["HOPSWORKS_API_KEY"]
)

feature_store = project.get_feature_store()
print("Connected to Hopsworks!")


feature_group = feature_store.get_feature_group(
    name="aqi_daily",
    version=1
)

print("Feature Group loaded!")

df = feature_group.read()

print("Data loaded from Hopsworks!")
print("Shape:", df.shape)
print(df.head())

target_col = ["target_aqi_day1", "target_aqi_day2", "target_aqi_day3"]

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

X = df.drop(columns=target_col)
y = df[target_col]


if "date" in X.columns:
    X["date"] = pd.to_datetime(X["date"])

    order = X["date"].sort_values().index

    X = X.loc[order].reset_index(drop=True)
    y = y.loc[order].reset_index(drop=True)


if "date" in X.columns:
    X = X.drop(columns=["date"])


X = X.fillna(0)

targets = [
    "target_aqi_day1",
    "target_aqi_day2",
    "target_aqi_day3"
]


split = int(len(X) * 0.80)

X_train = X.iloc[:split]
X_test = X.iloc[split:]
models = {

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Ridge Regression": Ridge(),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        random_state=42
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
}

results = []
trained_models = {}

for target in targets:


    print("TARGET:", target)


    y_train = y[target].iloc[:split]
    y_test = y[target].iloc[split:]


    models = {

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),

        "Ridge Regression": Ridge(),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            random_state=42
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    }


    for model_name, model in models.items():

        print("\nTraining:", model_name)

        model.fit(X_train,y_train)

        prediction = model.predict(X_test)

        mae = mean_absolute_error(y_test,prediction)

        rmse = np.sqrt(
            mean_squared_error(y_test,prediction))

        r2 = r2_score(y_test,prediction)

        print("MAE :", round(mae, 3))
        print("RMSE:", round(rmse, 3))
        print("R2  :", round(r2, 3))


        results.append({
            "Model": model_name,
            "Target": target,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        trained_models[(target, model_name)] = model

results_df = pd.DataFrame(results)
print("FINAL MODEL COMPARISON\n")

print(results_df)

print("BEST MODEL FOR EACH DAY\n")

best_models = {}

for target in targets:

    target_results = results_df[results_df["Target"] == target].copy()

    target_results["MAE_rank"] = target_results["MAE"].rank(
        ascending=True)

    target_results["RMSE_rank"] = target_results["RMSE"].rank(
        ascending=True)

    target_results["R2_rank"] = target_results["R2"].rank(
        ascending=False)


    target_results["Total_score"] = (
        target_results["MAE_rank"] + target_results["RMSE_rank"] + target_results["R2_rank"])

    best_model = target_results.loc[target_results["Total_score"].idxmin()]

    model_name = best_model["Model"]

    best_models[target] = {
        "model": trained_models[(target, model_name)],
        "name": model_name,
        "MAE": best_model["MAE"],
        "RMSE": best_model["RMSE"],
        "R2": best_model["R2"]
    }


    print("\n", target)

    print("Best Model:", best_model["Model"])
    print("MAE:", round(best_model["MAE"],3))
    print("RMSE:", round(best_model["RMSE"],3))
    print("R2:", round(best_model["R2"],3))


print("\nREGISTERING BEST MODELS IN HOPSWORKS MODEL REGISTRY...")

model_reg = project.get_model_registry()
print("Model Registry loaded!")

for target in targets:

    best = best_models[target]
    model = best["model"]
    model_name = best["name"]
    mae = best["MAE"]
    rmse = best["RMSE"]
    r2 = best["R2"]

   
    registry_name = target.replace("target_aqi_", "lahore_aqi_")
    model_file = f"{registry_name}.pkl"
    joblib.dump(model, model_file)

    print("\nModel saved locally:", model_file)

    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }

    registered_model = model_reg.sklearn.create_model(
        name=registry_name,
        metrics=metrics,
        description=f"Best model for Lahore AQI prediction {target}. "
                    f"Selected from Random Forest, Gradient Boosting, "
                    f"Extra Trees and XGBoost. "
                    f"Best algorithm: {model_name}."
    )
    registered_model.save(model_file)

    print("REGISTERED SUCCESSFULLY!")
    print("Model Name:", registry_name)
    print("Algorithm:", model_name)
    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2:", round(r2, 3))
