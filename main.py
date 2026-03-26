import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime   # ✅ FIX ADDED

API_KEY = "26E8negUjkBYnp1N2sAL0DWCAHhPZ8yFhEcIvr8L"

url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"

# ✅ DEFINE CURRENT TIME
current_time = datetime.now().strftime("%Y-%m-%dT%H")

params = {
    "api_key": API_KEY,
    "frequency": "hourly",
    "data[0]": "value",
    "facets[respondent][]": "PJM",
    "facets[type][]": "D",
    "start": "2018-01-01T00",
    "end": current_time
}

r = requests.get(url, params=params)

# ✅ Optional debug (safe)
if r.status_code != 200:
    print("API Error:", r.status_code, r.text)

data = r.json()["response"]["data"]

df = pd.DataFrame(data)
df = df[["period", "value"]]
df.columns = ["Datetime", "Load_MW"]

df["Datetime"] = pd.to_datetime(df["Datetime"])
df["Load_MW"] = pd.to_numeric(df["Load_MW"], errors="coerce")

df.dropna(inplace=True)
df.sort_values("Datetime", inplace=True)

print(df.head())
print("Rows:", df.shape[0])

df.to_csv("hourly_electricity_demand.csv", index=False)

# Plot 
sample_df = df.iloc[:168]   # 1 week

plt.figure(figsize=(12,4))
plt.plot(sample_df["Datetime"], sample_df["Load_MW"])
plt.title("Hourly Electricity Demand (1 Week)")
plt.xlabel("Datetime")
plt.ylabel("MW")
plt.show()

# Feature Engineering
df["hour"] = df["Datetime"].dt.hour
df["day"] = df["Datetime"].dt.day
df["month"] = df["Datetime"].dt.month
df["weekday"] = df["Datetime"].dt.weekday
df["is_weekend"] = df["weekday"].isin([5,6]).astype(int)

df["lag_1"] = df["Load_MW"].shift(1)
df["lag_24"] = df["Load_MW"].shift(24)
df["rolling_24"] = df["Load_MW"].rolling(24).mean()

df.dropna(inplace=True)

print(df.shape)
print(df.head())

# Train-Test Split
X = df.drop(["Datetime", "Load_MW"], axis=1)
y = df["Load_MW"]

split = int(len(df) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

# Evaluation Function
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate(model_name, y_true, y_pred):
    print("\n", model_name)
    print("MAE :", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R2  :", r2_score(y_true, y_pred))

# Linear Regression
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
evaluate("Linear Regression", y_test, y_pred_lr)

# Random Forest
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
evaluate("Random Forest", y_test, y_pred_rf)

# XGBoost
from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror"
)

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

# Actual vs Predicted Plot
plt.figure(figsize=(12,4))

plt.plot(y_test.values[:200], label="Actual Demand", linewidth=2)
plt.plot(y_pred_xgb[:200], label="Predicted Demand (XGBoost)", linestyle="--")

plt.title("Actual vs Predicted Electricity Demand (XGBoost)")
plt.xlabel("Time Steps")
plt.ylabel("Load (MW)")
plt.legend()
plt.grid(True)
plt.show()

# Residual Plot
residuals = y_test.values - y_pred_xgb

plt.figure(figsize=(12,4))
plt.plot(residuals[:200], color="red")
plt.axhline(0, color="black", linestyle="--")

plt.title("Residual Errors (Actual - Predicted) : XGBoost")
plt.xlabel("Time Steps")
plt.ylabel("Residual Error (MW)")
plt.grid(True)
plt.show()