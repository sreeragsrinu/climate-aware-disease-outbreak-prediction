import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ==============================
# LOAD DATASET
# =============================
file_path = os.path.join(os.path.dirname(__file__), "../data/Mergeddataset.xlsx")
print("Loading dataset from:", file_path)

df = pd.read_excel(file_path)

print("\n========= DATA PREVIEW =========")
print(df.head())

# Remove "Total" rows
df = df[df["State"] != "Total"]

# ==============================
# REMOVE OUTLIERS (Top 5%) BEFORE SPLIT? NO.
# First split, then clean training only
# ==============================

X = df.drop("Cases", axis=1)
y = df["Cases"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Remove outliers ONLY from training set
upper_limit = y_train.quantile(0.95)
mask = y_train <= upper_limit
X_train = X_train[mask]
y_train = y_train[mask]

# ==============================
# HANDLE MISSING VALUES (Train Mean)
# ==============================
for col in ["Actual_Rainfall", "Temperature", "Humidity"]:
    mean_value = X_train[col].mean()
    X_train[col] = X_train[col].fillna(mean_value)
    X_test[col] = X_test[col].fillna(mean_value)

# ==============================
# PREPROCESSING PIPELINE
# ==============================
categorical_features = ["State"]
numeric_features = ["Year", "Actual_Rainfall", "Temperature", "Humidity"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"
)

# ==============================
# MODEL + GRID SEARCH
# ==============================
rf = RandomForestRegressor(random_state=42)

param_grid = {
    "model__n_estimators": [200, 500],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5]
}

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", rf)
])

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Parameters:", grid_search.best_params_)

# ==============================
# CROSS VALIDATION SCORE
# ==============================
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="r2")
print("Cross Validation R2 Mean:", round(cv_scores.mean(), 3))

# ==============================
# TEST EVALUATION
# ==============================
y_pred = best_model.predict(X_test)

print("\n========= MODEL PERFORMANCE =========")
print("R2 Score:", round(r2_score(y_test, y_pred), 3))
print("Mean Absolute Error:", round(mean_absolute_error(y_test, y_pred), 2))

# ==============================
# YEARLY TREND GRAPH
# ==============================
yearly_cases = df.groupby("Year")["Cases"].sum()

plt.figure()
plt.plot(yearly_cases.index, yearly_cases.values, marker="o")
plt.xlabel("Year")
plt.ylabel("Total Cases")
plt.title("Total Dengue Cases by Year")
plt.show()

# ==============================
# ACTUAL vs PREDICTED
# ==============================
plt.figure(figsize=(10,5))

plt.plot(y_test.reset_index(drop=True), label="Actual")
plt.plot(pd.Series(y_pred), label="Predicted")

plt.legend()
plt.title("Actual vs Predicted Dengue Cases")
plt.xlabel("Test Data Points")
plt.ylabel("Cases")
plt.show()

# ==============================
# FEATURE IMPORTANCE
# ==============================
model_rf = best_model.named_steps["model"]
feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()

importances = model_rf.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=True)

plt.figure(figsize=(10,6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# ==============================
# MANUAL PREDICTION
# ==============================
print("\n========= MANUAL PREDICTION =========")

available_states = sorted(df["State"].unique())
print("Available States:", available_states)

rain = float(input("Enter Rainfall: "))
temp = float(input("Enter Temperature: "))
humidity = float(input("Enter Humidity: "))
year = int(input("Enter Year: "))
state_name = input("Enter State Name exactly as shown: ")

if state_name not in available_states:
    print("Invalid state. Please restart and enter correct state.")
else:
    input_data = pd.DataFrame({
        "Year": [year],
        "Actual_Rainfall": [rain],
        "Temperature": [temp],
        "Humidity": [humidity],
        "State": [state_name]
    })

    prediction = best_model.predict(input_data)
    print("\nPredicted Dengue Cases:", int(prediction[0]))

print("\n========= PROGRAM COMPLETED SUCCESSFULLY =========")
