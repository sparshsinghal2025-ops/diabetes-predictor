from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import pickle

warnings.filterwarnings("ignore")

# ===== 1. LOAD DATA =====
# Replace with your actual local file path: r"E:\Beginner ML projects datasets\diabetes.csv"
df = pd.read_csv(r"E:\Beginner ML projects datasets\diabetes.csv")

# Clean biological features where 0 represents missing/invalid measurements
numerical_features = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# Biological features where 0 is structurally impossible (replace 0 with median)
zero_invalid_features = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in zero_invalid_features:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

# Fill standard missing gaps in other columns if any exist
for col in numerical_features:
    if col in df.columns and col not in zero_invalid_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

# Handle data duplicates
df = df.drop_duplicates()

# ===== 2. SPLIT DATA =====
# Isolate feature matrix and targeted prediction outcome
X = df.drop(columns=["Outcome"], errors="ignore")
y = df["Outcome"].astype(int)

# ColumnTransformer handles numeric scaling across clean inputs
preprocessor = ColumnTransformer(
    transformers=[("num", "passthrough", X.columns)],
    remainder="drop",
)

# Fit and transform the data matrix
X_processed = preprocessor.fit_transform(X)

# Split using stratify to maintain proportional classification outcomes
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

# Scale all clinical parameters uniformly
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== 3. MODELS + TUNING =====
rf_estimators = [100, 200, 300]
rf_splits = [2, 5, 10]
xgb_estimators = [100, 200]
xgb_depths = [3, 5, 7]

models = {
    "Logistic": {
        "model": LogisticRegression(max_iter=1000), 
        "params": {}
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": rf_estimators,
            "max_depth": [5, 10, None],
            "min_samples_split": rf_splits
        }
    },
    "XGBoost": {
        "model": XGBClassifier(random_state=42, verbosity=0, eval_metric="logloss"),
        "params": {
            "n_estimators": xgb_estimators,
            "max_depth": xgb_depths,
            "learning_rate": [0.01, 0.1, 0.3]
        }
    }
}

results = {}

print("Training diabetes prediction models... grab chai ☕")
print("-" * 40)

for name, config in models.items():
    model = config["model"]
    params = config["params"]

    if params:
        search = RandomizedSearchCV(
            model, params, n_iter=10, cv=3, scoring="accuracy", random_state=42, n_jobs=-1
        )
        search.fit(X_train_scaled, y_train)
        best_model = search.best_estimator_
        print(f"{name}: Tuning done. Best params = {search.best_params_}")
    else:
        best_model = model
        best_model.fit(X_train_scaled, y_train)
        print(f"{name}: Trained")

    # Evaluate performance
    y_pred = best_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {"Accuracy": acc, "F1_Score": f1, "Model": best_model}

# ===== 4. SHOW WINNER =====
print("\n" + "=" * 40)
print("FINAL RESULTS")
print("=" * 40)

for name, scores in sorted(results.items(), key=lambda x: x[1]["Accuracy"], reverse=True):
    print(f"{name:15} | Accuracy = {scores['Accuracy']:.4f} | F1-Score = {scores['F1_Score']:.4f}")

winner_name = max(results, key=lambda x: results[x]["Accuracy"])
print("\n🏆 WINNER:", winner_name)
print(f"Best Accuracy: {results[winner_name]['Accuracy']:.4f}")
print(f"Use this model: results['{winner_name}']['Model']")

# Save artifacts tailored for Diabetes prediction
artifacts = {
    "model": results[winner_name]["Model"],
    "preprocessor": preprocessor,
    "scaler": scaler,
}

with open("DiabetesPrediction.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("\nModel, preprocessor, and scaler saved cleanly to DiabetesPrediction.pkl")
