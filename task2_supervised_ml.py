import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load and clean data
df = pd.read_csv("student_performance_raw.csv").drop_duplicates().reset_index(drop=True)

numeric_cols = [
    "Age", "Study_Hours_Per_Day", "Attendance_Percent",
    "Assignments_Score", "Final_Score"
]
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

df["Gender_Encoded"] = df["Gender"].map({"Male": 0, "Female": 1})
df["Passed_Encoded"] = df["Passed"].map({"No": 0, "Yes": 1})

features = [
    "Age", "Study_Hours_Per_Day", "Attendance_Percent",
    "Assignments_Score", "Gender_Encoded"
]
X = df[features]

# ---------------- REGRESSION ----------------
# Predict Final Score
y_reg = df["Final_Score"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.20, random_state=42
)

reg_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=100, random_state=42
    )
}

for name, model in reg_models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("\n", name)
    print("MAE:", mean_absolute_error(y_test, pred))
    print("RMSE:", mean_squared_error(y_test, pred) ** 0.5)
    print("R2:", r2_score(y_test, pred))

# ---------------- CLASSIFICATION ----------------
# Predict Passed / Not Passed
y_clf = df["Passed_Encoded"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y_clf, test_size=0.20, random_state=42, stratify=y_clf
)

clf_models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ])
}

for name, model in clf_models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred, zero_division=0))
    print("Recall:", recall_score(y_test, pred, zero_division=0))
    print("F1:", f1_score(y_test, pred, zero_division=0))
