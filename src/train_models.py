"""Train and evaluate baseline churn classification models."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
RANDOM_STATE = 42


def load_and_prepare_data(path: str | Path = DATA_PATH):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).drop_duplicates().reset_index(drop=True)
    df = df.drop(columns=["customerID"])
    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})
    return X, y


def build_preprocessor(X: pd.DataFrame):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])


def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    print(f"\n{name}")
    print("=" * len(name))
    print(classification_report(y_test, predictions, target_names=["No Churn", "Churn"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")


def main():
    X, y = load_and_prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(X_train)

    logistic = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])

    forest = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    for name, model in [("Logistic Regression", logistic), ("Random Forest", forest)]:
        model.fit(X_train, y_train)
        evaluate_model(name, model, X_test, y_test)


if __name__ == "__main__":
    main()
