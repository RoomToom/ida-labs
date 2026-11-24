"""Прогноз відтоку клієнтів Telco."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path(__file__).resolve().parent / "resources" / "Telco-Customer-Churn.csv"
SEED = 42
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "InternetAddons"]
ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
          "StreamingTV", "StreamingMovies"]
TARGET = "Churn"


def load_churn(path=DATA):
    data = pd.read_csv(path, dtype={"TotalCharges": "string"})
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"].str.strip(), errors="raise")
    if not data[TARGET].isin(["Yes", "No"]).all():
        raise ValueError("Churn має містити тільки Yes або No")
    return data


def prepare_churn(data):
    data = data.drop_duplicates().copy()
    if data["customerID"].isna().any() or data["customerID"].duplicated().any():
        raise ValueError("Очікується один рядок на клієнта")

    # За нуль місяців користування накопичена оплата ще не нарахована.
    no_bill = data["TotalCharges"].isna() & data["tenure"].eq(0)
    data.loc[no_bill, "TotalCharges"] = 0.0
    if data.isna().any().any():
        raise ValueError("У даних залишилися пропуски")

    data["InternetAddons"] = data[ADDONS].eq("Yes").sum(axis=1)
    data[TARGET] = data[TARGET].map({"No": 0, "Yes": 1})
    return data.set_index("customerID")


def split_churn(data):
    return train_test_split(data, test_size=0.2, stratify=data[TARGET], random_state=SEED)


def fit_models(train):
    X = train.drop(columns=TARGET)
    y = train[TARGET]
    categories = X.columns.difference(NUMERIC).tolist()
    preprocess = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categories),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    candidates = [
        ("Logistic Regression", LogisticRegression(max_iter=1000),
         {"logisticregression__C": [0.1, 1, 10]}),
        ("Random Forest", RandomForestClassifier(n_estimators=150, random_state=SEED),
         {"randomforestclassifier__max_depth": [5, 10, None],
          "randomforestclassifier__min_samples_leaf": [3, 10]}),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=SEED),
         {"gradientboostingclassifier__n_estimators": [75, 150],
          "gradientboostingclassifier__max_depth": [1, 2, 3]}),
    ]
    models = {}
    rows = []
    for name, classifier, grid in candidates:
        search = GridSearchCV(make_pipeline(preprocess, classifier), grid, cv=cv,
                              scoring="f1", error_score="raise")
        search.fit(X, y)
        models[name] = search.best_estimator_
        params = {key.split("__", 1)[1]: value for key, value in search.best_params_.items()}
        rows.append({
            "Модель": name,
            "CV F1": search.best_score_,
            "CV F1 std": search.cv_results_["std_test_score"][search.best_index_],
            "Параметри": params,
        })
    models["Baseline"] = DummyClassifier(strategy="most_frequent").fit(X, y)
    cv_results = pd.DataFrame(rows).set_index("Модель").sort_values("CV F1", ascending=False)
    return models, cv_results


def evaluate_models(models, test):
    X = test.drop(columns=TARGET)
    y = test[TARGET]
    rows = []
    predictions = pd.DataFrame(index=test.index)
    probabilities = pd.DataFrame(index=test.index)
    for name, model in models.items():
        predicted = model.predict(X)
        probability = model.predict_proba(X)[:, 1]
        predictions[name] = predicted
        probabilities[name] = probability
        rows.append({
            "Модель": name,
            "Accuracy": accuracy_score(y, predicted),
            "Precision": precision_score(y, predicted, zero_division=0),
            "Recall": recall_score(y, predicted),
            "F1": f1_score(y, predicted),
            "ROC AUC": roc_auc_score(y, probability),
            "Average precision": average_precision_score(y, probability),
        })
    return pd.DataFrame(rows).set_index("Модель"), predictions, probabilities


def mcnemar_test(actual, prediction_a, prediction_b):
    correct_a = actual.eq(prediction_a)
    correct_b = actual.eq(prediction_b)
    table = confusion_matrix(correct_a, correct_b, labels=[True, False])
    a_only = int(table[0, 1])
    b_only = int(table[1, 0])
    discordant = a_only + b_only
    p_value = binomtest(a_only, discordant, p=0.5).pvalue if discordant else 1.0
    return table, p_value


if __name__ == "__main__":
    train, test = split_churn(prepare_churn(load_churn()))
    models, cv_results = fit_models(train)
    metrics, predictions, _ = evaluate_models(models, test)
    print(cv_results.round(4).to_string())
    print("\nТестова вибірка")
    print(metrics.round(4).to_string())
    table, p_value = mcnemar_test(test[TARGET], predictions["Logistic Regression"],
                                  predictions["Random Forest"])
    print(f"\nМак-Немар (Logistic Regression / Random Forest):\n{table}\np = {p_value:.4f}")
