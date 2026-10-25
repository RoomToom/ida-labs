"""Assignment 3: regression and classification on the available original data"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             classification_report, f1_score, mean_absolute_error,
                             r2_score, root_mean_squared_error, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold, TimeSeriesSplit,
                                     cross_val_score, train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs" / "lab3"
FIGURES = ROOT / "figures" / "lab3"
SEED = 42
REVIEWS_DATASET = "septa97/100k-courseras-course-reviews-dataset"
HEART_NUMERIC = ["age", "trtbps", "chol", "thalachh", "oldpeak"]
HEART_CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exng", "slp", "caa", "thall"]


def extract_texts_from_file(file_path):
    """Read tabular CSV data (the function name is retained from the template)."""
    path = Path(file_path)
    if not path.is_absolute():
        path = ROOT / path
    return pd.read_csv(path)


def save_json(name, value):
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_plot(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def build_plot(file_path, x, y):
    data = extract_texts_from_file(file_path)
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.scatter(data[x], data[y], s=12, alpha=.4, marker="x")
    ax.set(xlabel=x, ylabel=y, title=f"{y} залежно від {x}")
    return fig


def regression_metrics(actual, predicted):
    return {"mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(root_mean_squared_error(actual, predicted)),
            "r2": float(r2_score(actual, predicted))}


def fit_regression(data, features, target, name, chronological=False):
    """Choose polynomial degree on train CV; evaluate once on held-out rows."""
    frame = data.drop_duplicates().copy()
    X = frame[features].apply(pd.to_numeric, errors="raise")
    y = pd.to_numeric(frame[target], errors="raise")
    if not np.isfinite(X.to_numpy()).all() or not np.isfinite(y.to_numpy()).all():
        raise ValueError("Regression input must contain finite numeric features and targets.")
    if chronological:
        boundary = int(len(frame) * .8)
        X_train, X_val = X.iloc[:boundary], X.iloc[boundary:]
        y_train, y_val = y.iloc[:boundary], y.iloc[boundary:]
        folds = TimeSeriesSplit(n_splits=5)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2, random_state=SEED)
        folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, candidates = [], {}
    for degree in (1, 2, 3):
        model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), LinearRegression())
        scores = -cross_val_score(model, X_train, y_train, scoring="neg_root_mean_squared_error", cv=folds)
        key = f"polynomial_degree_{degree}"
        rows.append({"model": key, "degree": degree, "cv_rmse_mean": float(scores.mean()),
                     "cv_rmse_std": float(scores.std())})
        candidates[key] = model
    ranking = pd.DataFrame(rows).sort_values(["cv_rmse_mean", "degree"])
    best_name = str(ranking.iloc[0]["model"])
    best = candidates[best_name].fit(X_train, y_train)
    predictions = best.predict(X_val)
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    terms = best.named_steps["polynomialfeatures"].get_feature_names_out(features).tolist()
    estimator = best.named_steps["linearregression"]
    theta = [float(estimator.intercept_), *estimator.coef_.astype(float).tolist()]
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(OUTPUTS / f"{name}_model_selection.csv", index=False)
    pd.DataFrame({"term": ["intercept", *terms], "coefficient": theta}).to_csv(
        OUTPUTS / f"{name}_coefficients.csv", index=False)
    validation = X_val.copy()
    if chronological:
        validation.insert(0, "Datetime", frame.loc[X_val.index, "Datetime"])
    validation["actual"] = y_val
    validation["predicted"] = predictions
    validation.to_csv(OUTPUTS / f"{name}_validation.csv", index_label="source_row")
    save_json(f"{name}_split.json", {"train_rows": X_train.index.tolist(), "validation_rows": X_val.index.tolist()})
    summary = {"status": "completed", "rows": len(frame), "duplicates_removed": len(data)-len(frame),
               "features": features, "target": target, "train_rows": len(X_train), "validation_rows": len(X_val),
               "selection": "5-fold expanding time CV on train, lowest mean RMSE" if chronological
                            else "5-fold CV on train, lowest mean RMSE",
               "split": "chronological_80_20" if chronological else "random_80_20_seed_42",
               "selected_model": best_name,
               "validation": regression_metrics(y_val, predictions),
               "baseline": regression_metrics(y_val, baseline.predict(X_val)),
               "theta_terms": ["intercept", *terms], "theta": theta}
    labels = {
        "co2": (["Об’єм двигуна, л"], "CO₂, г/км", "CO₂: прогноз лише за об’ємом двигуна"),
        "ice_cream": (["Температура, °C"], "Продажі, одиниці", "Продажі морозива залежно від температури"),
        "power": (["Температура", "Вологість", "Швидкість вітру"],
                  "Сума споживання 3 зон, одиниці файла", "Електроспоживання: зрізи моделі за погодою"),
    }
    x_labels, y_label, title = labels[name]
    if chronological:
        dates = pd.to_datetime(frame["Datetime"])
        summary["train_start"] = dates.loc[X_train.index].min().isoformat()
        summary["train_end"] = dates.loc[X_train.index].max().isoformat()
        summary["validation_start"] = dates.loc[X_val.index].min().isoformat()
        summary["validation_end"] = dates.loc[X_val.index].max().isoformat()
        summary["target_definition"] = "Sum of PowerConsumption_Zone1, Zone2 and Zone3 at the same timestamp."
        summary["target_units"] = "Original file units; no unit conversion or energy integration applied."
    save_json(f"{name}_summary.json", summary)
    if len(features) == 1:
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        ax.scatter(X_train.iloc[:, 0], y_train, s=14, alpha=.3, label="Train")
        ax.scatter(X_val.iloc[:, 0], y_val, s=20, alpha=.6, label="Validation")
        grid = pd.DataFrame({features[0]: np.linspace(X.iloc[:, 0].min(), X.iloc[:, 0].max(), 250)})
        ax.plot(grid.iloc[:, 0], best.predict(grid), c="#a93431", lw=2, label=best_name)
        ax.set(xlabel=x_labels[0], ylabel=y_label, title=title)
        ax.legend()
        ax.grid(alpha=.15)
    else:
        fig, axes = plt.subplots(1, len(features), figsize=(15, 4), layout="constrained")
        for j, (feature, ax) in enumerate(zip(features, axes)):
            grid = pd.DataFrame(np.tile(X_train.median().to_numpy(), (250, 1)), columns=features)
            grid[feature] = np.linspace(X_train[feature].min(), X_train[feature].max(), 250)
            ax.scatter(X_val[feature], y_val, s=3, alpha=.08, label="Validation: факт")
            ax.plot(grid[feature], best.predict(grid), c="#a93431", lw=2,
                    label="Модель; інші ознаки = медіани train")
            ax.set(xlabel=x_labels[j], ylabel=y_label if j == 0 else "")
        axes[0].legend(fontsize=8)
        fig.suptitle(title)
    save_plot(fig, f"{name}_model.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].scatter(y_val, predictions, s=15, alpha=.5)
    bounds = [min(y_val.min(), predictions.min()), max(y_val.max(), predictions.max())]
    axes[0].plot(bounds, bounds, "--", c="grey")
    axes[0].set(xlabel=f"Факт: {y_label}", ylabel=f"Прогноз: {y_label}", title="Validation: факт і прогноз")
    axes[1].scatter(predictions, y_val.to_numpy()-predictions, s=15, alpha=.5)
    axes[1].axhline(0, c="grey", ls="--")
    axes[1].set(xlabel=f"Прогноз: {y_label}", ylabel="Залишок в одиницях цілі", title="Validation: залишки")
    save_plot(fig, f"{name}_validation.png")
    if chronological:
        daily = validation.set_index("Datetime")[["actual", "predicted"]].resample("D").mean()
        fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
        ax.plot(daily.index, daily.actual, label="Факт")
        ax.plot(daily.index, daily.predicted, label="Прогноз за погодою")
        ax.set(title="Validation у часі: добові середні споживання", xlabel="Дата", ylabel=y_label)
        ax.legend()
        ax.tick_params(axis="x", rotation=20)
        save_plot(fig, "power_time.png")
    return theta


def CO2_emissions(data):
    return fit_regression(data, ["Engine_Size"], "CO2_Emissions", "co2")


def ice_cream_selling(data):
    return fit_regression(data, ["Temperature (°C)"], "Ice Cream Sales (units)", "ice_cream")


def prepare_power(data):
    frame = data.drop_duplicates().copy()
    frame["Datetime"] = pd.to_datetime(frame.Datetime, format="%m/%d/%Y %H:%M", errors="raise")
    if frame.Datetime.duplicated().any():
        raise ValueError("Conflicting rows share a timestamp; resolve them before training.")
    zones = ["PowerConsumption_Zone1", "PowerConsumption_Zone2", "PowerConsumption_Zone3"]
    if not np.isfinite(frame[zones].to_numpy(dtype=float)).all():
        raise ValueError("All three zone measurements are required to compute the total.")
    frame["PowerConsumption_Total"] = frame[zones].sum(axis=1)
    return frame.sort_values("Datetime")


def power_consumption(data):
    return fit_regression(prepare_power(data), ["Temperature", "Humidity", "WindSpeed"],
                          "PowerConsumption_Total", "power", chronological=True)


def heart_preprocessor():
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categories = make_pipeline(SimpleImputer(strategy="most_frequent"),
                               OneHotEncoder(handle_unknown="ignore"))
    return ColumnTransformer([("numeric", numeric, HEART_NUMERIC),
                              ("categorical", categories, HEART_CATEGORICAL)])


def save_classification_results(name, labels, actual, predicted, baseline_predicted):
    report = classification_report(actual, predicted, labels=labels, output_dict=True, zero_division=0)
    pd.DataFrame({key: row for key, row in report.items() if isinstance(row, dict)}).T.to_csv(
        OUTPUTS / f"{name}_classification_report.csv", index_label="class")
    save_json(f"{name}_classification_report.json", report)
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    display = ConfusionMatrixDisplay.from_predictions(actual, predicted, labels=labels, ax=ax,
                                                       cmap="Blues", colorbar=False, values_format="d")
    ax.set(title=f"{name}: матриця помилок на validation", xlabel="Передбачений клас", ylabel="Справжній клас")
    save_plot(fig, f"{name}_confusion.png")
    pd.DataFrame(display.confusion_matrix, index=labels, columns=labels).to_csv(
        OUTPUTS / f"{name}_confusion.csv", index_label="actual_class")
    return {"accuracy": float(accuracy_score(actual, predicted)),
            "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(actual, predicted, average="weighted", zero_division=0)),
            "baseline_accuracy": float(accuracy_score(actual, baseline_predicted)),
            "baseline_macro_f1": float(f1_score(actual, baseline_predicted, average="macro", zero_division=0))}


def heart_classification(data):
    frame = data.drop_duplicates().copy()
    features = HEART_NUMERIC + HEART_CATEGORICAL
    if set(frame.output.unique()) != {0, 1}:
        raise ValueError("Expected binary heart.csv target output.")
    X_train, X_val, y_train, y_val = train_test_split(
        frame[features], frame.output, test_size=.2, stratify=frame.output, random_state=SEED)
    pipeline = Pipeline([("preprocess", heart_preprocessor()),
                         ("classifier", LogisticRegression(max_iter=2000, random_state=SEED))])
    grids = [
        {"classifier": [LogisticRegression(max_iter=2000, random_state=SEED)], "classifier__C": [.1, 1, 10]},
        {"classifier": [RidgeClassifier()], "classifier__alpha": [.1, 1, 10]},
        {"classifier": [SGDClassifier(loss="log_loss", max_iter=3000, tol=1e-4, random_state=SEED)],
         "classifier__alpha": [.0001, .001, .01]},
    ]
    search = GridSearchCV(pipeline, grids, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                          scoring="f1_macro", refit=True, n_jobs=1)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    predicted = best.predict(X_val)
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train).predict(X_val)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    ranking = pd.DataFrame({"model": [str(p) for p in search.cv_results_["params"]],
                            "cv_macro_f1": search.cv_results_["mean_test_score"],
                            "cv_std": search.cv_results_["std_test_score"]}).sort_values("cv_macro_f1", ascending=False)
    ranking.to_csv(OUTPUTS / "heart_model_selection.csv", index=False)
    metrics = save_classification_results("heart", [0, 1], y_val, predicted, baseline)
    scores = best.decision_function(X_val)
    metrics["roc_auc"] = float(roc_auc_score(y_val, scores))
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    RocCurveDisplay.from_predictions(y_val, scores, ax=ax, name="Validation")
    ax.plot([0, 1], [0, 1], "--", c="grey")
    ax.set(title="heart.csv: ROC на validation")
    save_plot(fig, "heart_roc.png")
    validation = X_val.copy()
    validation["actual"] = y_val
    validation["predicted"] = predicted
    validation["decision_score"] = scores
    validation.to_csv(OUTPUTS / "heart_validation.csv", index_label="source_row")
    estimator = best.named_steps["classifier"]
    coefficients = estimator.coef_.ravel().astype(float)
    theta = [float(estimator.intercept_[0]), *coefficients.tolist()]
    terms = ["intercept", *best.named_steps["preprocess"].get_feature_names_out().tolist()]
    pd.DataFrame({"term": terms, "coefficient": theta}).to_csv(OUTPUTS / "heart_coefficients.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    selected = np.argsort(np.abs(coefficients))[-10:]
    ax.barh(np.array(terms[1:])[selected], coefficients[selected], color="#326f92")
    ax.set(title="heart.csv: найбільші коефіцієнти обраної моделі", xlabel="Коефіцієнт у перетвореному просторі")
    save_plot(fig, "heart_coefficients.png")
    save_json("heart_split.json", {"train_rows": X_train.index.tolist(), "validation_rows": X_val.index.tolist()})
    save_json("heart_summary.json", {"status": "completed", "rows": len(frame),
              "duplicates_removed": len(data)-len(frame), "features": features,
              "train_rows": len(X_train), "validation_rows": len(X_val),
              "selected_model": type(estimator).__name__, "selected_params": str(search.best_params_),
              "selection": "5-fold stratified CV on train, highest macro F1",
              "cv_macro_f1": float(search.best_score_), "validation": metrics,
              "target_note": "Predicts output labels in the supplied dataset, not calibrated clinical risk.",
              "theta_terms": terms, "theta": theta})
    return theta


def load_dataset(url=REVIEWS_DATASET, csv_file="reviews.csv"):
    """Read the locally preserved original Kaggle CSV."""
    if url != REVIEWS_DATASET or csv_file != "reviews.csv":
        raise ValueError("Use the documented Coursera Reviews dataset and reviews.csv.")
    return extract_texts_from_file("resources/reviews.csv")


def prepare_reviews(data):
    """Remove empty reviews, duplicate texts and conflicting labels before split."""
    frame = data[["Review", "Label"]].dropna().copy()
    frame["Label"] = pd.to_numeric(frame.Label, errors="raise")
    if not frame.Label.isin([1, 2, 3, 4, 5]).all():
        raise ValueError("Review ratings must be integers from 1 to 5.")
    frame["Label"] = frame.Label.astype(int)
    frame["Review"] = frame.Review.astype(str).str.replace(r"<[^>]+>", " ", regex=True)
    frame["key"] = frame.Review.str.casefold().str.replace(r"[\W_]+", " ", regex=True).str.strip()
    empty = frame.key.eq("")
    conflicts = frame.groupby("key").Label.nunique()
    conflicting = frame.key.isin(conflicts[conflicts > 1].index)
    valid = frame.loc[~empty & ~conflicting]
    cleaned = valid.drop_duplicates("key").copy()
    stats = {"original_rows": len(data), "missing_rows_removed": len(data)-len(frame),
             "empty_rows_removed": int(empty.sum()), "conflicting_rows_removed": int((conflicting & ~empty).sum()),
             "duplicates_removed": len(valid)-len(cleaned), "rows_after_cleaning": len(cleaned)}
    return cleaned, stats


def text_pipeline(classifier):
    return Pipeline([("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                               stop_words="english", min_df=3, max_features=20000,
                                               ngram_range=(1, 2), sublinear_tf=True)),
                     ("classifier", classifier)])


def classify_text_ml(data):
    frame, cleaning = prepare_reviews(data)
    train, val = train_test_split(frame, test_size=.2, stratify=frame.Label, random_state=SEED)
    if set(train.key) & set(val.key):
        raise AssertionError("The same normalized review occurs in both partitions.")
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    classifiers = {
        "multinomial_nb": MultinomialNB(alpha=.5),
        "sgd_logistic": SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=2000,
                                       tol=1e-3, random_state=SEED),
        "linear_svc": LinearSVC(C=1, class_weight="balanced", max_iter=5000, dual=False, random_state=SEED),
    }
    rows, candidates = [], {}
    for name, classifier in classifiers.items():
        pipeline = text_pipeline(classifier)
        scores = cross_val_score(pipeline, train.Review, train.Label, cv=cv, scoring="f1_macro", n_jobs=1)
        print(f"Text CV: {name}, macro F1={scores.mean():.4f}", flush=True)
        rows.append({"model": name, "cv_macro_f1": float(scores.mean()), "cv_std": float(scores.std())})
        candidates[name] = pipeline
    ranking = pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False)
    selected = str(ranking.iloc[0]["model"])
    best = candidates[selected].fit(train.Review, train.Label)
    predicted = best.predict(val.Review)
    baseline = DummyClassifier(strategy="most_frequent").fit(
        np.zeros((len(train), 1)), train.Label).predict(np.zeros((len(val), 1)))
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(OUTPUTS / "text_model_selection.csv", index=False)
    metrics = save_classification_results("text", [1, 2, 3, 4, 5], val.Label, predicted, baseline)
    pd.DataFrame({"source_row": val.index, "actual": val.Label, "predicted": predicted}).to_csv(
        OUTPUTS / "text_validation.csv", index=False)
    save_json("text_split.json", {"train_rows": train.index.tolist(), "validation_rows": val.index.tolist()})
    vocabulary = best.named_steps["tfidf"].get_feature_names_out()
    estimator = best.named_steps["classifier"]
    weights = estimator.coef_ if hasattr(estimator, "coef_") else estimator.feature_log_prob_
    top_terms = []
    for row, label in enumerate(estimator.classes_):
        terms = vocabulary[np.argsort(weights[row])[-10:][::-1]].tolist()
        top_terms.append({"label": int(label), "terms": terms})
    save_json("text_top_terms.json", top_terms)
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    counts = pd.DataFrame({"train": train.Label.value_counts(), "validation": val.Label.value_counts()}).sort_index()
    counts.plot.bar(ax=ax, color=["#326f92", "#bc7941"], rot=0)
    ax.set(title="Coursera: розподіл оцінок після очищення", xlabel="Оцінка", ylabel="Кількість відгуків")
    save_plot(fig, "text_class_balance.png")
    save_json("text_summary.json", {"status": "completed", "dataset": REVIEWS_DATASET,
              "cleaning": cleaning, "train_rows": len(train), "validation_rows": len(val),
              "label_counts": {str(k): int(v) for k, v in frame.Label.value_counts().sort_index().items()},
              "selected_model": selected, "vocabulary_size": len(vocabulary),
              "selection": "3-fold stratified CV on train, highest macro F1; TF-IDF fitted inside each fold",
              "cv_macro_f1": float(ranking.iloc[0]["cv_macro_f1"]), "validation": metrics,
              "target": "Review rating, 1 to 5; Id is not a feature"})


def write_summary():
    result = {}
    for name in ("co2", "ice_cream", "power", "heart", "text"):
        result[name] = json.loads((OUTPUTS / f"{name}_summary.json").read_text(encoding="utf-8"))
    save_json("summary.json", result)
    return result


if __name__ == "__main__":
    CO2_emissions(extract_texts_from_file("resources/CO2_emission.csv"))
    ice_cream_selling(extract_texts_from_file("resources/ice_cream_selling_data.csv"))
    power_consumption(extract_texts_from_file("resources/powerconsumption.csv"))
    heart_classification(extract_texts_from_file("resources/heart.csv"))
    classify_text_ml(load_dataset())
    print(json.dumps(write_summary(), ensure_ascii=False, indent=2))
