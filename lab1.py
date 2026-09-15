"""Assignment 1: collection, weather visualization and missing-data experiments."""
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import hashlib
import json
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
OUTPUTS = ROOT / "outputs"
DATASET = "ilayaraja07/data-cleaning-feature-imputation"
FEATURES = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
            "pH", "sulphates", "alcohol"]
METHODS = ("mean", "median", "mode", "knn")


def get_response(url):
    """GET with timeout; network and HTTP errors remain visible to the caller."""
    response = requests.get(url, timeout=45,
                            headers={"User-Agent": "IDA-Lab1/1.0 (educational)"})
    response.raise_for_status()
    return response


def parse_web_page(url):
    """Return static HTML text without markup, scripts, styles or comments."""
    response = get_response(url)
    soup = BeautifulSoup(response.content, "html.parser")
    for element in soup(["script", "style", "template"]):
        element.decompose()
    return " ".join(soup.stripped_strings)


def parse_api(api_url, output_path=None):
    """Fetch JSON and save it to result.json; invalid JSON raises an exception."""
    data = get_response(api_url).json()
    path = Path(output_path) if output_path is not None else ROOT / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def canonical_date(value):
    """Accept both YYYY-M-D and YYYY-MM-DD; reject invalid calendar dates."""
    return datetime.strptime(value, "%Y-%m-%d").date().isoformat()


def parse_json(date, path=None):
    """Return matching records as a list of dicts, or [] when date is absent."""
    target = canonical_date(date)
    source = Path(path) if path is not None else ROOT / "resources" / "weather.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    return [dict(row, date=canonical_date(row["date"])) for row in data["daily"]
            if canonical_date(row["date"]) == target]


def read_weather(path=None):
    source = Path(path) if path is not None else ROOT / "resources" / "weather.csv"
    frame = pd.read_csv(source)
    frame.columns = frame.columns.str.strip()
    frame["CET"] = pd.to_datetime(frame["CET"], format="%Y-%m-%d", errors="raise")
    return frame.sort_values("CET").reset_index(drop=True)


def parse_csv(date, path=None):
    """Return a list of typed records; missing CSV fields become None."""
    target = canonical_date(date)
    frame = read_weather(path)
    selected = frame.loc[frame["CET"].eq(pd.Timestamp(target))].copy()
    selected["CET"] = selected["CET"].dt.strftime("%Y-%m-%d")
    return selected.astype(object).where(selected.notna(), None).to_dict("records")


def save_figure(fig, name):
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def visualize_weather():
    """Save several plots; missing observations are never replaced by zero."""
    frame = read_weather()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    monthly = frame.set_index("CET")[["Max TemperatureC", "Mean TemperatureC",
                                      "Min TemperatureC"]].resample("MS").mean()
    fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
    ax.fill_between(monthly.index, monthly["Min TemperatureC"],
                    monthly["Max TemperatureC"], color="#bed9ed", label="Середні min–max")
    ax.plot(monthly.index, monthly["Mean TemperatureC"], color="#1e5685", lw=1.2,
            label="Середня температура")
    ax.set(title="Погода: помісячна температура за наявними спостереженнями",
           xlabel="Рік", ylabel="Температура, °C")
    ax.legend(loc="upper left", ncols=2)
    ax.grid(alpha=.2)
    paths = [save_figure(fig, "weather_temperature.png")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].hist(frame["Mean TemperatureC"].dropna(), bins=30, color="#2677a5", edgecolor="white")
    axes[0].set(title="Розподіл середньодобової температури", xlabel="Температура, °C", ylabel="Кількість днів")
    axes[1].scatter(frame["Mean TemperatureC"], frame["Mean Humidity"], s=6, alpha=.12, color="#368477")
    axes[1].set(title="Температура та вологість", xlabel="Температура, °C", ylabel="Середня вологість, %")
    paths.append(save_figure(fig, "weather_distribution.png"))

    indexed = frame.set_index("CET")
    expected = pd.date_range(indexed.index.min(), indexed.index.max(), freq="D")
    coverage = indexed.reindex(expected)["Mean TemperatureC"].notna().groupby(expected.year).mean() * 100
    fig, ax = plt.subplots(figsize=(11, 3), layout="constrained")
    ax.bar(coverage.index, coverage.values, color="#368477")
    ax.set_xticks(coverage.index[::2])
    ax.set(title="Повнота спостережень температури за роками", xlabel="Рік",
           ylabel="Днів зі значенням, %", ylim=(0, 105))
    paths.append(save_figure(fig, "weather_coverage.png"))
    return paths


def load_dataset(url=DATASET, csv_file="Wine_Quality.csv"):
    """Load an authentic Kaggle CSV, downloading a public archive if absent.

    The direct Kaggle API avoids an unnecessary account dependency. A cached
    original file permits offline re-runs. Authentication failures are not hidden.
    """
    if url != DATASET or Path(csv_file).name != csv_file:
        raise ValueError("This lab loader supports the documented dataset and a CSV filename.")
    path = ROOT / "data" / "kaggle" / csv_file
    if not path.exists():
        endpoint = f"https://www.kaggle.com/api/v1/datasets/download/{url}"
        response = get_response(endpoint)
        with zipfile.ZipFile(BytesIO(response.content)) as archive:
            content = archive.read(csv_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        provenance = {"dataset": f"https://www.kaggle.com/datasets/{url}",
                      "download_url": endpoint, "file": csv_file,
                      "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                      "sha256": hashlib.sha256(content).hexdigest()}
        (ROOT / "data" / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return pd.read_csv(path)


def make_imputer(method):
    if method not in METHODS:
        raise ValueError(f"Unknown imputation method: {method}")
    if method == "knn":
        return KNNImputer(n_neighbors=5, weights="distance")
    return SimpleImputer(strategy="most_frequent" if method == "mode" else method)


def impute_data(df, method="median"):
    """Impute numeric predictors and categorical values, preserving observations.

    quality is an ordinal target and is deliberately excluded from distances.
    For predictive use, fit on training data only (see evaluate_imputation).
    """
    result = df.copy()
    features = [c for c in FEATURES if c in result]
    if result[features].isna().all().any():
        raise ValueError("A completely missing feature cannot be inferred from this dataset.")
    values = result[features].astype(float)
    if method == "knn":
        # Estimate distances on comparable scales, using observed data only.
        scaler = StandardScaler()
        scaled = scaler.fit_transform(values)
        filled = scaler.inverse_transform(make_imputer(method).fit_transform(scaled))
    else:
        filled = make_imputer(method).fit_transform(values)
    filled = pd.DataFrame(filled, columns=features, index=result.index)
    result[features] = values.where(values.notna(), filled)
    for col in result.columns.difference(features):
        if result[col].isna().any():
            if col == "quality":
                raise ValueError("Target quality is missing; do not impute labels.")
            modes = result[col].mode(dropna=True)
            if modes.empty:
                raise ValueError(f"No observed values in {col}")
            result[col] = result[col].fillna(modes.iloc[0])
    return result


def normalize_data(df, method="robust"):
    """Scale numeric predictors; preserve type and quality unchanged."""
    scalers = {"minmax": MinMaxScaler, "z-score": StandardScaler, "robust": RobustScaler}
    if method not in scalers:
        raise ValueError(f"Unknown normalization method: {method}")
    result = df.copy()
    if result[FEATURES].isna().any().any():
        raise ValueError("Impute missing predictors before normalization.")
    result[FEATURES] = scalers[method]().fit_transform(result[FEATURES])
    return result


def evaluate_imputation(df, seed=42):
    """Mask 10% of observed validation values to compare against known truth.

    No synthetic values are added to the source. The mask exists only in an
    experimental copy. No validation rows or quality labels enter fitting.
    """
    train, validation = train_test_split(df[FEATURES].astype(float), test_size=.2, random_state=seed)
    rng = np.random.default_rng(seed)
    mask = (rng.random(validation.shape) < .1) & validation.notna().to_numpy()
    masked = validation.mask(mask)
    scaler = StandardScaler().fit(train)
    train_scaled = scaler.transform(train)
    val_scaled = scaler.transform(masked)
    truth = validation.to_numpy()
    rows = []
    for method in METHODS:
        imputer = make_imputer(method)
        if method == "knn":
            imputer.fit(train_scaled)
            predicted = scaler.inverse_transform(imputer.transform(val_scaled))
        else:
            imputer.fit(train)
            predicted = imputer.transform(masked)
        for j, feature in enumerate(FEATURES):
            selected = mask[:, j]
            error = predicted[selected, j] - truth[selected, j]
            rows.append({"method": method, "feature": feature,
                         "masked_values": int(selected.sum()),
                         "mae": float(np.abs(error).mean()),
                         "standardized_mae": float(np.abs(error).mean() / scaler.scale_[j])})
    detail = pd.DataFrame(rows)
    ranking = detail.groupby("method")["standardized_mae"].mean().sort_values()
    return detail, ranking


def visualize_imputation(original, versions, ranking):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    missing = original.isna().sum()
    missing = missing[missing > 0].sort_values()
    axes[0].barh(missing.index, missing.values, color="#be7545")
    axes[0].set(title="Реальні пропуски у Wine_Quality.csv", xlabel="Кількість значень")
    axes[1].bar(ranking.index, ranking.values, color="#2677a5")
    axes[1].set(title="Похибка на прихованих відомих значеннях", ylabel="Середня MAE / σ навчальної ознаки")
    paths = [save_figure(fig, "imputation_comparison.png")]
    feature = "fixed acidity"
    idx = original[feature].isna()
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    for method in METHODS:
        ax.plot(np.arange(idx.sum()), versions[method].loc[idx, feature], "o-", label=method)
    ax.set(title="Оцінки для реальних пропусків fixed acidity", xlabel="Номер пропуску (порядок у файлі)",
           ylabel="Значення у вихідній шкалі")
    ax.legend(ncols=4)
    paths.append(save_figure(fig, "imputed_values.png"))
    return paths


def visualize_normalization(imputed):
    columns = ["residual sugar", "total sulfur dioxide", "alcohol"]
    variants = {"Вихідні значення": imputed,
                "MinMax": normalize_data(imputed, "minmax"),
                "Z-score": normalize_data(imputed, "z-score"),
                "Robust": normalize_data(imputed, "robust")}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for ax, (title, values) in zip(axes.flat, variants.items()):
        ax.boxplot([values[c] for c in columns], tick_labels=columns, showfliers=True,
                   flierprops={"markersize": 2, "alpha": .15})
        ax.set(title=title, ylabel="Значення" if title == "Вихідні значення" else "Масштабоване значення")
        ax.tick_params(axis="x", labelrotation=12)
    return [save_figure(fig, "normalization_comparison.png")]


def run_data_experiment():
    OUTPUTS.mkdir(exist_ok=True)
    df = load_dataset()
    detail, ranking = evaluate_imputation(df)
    best = str(ranking.index[0])
    versions = {method: impute_data(df, method) for method in METHODS}
    for method, frame in versions.items():
        frame.to_csv(OUTPUTS / f"wine_imputed_{method}.csv", index=False)
    for method in ("minmax", "z-score", "robust"):
        normalize_data(versions[best], method).to_csv(OUTPUTS / f"wine_normalized_{method}.csv", index=False)
    detail.to_csv(OUTPUTS / "imputation_validation.csv", index=False)
    paths = visualize_imputation(df, versions, ranking)
    paths += visualize_normalization(versions[best])
    summary = {"rows": len(df), "columns": len(df.columns),
               "missing_before": int(df.isna().sum().sum()),
               "missing_after": int(versions[best].isna().sum().sum()),
               "validation_seed": 42, "validation_fraction": .2,
               "mask_probability": .1, "selected_imputation": best,
               "selected_scaling": "robust", "standardized_mae": ranking.to_dict(),
               "source_sha256": hashlib.sha256((ROOT / "data/kaggle/Wine_Quality.csv").read_bytes()).hexdigest()}
    (OUTPUTS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return df, versions, detail, ranking, paths, summary


def collect_examples():
    OUTPUTS.mkdir(exist_ok=True)
    urls = ["https://example.com/", "https://docs.python.org/3/tutorial/index.html"]
    collection = []
    for i, url in enumerate(urls, 1):
        content = parse_web_page(url)
        (OUTPUTS / f"webpage_{i}.txt").write_text(content + "\n", encoding="utf-8")
        collection.append({"url": url, "characters": len(content), "file": f"webpage_{i}.txt"})
    parse_api("https://api.github.com/")
    collection.append({"url": "https://api.github.com/", "file": "../result.json"})
    (OUTPUTS / "collection_sources.json").write_text(json.dumps({
        "collected_at_utc": datetime.now(timezone.utc).isoformat(), "sources": collection
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return collection


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true", help="Fetch the live webpage and API examples")
    args = parser.parse_args()
    if args.collect:
        print(collect_examples())
    print(parse_json("2024-8-19"))
    print(parse_csv("1997-5-22"))
    print([str(path) for path in visualize_weather()])
    print(json.dumps(run_data_experiment()[-1], indent=2))
