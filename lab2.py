"""Assignment 2: English text preprocessing, TF-IDF and cosine similarity."""
from functools import lru_cache
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spellchecker import SpellChecker

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs" / "lab2"
FIGURES = ROOT / "figures" / "lab2"
ABBREVIATIONS = {
    "e.g.": "for example", "i.e.": "that is", "u.s.a.": "united states america",
    "ai": "artificial intelligence", "ml": "machine learning",
    "nlp": "natural language processing", "api": "application programming interface",
    "cpu": "central processing unit", "gpu": "graphics processing unit",
    "u": "you", "ur": "your", "r": "are", "pls": "please", "thx": "thanks",
    "can't": "cannot", "won't": "will not", "don't": "do not",
    "isn't": "is not", "it's": "it is", "tf-idf": "tfidf",
}
ABBREVIATION_PATTERN = re.compile(
    r"(?<![a-z])(" + "|".join(re.escape(key) for key in sorted(ABBREVIATIONS, key=len, reverse=True)) + r")(?![a-z])"
)
STEMMER = PorterStemmer()
# Keep negation: 'not useful' should not become 'useful' during cleaning.
STOP_WORDS = ENGLISH_STOP_WORDS - {"not", "no", "never", "nor", "cannot"}


def extract_texts_from_file(file_path):
    """Read a JSON list of strings, or an object with a 'texts' list.

    Preserve document order and empty documents. Never treat metadata as text.
    """
    path = Path(file_path)
    if not path.is_absolute():
        path = ROOT / path
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        data = data.get("texts")
    if not isinstance(data, list) or not data or not all(isinstance(text, str) for text in data):
        raise ValueError("Expected a nonempty JSON list of strings or {'texts': [...]}.")
    return data


@lru_cache(maxsize=1)
def spell_checker():
    checker = SpellChecker(language="en", distance=1)
    checker.word_frequency.load_words([
        "python", "numpy", "pandas", "tfidf", "json", "csv", "github", "inception",
        "matplotlib", "cosine", "tokenization", "stemming", "lemmatization",
        "postgresql", "kharkiv", "kyiv", "embeddings",
        "oppenheimer", "cillian", "murphy", "nolan", "dijkstra", "kruskal",
    ])
    return checker


def process_texts_with_audit(texts):
    """Expand abbreviations, clean, correct spelling, remove stopwords and stem.

    Spelling is corrected BEFORE stemming. An audit records every replacement.
    The chosen stemmer and dictionary support English; other scripts are rejected.
    """
    if not isinstance(texts, list) or not texts or not all(isinstance(text, str) for text in texts):
        raise ValueError("Expected a nonempty list of strings.")
    checker = spell_checker()
    processed, audit = [], []
    for document, original in enumerate(texts, 1):
        if any(char.isalpha() and not char.isascii() for char in original):
            raise ValueError(f"Document {document}: this pipeline supports English text only.")
        lower = original.lower().replace("’", "'")
        expanded = ABBREVIATION_PATTERN.sub(lambda match: ABBREVIATIONS[match.group()], lower)
        expanded = re.sub(r"\b([a-z]+)'s\b", r"\1", expanded)
        tokens = re.findall(r"[a-z]+", expanded)
        replacements = {}
        for word in sorted(checker.unknown(tokens)):
            replacement = checker.correction(word) if len(word) > 2 else None
            if replacement and replacement != word:
                replacements[word] = replacement
        corrected = [replacements.get(word, word) for word in tokens]
        kept = [word for word in corrected if word not in STOP_WORDS]
        result = " ".join(STEMMER.stem(word) for word in kept)
        processed.append(result)
        audit.append({"document": document, "original": original,
                      "expanded": expanded, "tokens_before_correction": tokens,
                      "spelling_corrections": replacements, "processed": result})
    return processed, audit


def process_texts(texts):
    """Return processed documents, preserving their number and order."""
    return process_texts_with_audit(texts)[0]


def tfidf_features(texts_processed):
    if not isinstance(texts_processed, list) or not texts_processed or not all(
        isinstance(text, str) for text in texts_processed
    ):
        raise ValueError("Expected a nonempty list of processed strings.")
    if not any(re.search(r"[a-z]+", text) for text in texts_processed):
        raise ValueError("All documents are empty after preprocessing; TF-IDF has no vocabulary.")
    vectorizer = TfidfVectorizer(lowercase=False, token_pattern=r"(?u)\b[a-z]+\b",
                                 norm="l2", smooth_idf=True)
    matrix = vectorizer.fit_transform(texts_processed)
    return vectorizer, matrix


def calcuate_tfidf(texts_processed):
    """Return an n x n cosine matrix as lists (name retained from the template)."""
    _, matrix = tfidf_features(texts_processed)
    return np.clip(cosine_similarity(matrix), 0, 1).tolist()


def visualize_data(cosine_matrix, texts_processed, filename="cosine_similarity.png", title=None):
    """Save an annotated heatmap labelled with document IDs and initial words."""
    values = np.asarray(cosine_matrix, dtype=float)
    count = len(texts_processed)
    if count == 0 or values.shape != (count, count):
        raise ValueError("Matrix dimensions must match the number of documents.")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Cosine scores must be finite values between 0 and 1.")
    labels = [f"D{i}: {' '.join(text.split()[:4]) or '[empty]'}"
              for i, text in enumerate(texts_processed, 1)]
    fig, ax = plt.subplots(figsize=(max(8, count * 1.1), max(6, count * .85)), layout="constrained")
    heat = ax.imshow(values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(count), labels, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(count), labels)
    ax.set(title=title or "Косинусна схожість текстів за TF-IDF",
           xlabel="Документ", ylabel="Документ")
    ax.tick_params(labelsize=9)
    for row in range(count):
        for col in range(count):
            ax.text(col, row, f"{values[row, col]:.2f}", ha="center", va="center",
                    color="white" if values[row, col] > .55 else "#153447", fontsize=10)
    fig.colorbar(heat, ax=ax, shrink=.8, label="Косинусна схожість (0–1)")
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / filename
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def get_texts():
    """Load the separate original corpus for task 5."""
    return extract_texts_from_file("resources/lab2_own_texts.json")


def analyse_corpus(texts, name):
    processed, audit = process_texts_with_audit(texts)
    vectorizer, features = tfidf_features(processed)
    cosine_matrix = np.asarray(calcuate_tfidf(processed))
    ids = [f"D{i}" for i in range(1, len(texts) + 1)]
    terms = vectorizer.get_feature_names_out()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(features.toarray(), index=ids, columns=terms).to_csv(
        OUTPUTS / f"{name}_tfidf.csv", index_label="document")
    pd.DataFrame(cosine_matrix, index=ids, columns=ids).to_csv(
        OUTPUTS / f"{name}_cosine.csv", index_label="document")
    pd.DataFrame({"term": terms, "idf": vectorizer.idf_}).to_csv(
        OUTPUTS / f"{name}_idf.csv", index=False)
    (OUTPUTS / f"{name}_processing.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Sort only unique pairs: skip the diagonal and mirrored duplicates.
    pairs = [{"document_1": ids[i], "document_2": ids[j], "cosine": float(cosine_matrix[i, j])}
             for i in range(len(texts)) for j in range(i + 1, len(texts))]
    pairs.sort(key=lambda pair: (-pair["cosine"], pair["document_1"], pair["document_2"]))
    pd.DataFrame(pairs, columns=["document_1", "document_2", "cosine"]).to_csv(
        OUTPUTS / f"{name}_pairs.csv", index=False)
    top_terms = []
    for row in range(len(texts)):
        weights = features.getrow(row).toarray().ravel()
        indices = sorted(np.flatnonzero(weights), key=lambda i: (-weights[i], terms[i]))[:5]
        top_terms.append({"document": ids[row], "terms": ", ".join(terms[indices])})
    image = visualize_data(cosine_matrix, processed, f"{name}_cosine.png",
                           "Тексти викладача: TF-IDF" if name == "teacher" else "Власні тексти: TF-IDF")
    return {"processed": processed, "audit": audit, "cosine": cosine_matrix,
            "features": features, "terms": terms, "top_terms": top_terms,
            "pairs": pairs, "image": image}


def main():
    summaries = {}
    for name, texts in (("teacher", extract_texts_from_file("resources/data.json")),
                        ("own", get_texts())):
        result = analyse_corpus(texts, name)
        summaries[name] = {"documents": len(texts), "vocabulary": len(result["terms"]),
                           "empty_documents": [i + 1 for i, text in enumerate(result["processed"]) if not text],
                           "most_similar_pair": result["pairs"][0] if result["pairs"] else None}
    (OUTPUTS / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summaries


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
