import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

import lab3


class PreparationTests(unittest.TestCase):
    def test_review_duplicates_and_conflicting_labels(self):
        frame = pd.DataFrame({"Review": ["Great course!", "great COURSE", "Bad", "bad!", "!!!", "Helpful"],
                              "Label": [5, 5, 1, 2, 3, 4]})
        cleaned, stats = lab3.prepare_reviews(frame)
        self.assertEqual(cleaned.Label.tolist(), [5, 4])
        self.assertEqual(cleaned.key.tolist(), ["great course", "helpful"])
        self.assertEqual(stats["duplicates_removed"], 1)
        self.assertEqual(stats["conflicting_rows_removed"], 2)
        self.assertEqual(stats["empty_rows_removed"], 1)
        self.assertEqual(stats["rows_after_cleaning"], 2)

    def test_invalid_review_rating_is_not_silently_truncated(self):
        with self.assertRaises(ValueError):
            lab3.prepare_reviews(pd.DataFrame({"Review": ["Review"], "Label": [1.5]}))

    def test_vectorizer_does_not_learn_validation_words(self):
        from sklearn.naive_bayes import MultinomialNB
        model = lab3.text_pipeline(MultinomialNB())
        model.fit(["good helpful course"] * 3 + ["bad confusing course"] * 3, [5]*3 + [1]*3)
        model.predict(["validationonlyword helpful"])
        self.assertNotIn("validationonlyword", model.named_steps["tfidf"].vocabulary_)

    def test_heart_scaler_fits_only_provided_rows(self):
        source = lab3.extract_texts_from_file("resources/heart.csv")
        train = source.iloc[:30]
        preprocessor = lab3.heart_preprocessor().fit(train)
        scaler = preprocessor.named_transformers_["numeric"].named_steps["standardscaler"]
        np.testing.assert_allclose(scaler.mean_, train[lab3.HEART_NUMERIC].mean().to_numpy())
        before = scaler.mean_.copy()
        preprocessor.transform(source.iloc[30:40])
        np.testing.assert_array_equal(scaler.mean_, before)


class ModelChecks(unittest.TestCase):
    def test_polynomial_theta_matches_predictions(self):
        X = pd.DataFrame({"Engine_Size": np.arange(1, 7, dtype=float)})
        y = 2 + 3 * X.Engine_Size + 4 * X.Engine_Size**2
        model = make_pipeline(PolynomialFeatures(2, include_bias=False), LinearRegression()).fit(X, y)
        estimator = model.named_steps["linearregression"]
        theta = [estimator.intercept_, *estimator.coef_]
        independent = theta[0] + theta[1]*X.Engine_Size + theta[2]*X.Engine_Size**2
        np.testing.assert_allclose(model.predict(X), independent, atol=1e-10)
        np.testing.assert_allclose(theta, [2, 3, 4], atol=1e-10)

    def test_regression_metrics_against_manual_result(self):
        result = lab3.regression_metrics([1, 2, 3], [1, 3, 2])
        self.assertAlmostEqual(result["mae"], 2/3)
        self.assertAlmostEqual(result["rmse"], np.sqrt(2/3))
        self.assertAlmostEqual(result["r2"], 0)


if __name__ == "__main__":
    unittest.main()
