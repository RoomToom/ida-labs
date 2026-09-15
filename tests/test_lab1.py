import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

import lab1


class ParsingTests(unittest.TestCase):
    def test_json_date_formats_and_known_values(self):
        result = lab1.parse_json("2024-8-19")
        self.assertEqual(result, lab1.parse_json("2024-08-19"))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["precipitation"], 5.0)
        self.assertEqual(result[0]["max_temperature"], 30.0)

    def test_csv_date_formats_and_missing_fields(self):
        self.assertEqual(lab1.parse_csv("1997-1-1"), lab1.parse_csv("1997-01-01"))
        result = lab1.parse_csv("1997-1-1")[0]
        self.assertEqual(result["Mean TemperatureC"], 4.0)
        self.assertIsNone(result["Max Gust SpeedKm/h"])
        self.assertIsNone(result["Events"])

    def test_absent_and_invalid_dates(self):
        for parser in (lab1.parse_json, lab1.parse_csv):
            self.assertEqual(parser("1900-01-01"), [])
            with self.assertRaises(ValueError):
                parser("2024-02-30")

    @patch("lab1.get_response")
    def test_scraping_nested_text_entities_and_scripts(self, response):
        response.return_value.content = b'<title>Title</title><p>Hello <b>world</b> &amp; friends</p><script>secret()</script><style>p{}</style><!--comment-->'
        self.assertEqual(lab1.parse_web_page("https://example.com"), "Title Hello world & friends")

    @patch("lab1.requests.get")
    def test_http_failure_propagates(self, get):
        get.return_value.raise_for_status.side_effect = requests.HTTPError("404")
        with self.assertRaises(requests.HTTPError):
            lab1.parse_web_page("https://example.com/missing")

    @patch("lab1.get_response")
    def test_api_saves_json_and_does_not_overwrite_on_invalid_json(self, response):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            response.return_value.json.return_value = {"temperature": 0, "city": "Київ"}
            lab1.parse_api("https://example.com/api", path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["city"], "Київ")
            before = path.read_bytes()
            response.return_value.json.side_effect = ValueError("not JSON")
            with self.assertRaises(ValueError):
                lab1.parse_api("https://example.com/api", path)
            self.assertEqual(path.read_bytes(), before)


class TransformationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = lab1.load_dataset()

    def test_all_methods_fill_missing_and_preserve_observations(self):
        before = self.source.copy(deep=True)
        for method in lab1.METHODS:
            with self.subTest(method=method):
                result = lab1.impute_data(self.source, method)
                self.assertEqual(result.shape, self.source.shape)
                self.assertFalse(result.isna().any().any())
                for column in self.source:
                    observed = self.source[column].notna()
                    np.testing.assert_array_equal(result.loc[observed, column], self.source.loc[observed, column])
        pd.testing.assert_frame_equal(self.source, before)

    def test_normalizations_preserve_labels_and_have_expected_scales(self):
        filled = lab1.impute_data(self.source)
        for method in ("minmax", "z-score", "robust"):
            result = lab1.normalize_data(filled, method)
            pd.testing.assert_frame_equal(result[["quality", "type"]], filled[["quality", "type"]])
            values = result[lab1.FEATURES].to_numpy()
            self.assertTrue(np.isfinite(values).all())
            if method == "minmax":
                np.testing.assert_allclose(values.min(axis=0), 0, atol=1e-10)
                np.testing.assert_allclose(values.max(axis=0), 1, atol=1e-10)
            elif method == "z-score":
                np.testing.assert_allclose(values.mean(axis=0), 0, atol=1e-10)
                np.testing.assert_allclose(values.std(axis=0), 1, atol=1e-10)
            else:
                np.testing.assert_allclose(np.median(values, axis=0), 0, atol=1e-10)
                np.testing.assert_allclose(np.percentile(values, 75, axis=0) - np.percentile(values, 25, axis=0), 1, atol=1e-10)

    def test_validation_has_no_train_test_leakage(self):
        from sklearn.model_selection import train_test_split
        train, val = train_test_split(self.source[lab1.FEATURES].astype(float), test_size=.2, random_state=42)
        mask = (np.random.default_rng(42).random(val.shape) < .1) & val.notna().to_numpy()
        detail, ranking = lab1.evaluate_imputation(self.source)
        # Independently recompute mean-imputation MAE using TRAIN means only.
        for j, feature in enumerate(lab1.FEATURES):
            expected = np.abs(val.loc[mask[:, j], feature] - train[feature].mean()).mean()
            actual = detail.loc[(detail.method == "mean") & (detail.feature == feature), "mae"].iloc[0]
            self.assertAlmostEqual(expected, actual)
        self.assertEqual(set(ranking.index), set(lab1.METHODS))


if __name__ == "__main__":
    unittest.main()
