import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import lab2


class TextProcessingTests(unittest.TestCase):
    def test_json_preserves_order_and_does_not_include_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "texts.json"
            path.write_text(json.dumps({"source": "not a document", "texts": ["First", "", "Third"]}), encoding="utf-8")
            self.assertEqual(lab2.extract_texts_from_file(path), ["First", "", "Third"])
            path.write_text('["valid", 42]', encoding="utf-8")
            with self.assertRaises(ValueError):
                lab2.extract_texts_from_file(path)

    def test_required_preprocessing_and_spelling_before_stemming(self):
        processed, audit = lab2.process_texts_with_audit(["NLP: langauge, running!!! 123 @#$"])
        self.assertEqual(processed, ["natur languag process languag run"])
        self.assertEqual(audit[0]["spelling_corrections"], {"langauge": "language"})
        self.assertIn("natural language processing", audit[0]["expanded"])

    def test_abbreviations_use_boundaries_and_preserve_negation(self):
        _, audit = lab2.process_texts_with_audit(["E.g. AI, ML and GPU. Rain is not useful."])
        self.assertIn("for example artificial intelligence, machine learning", audit[0]["expanded"])
        self.assertIn("graphics processing unit", audit[0]["expanded"])
        self.assertIn("rain", audit[0]["tokens_before_correction"])
        self.assertIn("not", audit[0]["processed"].split())
        self.assertEqual(lab2.process_texts(["vector's direction"]), ["vector direct"])

    def test_empty_documents_and_language_validation(self):
        self.assertEqual(lab2.process_texts(["123 !!!", "the and a"]), ["", ""])
        with self.assertRaises(ValueError):
            lab2.process_texts(["Український текст"])


class SimilarityTests(unittest.TestCase):
    def test_cosine_has_expected_pairs_and_zero_vector(self):
        result = np.array(lab2.calcuate_tfidf(["cat dog", "cat dog", "sun rain", ""]))
        np.testing.assert_allclose(result, [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], atol=1e-12)

    def test_tfidf_against_manual_smoothed_idf_and_cosine(self):
        # Two docs: df(cat)=2 and df(dog)=1, so IDF=[1, log(3/2)+1].
        result = np.array(lab2.calcuate_tfidf(["cat dog", "cat"]))
        expected = 1 / np.sqrt(1 + (np.log(3 / 2) + 1) ** 2)
        self.assertAlmostEqual(result[0, 1], expected)
        np.testing.assert_allclose(result, result.T, atol=1e-12)

    def test_all_empty_corpus_is_rejected(self):
        for texts in ([], ["", ""]):
            with self.assertRaises(ValueError):
                lab2.calcuate_tfidf(texts)

    def test_heatmap_rejects_inconsistent_matrix(self):
        with self.assertRaises(ValueError):
            lab2.visualize_data([[1]], ["first", "second"])
        with self.assertRaises(ValueError):
            lab2.visualize_data([[float("nan")]], ["first"])


if __name__ == "__main__":
    unittest.main()
