import unittest

import numpy as np
import pandas as pd

from lab4 import mcnemar_test


class McNemarTest(unittest.TestCase):
    def test_disagreements_and_identical_predictions(self):
        actual = pd.Series([0] * 9)
        first = pd.Series([0] * 7 + [1, 1])
        second = pd.Series([1] * 6 + [0, 0, 1])

        table, p_value = mcnemar_test(actual, first, second)

        np.testing.assert_array_equal(table, [[1, 6], [1, 1]])
        self.assertAlmostEqual(p_value, 0.125)
        self.assertEqual(mcnemar_test(actual, first, first)[1], 1.0)
