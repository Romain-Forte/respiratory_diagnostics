import unittest

import numpy as np
import pandas as pd

from utils.data_quality import nettoyer_nan_par_colonne


class TestNettoyerNanParColonne(unittest.TestCase):
    def test_mice_imputes_numeric_columns_and_preserves_other_strategies(self) -> None:
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, np.nan, 4.0],
                "b": [2.0, np.nan, 6.0, 8.0],
                "c": ["x", None, "y", "z"],
            }
        )

        result = nettoyer_nan_par_colonne(
            df,
            strategies={"a": "mice", "b": "mice", "c": "inconnu"},
            mice_random_state=0,
            mice_max_iter=20,
        )

        self.assertFalse(result[["a", "b"]].isna().any().any())
        self.assertEqual(result.loc[1, "c"], "inconnu")
        self.assertAlmostEqual(result.loc[2, "a"], 3.0, places=3)
        self.assertAlmostEqual(result.loc[1, "b"], 4.0, places=3)
        self.assertEqual(result.loc[0, "a"], 1.0)
        self.assertEqual(result.loc[3, "b"], 8.0)


if __name__ == "__main__":
    unittest.main()
