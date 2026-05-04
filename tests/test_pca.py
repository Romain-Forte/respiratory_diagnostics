import unittest

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.pca import PCA, pca_from_dataframe


class TestPcaFromDataFrame(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            {
                "feat_1": [0.2, 0.5, 1.1, 1.4, 2.0, 2.3],
                "feat_2": [1.0, 0.8, 0.2, -0.1, -0.5, -0.9],
                "feat_3": [10, 11, 10, 9, 8, 7],
                "label_a": [1, 0, 1, 0, 1, 0],
                "label_b": [False, True, False, True, False, True],
            },
            index=[101, 102, 103, 104, 105, 106],
        )

    def tearDown(self) -> None:
        plt.close("all")

    def test_wrapper_matches_direct_pca(self) -> None:
        direct_result = PCA(
            labels=self.df[["label_a", "label_b"]],
            features=self.df[["feat_1", "feat_2", "feat_3"]],
            render_mode="scatter",
            show_ellipse=False,
            max_points_per_label=10,
        )
        wrapped_result = pca_from_dataframe(
            self.df,
            ["label_a", "label_b"],
            render_mode="scatter",
            show_ellipse=False,
            max_points_per_label=10,
        )

        np.testing.assert_allclose(
            direct_result["explained_variance_ratio"],
            wrapped_result["explained_variance_ratio"],
        )
        np.testing.assert_allclose(
            direct_result["coordinates"].to_numpy(),
            wrapped_result["coordinates"].to_numpy(),
        )
        self.assertListEqual(
            list(wrapped_result["sampled_points"]["label_a"]),
            [101, 103, 105],
        )

    def test_missing_color_column_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "introuvables"):
            pca_from_dataframe(self.df, ["label_a", "missing_column"])

    def test_non_binary_color_column_raises(self) -> None:
        df_invalid = self.df.copy()
        df_invalid["label_a"] = [0, 1, 2, 0, 1, 0]

        with self.assertRaisesRegex(ValueError, "binaires/boolennes"):
            pca_from_dataframe(df_invalid, ["label_a"])

    def test_requires_two_numeric_features_after_excluding_color_columns(self) -> None:
        df_small = pd.DataFrame(
            {
                "feat_1": [0.1, 0.2, 0.3, 0.4],
                "label_a": [1, 0, 1, 0],
                "label_b": [0, 1, 0, 1],
            }
        )

        with self.assertRaisesRegex(ValueError, "deux colonnes numeriques"):
            pca_from_dataframe(df_small, ["label_a", "label_b"])


if __name__ == "__main__":
    unittest.main()
