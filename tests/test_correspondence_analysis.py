import unittest

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.correspondence_analysis import (
    correspondence_analysis,
    correspondence_analysis_from_dataframe,
)


class TestCorrespondenceAnalysisFromDataFrame(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            {
                "diag_a": [1, 1, 0, 0, 1, 0],
                "diag_b": [0, 1, 1, 0, 0, 1],
                "cond_x": [1, 0, 1, 0, 1, 0],
                "cond_y": [0, 1, 1, 0, 0, 1],
                "cond_z": [1, 1, 0, 0, 0, 0],
            },
            index=[101, 102, 103, 104, 105, 106],
        )

    def tearDown(self) -> None:
        plt.close("all")

    def test_wrapper_matches_direct_call_and_contingency_table(self) -> None:
        direct_result = correspondence_analysis(
            diagnoses=self.df[["diag_a", "diag_b"]],
            underlying_conditions=self.df[["cond_x", "cond_y", "cond_z"]],
        )
        wrapped_result = correspondence_analysis_from_dataframe(
            self.df,
            ["diag_a", "diag_b"],
            ["cond_x", "cond_y", "cond_z"],
        )

        expected_contingency = pd.DataFrame(
            {
                "cond_x": [2, 1],
                "cond_y": [1, 3],
                "cond_z": [2, 1],
            },
            index=["diag_a", "diag_b"],
        )

        pd.testing.assert_frame_equal(
            wrapped_result["contingency_table"],
            expected_contingency,
        )
        np.testing.assert_allclose(
            direct_result["row_coordinates"].to_numpy(),
            wrapped_result["row_coordinates"].to_numpy(),
        )
        np.testing.assert_allclose(
            direct_result["column_coordinates"].to_numpy(),
            wrapped_result["column_coordinates"].to_numpy(),
        )
        self.assertEqual(list(wrapped_result["row_coordinates"].columns), ["CA1", "CA2"])
        self.assertEqual(list(wrapped_result["column_coordinates"].columns), ["CA1", "CA2"])

    def test_missing_column_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "introuvables"):
            correspondence_analysis_from_dataframe(
                self.df,
                ["diag_a", "missing_diag"],
                ["cond_x", "cond_y"],
            )

    def test_non_binary_column_raises(self) -> None:
        invalid_df = self.df.copy()
        invalid_df["diag_a"] = [0, 1, 2, 0, 1, 0]

        with self.assertRaisesRegex(ValueError, "binaires/boolennes"):
            correspondence_analysis_from_dataframe(
                invalid_df,
                ["diag_a", "diag_b"],
                ["cond_x", "cond_y"],
            )

    def test_overlapping_column_names_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "distinctes"):
            correspondence_analysis_from_dataframe(
                self.df,
                ["diag_a", "diag_b"],
                ["diag_b", "cond_x"],
            )


if __name__ == "__main__":
    unittest.main()
