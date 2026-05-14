import shutil
import sys
import traceback
import unittest
import uuid
from pathlib import Path
from pprint import pprint
from unittest import mock

import joblib
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.saved_model_inference import predict_all_saved_models


class FakePredictorWithFeatureNames:
    def __init__(self, expected_columns, positive_probability):
        self.feature_names_in_ = np.asarray(expected_columns, dtype=object)
        self.positive_probability = float(positive_probability)

    def predict_proba(self, X):
        if list(X.columns) != list(self.feature_names_in_):
            raise AssertionError("Unexpected feature order received by predictor.")
        return np.array([[1.0 - self.positive_probability, self.positive_probability]])


class FakeScaler:
    def __init__(self, expected_columns):
        self.colonnes_numeriques = list(expected_columns)


class FakePredictorWithScalerFallback:
    def __init__(self, expected_columns, positive_probability):
        self.named_steps = {"scaler": FakeScaler(expected_columns)}
        self.positive_probability = float(positive_probability)

    def predict_proba(self, X):
        if list(X.columns) != self.named_steps["scaler"].colonnes_numeriques:
            raise AssertionError("Unexpected feature order received by fallback predictor.")
        return np.array([[1.0 - self.positive_probability, self.positive_probability]])


class FakePredictorWithoutProba:
    def __init__(self, expected_columns, prediction_value):
        self.feature_names_in_ = np.asarray(expected_columns, dtype=object)
        self.prediction_value = int(prediction_value)

    def predict(self, X):
        if list(X.columns) != list(self.feature_names_in_):
            raise AssertionError("Unexpected feature order received by predictor without proba.")
        return np.array([self.prediction_value])


class TestPredictAllSavedModels(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.real_model_dir = self.repo_root / "models"
        self.model_dir = self.repo_root / "tests" / f"_tmp_saved_model_inference_{uuid.uuid4().hex}"
        self.model_dir.mkdir(parents=True, exist_ok=False)
        self.feature_names = ["feat_a", "feat_b", "feat_c"]
        self.feature_values = [0, 0, 0]

        joblib.dump(
            {
                "diagnostic": "Diagnosis A",
                "pipe_inference": FakePredictorWithFeatureNames(
                    expected_columns=["feat_b", "feat_a"],
                    positive_probability=0.2,
                ),
                "Youden_threshold": 0.1,
            },
            self.model_dir / "diagnosis_a.joblib",
        )
        joblib.dump(
            {
                "diagnostic": "Diagnosis B",
                "pipe_inference": FakePredictorWithScalerFallback(
                    expected_columns=["feat_c", "feat_a"],
                    positive_probability=0.7,
                ),
            },
            self.model_dir / "diagnosis_b.joblib",
        )
        joblib.dump(
            {
                "diagnostic": "Diagnosis C",
                "pipe_train": FakePredictorWithoutProba(
                    expected_columns=["feat_a"],
                    prediction_value=1,
                ),
                "Youden_threshold": 0.3,
            },
            self.model_dir / "diagnosis_c.joblib",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.model_dir, ignore_errors=True)

    def test_returns_all_saved_models_for_zero_features(self) -> None:
        results = predict_all_saved_models(
            feature_values=self.feature_values,
            feature_names=self.feature_names,
            model_dir=self.model_dir,
        )

        self.assertEqual(set(results.keys()), {"Diagnosis A", "Diagnosis B", "Diagnosis C"})
        for result in results.values():
            self.assertIn("diagnostic", result)
            self.assertIn("model_path", result)
            self.assertIn("probability", result)
            self.assertIn("threshold", result)
            self.assertIn("prediction", result)

        self.assertAlmostEqual(results["Diagnosis A"]["probability"], 0.2)
        self.assertEqual(results["Diagnosis A"]["threshold"], 0.1)
        self.assertEqual(results["Diagnosis A"]["prediction"], 1)

        self.assertAlmostEqual(results["Diagnosis B"]["probability"], 0.7)
        self.assertEqual(results["Diagnosis B"]["threshold"], 0.5)
        self.assertEqual(results["Diagnosis B"]["prediction"], 1)

        self.assertIsNone(results["Diagnosis C"]["probability"])
        self.assertEqual(results["Diagnosis C"]["threshold"], 0.3)
        self.assertEqual(results["Diagnosis C"]["prediction"], 1)

    def test_feature_names_are_inferred_when_omitted(self) -> None:
        results = predict_all_saved_models(
            feature_values=self.feature_values,
            model_dir=self.model_dir,
        )

        self.assertEqual(set(results.keys()), {"Diagnosis A", "Diagnosis B", "Diagnosis C"})
        self.assertAlmostEqual(results["Diagnosis A"]["probability"], 0.2)
        self.assertAlmostEqual(results["Diagnosis B"]["probability"], 0.7)
        self.assertIsNone(results["Diagnosis C"]["probability"])

    def test_models_are_loaded_once_when_feature_names_are_inferred(self) -> None:
        with mock.patch("utils.saved_model_inference.joblib.load", wraps=joblib.load) as mocked_load:
            predict_all_saved_models(
                feature_values=self.feature_values,
                model_dir=self.model_dir,
            )

        self.assertEqual(mocked_load.call_count, 3)

    def test_mismatched_feature_lengths_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "meme longueur"):
            predict_all_saved_models(
                feature_values=[0, 0],
                feature_names=self.feature_names,
                model_dir=self.model_dir,
            )

    def test_missing_required_columns_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "Diagnosis B.*feat_c"):
            predict_all_saved_models(
                feature_values=[0, 0],
                feature_names=["feat_a", "feat_b"],
                model_dir=self.model_dir,
            )

    def test_default_threshold_is_used_when_missing(self) -> None:
        results = predict_all_saved_models(
            feature_values=self.feature_values,
            feature_names=self.feature_names,
            model_dir=self.model_dir,
        )

        self.assertEqual(results["Diagnosis B"]["threshold"], 0.5)

    def test_real_use_case(self) -> None:
        self.assertTrue(
            self.real_model_dir.exists(),
            f"Missing real model directory: {self.real_model_dir}",
        )

        feature_values = [
            np.float64(0.9830508474576272), np.float64(0.0),
            np.float64(0.9830508474576272), np.float64(0.0), np.float64(0.0),
            np.int64(0), np.int64(0), np.int64(0), np.float64(0.0),
            np.float64(1.0), 0, np.float64(1.0), np.float64(3.0),
            np.float64(0.0), np.float64(0.98), np.float64(0.0),
            np.float64(2.0), np.float64(0.0), np.float64(0.0), np.float64(1.0),
            np.float64(1.0), np.float64(0.0), np.float64(0.0), np.float64(0.0),
            np.float64(1.0), np.float64(1.0), np.float64(1.0), False,
            np.False_, np.False_, np.True_, np.float64(156.1343154),
            np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0),
            np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0),
            np.float64(0.5531914893617021), np.float64(0.0), np.float64(0.0),
            np.float64(1.0), np.float64(0.0), np.float64(0.0), np.float64(0.0),
            np.float64(0.0), np.float64(0.0), np.float64(0.0),
            np.float64(0.125), np.float64(1.0), np.float64(0.6),
            np.float64(0.0), np.float64(0.0), np.float64(1.0), np.float64(0.0),
        ]
        feature_names = [
            "Time H-ICU", "TIME SYMPTOMES-ICU", "Time  DG-ICU",
            "HSCT_BMT_Allograft", "HSCT_BMT_Autograft", "Sys_dis", "Solid_tumor",
            "Organ_transpl", "Immuno_drugs", "Steroids_YN",
            "Prophylaxis_antifungal", "Prophylaxis_viral", "Charlson_index",
            "Hemoptysis", "SaO2", "Neutropenie", "Quad_no", "Septal_line",
            "Halo_sign", "Disease_status_remission", "Hypotension", "Pleural_eff",
            "Excavation", "Lymph_bulky", "GGO", "Nodules_any", "Alveolar", "GvHD",
            "Ibr_Flu_Met", "Sex", "Chemotherapy", "PaO2/FiO2 VALUE VALUE",
            "SOFA_Nervous", "Tar_ther", "Immunotherapy", "Carttcells",
            "Prophylaxis_bacterial", "Vaccins#Flu", "Vaccins#COVID",
            "Vaccins#Other", "Disease_status_inaugural", "Disease_status_evolutive",
            "Age_scaled", "Hem_mal_AML", "Hem_mal_ALL",
            "Hem_mal_Non_hodgkin_lymphoma", "Hem_mal_myeloma",
            "Hem_mal_hodgkin_lymphoma", "Hem_mal_CLL", "Hem_mal_CML", "Hem_mal_MDS",
            "Hem_mal_other", "SOFA_scaled", "Resp_severity", "Temp_gravité",
            "Leukostase", "Indication_prophy_anti_fun",
            "Indication_prophy_pneumocystose_taken",
            "Indication_prophy_pneumocystose_not_taken",
        ]

        try:
            results = predict_all_saved_models(
                feature_values=feature_values,
                feature_names=feature_names,
                model_dir=self.real_model_dir,
            )
        except Exception as exc:
            print("\nDetailed failure in test_real_use_case")
            print(f"Exception type: {type(exc).__name__}")
            print(f"Exception message: {exc}")
            print(f"Model directory: {self.real_model_dir}")
            print(f"Feature count: {len(feature_values)}")
            print(f"Feature names count: {len(feature_names)}")
            print(f"First 10 feature names: {feature_names[:10]}")
            traceback.print_exc()
            raise AssertionError(
                "predict_all_saved_models failed in test_real_use_case. "
                f"model_dir={self.real_model_dir}, "
                f"feature_count={len(feature_values)}, "
                f"feature_name_count={len(feature_names)}. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        self.assertTrue(all(isinstance(key, str) for key in results))
        self.assertTrue(all(isinstance(value, dict) for value in results.values()))
        pprint(results)


if __name__ == "__main__":
    unittest.main(verbosity=2, buffer=False)
