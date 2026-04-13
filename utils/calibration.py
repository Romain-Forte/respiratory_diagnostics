from __future__ import annotations

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def _as_binary_array(y: Any) -> np.ndarray:
    y_array = np.asarray(y).reshape(-1)
    unique = np.unique(y_array)
    if unique.size > 2:
        raise ValueError("La calibration binaire requiert une cible avec au plus 2 classes.")
    return y_array.astype(int)


class CalibratedPredictor(BaseEstimator, ClassifierMixin):
    """Wrap a fitted predictor and calibrate its binary scores with Platt scaling."""

    def __init__(
        self,
        predictor: Any,
        method: str = "sigmoid",
        threshold: float = 0.5,
        eps: float = 1e-6,
        show_calibration_fitness: bool = False,
    ):
        self.predictor = predictor
        self.method = method
        self.threshold = threshold
        self.eps = eps
        self.show_calibration_fitness = show_calibration_fitness

    def _normalized_method(self) -> str:
        method = str(self.method).strip().lower()
        if method == "platt":
            return "sigmoid"
        if method not in {"sigmoid", "isotonic"}:
            raise ValueError("`method` doit etre 'sigmoid', 'platt' ou 'isotonic'.")
        return method

    def _raw_scores(self, X: Any) -> np.ndarray:
        if hasattr(self.predictor, "decision_function"):
            scores = np.asarray(self.predictor.decision_function(X)).reshape(-1)
            return scores
        if not hasattr(self.predictor, "predict_proba"):
            raise AttributeError(
                "Le predictor doit exposer `decision_function` ou `predict_proba` pour etre calibre."
            )
        proba = np.asarray(self.predictor.predict_proba(X))
        if proba.ndim == 2 and proba.shape[1] >= 2:
            positive_scores = proba[:, -1]
        else:
            positive_scores = proba.reshape(-1)
        clipped = np.clip(positive_scores, self.eps, 1.0 - self.eps)
        return np.log(clipped / (1.0 - clipped))

    def _base_positive_proba(self, X: Any) -> np.ndarray:
        if hasattr(self.predictor, "predict_proba"):
            proba = np.asarray(self.predictor.predict_proba(X))
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return np.clip(proba[:, -1], 0.0, 1.0)
            return np.clip(proba.reshape(-1), 0.0, 1.0)

        raw_scores = self._raw_scores(X)
        return 1.0 / (1.0 + np.exp(-raw_scores))

    def _calibrated_positive_proba_from_scores(self, score_vector: np.ndarray) -> np.ndarray:
        if self.calibration_method_ == "isotonic":
            positive_proba = np.asarray(self.calibrator_.predict(score_vector.reshape(-1)))
        else:
            positive_proba = self.calibrator_.predict_proba(score_vector)[:, 1]
        return np.clip(positive_proba, 0.0, 1.0)

    def plot_calibration_curve(self, X: Any, y: Any, n_bins: int = 10):
        y_array = _as_binary_array(y)
        base_proba = self._base_positive_proba(X)
        calibrated_proba = self.predict_proba(X)[:, 1]

        frac_pos_base, mean_pred_base = calibration_curve(
            y_array,
            base_proba,
            n_bins=n_bins,
            strategy="uniform",
        )
        frac_pos_cal, mean_pred_cal = calibration_curve(
            y_array,
            calibrated_proba,
            n_bins=n_bins,
            strategy="uniform",
        )

        brier_base = brier_score_loss(y_array, base_proba)
        brier_cal = brier_score_loss(y_array, calibrated_proba)

        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Calibration parfaite")
        plt.plot(
            mean_pred_base,
            frac_pos_base,
            marker="o",
            linewidth=2,
            label=f"Avant calibration (Brier={brier_base:.3f})",
        )
        plt.plot(
            mean_pred_cal,
            frac_pos_cal,
            marker="o",
            linewidth=2,
            label=f"Apres calibration {self.calibration_method_} (Brier={brier_cal:.3f})",
        )
        plt.xlabel("Probabilite predite moyenne")
        plt.ylabel("Frequence observee")
        plt.title("Courbe de calibration")
        plt.legend(loc="best")
        plt.grid(False)
        plt.tight_layout()
        plt.show()

    def fit(self, X: Any, y: Any):
        calibration_method = self._normalized_method()
        y_array = _as_binary_array(y)
        if len(y_array) == 0:
            raise ValueError("Impossible de calibrer sur un jeu vide.")
        score_vector = self._raw_scores(X).reshape(-1, 1)
        if calibration_method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(score_vector.reshape(-1), y_array)
        else:
            calibrator = LogisticRegression(solver="lbfgs")
            calibrator.fit(score_vector, y_array)

        self.calibrator_ = calibrator
        self.calibration_method_ = calibration_method
        self.is_calibrated_ = True
        self.classes_ = np.array([0, 1], dtype=int)

        if hasattr(self.predictor, "feature_names_in_"):
            self.feature_names_in_ = self.predictor.feature_names_in_
        if hasattr(self.predictor, "n_features_in_"):
            self.n_features_in_ = self.predictor.n_features_in_

        if self.show_calibration_fitness:
            self.plot_calibration_curve(X, y_array)

        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "calibrator_"):
            return self.predictor.predict_proba(X)
        score_vector = self._raw_scores(X).reshape(-1, 1)
        positive_proba = self._calibrated_positive_proba_from_scores(score_vector)
        return np.column_stack([1.0 - positive_proba, positive_proba])

    def predict(self, X: Any, threshold: Optional[float] = None) -> np.ndarray:
        active_threshold = self.threshold if threshold is None else threshold
        y_proba = self.predict_proba(X)[:, 1]
        return (y_proba >= active_threshold).astype(int)

    def decision_function(self, X: Any) -> np.ndarray:
        return self._raw_scores(X)

    def get_calibration_info(self) -> dict[str, Any]:
        info = {
            "is_calibrated": bool(getattr(self, "is_calibrated_", False)),
            "method": getattr(self, "calibration_method_", None),
            "threshold": float(self.threshold),
        }
        if hasattr(self, "calibrator_"):
            if self.calibration_method_ == "isotonic":
                info["x_thresholds"] = np.asarray(self.calibrator_.X_thresholds_).tolist()
                info["y_thresholds"] = np.asarray(self.calibrator_.y_thresholds_).tolist()
            else:
                coef = np.asarray(self.calibrator_.coef_).reshape(-1)
                intercept = np.asarray(self.calibrator_.intercept_).reshape(-1)
                info["coef"] = coef.tolist()
                info["intercept"] = intercept.tolist()
        return info

    def __getattr__(self, item: str):
        return getattr(self.predictor, item)


def CalibratedModel(
    predictor: Any,
    X_calibration: Any = None,
    y_calibration: Any = None,
    method: str = "sigmoid",
    threshold: float = 0.5,
    show_calibration_fitness: bool = False,
) -> CalibratedPredictor:
    """Factory returning a predictor wrapper calibrated with sigmoid/platt or isotonic scaling."""

    model = CalibratedPredictor(
        predictor=predictor,
        method=method,
        threshold=threshold,
        show_calibration_fitness=show_calibration_fitness,
    )
    if X_calibration is not None and y_calibration is not None:
        model.fit(X_calibration, y_calibration)
    return model
