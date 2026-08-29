from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error, r2_score


def score_to_grade(scores: np.ndarray) -> np.ndarray:
    grades = np.floor(np.clip(scores, 0.0, 0.999999) / 0.2).astype(np.int32) + 1
    grades = np.clip(grades, 1, 5)
    grades[scores >= 1.0] = 5
    return grades


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def compute_grade_metrics(y_true_grade: np.ndarray, y_pred_score: np.ndarray) -> dict[str, float]:
    y_pred_grade = score_to_grade(y_pred_score)
    oa = float((y_pred_grade == y_true_grade).mean())
    kappa = float(cohen_kappa_score(y_true_grade, y_pred_grade, labels=[1, 2, 3, 4, 5]))
    return {"oa": oa, "kappa": kappa}


def compute_all_metrics(
    y_true_score: np.ndarray,
    y_pred_score: np.ndarray,
    y_true_grade: np.ndarray,
) -> dict[str, float]:
    metrics = compute_regression_metrics(y_true_score, y_pred_score)
    metrics.update(compute_grade_metrics(y_true_grade, y_pred_score))
    return metrics
