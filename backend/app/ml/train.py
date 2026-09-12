"""Offline training pipeline.

    Scenario generator -> synthetic environment -> receiver simulator
      -> historical observation dataset -> feature engineering
      -> train / validation / test -> model training
      -> evaluation + calibration -> model registry -> production model

XGBoost is the primary model.  Random Forest is trained under the identical
split and feature set so the choice of primary model rests on measurement.
Probabilities are calibrated on the validation split, because the mathematical
layer consumes them as a prior: a miscalibrated probability corrupts the Beta
prior even when its ranking is fine.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..config import Config
from .dataset import DATASET_VERSION, DatasetBundle
from .features import FEATURE_NAMES, FEATURE_VERSION
from .model_registry import ModelMetadata, ModelRegistry, next_model_id

MODEL_VERSION = "1.0"


@dataclass(slots=True)
class TrainingResult:
    kind: str
    model_id: str
    metrics: Dict[str, float]
    calibration: Dict[str, float]
    hyperparameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "model_id": self.model_id,
            "metrics": self.metrics,
            "calibration": self.calibration,
            "hyperparameters": self.hyperparameters,
        }


def evaluate(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    """Classification and probability-quality metrics."""
    pred = (proba >= 0.5).astype(int)
    y = np.asarray(y_true).astype(int)
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, proba)),
        "positive_rate": float(y.mean()),
    }
    if len(np.unique(y)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y, proba))
        metrics["log_loss"] = float(log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6)))
    else:  # pragma: no cover - degenerate split
        metrics["roc_auc"] = float("nan")
        metrics["log_loss"] = float("nan")
    return {k: round(v, 5) for k, v in metrics.items()}


def calibration_report(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """Expected/maximum calibration error plus the reliability curve."""
    y = np.asarray(y_true).astype(float)
    p = np.asarray(proba, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    mce = 0.0
    curve = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = float(p[mask].mean())
        freq = float(y[mask].mean())
        weight = float(mask.sum()) / len(p)
        gap = abs(conf - freq)
        ece += weight * gap
        mce = max(mce, gap)
        curve.append({"bin": b, "confidence": round(conf, 4),
                      "frequency": round(freq, 4), "count": int(mask.sum())})
    return {
        "expected_calibration_error": round(float(ece), 5),
        "max_calibration_error": round(float(mce), 5),
        "reliability_curve": curve,
    }


def _fit_xgboost(X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                 params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    from xgboost import XGBClassifier

    positive = max(1.0, float(y.sum()))
    scale_pos_weight = float(len(y) - positive) / positive
    hyper = {
        "n_estimators": int(params.get("n_estimators", 400)),
        "max_depth": int(params.get("max_depth", 6)),
        "learning_rate": float(params.get("learning_rate", 0.06)),
        "subsample": float(params.get("subsample", 0.9)),
        "colsample_bytree": float(params.get("colsample_bytree", 0.9)),
        "min_child_weight": float(params.get("min_child_weight", 2)),
        "reg_lambda": float(params.get("reg_lambda", 1.5)),
        "n_jobs": int(params.get("n_jobs", 0)),
        "scale_pos_weight": round(scale_pos_weight, 4),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": 17,
        "early_stopping_rounds": 30,
    }
    model = XGBClassifier(**hyper)
    model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
    return model, hyper


def _fit_random_forest(X: np.ndarray, y: np.ndarray, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    hyper = {
        "n_estimators": int(params.get("n_estimators", 300)),
        "max_depth": int(params.get("max_depth", 14)),
        "min_samples_leaf": int(params.get("min_samples_leaf", 5)),
        "n_jobs": int(params.get("n_jobs", -1)),
        "class_weight": "balanced_subsample",
        "random_state": 17,
    }
    model = RandomForestClassifier(**hyper)
    model.fit(X, y)
    return model, hyper


def _prefit_calibrator(model: Any) -> CalibratedClassifierCV:
    """Isotonic calibrator over an already-fitted estimator.

    scikit-learn replaced ``cv="prefit"`` with ``FrozenEstimator``; use whichever
    the installed version provides so the pipeline survives either.
    """
    try:
        from sklearn.frozen import FrozenEstimator

        return CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    except ImportError:  # pragma: no cover - older scikit-learn
        return CalibratedClassifierCV(model, method="isotonic", cv="prefit")


def train_model(
    kind: str,
    bundle: DatasetBundle,
    cfg: Config,
    registry: Optional[ModelRegistry] = None,
    calibrate: bool = True,
    promote: bool = True,
) -> TrainingResult:
    """Train, evaluate, calibrate and register one model."""
    registry = registry or ModelRegistry()
    splits = bundle.split_by_group(seed=int(cfg.get_path("environment.seed", 1234)) % 10_000)
    X_tr, y_tr = splits["train"]
    X_val, y_val = splits["val"]
    X_te, y_te = splits["test"]

    if kind == "xgboost":
        model, hyper = _fit_xgboost(X_tr, y_tr, X_val, y_val,
                                    (cfg.get_path("ml.xgboost", {}) or {}))
    elif kind == "random_forest":
        model, hyper = _fit_random_forest(X_tr, y_tr,
                                          (cfg.get_path("ml.random_forest", {}) or {}))
    else:
        raise ValueError(f"unsupported model kind {kind!r}")

    raw_test = model.predict_proba(X_te)[:, 1]
    pre_calibration = calibration_report(y_te, raw_test)

    final_model = model
    if calibrate and len(np.unique(y_val)) > 1:
        calibrated = _prefit_calibrator(model)
        calibrated.fit(X_val, y_val)
        cal_test = calibrated.predict_proba(X_te)[:, 1]
        if calibration_report(y_te, cal_test)["expected_calibration_error"] <= \
                pre_calibration["expected_calibration_error"]:
            final_model = calibrated

    proba_test = final_model.predict_proba(X_te)[:, 1]
    metrics = evaluate(y_te, proba_test)
    calibration = calibration_report(y_te, proba_test)
    metrics["validation_roc_auc"] = evaluate(y_val, final_model.predict_proba(X_val)[:, 1]).get(
        "roc_auc", float("nan"))

    hyper_dict = dict(hyper)
    if hasattr(hyper, "to_dict"):  # pragma: no cover - Config passthrough
        hyper_dict = hyper.to_dict()
    hyper_dict["calibrated"] = final_model is not model

    metadata = ModelMetadata(
        model_id=next_model_id(kind),
        kind=kind,
        version=MODEL_VERSION,
        feature_version=FEATURE_VERSION,
        feature_names=list(FEATURE_NAMES),
        dataset_version=str(bundle.metadata.get("dataset_version", DATASET_VERSION)),
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        horizon=int(bundle.metadata.get("horizon", 1)),
        label_source=str(bundle.metadata.get("label_source", "truth")),
        scenario_coverage=list(bundle.metadata.get("scenarios", [])),
        hyperparameters=hyper_dict,
        validation_metrics=metrics,
        calibration_metrics={k: v for k, v in calibration.items() if k != "reliability_curve"},
        n_train_rows=int(len(y_tr)),
        n_test_rows=int(len(y_te)),
        notes="Trained on synthetic simulation traces. Labels never enter the feature set.",
    )
    registry.save(final_model, metadata)
    if promote:
        registry.promote(metadata.model_id)

    return TrainingResult(
        kind=kind,
        model_id=metadata.model_id,
        metrics=metrics,
        calibration=calibration,
        hyperparameters=hyper_dict,
    )


def train_all(bundle: DatasetBundle, cfg: Config,
              registry: Optional[ModelRegistry] = None) -> Dict[str, TrainingResult]:
    """Train the primary model and its baseline under the same methodology."""
    registry = registry or ModelRegistry()
    results = {
        "xgboost": train_model("xgboost", bundle, cfg, registry),
        "random_forest": train_model("random_forest", bundle, cfg, registry),
    }
    return results


def feature_importance(model: Any, top_n: int = 15) -> Dict[str, float]:
    """Feature importances, unwrapping a calibration wrapper when present."""
    estimator = model
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        inner = model.calibrated_classifiers_[0]
        estimator = getattr(inner, "estimator", getattr(inner, "base_estimator", model))
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return {}
    order = np.argsort(importances)[::-1][:top_n]
    return {FEATURE_NAMES[i]: round(float(importances[i]), 5) for i in order}


__all__ = ["train_model", "train_all", "evaluate", "calibration_report",
           "feature_importance", "TrainingResult", "MODEL_VERSION"]
