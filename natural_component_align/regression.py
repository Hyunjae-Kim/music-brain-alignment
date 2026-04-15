from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgWarning
from sklearn.linear_model import RidgeCV


DEFAULT_ALPHAS = np.power(10.0, np.arange(-50, 50, dtype=np.float64))[::-1]


@dataclass(frozen=True)
class Split:
    train_mask: np.ndarray
    test_mask: np.ndarray


@dataclass(frozen=True)
class ComponentRegressionResult:
    component_names: tuple[str, ...]
    split_r2: np.ndarray
    split_train_r2: np.ndarray
    split_alpha: np.ndarray
    splits: tuple[Split, ...]
    source_dim: int


def make_splits(
    n_stimuli: int,
    *,
    n_splits: int = 10,
    n_train: int = 83,
    n_test: int = 82,
    seed: int = 0,
) -> tuple[Split, ...]:
    if n_train <= 0 or n_test <= 0:
        raise ValueError("n_train and n_test must both be positive.")
    if n_train + n_test > n_stimuli:
        raise ValueError(
            f"Requested n_train + n_test = {n_train + n_test}, but only {n_stimuli} stimuli are available."
        )

    rng = np.random.RandomState(seed)
    splits: list[Split] = []
    for _ in range(n_splits):
        train_indices = rng.choice(n_stimuli, size=n_train, replace=False)
        remaining = np.setdiff1d(np.arange(n_stimuli), train_indices, assume_unique=False)
        test_indices = rng.choice(remaining, size=n_test, replace=False)
        train_mask = np.zeros(n_stimuli, dtype=bool)
        test_mask = np.zeros(n_stimuli, dtype=bool)
        train_mask[train_indices] = True
        test_mask[test_indices] = True
        splits.append(Split(train_mask=train_mask, test_mask=test_mask))
    return tuple(splits)


def _columnwise_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_centered = a - a.mean(axis=0, keepdims=True)
    b_centered = b - b.mean(axis=0, keepdims=True)
    a_norm = np.sqrt(np.sum(a_centered**2, axis=0))
    b_norm = np.sqrt(np.sum(b_centered**2, axis=0))
    denom = a_norm * b_norm
    corr = np.zeros(a.shape[1], dtype=np.float64)
    nonzero = denom > 0
    corr[nonzero] = np.sum(a_centered[:, nonzero] * b_centered[:, nonzero], axis=0) / denom[nonzero]
    return corr


def _ridge_loocv_predictions(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    *,
    alphas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_train_centered = x_train - x_mean
    x_test_centered = x_test - x_mean

    y_mean = y_train.mean(axis=0, keepdims=True)
    y_train_centered = y_train - y_mean
    model = RidgeCV(alphas=alphas, fit_intercept=True, alpha_per_target=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=LinAlgWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        model.fit(x_train_centered, y_train_centered)

    y_pred_train = np.asarray(model.predict(x_train_centered), dtype=np.float64)
    y_pred_test = np.asarray(model.predict(x_test_centered), dtype=np.float64)
    selected_alphas = np.asarray(model.alpha_, dtype=np.float64).reshape(-1)
    return y_train_centered, y_pred_train, y_pred_test, selected_alphas


def run_component_regression(
    source_features: np.ndarray,
    component_responses: np.ndarray,
    component_names: tuple[str, ...],
    *,
    alphas: np.ndarray | None = None,
    n_splits: int = 10,
    n_train: int = 83,
    n_test: int = 82,
    seed: int = 0,
) -> ComponentRegressionResult:
    source_features = np.asarray(source_features, dtype=np.float64)
    component_responses = np.asarray(component_responses, dtype=np.float64)
    if source_features.ndim != 2:
        raise ValueError("source_features must have shape [stimuli, features].")
    if component_responses.ndim != 2:
        raise ValueError("component_responses must have shape [stimuli, components].")
    if source_features.shape[0] != component_responses.shape[0]:
        raise ValueError("source_features and component_responses must share the stimulus dimension.")
    if component_responses.shape[1] != len(component_names):
        raise ValueError("component_names must align with component_responses.")

    alphas = DEFAULT_ALPHAS if alphas is None else np.asarray(alphas, dtype=np.float64)
    splits = make_splits(
        source_features.shape[0],
        n_splits=n_splits,
        n_train=n_train,
        n_test=n_test,
        seed=seed,
    )

    n_components = component_responses.shape[1]
    split_r2 = np.zeros((n_components, len(splits)), dtype=np.float32)
    split_train_r2 = np.zeros((n_components, len(splits)), dtype=np.float32)
    split_alpha = np.zeros((n_components, len(splits)), dtype=np.float64)

    for split_index, split in enumerate(splits):
        x_train = source_features[split.train_mask]
        x_test = source_features[split.test_mask]
        y_train = component_responses[split.train_mask]
        y_test = component_responses[split.test_mask]

        y_train_centered, y_pred_train, y_pred_test, selected_alphas = _ridge_loocv_predictions(
            x_train,
            x_test,
            y_train,
            alphas=alphas,
        )
        y_test_centered = y_test - y_train.mean(axis=0, keepdims=True)

        r_train = np.clip(_columnwise_correlation(y_pred_train, y_train_centered), a_min=0.0, a_max=None)
        r_test = np.clip(_columnwise_correlation(y_pred_test, y_test_centered), a_min=0.0, a_max=None)
        split_train_r2[:, split_index] = np.square(r_train).astype(np.float32)
        split_r2[:, split_index] = np.square(r_test).astype(np.float32)
        split_alpha[:, split_index] = selected_alphas

    return ComponentRegressionResult(
        component_names=component_names,
        split_r2=split_r2,
        split_train_r2=split_train_r2,
        split_alpha=split_alpha,
        splits=splits,
        source_dim=int(source_features.shape[1]),
    )


def regression_result_to_frame(result: ComponentRegressionResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "component_name": list(result.component_names),
            "mean_r2": np.mean(result.split_r2, axis=1),
            "median_r2": np.median(result.split_r2, axis=1),
            "mean_train_r2": np.mean(result.split_train_r2, axis=1),
            "median_train_r2": np.median(result.split_train_r2, axis=1),
            "mean_alpha": np.mean(result.split_alpha, axis=1),
            "median_alpha": np.median(result.split_alpha, axis=1),
            "n_splits": result.split_r2.shape[1],
            "source_dim": result.source_dim,
        }
    )
