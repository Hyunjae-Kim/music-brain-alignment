from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ComponentRSAResult:
    component_names: tuple[str, ...]
    split_scores: np.ndarray
    model_rdm: np.ndarray
    component_rdms: np.ndarray


def model_rdm(feature_matrix: np.ndarray) -> np.ndarray:
    feature_matrix = np.asarray(feature_matrix, dtype=np.float64)
    corr = np.corrcoef(feature_matrix)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return 1.0 - corr


def component_rdm(component_values: np.ndarray) -> np.ndarray:
    component_values = np.asarray(component_values, dtype=np.float64).reshape(-1)
    return np.abs(component_values[:, None] - component_values[None, :])


def _spearman_upper(matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
    upper = np.triu_indices_from(matrix_a, k=1)
    statistic = stats.spearmanr(matrix_a[upper], matrix_b[upper]).statistic
    return float(0.0 if np.isnan(statistic) else statistic)


def _maybe_standardize_features(
    train_data: np.ndarray,
    test_data: np.ndarray | None,
    *,
    mean_subtract: bool,
    with_std: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    if not mean_subtract:
        return train_data, test_data
    scaler = StandardScaler(with_std=with_std).fit(train_data)
    transformed_test = None if test_data is None else scaler.transform(test_data)
    return scaler.transform(train_data), transformed_test


def _maybe_standardize_targets(
    train_values: np.ndarray,
    test_values: np.ndarray | None,
    *,
    mean_subtract: bool,
    with_std: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    train_values = np.asarray(train_values, dtype=np.float64).reshape(-1, 1)
    transformed_train, transformed_test = _maybe_standardize_features(
        train_values,
        None if test_values is None else np.asarray(test_values, dtype=np.float64).reshape(-1, 1),
        mean_subtract=mean_subtract,
        with_std=with_std,
    )
    return transformed_train[:, 0], None if transformed_test is None else transformed_test[:, 0]


def run_component_rsa(
    model_features: np.ndarray,
    component_responses: np.ndarray,
    component_names: tuple[str, ...],
    *,
    n_splits: int = 10,
    n_train: int = 83,
    n_test: int = 82,
    random_seed: int = 0,
    mean_subtract: bool = True,
    with_std: bool = True,
) -> ComponentRSAResult:
    from .regression import make_splits

    model_features = np.asarray(model_features, dtype=np.float64)
    component_responses = np.asarray(component_responses, dtype=np.float64)
    if model_features.shape[0] != component_responses.shape[0]:
        raise ValueError("model_features and component_responses must align on stimuli.")
    if component_responses.shape[1] != len(component_names):
        raise ValueError("component_names must align with component_responses.")

    all_model_features = model_features
    if mean_subtract:
        all_model_features, _ = _maybe_standardize_features(
            model_features,
            None,
            mean_subtract=mean_subtract,
            with_std=with_std,
        )

    full_model_rdm = model_rdm(all_model_features)
    full_component_rdms = []
    for component_index in range(component_responses.shape[1]):
        component_values = component_responses[:, component_index]
        if mean_subtract:
            component_values, _ = _maybe_standardize_targets(
                component_values,
                None,
                mean_subtract=mean_subtract,
                with_std=with_std,
            )
        full_component_rdms.append(component_rdm(component_values))

    splits = make_splits(
        model_features.shape[0],
        n_splits=n_splits,
        n_train=n_train,
        n_test=n_test,
        seed=random_seed,
    )
    split_scores = np.zeros((component_responses.shape[1], len(splits)), dtype=np.float32)

    for split_index, split in enumerate(splits):
        train_model = model_features[split.train_mask]
        test_model = model_features[split.test_mask]
        _, test_model = _maybe_standardize_features(
            train_model,
            test_model,
            mean_subtract=mean_subtract,
            with_std=with_std,
        )
        if test_model is None:
            raise RuntimeError("Unexpected None test_model after standardization.")
        test_model_rdm = model_rdm(test_model)

        for component_index in range(component_responses.shape[1]):
            train_component = component_responses[split.train_mask, component_index]
            test_component = component_responses[split.test_mask, component_index]
            _, test_component = _maybe_standardize_targets(
                train_component,
                test_component,
                mean_subtract=mean_subtract,
                with_std=with_std,
            )
            if test_component is None:
                raise RuntimeError("Unexpected None test_component after standardization.")
            test_component_rdm = component_rdm(test_component)
            split_scores[component_index, split_index] = _spearman_upper(test_model_rdm, test_component_rdm)

    return ComponentRSAResult(
        component_names=component_names,
        split_scores=split_scores,
        model_rdm=full_model_rdm,
        component_rdms=np.asarray(full_component_rdms, dtype=np.float64),
    )


def rsa_result_to_frame(result: ComponentRSAResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "component_name": list(result.component_names),
            "mean_spearman": np.mean(result.split_scores, axis=1),
            "median_spearman": np.median(result.split_scores, axis=1),
            "n_splits": result.split_scores.shape[1],
        }
    )
