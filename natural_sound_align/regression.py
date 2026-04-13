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
class RegressionResult:
    split_r2: np.ndarray
    split_corrected_r2: np.ndarray
    split_train_r2: np.ndarray
    split_alpha: np.ndarray
    voxel_ids: np.ndarray
    splits: tuple[Split, ...]
    source_dim: int


@dataclass(frozen=True)
class SubjectAggregationResult:
    subject_scores: pd.DataFrame
    iteration_scores: pd.DataFrame
    value_column: str


def make_splits(
    n_stimuli: int,
    *,
    n_splits: int = 10,
    n_train: int = 83,
    n_test: int = 82,
    seed: int = 0,
) -> tuple[Split, ...]:
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


def _columnwise_correlation(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a_centered = a - a.mean(axis=0, keepdims=True)
    b_centered = b - b.mean(axis=0, keepdims=True)
    a_norm = np.sqrt(np.sum(a_centered**2, axis=0))
    b_norm = np.sqrt(np.sum(b_centered**2, axis=0))
    denom = a_norm * b_norm
    corr = np.zeros(a.shape[1], dtype=np.float64)
    nonzero = denom > 0
    corr[nonzero] = np.sum(a_centered[:, nonzero] * b_centered[:, nonzero], axis=0) / denom[nonzero]
    return corr, a_norm == 0


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


def run_noise_corrected_regression(
    source_features: np.ndarray,
    voxel_data: np.ndarray,
    voxel_ids: np.ndarray,
    *,
    alphas: np.ndarray | None = None,
    n_splits: int = 10,
    n_train: int = 83,
    n_test: int = 82,
    seed: int = 0,
) -> RegressionResult:
    source_features = np.asarray(source_features, dtype=np.float64)
    voxel_data = np.asarray(voxel_data, dtype=np.float64)
    if voxel_data.ndim != 3 or voxel_data.shape[2] != 3:
        raise ValueError("voxel_data must have shape [stimuli, voxels, 3 repetitions].")

    alphas = DEFAULT_ALPHAS if alphas is None else np.asarray(alphas, dtype=np.float64)
    splits = make_splits(
        source_features.shape[0],
        n_splits=n_splits,
        n_train=n_train,
        n_test=n_test,
        seed=seed,
    )

    n_voxels = voxel_data.shape[1]
    split_r2 = np.zeros((n_voxels, len(splits)), dtype=np.float32)
    split_corrected_r2 = np.zeros((n_voxels, len(splits)), dtype=np.float32)
    split_train_r2 = np.zeros((n_voxels, len(splits)), dtype=np.float32)
    split_alpha = np.zeros((n_voxels, len(splits)), dtype=np.float64)

    mean_targets = voxel_data.mean(axis=2)

    for split_index, split in enumerate(splits):
        x_train = source_features[split.train_mask]
        x_test = source_features[split.test_mask]

        y_mean_train, mean_pred_train, mean_pred_test, selected_alphas = _ridge_loocv_predictions(
            x_train,
            x_test,
            mean_targets[split.train_mask],
            alphas=alphas,
        )
        y_mean_test = mean_targets[split.test_mask] - mean_targets[split.train_mask].mean(axis=0, keepdims=True)

        r_train, _ = _columnwise_correlation(mean_pred_train, y_mean_train)
        r_train = np.clip(r_train, a_min=0.0, a_max=None)
        split_train_r2[:, split_index] = np.square(r_train).astype(np.float32)

        r_test, _ = _columnwise_correlation(mean_pred_test, y_mean_test)
        r_test = np.clip(r_test, a_min=0.0, a_max=None)
        r2_mean = np.square(r_test)
        split_r2[:, split_index] = r2_mean.astype(np.float32)
        split_alpha[:, split_index] = selected_alphas

        rep_predictions: list[np.ndarray] = []
        for rep_index in range(voxel_data.shape[2]):
            _, _, rep_pred_test, _ = _ridge_loocv_predictions(
                x_train,
                x_test,
                voxel_data[split.train_mask, :, rep_index],
                alphas=alphas,
            )
            rep_predictions.append(rep_pred_test)

        observed_pairwise = []
        predicted_pairwise = []
        for rep_a, rep_b in ((0, 1), (1, 2), (0, 2)):
            observed_corr, _ = _columnwise_correlation(
                voxel_data[split.train_mask, :, rep_a],
                voxel_data[split.train_mask, :, rep_b],
            )
            predicted_corr, _ = _columnwise_correlation(
                rep_predictions[rep_a],
                rep_predictions[rep_b],
            )
            observed_pairwise.append(observed_corr)
            predicted_pairwise.append(predicted_corr)

        rv_med = np.nanmedian(np.stack(observed_pairwise, axis=0), axis=0)
        rv_hat_med = np.nanmedian(np.stack(predicted_pairwise, axis=0), axis=0)
        rv = np.maximum((3.0 * rv_med) / (1.0 + 2.0 * rv_med), 0.182)
        rv_hat = np.maximum((3.0 * rv_hat_med) / (1.0 + 2.0 * rv_hat_med), 0.183)
        denom = rv * rv_hat
        corrected = np.divide(r2_mean, denom, out=np.zeros_like(r2_mean), where=np.isfinite(denom) & (denom > 0))
        corrected = np.clip(corrected, a_min=0.0, a_max=None)
        split_corrected_r2[:, split_index] = corrected.astype(np.float32)

    return RegressionResult(
        split_r2=split_r2,
        split_corrected_r2=split_corrected_r2,
        split_train_r2=split_train_r2,
        split_alpha=split_alpha,
        voxel_ids=np.asarray(voxel_ids, dtype=int),
        splits=splits,
        source_dim=int(source_features.shape[1]),
    )


def summarize_regression_result(result: RegressionResult, *, model_name: str, representation_name: str) -> dict[str, object]:
    voxel_median_corrected_raw = np.median(result.split_corrected_r2, axis=1)
    voxel_median_corrected = np.clip(voxel_median_corrected_raw, a_min=0.0, a_max=1.0)
    voxel_median_r2 = np.median(result.split_r2, axis=1)
    voxel_median_train_r2 = np.median(result.split_train_r2, axis=1)

    return {
        "model_name": model_name,
        "representation_name": representation_name,
        "n_voxels": int(result.split_corrected_r2.shape[0]),
        "n_splits": int(result.split_corrected_r2.shape[1]),
        "source_dim": int(result.source_dim),
        "median_noise_corrected_r2": float(np.median(voxel_median_corrected)),
        "mean_noise_corrected_r2": float(np.mean(voxel_median_corrected)),
        "median_noise_corrected_r2_raw": float(np.median(voxel_median_corrected_raw)),
        "mean_noise_corrected_r2_raw": float(np.mean(voxel_median_corrected_raw)),
        "median_r2": float(np.median(voxel_median_r2)),
        "mean_r2": float(np.mean(voxel_median_r2)),
        "median_train_r2": float(np.median(voxel_median_train_r2)),
        "mean_alpha": float(np.mean(result.split_alpha)),
        "median_alpha": float(np.median(result.split_alpha)),
    }


def regression_result_to_frame(result: RegressionResult) -> pd.DataFrame:
    median_corrected_raw = np.median(result.split_corrected_r2, axis=1)
    return pd.DataFrame(
        {
            "voxel_id": result.voxel_ids,
            "median_r2": np.median(result.split_r2, axis=1),
            "median_noise_corrected_r2_raw": median_corrected_raw,
            "median_noise_corrected_r2": np.clip(median_corrected_raw, a_min=0.0, a_max=1.0),
            "median_train_r2": np.median(result.split_train_r2, axis=1),
            "median_alpha": np.median(result.split_alpha, axis=1),
            "mean_alpha": np.mean(result.split_alpha, axis=1),
        }
    )


def aggregate_subject_cv_splits(
    split_values: np.ndarray,
    voxel_metadata: pd.DataFrame,
    *,
    value_column: str,
    nit: int = 10,
    collapse_over_splits: str = "median",
    clip_upper: float | None = None,
) -> SubjectAggregationResult:
    split_values = np.asarray(split_values, dtype=np.float64)
    if split_values.ndim != 2 or split_values.shape[1] != 10:
        raise ValueError("split_values must have shape [voxels, 10].")
    if len(voxel_metadata) != split_values.shape[0]:
        raise ValueError("voxel_metadata must align with split_values by voxel.")
    if "subj_idx" not in voxel_metadata.columns:
        raise ValueError("voxel_metadata must include a subj_idx column.")

    subject_iterations: list[pd.DataFrame] = []

    for iteration in range(nit):
        np.random.seed(iteration)
        split_mask = np.zeros(split_values.shape, dtype=bool)
        for voxel_index in range(split_values.shape[0]):
            mask_row = np.tile([0, 1], reps=5)
            np.random.shuffle(mask_row)
            split_mask[voxel_index] = mask_row.astype(bool)

        readout_values = split_values.copy()
        readout_values[split_mask] = np.nan

        if collapse_over_splits == "median":
            collapsed = np.nanmedian(readout_values, axis=1)
        elif collapse_over_splits == "mean":
            collapsed = np.nanmean(readout_values, axis=1)
        else:
            raise ValueError("collapse_over_splits must be either 'median' or 'mean'.")

        if clip_upper is not None:
            collapsed = np.clip(collapsed, a_min=None, a_max=clip_upper)

        subject_frame = pd.DataFrame(
            {
                value_column: collapsed,
                "subj_idx": voxel_metadata["subj_idx"].to_numpy(),
                "nit": iteration,
            }
        )
        subject_grouped = subject_frame.groupby("subj_idx", sort=True)[[value_column]].median()
        subject_grouped["nit"] = iteration
        subject_iterations.append(subject_grouped.reset_index())

    iteration_scores = pd.concat(subject_iterations, ignore_index=True)
    grouped = iteration_scores.groupby("subj_idx", sort=True)[value_column]
    subject_scores = pd.DataFrame(
        {
            "subj_idx": grouped.mean().index,
            value_column: grouped.mean().to_numpy(),
            f"{value_column}_std_over_it": grouped.std(ddof=1).to_numpy(),
            f"{value_column}_sem_over_it": grouped.sem(ddof=1).to_numpy(),
        }
    )
    return SubjectAggregationResult(
        subject_scores=subject_scores,
        iteration_scores=iteration_scores,
        value_column=value_column,
    )


def summarize_subject_aggregation(
    aggregation: SubjectAggregationResult,
    *,
    model_name: str,
    representation_name: str,
) -> dict[str, object]:
    values = aggregation.subject_scores[aggregation.value_column].to_numpy(dtype=np.float64)
    return {
        "model_name": model_name,
        "representation_name": representation_name,
        "n_subjects": int(len(values)),
        "cv_subject_mean": float(np.mean(values)),
        "cv_subject_median": float(np.median(values)),
        "cv_subject_sem": float(pd.Series(values).sem(ddof=1)),
        "cv_subject_std": float(np.std(values, ddof=1)),
        "value_column": aggregation.value_column,
    }
