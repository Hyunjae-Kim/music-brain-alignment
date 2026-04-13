from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

from .data import category_order, category_sort_index


@dataclass(frozen=True)
class RSAResult:
    model_name: str
    representation_name: str
    scores_by_repetition: np.ndarray
    mean_spearman: float
    model_correlation_matrix: np.ndarray
    human_correlation_matrices: np.ndarray
    mean_human_correlation_matrix: np.ndarray
    participant_ids: np.ndarray
    split_scores: np.ndarray
    leave_one_out_neural_scores: np.ndarray


def correlation_matrix(feature_matrix: np.ndarray) -> np.ndarray:
    return np.corrcoef(np.asarray(feature_matrix, dtype=np.float64))


def spearman_rsa(matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
    upper = np.triu_indices_from(matrix_a, k=1)
    statistic = stats.spearmanr(matrix_a[upper], matrix_b[upper]).statistic
    return float(statistic)


def _train_test_splits(
    n_stimuli: int,
    *,
    n_splits: int,
    n_train: int,
    n_test: int,
    random_seed: int,
) -> list[dict[str, np.ndarray]]:
    rng = np.random.RandomState(random_seed)
    splits: list[dict[str, np.ndarray]] = []
    for _ in range(n_splits):
        train_indices = rng.choice(n_stimuli, size=n_train, replace=False)
        remaining = np.setdiff1d(np.arange(n_stimuli), train_indices, assume_unique=False)
        test_indices = rng.choice(remaining, size=n_test, replace=False) if n_test > 0 else np.array([], dtype=int)
        train_mask = np.zeros(n_stimuli, dtype=bool)
        test_mask = np.zeros(n_stimuli, dtype=bool)
        train_mask[train_indices] = True
        test_mask[test_indices] = True
        splits.append({"train": train_mask, "test": test_mask})
    return splits


def _maybe_standardize(
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


def run_rsa(
    model_features: np.ndarray,
    brain_voxel_data: np.ndarray,
    metadata: pd.DataFrame,
    *,
    voxel_metadata: pd.DataFrame,
    model_name: str,
    representation_name: str,
    n_splits: int = 10,
    n_train: int = 83,
    n_test: int = 82,
    random_seed: int = 0,
    mean_subtract: bool = False,
    with_std: bool = False,
    reorder_by_category: bool = True,
) -> RSAResult:
    model_features = np.asarray(model_features, dtype=np.float64)
    brain_voxel_data = np.asarray(brain_voxel_data, dtype=np.float64)
    if reorder_by_category:
        sound_index = category_sort_index(metadata)
        split_model_features = model_features[sound_index]
        split_brain_voxel_data = brain_voxel_data[sound_index]
    else:
        split_model_features = model_features
        split_brain_voxel_data = brain_voxel_data

    all_splits = _train_test_splits(
        metadata.shape[0],
        n_splits=n_splits,
        n_train=n_train,
        n_test=n_test,
        random_seed=random_seed,
    )

    if mean_subtract:
        all_model_data, _ = _maybe_standardize(
            model_features,
            None,
            mean_subtract=mean_subtract,
            with_std=with_std,
        )
    else:
        all_model_data = model_features

    model_corr = correlation_matrix(all_model_data)

    split_model_test_corrs = []
    for split in all_splits:
        train_features = split_model_features[split["train"]]
        test_features = split_model_features[split["test"]]
        train_features, test_features = _maybe_standardize(
            train_features,
            test_features,
            mean_subtract=mean_subtract,
            with_std=with_std,
        )
        _ = correlation_matrix(train_features)
        split_model_test_corrs.append(correlation_matrix(test_features))

    participant_ids = np.sort(pd.Series(voxel_metadata["subj_idx"]).astype(int).unique())
    participant_corrs = []
    split_scores = np.full((participant_ids.shape[0], n_splits), np.nan, dtype=np.float64)
    test_corr_mats = np.empty((participant_ids.shape[0], n_splits), dtype=object)

    subj_ids = pd.Series(voxel_metadata["subj_idx"]).astype(int).to_numpy()
    for participant_position, participant_id in enumerate(participant_ids):
        participant_mask = subj_ids == participant_id
        participant_data = brain_voxel_data[:, participant_mask, :]
        split_participant_data = split_brain_voxel_data[:, participant_mask, :]

        all_data_avg = participant_data.mean(axis=2)
        all_data_avg, _ = _maybe_standardize(
            all_data_avg,
            None,
            mean_subtract=mean_subtract,
            with_std=with_std,
        )
        participant_corrs.append(correlation_matrix(all_data_avg))

        for split_index, split in enumerate(all_splits):
            train_avg = split_participant_data[split["train"]].mean(axis=2)
            test_avg = split_participant_data[split["test"]].mean(axis=2)
            train_avg, test_avg = _maybe_standardize(
                train_avg,
                test_avg,
                mean_subtract=mean_subtract,
                with_std=with_std,
            )
            test_corr = correlation_matrix(test_avg)
            test_corr_mats[participant_position, split_index] = test_corr
            split_scores[participant_position, split_index] = spearman_rsa(
                split_model_test_corrs[split_index],
                test_corr,
            )

    leave_one_out_scores = np.full((participant_ids.shape[0], n_splits), np.nan, dtype=np.float64)
    for participant_position in range(participant_ids.shape[0]):
        for split_index in range(n_splits):
            this_corr = test_corr_mats[participant_position, split_index]
            other_corrs = [
                test_corr_mats[other_position, split_index]
                for other_position in range(participant_ids.shape[0])
                if other_position != participant_position
            ]
            other_average = np.mean(np.stack(other_corrs, axis=0), axis=0)
            leave_one_out_scores[participant_position, split_index] = spearman_rsa(this_corr, other_average)

    participant_scores = np.median(split_scores, axis=1)
    human_corrs_array = np.stack(participant_corrs, axis=0)
    return RSAResult(
        model_name=model_name,
        representation_name=representation_name,
        scores_by_repetition=participant_scores,
        mean_spearman=float(np.mean(participant_scores)),
        model_correlation_matrix=model_corr,
        human_correlation_matrices=human_corrs_array,
        mean_human_correlation_matrix=np.mean(human_corrs_array, axis=0),
        participant_ids=participant_ids.astype(int),
        split_scores=split_scores,
        leave_one_out_neural_scores=leave_one_out_scores,
    )


def summarize_rsa_result(result: RSAResult) -> dict[str, object]:
    centered = result.scores_by_repetition - np.mean(result.scores_by_repetition)
    return {
        "model_name": result.model_name,
        "representation_name": result.representation_name,
        "n_repetitions": int(result.scores_by_repetition.shape[0]),
        "n_participants": int(result.scores_by_repetition.shape[0]),
        "n_splits": int(result.split_scores.shape[1]),
        "mean_spearman": float(result.mean_spearman),
        "median_spearman": float(np.median(result.scores_by_repetition)),
        "participant_sem": float(np.std(centered, ddof=1) / np.sqrt(len(centered))) if len(centered) > 1 else 0.0,
        "mean_leave_one_out_neural_spearman": float(np.mean(np.median(result.leave_one_out_neural_scores, axis=1))),
    }


def plot_model_vs_human_rdm(
    model_corr: np.ndarray,
    human_corr: np.ndarray,
    metadata: pd.DataFrame,
    *,
    model_name: str,
    reorder_by_category: bool = True,
    figure: plt.Figure | None = None,
    axes: np.ndarray | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    index = category_sort_index(metadata) if reorder_by_category else np.arange(len(metadata))
    model_rdm = 1.0 - model_corr[np.ix_(index, index)]
    human_rdm = 1.0 - human_corr[np.ix_(index, index)]
    vmax = float(np.nanmax([model_rdm.max(), human_rdm.max()]))

    if figure is None or axes is None:
        figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for ax, matrix, title in zip(
        axes,
        (human_rdm, model_rdm),
        ("Human fMRI RDM", f"{model_name} RDM"),
        strict=True,
    ):
        image = ax.imshow(matrix, cmap="Greys", vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

        if reorder_by_category:
            ordered_labels = metadata.iloc[index]["category_label"].tolist()
            positions = []
            running = 0
            for label in category_order(metadata.iloc[index].reset_index(drop=True)):
                count = sum(item == label for item in ordered_labels)
                running += count
                positions.append(running)
            for boundary in positions[:-1]:
                ax.axhline(boundary - 0.5, color="white", linewidth=0.5)
                ax.axvline(boundary - 0.5, color="white", linewidth=0.5)

    colorbar = figure.colorbar(image, ax=list(axes), fraction=0.046, pad=0.04)
    colorbar.set_label("1 - Pearson correlation")
    return figure, axes
