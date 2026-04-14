from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, mean_absolute_error, r2_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler


MODEL_REPRESENTATIONS = {
    "clap": "audio_embedding",
    "clap_music": "audio_embedding",
    "clap_music_speech": "audio_embedding",
    "mulan": "final",
    "mert": "final",
    "mert95": "final",
    "wav2vec2": "final",
    "musicgen_small": "final",
    "musicgen_medium": "final",
    "musicgen_large": "final",
    "musicgen_melody_large": "final",
}

TASK_SPECS = (
    {"dataset_dir": "gtzan", "task_dir": "genre_classification"},
    {"dataset_dir": "vocalset", "task_dir": "suffering_technique"} if False else {"dataset_dir": "vocalset", "task_dir": "singing_technique"},
    {"dataset_dir": "emo", "task_dir": "emotion_regression"},
    {"dataset_dir": "nsynthpitch5h", "task_dir": "pitch_estimation"},
    {"dataset_dir": "magnatagatune", "task_dir": "music_tagging"},
    {"dataset_dir": "giantsteps", "task_dir": "key_detection"},
    {"dataset_dir": "mtgtop50", "task_dir": "music_tagging"},
    {"dataset_dir": "mtgmoodtheme", "task_dir": "emotion_classification"},
    {"dataset_dir": "mtggenre", "task_dir": "genre_classification"},
    {"dataset_dir": "mtginstrument", "task_dir": "instrument_classification"},
)

LOGISTIC_C_GRID = (1.0,)
RIDGE_ALPHA_GRID = (1.0,)


@dataclass(frozen=True)
class EmbeddingPartition:
    partition_name: str
    embeddings: np.ndarray
    sample_id: np.ndarray
    target_json: np.ndarray
    label_text: np.ndarray
    split: np.ndarray
    fold: np.ndarray
    dataset_name: str
    task_name: str
    target_type: str
    label_vocabulary: tuple[str, ...]
    target_names: tuple[str, ...]
    model_name: str
    representation_name: str


def _safe_item(array: np.ndarray) -> str:
    if array.ndim == 0:
        return str(array.item())
    return str(array[0])


def load_partition(path: str | Path) -> EmbeddingPartition:
    path = Path(path)
    payload = np.load(path, allow_pickle=True)
    return EmbeddingPartition(
        partition_name=path.stem,
        embeddings=np.asarray(payload["embeddings"], dtype=np.float32),
        sample_id=np.asarray(payload["sample_id"]).astype(str),
        target_json=np.asarray(payload["target_json"]).astype(str),
        label_text=np.asarray(payload["label_text"]).astype(str),
        split=np.asarray(payload["split"]).astype(str),
        fold=np.asarray(payload["fold"]).astype(str),
        dataset_name=_safe_item(payload["dataset_name"]),
        task_name=_safe_item(payload["task_name"]),
        target_type=_safe_item(payload["target_type"]),
        label_vocabulary=tuple(np.asarray(payload["label_vocabulary"], dtype=object).tolist()),
        target_names=tuple(np.asarray(payload["target_names"], dtype=object).tolist()),
        model_name=_safe_item(payload["model_name"]),
        representation_name=_safe_item(payload["representation_name"]),
    )


def load_embedding_bundle(representation_dir: str | Path) -> dict[str, EmbeddingPartition]:
    representation_dir = Path(representation_dir)
    return {
        partition_path.stem: load_partition(partition_path)
        for partition_path in sorted(representation_dir.glob("*.npz"))
    }


def parse_target_array(target_json: np.ndarray) -> list[Any]:
    return [json.loads(value) for value in target_json.tolist()]


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(pearsonr(x, y).statistic)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(spearmanr(x, y).statistic)


def _metric_safe_name(name: str) -> str:
    text = str(name).strip().lower()
    for old, new in (("/", "_"), (" ", "_"), ("-", "_")):
        text = text.replace(old, new)
    return text


def _safe_macro_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    scores: list[float] = []
    for index in range(y_true.shape[1]):
        truth = y_true[:, index]
        if truth.max() == truth.min():
            continue
        scores.append(float(average_precision_score(truth, y_score[:, index])))
    return float(np.mean(scores)) if scores else float("nan")


def _is_integer_like(values: np.ndarray) -> bool:
    return np.allclose(values, np.rint(values))


def _single_label_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def _multi_label_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "average_precision_macro": _safe_macro_average_precision(y_true, y_score),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }


def _regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: tuple[str, ...] | list[str] | None = None,
) -> dict[str, float]:
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_pred = y_pred[:, None]

    pearsons = [_safe_pearson(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    spearmans = [_safe_spearman(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    r2_scores = [r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    mae_scores = [mean_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    metrics = {
        "mae_mean": float(np.mean(mae_scores)),
        "r2_mean": float(np.mean(r2_scores)),
        "pearson_mean": float(np.nanmean(pearsons)),
        "spearman_mean": float(np.nanmean(spearmans)),
    }
    names = list(target_names or [f"target_{index}" for index in range(y_true.shape[1])])
    if len(names) < y_true.shape[1]:
        names.extend(f"target_{index}" for index in range(len(names), y_true.shape[1]))
    for index in range(y_true.shape[1]):
        suffix = _metric_safe_name(names[index])
        metrics[f"pearson_{suffix}"] = float(pearsons[index])
        metrics[f"spearman_{suffix}"] = float(spearmans[index])
        metrics[f"r2_{suffix}"] = float(r2_scores[index])
        metrics[f"mae_{suffix}"] = float(mae_scores[index])

    if y_true.shape[1] == 1 and _is_integer_like(y_true[:, 0]):
        rounded = np.rint(y_pred[:, 0]).astype(int)
        metrics["rounded_accuracy"] = float(accuracy_score(y_true[:, 0].astype(int), rounded))
    return metrics


def _primary_metric_name(target_type: str, target_names: tuple[str, ...], y_train: np.ndarray) -> str:
    if target_type == "single_label_classification":
        return "accuracy"
    if target_type == "multi_label_classification":
        return "average_precision_macro"
    if y_train.ndim == 1 or (y_train.ndim == 2 and y_train.shape[1] == 1):
        flat = y_train.reshape(-1)
        if _is_integer_like(flat):
            return "rounded_accuracy"
    return "pearson_mean"


def _build_single_label_probe(c_value: float) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    model = LogisticRegression(
        C=c_value,
        max_iter=3000,
        solver="lbfgs",
    )
    return scaler, model


def _build_multi_label_probe(c_value: float) -> tuple[StandardScaler, OneVsRestClassifier]:
    scaler = StandardScaler()
    model = OneVsRestClassifier(
        LogisticRegression(
            C=c_value,
            max_iter=2000,
            solver="liblinear",
        )
    )
    return scaler, model


def _build_regression_probe(alpha: float) -> tuple[StandardScaler, Ridge]:
    scaler = StandardScaler()
    model = Ridge(alpha=alpha)
    return scaler, model


def _fit_and_score_single_label(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    primary_name = _primary_metric_name("single_label_classification", (), y_train)
    best_c = None
    best_score = -np.inf

    for c_value in LOGISTIC_C_GRID:
        scaler, model = _build_single_label_probe(c_value)
        x_train_scaled = scaler.fit_transform(x_train)
        x_valid_scaled = scaler.transform(x_valid)
        model.fit(x_train_scaled, y_train)
        valid_pred = model.predict(x_valid_scaled)
        valid_metrics = _single_label_metrics(y_valid, valid_pred)
        score = valid_metrics[primary_name]
        if score > best_score:
            best_score = score
            best_c = c_value

    scaler, model = _build_single_label_probe(float(best_c))
    x_train_valid = np.concatenate([x_train, x_valid], axis=0)
    y_train_valid = np.concatenate([y_train, y_valid], axis=0)
    x_train_valid_scaled = scaler.fit_transform(x_train_valid)
    x_test_scaled = scaler.transform(x_test)
    model.fit(x_train_valid_scaled, y_train_valid)
    test_pred = model.predict(x_test_scaled)
    return _single_label_metrics(y_test, test_pred), {"best_c": float(best_c), "selection_metric": primary_name}


def _fit_and_score_multi_label(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    primary_name = _primary_metric_name("multi_label_classification", (), y_train)
    best_c = None
    best_score = -np.inf

    for c_value in LOGISTIC_C_GRID:
        scaler, model = _build_multi_label_probe(c_value)
        x_train_scaled = scaler.fit_transform(x_train)
        x_valid_scaled = scaler.transform(x_valid)
        model.fit(x_train_scaled, y_train)
        valid_scores = model.predict_proba(x_valid_scaled)
        valid_pred = (valid_scores >= 0.5).astype(int)
        valid_metrics = _multi_label_metrics(y_valid, np.asarray(valid_scores), valid_pred)
        score = valid_metrics[primary_name]
        if score > best_score:
            best_score = score
            best_c = c_value

    scaler, model = _build_multi_label_probe(float(best_c))
    x_train_valid = np.concatenate([x_train, x_valid], axis=0)
    y_train_valid = np.concatenate([y_train, y_valid], axis=0)
    x_train_valid_scaled = scaler.fit_transform(x_train_valid)
    x_test_scaled = scaler.transform(x_test)
    model.fit(x_train_valid_scaled, y_train_valid)
    test_scores = np.asarray(model.predict_proba(x_test_scaled))
    test_pred = (test_scores >= 0.5).astype(int)
    return _multi_label_metrics(y_test, test_scores, test_pred), {"best_c": float(best_c), "selection_metric": primary_name}


def _fit_and_score_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    target_names: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, float]]:
    primary_name = _primary_metric_name("regression", target_names, y_train)
    best_alpha = None
    best_score = -np.inf

    for alpha in RIDGE_ALPHA_GRID:
        scaler, model = _build_regression_probe(alpha)
        x_train_scaled = scaler.fit_transform(x_train)
        x_valid_scaled = scaler.transform(x_valid)
        model.fit(x_train_scaled, y_train)
        valid_pred = model.predict(x_valid_scaled)
        valid_metrics = _regression_metrics(y_valid, valid_pred, target_names=target_names)
        score = valid_metrics[primary_name]
        if score > best_score:
            best_score = score
            best_alpha = alpha

    scaler, model = _build_regression_probe(float(best_alpha))
    x_train_valid = np.concatenate([x_train, x_valid], axis=0)
    y_train_valid = np.concatenate([y_train, y_valid], axis=0)
    x_train_valid_scaled = scaler.fit_transform(x_train_valid)
    x_test_scaled = scaler.transform(x_test)
    model.fit(x_train_valid_scaled, y_train_valid)
    test_pred = model.predict(x_test_scaled)
    return _regression_metrics(y_test, test_pred, target_names=target_names), {
        "best_alpha": float(best_alpha),
        "selection_metric": primary_name,
    }


def _decode_single_label(partition: EmbeddingPartition) -> np.ndarray:
    return np.asarray(parse_target_array(partition.target_json))


def _decode_multi_label(partition: EmbeddingPartition) -> np.ndarray:
    label_lists = parse_target_array(partition.target_json)
    binarizer = MultiLabelBinarizer(classes=list(partition.label_vocabulary))
    return binarizer.fit_transform(label_lists).astype(np.int64)


def _decode_regression(partition: EmbeddingPartition) -> np.ndarray:
    targets = parse_target_array(partition.target_json)
    matrix = np.asarray(targets, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    return matrix


def _partition_metrics_row(
    *,
    partition: EmbeddingPartition,
    metrics: dict[str, float],
    selection: dict[str, float],
    task_identifier: str,
    evaluation_scheme: str,
    fold_name: str | None = None,
) -> dict[str, Any]:
    target_matrix = (
        _decode_regression(partition)
        if partition.target_type == "regression"
        else (_decode_multi_label(partition) if partition.target_type == "multi_label_classification" else _decode_single_label(partition))
    )
    primary_name = _primary_metric_name(partition.target_type, partition.target_names, target_matrix)
    row = {
        "dataset_name": partition.dataset_name,
        "task_name": partition.task_name,
        "dataset_task": task_identifier,
        "model_name": partition.model_name.lower(),
        "representation_name": partition.representation_name,
        "target_type": partition.target_type,
        "evaluation_scheme": evaluation_scheme,
        "fold_name": fold_name or "",
        "n_examples": int(partition.embeddings.shape[0]),
        "primary_metric_name": primary_name,
        "primary_metric_value": float(metrics[primary_name]),
    }
    row.update({key: float(value) for key, value in metrics.items()})
    row.update(selection)
    return row


def evaluate_standard_splits(bundle: dict[str, EmbeddingPartition]) -> dict[str, Any]:
    train = bundle["train"]
    valid = bundle["valid"]
    test = bundle["test"]

    if train.target_type == "single_label_classification":
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(_decode_single_label(train))
        y_valid = label_encoder.transform(_decode_single_label(valid))
        y_test = label_encoder.transform(_decode_single_label(test))
        metrics, selection = _fit_and_score_single_label(
            train.embeddings,
            y_train,
            valid.embeddings,
            y_valid,
            test.embeddings,
            y_test,
        )
    elif train.target_type == "multi_label_classification":
        y_train = _decode_multi_label(train)
        y_valid = _decode_multi_label(valid)
        y_test = _decode_multi_label(test)
        metrics, selection = _fit_and_score_multi_label(
            train.embeddings,
            y_train,
            valid.embeddings,
            y_valid,
            test.embeddings,
            y_test,
        )
    else:
        y_train = _decode_regression(train)
        y_valid = _decode_regression(valid)
        y_test = _decode_regression(test)
        metrics, selection = _fit_and_score_regression(
            train.embeddings,
            y_train,
            valid.embeddings,
            y_valid,
            test.embeddings,
            y_test,
            train.target_names,
        )

    return {
        "summary": _partition_metrics_row(
            partition=test,
            metrics=metrics,
            selection=selection,
            task_identifier=f"{test.dataset_name}/{test.task_name}",
            evaluation_scheme="train_valid_test",
        ),
        "details": [],
    }


def evaluate_fold_splits(bundle: dict[str, EmbeddingPartition]) -> dict[str, Any]:
    partitions = list(bundle.values())
    fold_names = sorted(part.partition_name for part in partitions)
    summary_partition = partitions[0]
    test_fold = fold_names[0]
    valid_fold = fold_names[1]
    train_folds = [name for name in fold_names if name not in {test_fold, valid_fold}]

    train_parts = [bundle[name] for name in train_folds]
    valid_part = bundle[valid_fold]
    test_part = bundle[test_fold]

    x_train = np.concatenate([part.embeddings for part in train_parts], axis=0)
    x_valid = valid_part.embeddings
    x_test = test_part.embeddings

    if summary_partition.target_type != "single_label_classification":
        raise ValueError("Fold-based evaluation is only implemented for single-label classification tasks.")

    label_encoder = LabelEncoder()
    train_labels = np.concatenate([_decode_single_label(part) for part in train_parts], axis=0)
    y_train = label_encoder.fit_transform(train_labels)
    y_valid = label_encoder.transform(_decode_single_label(valid_part))
    y_test = label_encoder.transform(_decode_single_label(test_part))
    metrics, selection = _fit_and_score_single_label(x_train, y_train, x_valid, y_valid, x_test, y_test)

    row = _partition_metrics_row(
        partition=test_part,
        metrics=metrics,
        selection=selection,
        task_identifier=f"{test_part.dataset_name}/{test_part.task_name}",
        evaluation_scheme="representative_fold_holdout",
        fold_name=test_fold,
    )
    row["n_examples"] = int(sum(part.embeddings.shape[0] for part in partitions))
    return {"summary": row, "details": [row]}


def evaluate_embedding_bundle(bundle: dict[str, EmbeddingPartition]) -> dict[str, Any]:
    partition_names = set(bundle)
    if {"train", "valid", "test"}.issubset(partition_names):
        return evaluate_standard_splits(bundle)
    return evaluate_fold_splits(bundle)


def resolve_representation_dir(
    *,
    embeddings_root: str | Path,
    dataset_dir: str,
    task_dir: str,
    model_name: str,
) -> Path:
    embeddings_root = Path(embeddings_root)
    return embeddings_root / dataset_dir / task_dir / model_name / MODEL_REPRESENTATIONS[model_name]


def build_probe_jobs(embeddings_root: str | Path) -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    for task in TASK_SPECS:
        for model_name in MODEL_REPRESENTATIONS:
            representation_dir = resolve_representation_dir(
                embeddings_root=embeddings_root,
                dataset_dir=task["dataset_dir"],
                task_dir=task["task_dir"],
                model_name=model_name,
            )
            jobs.append(
                {
                    "dataset_dir": task["dataset_dir"],
                    "task_dir": task["task_dir"],
                    "model_name": model_name,
                    "representation_dir": str(representation_dir),
                }
            )
    return jobs


def load_final_alignment(alignment_root: str | Path) -> pd.DataFrame:
    alignment_root = Path(alignment_root)
    rows: list[pd.DataFrame] = []
    for csv_path in (
        alignment_root / "embeddings" / "nh2015" / "rsa_summary.csv",
        alignment_root / "embeddings" / "b2021" / "rsa_summary.csv",
    ):
        frame = pd.read_csv(csv_path)
        frame["model_name"] = frame["model_name"].str.lower()
        rows.append(frame[["dataset_name", "model_name", "representation_name", "mean_spearman"]])
    result = pd.concat(rows, ignore_index=True)
    result = result.rename(columns={"dataset_name": "alignment_dataset", "mean_spearman": "alignment_mean_spearman"})
    averaged = (
        result.groupby("model_name", as_index=False)["alignment_mean_spearman"]
        .mean()
        .rename(columns={"alignment_mean_spearman": "alignment_mean_spearman_avg"})
    )
    return result.merge(averaged, on="model_name", how="left")
