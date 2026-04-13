from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgWarning
from scipy.stats import pearsonr, spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, mean_absolute_error, r2_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=LinAlgWarning)


TASK_SPECS = (
    {"dataset_dir": "gtzan", "task_dir": "genre_classification", "display_name": "GTZAN"},
    {"dataset_dir": "vocalset", "task_dir": "singing_technique", "display_name": "VocalSet"},
    {"dataset_dir": "emo", "task_dir": "emotion_regression", "display_name": "EmoMusic"},
    {"dataset_dir": "nsynthpitch5h", "task_dir": "pitch_estimation", "display_name": "NSynth-pitch 5h"},
    {"dataset_dir": "magnatagatune", "task_dir": "music_tagging", "display_name": "MagnaTagATune"},
    {"dataset_dir": "giantsteps", "task_dir": "key_detection", "display_name": "GiantSteps"},
    {"dataset_dir": "mtgtop50", "task_dir": "music_tagging", "display_name": "MTG-Top50"},
    {"dataset_dir": "mtgmoodtheme", "task_dir": "emotion_classification", "display_name": "MTG-MoodTheme"},
    {"dataset_dir": "mtggenre", "task_dir": "genre_classification", "display_name": "MTG-Genre"},
    {"dataset_dir": "mtginstrument", "task_dir": "instrument_classification", "display_name": "MTG-Instrument"},
)

PREFERRED_MODEL_ORDER = (
    "clap",
    "clap_music",
    "clap_music_speech",
    "mulan",
    "mert",
    "mert95",
    "musicgen_small",
    "musicgen_medium",
    "musicgen_large",
    "musicgen_melody_large",
)

DATASET_DISPLAY_NAME_BY_DIR = {
    "gtzan": "GTZAN",
    "vocalset": "VocalSet",
    "emo": "EMO",
    "nsynthpitch5h": "NSynthPitch5h",
    "magnatagatune": "MagnaTagATune",
    "giantsteps": "GiantSteps",
    "mtgtop50": "MTGTop50",
    "mtgmoodtheme": "MTGMoodTheme",
    "mtggenre": "MTGGenre",
    "mtginstrument": "MTGInstrument",
}

ALIGNMENT_SPECS = {
    "rsa": {
        "path": Path("compiled_results/rsa_layerwise_summary.csv"),
        "score_name": "mean_spearman",
        "score_column": "mean_spearman",
        "secondary_name": "median_spearman",
        "secondary_column": "median_spearman",
    },
    "regression": {
        "path": Path("compiled_results/regression_layerwise_summary.csv"),
        "score_name": "mean_noise_corrected_r2",
        "score_column": "mean_noise_corrected_r2",
        "secondary_name": "median_noise_corrected_r2",
        "secondary_column": "median_noise_corrected_r2",
    },
}

RESUME_SOURCE_DIR_NAMES = (
    "probe_results_final_only",
    "probe_results",
)


@dataclass(frozen=True)
class EmbeddingPartition:
    name: str
    embeddings: np.ndarray
    target_json: np.ndarray
    label_text: np.ndarray
    split: np.ndarray
    fold: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ProbeResult:
    primary_metric_name: str
    primary_metric_value: float
    best_param_name: str
    best_param_value: float
    metrics: dict[str, float]


def _load_partition_npz(path: Path) -> EmbeddingPartition:
    with np.load(path, allow_pickle=True) as payload:
        metadata = {
            "dataset_name": str(payload["dataset_name"].item()),
            "task_name": str(payload["task_name"].item()),
            "target_type": str(payload["target_type"].item()),
            "model_name": str(payload["model_name"].item()),
            "representation_name": str(payload["representation_name"].item()),
            "label_vocabulary": [str(value) for value in payload["label_vocabulary"].tolist()],
            "target_names": [str(value) for value in payload["target_names"].tolist()],
        }
        return EmbeddingPartition(
            name=path.stem,
            embeddings=np.asarray(payload["embeddings"], dtype=np.float32),
            target_json=np.asarray(payload["target_json"]).astype(str),
            label_text=np.asarray(payload["label_text"]).astype(str),
            split=np.asarray(payload["split"]).astype(str),
            fold=np.asarray(payload["fold"]).astype(str),
            metadata=metadata,
        )


def _load_representation_dir(path: Path) -> tuple[dict[str, EmbeddingPartition], dict[str, Any]]:
    partitions = {}
    metadata: dict[str, Any] | None = None
    for npz_path in sorted(path.glob("*.npz")):
        partition = _load_partition_npz(npz_path)
        partitions[partition.name] = partition
        if metadata is None:
            metadata = partition.metadata
    if metadata is None:
        raise FileNotFoundError(f"No embedding files found under {path}")
    return partitions, metadata


def _parse_targets(target_json: np.ndarray) -> list[Any]:
    return [json.loads(str(value)) for value in target_json.tolist()]


def _safe_mean(values: list[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    return float(np.mean(valid)) if valid else float("nan")


def _safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(pearsonr(y_true, y_pred).statistic)


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(spearmanr(y_true, y_pred).statistic)


def _metric_safe_name(name: str) -> str:
    text = str(name).strip().lower()
    for old, new in (("/", "_"), (" ", "_"), ("-", "_")):
        text = text.replace(old, new)
    return text


def _multilabel_matrix(targets: list[Any], label_vocabulary: list[str]) -> np.ndarray:
    binarizer = MultiLabelBinarizer(classes=label_vocabulary)
    return binarizer.fit_transform(targets).astype(np.int64, copy=False)


def _macro_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    values = []
    for column_index in range(y_true.shape[1]):
        column_true = y_true[:, column_index]
        if int(column_true.sum()) == 0:
            continue
        values.append(float(average_precision_score(column_true, y_score[:, column_index])))
    return _safe_mean(values)


def _macro_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    values = []
    for column_index in range(y_true.shape[1]):
        column_true = y_true[:, column_index]
        positives = int(column_true.sum())
        negatives = int((1 - column_true).sum())
        if positives == 0 or negatives == 0:
            continue
        rank = np.argsort(np.argsort(y_score[:, column_index]))
        positive_ranks = rank[column_true == 1]
        auc = (positive_ranks.sum() - positives * (positives - 1) / 2.0) / (positives * negatives)
        values.append(float(auc))
    return _safe_mean(values)


def _single_label_targets(targets: list[Any], label_vocabulary: list[str]) -> np.ndarray:
    vocab = label_vocabulary or sorted({str(value) for value in targets})
    label_to_index = {label: index for index, label in enumerate(vocab)}
    return np.asarray([label_to_index[str(value)] for value in targets], dtype=np.int64)


def _inferred_single_label_vocabulary(targets: list[Any]) -> list[str]:
    values = list(targets)
    try:
        numeric = [float(value) for value in values]
    except (TypeError, ValueError):
        return sorted({str(value) for value in values})

    if all(abs(value - round(value)) <= 1e-6 for value in numeric):
        return [str(value) for value in sorted({int(round(value)) for value in numeric})]
    return [str(value) for value in sorted({float(value) for value in numeric})]


def _effective_target_type(metadata: dict[str, Any]) -> str:
    task_name = str(metadata.get("task_name", ""))
    if task_name == "pitch_estimation":
        return "single_label_classification"
    return str(metadata["target_type"])


def _regression_array(targets: list[Any]) -> np.ndarray:
    parsed = np.asarray(targets, dtype=np.float64)
    if parsed.ndim == 1:
        parsed = parsed[:, None]
    return parsed


def _is_integer_regression_target(y_true: np.ndarray) -> bool:
    return bool(np.allclose(y_true, np.rint(y_true), atol=1e-6))


def _evaluate_single_label(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> ProbeResult:
    alpha = 1.0
    final_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("probe", RidgeClassifier(alpha=alpha)),
        ]
    )
    final_model.fit(np.concatenate([x_train, x_valid], axis=0), np.concatenate([y_train, y_valid], axis=0))
    test_pred = final_model.predict(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "macro_f1": float(f1_score(y_test, test_pred, average="macro")),
    }
    return ProbeResult(
        primary_metric_name="accuracy",
        primary_metric_value=metrics["accuracy"],
        best_param_name="alpha",
        best_param_value=float(alpha),
        metrics=metrics,
    )


def _evaluate_multilabel(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> ProbeResult:
    alpha = 1.0
    final_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "probe",
                OneVsRestClassifier(
                    RidgeClassifier(alpha=alpha),
                    n_jobs=-1,
                ),
            ),
        ]
    )
    final_model.fit(np.concatenate([x_train, x_valid], axis=0), np.concatenate([y_train, y_valid], axis=0))
    test_score = np.asarray(final_model.decision_function(x_test), dtype=np.float64)
    if test_score.ndim == 1:
        test_score = test_score[:, None]
    test_pred = (test_score >= 0.0).astype(np.int64, copy=False)

    metrics = {
        "average_precision_macro": _macro_average_precision(y_test, test_score),
        "roc_auc_macro": _macro_roc_auc(y_test, test_score),
        "macro_f1": float(f1_score(y_test, test_pred, average="macro", zero_division=0)),
    }
    return ProbeResult(
        primary_metric_name="average_precision_macro",
        primary_metric_value=metrics["average_precision_macro"],
        best_param_name="alpha",
        best_param_value=float(alpha),
        metrics=metrics,
    )


def _regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: tuple[str, ...] | list[str] | None = None,
) -> dict[str, float]:
    target_count = y_true.shape[1]
    pearsons = [_safe_pearson(y_true[:, index], y_pred[:, index]) for index in range(target_count)]
    spearmans = [_safe_spearman(y_true[:, index], y_pred[:, index]) for index in range(target_count)]
    r2_scores = [float(r2_score(y_true[:, index], y_pred[:, index])) for index in range(target_count)]
    maes = [float(mean_absolute_error(y_true[:, index], y_pred[:, index])) for index in range(target_count)]

    metrics = {
        "pearson_mean": _safe_mean(pearsons),
        "spearman_mean": _safe_mean(spearmans),
        "r2_mean": _safe_mean(r2_scores),
        "mae_mean": _safe_mean(maes),
    }
    names = list(target_names or [f"target_{index}" for index in range(target_count)])
    if len(names) < target_count:
        names.extend(f"target_{index}" for index in range(len(names), target_count))
    for index in range(target_count):
        suffix = _metric_safe_name(names[index])
        metrics[f"pearson_{suffix}"] = float(pearsons[index])
        metrics[f"spearman_{suffix}"] = float(spearmans[index])
        metrics[f"r2_{suffix}"] = float(r2_scores[index])
        metrics[f"mae_{suffix}"] = float(maes[index])
    if target_count == 1 and _is_integer_regression_target(y_true):
        rounded_pred = np.rint(y_pred[:, 0]).astype(np.int64)
        rounded_true = np.rint(y_true[:, 0]).astype(np.int64)
        metrics["rounded_accuracy"] = float(accuracy_score(rounded_true, rounded_pred))
    return metrics


def _regression_primary_metric_name(y_true: np.ndarray) -> str:
    if y_true.shape[1] == 1 and _is_integer_regression_target(y_true):
        return "rounded_accuracy"
    return "pearson_mean"


def _evaluate_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    target_names: tuple[str, ...] | list[str] | None = None,
) -> ProbeResult:
    primary_name = _regression_primary_metric_name(y_valid)
    alpha = 1.0
    final_model = Pipeline([("scaler", StandardScaler()), ("probe", Ridge(alpha=alpha))])
    final_model.fit(np.concatenate([x_train, x_valid], axis=0), np.concatenate([y_train, y_valid], axis=0))
    test_pred = final_model.predict(x_test)
    test_pred = np.asarray(test_pred, dtype=np.float64)
    if test_pred.ndim == 1:
        test_pred = test_pred[:, None]
    metrics = _regression_metrics(y_test, test_pred, target_names=target_names)
    return ProbeResult(
        primary_metric_name=primary_name,
        primary_metric_value=float(metrics[primary_name]),
        best_param_name="alpha",
        best_param_value=float(alpha),
        metrics=metrics,
    )


def _evaluate_regular_task(partitions: dict[str, EmbeddingPartition], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    required = ("train", "valid", "test")
    missing = [name for name in required if name not in partitions]
    if missing:
        raise ValueError(f"Missing partitions for {metadata['dataset_name']}: {missing}")

    train = partitions["train"]
    valid = partitions["valid"]
    test = partitions["test"]
    targets_train = _parse_targets(train.target_json)
    targets_valid = _parse_targets(valid.target_json)
    targets_test = _parse_targets(test.target_json)

    target_type = _effective_target_type(metadata)
    if target_type == "single_label_classification":
        label_vocabulary = list(metadata["label_vocabulary"])
        if len(label_vocabulary) == 0:
            label_vocabulary = _inferred_single_label_vocabulary(targets_train + targets_valid + targets_test)
        y_train = _single_label_targets(targets_train, label_vocabulary)
        y_valid = _single_label_targets(targets_valid, label_vocabulary)
        y_test = _single_label_targets(targets_test, label_vocabulary)
        result = _evaluate_single_label(train.embeddings, y_train, valid.embeddings, y_valid, test.embeddings, y_test)
    elif target_type == "multi_label_classification":
        y_train = _multilabel_matrix(targets_train, metadata["label_vocabulary"])
        y_valid = _multilabel_matrix(targets_valid, metadata["label_vocabulary"])
        y_test = _multilabel_matrix(targets_test, metadata["label_vocabulary"])
        result = _evaluate_multilabel(train.embeddings, y_train, valid.embeddings, y_valid, test.embeddings, y_test)
    elif target_type == "regression":
        y_train = _regression_array(targets_train)
        y_valid = _regression_array(targets_valid)
        y_test = _regression_array(targets_test)
        result = _evaluate_regression(
            train.embeddings,
            y_train,
            valid.embeddings,
            y_valid,
            test.embeddings,
            y_test,
            target_names=tuple(metadata.get("target_names", [])),
        )
    else:
        raise ValueError(f"Unsupported target type: {target_type}")

    row = {
        "dataset_name": metadata["dataset_name"],
        "task_name": metadata["task_name"],
        "model_name": metadata["model_name"],
        "representation_name": metadata["representation_name"],
        "evaluation_protocol": "train_valid_test",
        "fold_name": "",
        "n_train": int(train.embeddings.shape[0]),
        "n_valid": int(valid.embeddings.shape[0]),
        "n_test": int(test.embeddings.shape[0]),
        "primary_metric_name": result.primary_metric_name,
        "primary_metric_value": result.primary_metric_value,
        "best_param_name": result.best_param_name,
        "best_param_value": result.best_param_value,
    }
    row.update(result.metrics)
    return [row]


def _concatenate_parts(parts: list[EmbeddingPartition]) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.concatenate([part.embeddings for part in parts], axis=0)
    targets = np.concatenate([part.target_json for part in parts], axis=0)
    return embeddings, targets


def _evaluate_fold_task(partitions: dict[str, EmbeddingPartition], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    fold_names = sorted(partitions.keys(), key=lambda value: int(value))
    rows: list[dict[str, Any]] = []

    for index, test_fold in enumerate(fold_names):
        valid_fold = fold_names[(index + 1) % len(fold_names)]
        train_folds = [name for name in fold_names if name not in {test_fold, valid_fold}]

        train_embeddings, train_targets = _concatenate_parts([partitions[name] for name in train_folds])
        valid_embeddings, valid_targets = _concatenate_parts([partitions[valid_fold]])
        test_embeddings, test_targets = _concatenate_parts([partitions[test_fold]])

        target_type = metadata["target_type"]
        if target_type != "single_label_classification":
            raise ValueError(f"Fold evaluation currently expects single-label classification, got {target_type}")

        y_train = _single_label_targets(_parse_targets(train_targets), metadata["label_vocabulary"])
        y_valid = _single_label_targets(_parse_targets(valid_targets), metadata["label_vocabulary"])
        y_test = _single_label_targets(_parse_targets(test_targets), metadata["label_vocabulary"])
        result = _evaluate_single_label(train_embeddings, y_train, valid_embeddings, y_valid, test_embeddings, y_test)

        row = {
            "dataset_name": metadata["dataset_name"],
            "task_name": metadata["task_name"],
            "model_name": metadata["model_name"],
            "representation_name": metadata["representation_name"],
            "evaluation_protocol": "cross_validation",
            "fold_name": test_fold,
            "n_train": int(train_embeddings.shape[0]),
            "n_valid": int(valid_embeddings.shape[0]),
            "n_test": int(test_embeddings.shape[0]),
            "primary_metric_name": result.primary_metric_name,
            "primary_metric_value": result.primary_metric_value,
            "best_param_name": result.best_param_name,
            "best_param_value": result.best_param_value,
        }
        row.update(result.metrics)
        rows.append(row)

    summary = {
        "dataset_name": metadata["dataset_name"],
        "task_name": metadata["task_name"],
        "model_name": metadata["model_name"],
        "representation_name": metadata["representation_name"],
        "evaluation_protocol": "cross_validation_mean",
        "fold_name": "mean",
        "n_train": float(np.mean([row["n_train"] for row in rows])),
        "n_valid": float(np.mean([row["n_valid"] for row in rows])),
        "n_test": float(np.mean([row["n_test"] for row in rows])),
        "primary_metric_name": rows[0]["primary_metric_name"],
        "primary_metric_value": float(np.mean([row["primary_metric_value"] for row in rows])),
        "best_param_name": rows[0]["best_param_name"],
        "best_param_value": float(np.mean([row["best_param_value"] for row in rows])),
    }
    metric_names = set().union(*(row.keys() for row in rows))
    for metric_name in metric_names:
        if metric_name in summary or metric_name.endswith("_name"):
            continue
        if all(isinstance(row.get(metric_name), (int, float, np.floating)) for row in rows if metric_name in row):
            summary[metric_name] = float(np.mean([float(row[metric_name]) for row in rows]))
    rows.append(summary)
    return rows


def _representation_sort_key(name: str) -> tuple[int, int | str]:
    if name.startswith("layer_"):
        return (0, int(name.split("_", 1)[1]))
    if name in {"audio_embedding", "final", "encoder"}:
        return (1, name)
    return (2, name)


def _discover_representation_dirs(task_root: Path, model_name: str) -> list[Path]:
    model_root = task_root / model_name
    if not model_root.exists():
        return []
    return sorted(
        [path for path in model_root.iterdir() if path.is_dir()],
        key=lambda path: _representation_sort_key(path.name),
    )


def _discover_model_names(task_root: Path) -> list[str]:
    discovered = sorted(
        [path.name for path in task_root.iterdir() if path.is_dir()],
        key=lambda name: (
            PREFERRED_MODEL_ORDER.index(name)
            if name in PREFERRED_MODEL_ORDER
            else len(PREFERRED_MODEL_ORDER),
            name,
        ),
    )
    return discovered


def _alignment_representation_name(model_name: str, representation_name: str) -> str:
    if representation_name.startswith("layer_"):
        return representation_name
    if model_name == "clap" and representation_name in {"audio_embedding", "encoder"}:
        return "final"
    if representation_name == "final":
        return "final"
    return representation_name


def _is_final_representation_name(model_name: str, representation_name: str) -> bool:
    model_name = str(model_name).lower()
    representation_name = str(representation_name)
    if representation_name.startswith("layer_"):
        return False
    if model_name == "clap":
        return representation_name in {"audio_embedding", "encoder", "final"}
    return representation_name == "final"


def _layer_index_from_representation(representation_name: str) -> float:
    if representation_name.startswith("layer_"):
        return float(int(representation_name.split("_", 1)[1]))
    return float("nan")


def _annotate_probe_row(row: dict[str, Any]) -> dict[str, Any]:
    representation_name = str(row["representation_name"])
    model_name = str(row["model_name"]).lower()
    row["alignment_representation_name"] = _alignment_representation_name(model_name, representation_name)
    row["layer_index"] = _layer_index_from_representation(representation_name)
    row["is_final"] = bool(not representation_name.startswith("layer_"))
    row["dataset_task"] = f"{row['dataset_name']} / {row['task_name']}"
    return row


def _representation_key(
    dataset_name: str,
    task_name: str,
    model_name: str,
    representation_name: str,
) -> tuple[str, str, str, str]:
    return (
        str(dataset_name),
        str(task_name),
        str(model_name).lower(),
        str(representation_name),
    )


def _detail_frame_from_rows(detail_rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not detail_rows:
        return pd.DataFrame()
    return pd.DataFrame(detail_rows).sort_values(
        ["dataset_name", "task_name", "model_name", "layer_index", "representation_name", "evaluation_protocol", "fold_name"],
        kind="stable",
    )


def _existing_completed_keys(detail_rows: list[dict[str, Any]]) -> set[tuple[str, str, str, str]]:
    details = _detail_frame_from_rows(detail_rows)
    if details.empty:
        return set()

    summary = details[
        details["evaluation_protocol"].isin(["train_valid_test", "cross_validation_mean"])
    ].copy()
    if summary.empty:
        return set()

    return {
        _representation_key(
            row.dataset_name,
            row.task_name,
            row.model_name,
            row.representation_name,
        )
        for row in summary.itertuples(index=False)
    }


def _load_resume_rows(output_root: Path, representation_mode: str) -> list[dict[str, Any]]:
    candidate_paths = [output_root / "probe_details.csv"]
    if representation_mode == "all":
        for sibling_name in RESUME_SOURCE_DIR_NAMES:
            sibling_path = output_root.parent / sibling_name / "probe_details.csv"
            if sibling_path != candidate_paths[0]:
                candidate_paths.append(sibling_path)

    merged_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str, str, str, str]] = set()

    for csv_path in candidate_paths:
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            continue
        frame = pd.read_csv(csv_path)
        if frame.empty:
            continue
        records = frame.replace({np.nan: None}).to_dict(orient="records")
        for row in records:
            normalized = _annotate_probe_row(dict(row))
            dedupe_key = (
                str(normalized.get("dataset_name")),
                str(normalized.get("task_name")),
                str(normalized.get("model_name")).lower(),
                str(normalized.get("representation_name")),
                str(normalized.get("evaluation_protocol")),
                str(normalized.get("fold_name")),
            )
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            merged_rows.append(normalized)

    return merged_rows


def _load_alignment_layerwise(alignment_root: Path) -> pd.DataFrame:
    all_frames: list[pd.DataFrame] = []

    for alignment_kind, spec in ALIGNMENT_SPECS.items():
        frame = pd.read_csv(alignment_root / spec["path"]).copy()
        frame["alignment_kind"] = alignment_kind
        frame["alignment_dataset"] = frame["dataset_name"].astype(str)
        frame["model_name"] = frame["model_name"].astype(str).str.lower()
        frame["alignment_representation_name"] = frame["representation_name"].astype(str)
        frame["alignment_score_name"] = spec["score_name"]
        frame["alignment_score"] = frame[spec["score_column"]].astype(float)
        frame["alignment_secondary_name"] = spec["secondary_name"]
        frame["alignment_secondary_score"] = frame[spec["secondary_column"]].astype(float)
        frame["layer_index"] = pd.to_numeric(frame.get("layer_index"), errors="coerce")
        frame["layer_depth"] = pd.to_numeric(frame.get("layer_depth"), errors="coerce")
        frame["is_final"] = frame["is_final"].astype(bool)
        all_frames.append(
            frame[
                [
                    "alignment_kind",
                    "alignment_dataset",
                    "model_name",
                    "alignment_representation_name",
                    "layer_index",
                    "layer_depth",
                    "is_final",
                    "alignment_score_name",
                    "alignment_score",
                    "alignment_secondary_name",
                    "alignment_secondary_score",
                    "result_dir",
                ]
            ].copy()
        )

    alignment = pd.concat(all_frames, ignore_index=True)

    mean_alignment = (
        alignment.groupby(
            ["alignment_kind", "model_name", "alignment_representation_name"],
            as_index=False,
        )[
            ["layer_index", "layer_depth", "alignment_score", "alignment_secondary_score"]
        ]
        .mean()
        .merge(
            alignment.groupby(
                ["alignment_kind", "model_name", "alignment_representation_name"],
                as_index=False,
            )[["is_final"]]
            .max(),
            on=["alignment_kind", "model_name", "alignment_representation_name"],
            how="left",
        )
    )
    mean_alignment["alignment_dataset"] = "mean"
    mean_alignment["alignment_score_name"] = mean_alignment["alignment_kind"].map(
        lambda kind: ALIGNMENT_SPECS[str(kind)]["score_name"]
    )
    mean_alignment["alignment_secondary_name"] = mean_alignment["alignment_kind"].map(
        lambda kind: ALIGNMENT_SPECS[str(kind)]["secondary_name"]
    )
    mean_alignment["result_dir"] = ""

    alignment = pd.concat([alignment, mean_alignment], ignore_index=True)
    alignment = alignment.sort_values(
        ["alignment_kind", "alignment_dataset", "model_name", "layer_depth", "alignment_representation_name"],
        kind="stable",
    ).reset_index(drop=True)
    return alignment


def _write_snapshots(
    detail_rows: list[dict[str, Any]],
    alignment: pd.DataFrame,
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    details = _detail_frame_from_rows(detail_rows)
    if details.empty:
        details.to_csv(output_root / "probe_details.csv", index=False)
        pd.DataFrame().to_csv(output_root / "probe_summary.csv", index=False)
        alignment.to_csv(output_root / "alignment_layerwise.csv", index=False)
        pd.DataFrame().to_csv(output_root / "probe_alignment_layerwise_merged.csv", index=False)
        pd.DataFrame().to_csv(output_root / "probe_summary_final.csv", index=False)
        alignment[alignment["is_final"]].copy().to_csv(output_root / "alignment_final.csv", index=False)
        pd.DataFrame().to_csv(output_root / "probe_alignment_merged.csv", index=False)
        return details, pd.DataFrame(), pd.DataFrame()

    summary = details[
        details["evaluation_protocol"].isin(["train_valid_test", "cross_validation_mean"])
    ].copy()
    merged = summary.merge(
        alignment,
        on=["model_name", "alignment_representation_name"],
        how="left",
        suffixes=("", "_alignment"),
    )

    details.to_csv(output_root / "probe_details.csv", index=False)
    summary.to_csv(output_root / "probe_summary.csv", index=False)
    alignment.to_csv(output_root / "alignment_layerwise.csv", index=False)
    merged.to_csv(output_root / "probe_alignment_layerwise_merged.csv", index=False)

    final_summary = summary[summary["is_final"]].copy()
    final_alignment = alignment[alignment["is_final"]].copy()
    final_merged = merged[merged["is_final"]].copy()
    final_summary.to_csv(output_root / "probe_summary_final.csv", index=False)
    final_alignment.to_csv(output_root / "alignment_final.csv", index=False)
    final_merged.to_csv(output_root / "probe_alignment_merged.csv", index=False)
    return details, summary, merged


def run_probes(
    embeddings_root: Path,
    alignment_root: Path,
    output_root: Path,
    representation_mode: str = "all",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = _load_resume_rows(output_root, representation_mode=representation_mode)
    completed_keys = _existing_completed_keys(detail_rows)
    alignment = _load_alignment_layerwise(alignment_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if completed_keys:
        print(
            f"resuming from {len(completed_keys)} completed representations "
            f"({len(detail_rows)} detail rows loaded)",
            flush=True,
        )

    for task_spec in TASK_SPECS:
        task_root = embeddings_root / task_spec["dataset_dir"] / task_spec["task_dir"]
        for model_name in _discover_model_names(task_root):
            representation_dirs = _discover_representation_dirs(task_root, model_name)
            if len(representation_dirs) == 0:
                continue
            print(
                f"running {task_spec['dataset_dir']}/{task_spec['task_dir']} :: "
                f"{model_name} ({len(representation_dirs)} reps)",
                flush=True,
            )
            for representation_dir in representation_dirs:
                if representation_mode == "final_only" and not _is_final_representation_name(model_name, representation_dir.name):
                    continue
                rep_key = _representation_key(
                    DATASET_DISPLAY_NAME_BY_DIR[task_spec["dataset_dir"]],
                    task_spec["task_dir"],
                    model_name,
                    representation_dir.name,
                )
                if rep_key in completed_keys:
                    print(f"skip completed: {task_spec['dataset_dir']}/{task_spec['task_dir']} :: {model_name} / {representation_dir.name}", flush=True)
                    continue
                try:
                    partitions, metadata = _load_representation_dir(representation_dir)
                    if task_spec["dataset_dir"] == "vocalset":
                        rows = _evaluate_fold_task(partitions, metadata)
                    else:
                        rows = _evaluate_regular_task(partitions, metadata)
                except (FileNotFoundError, ValueError) as exc:
                    print(f"skipping incomplete representation: {representation_dir} ({exc})", flush=True)
                    continue
                for row in rows:
                    row["dataset_dir"] = task_spec["dataset_dir"]
                    row["task_dir"] = task_spec["task_dir"]
                    row["display_name"] = task_spec["display_name"]
                    detail_rows.append(_annotate_probe_row(row))

                completed_keys.add(
                    _representation_key(
                        str(metadata["dataset_name"]),
                        str(metadata["task_name"]),
                        str(metadata["model_name"]),
                        str(metadata["representation_name"]),
                    )
                )

                _write_snapshots(detail_rows, alignment, output_root)

    details, summary, _ = _write_snapshots(detail_rows, alignment, output_root)
    return details, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight linear probes on prepared music embeddings.")
    parser.add_argument(
        "--embeddings-root",
        type=Path,
        default=Path("/home/hjkim/selective_auditory/ismir2026/music_task_embed/embeddings"),
    )
    parser.add_argument(
        "--alignment-root",
        type=Path,
        default=Path("/home/hjkim/selective_auditory/ismir2026/natural_sound_align"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/hjkim/selective_auditory/ismir2026/music_task_probe/probe_results"),
    )
    parser.add_argument(
        "--representation-mode",
        choices=["all", "final_only"],
        default="all",
    )
    args = parser.parse_args()

    details, summary = run_probes(
        args.embeddings_root,
        args.alignment_root,
        args.output_root,
        representation_mode=args.representation_mode,
    )
    print(args.output_root / "probe_summary.csv")
    print(args.output_root / "probe_details.csv")
    print(f"summary_rows={len(summary)} detail_rows={len(details)}")


if __name__ == "__main__":
    main()
