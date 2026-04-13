from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_MANIFEST_COLUMNS = (
    "sample_id",
    "audio_relpath",
    "audio_path",
    "split",
    "fold",
    "target_json",
    "label_text",
)

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
CURRENT_DATA_ROOT = PROJECT_ROOT / "dataset" / "music_dataset"
LEGACY_DATA_ROOTS: tuple[Path, ...] = ()


@dataclass(frozen=True)
class ManifestBundle:
    dataset_name: str
    task_name: str
    target_type: str
    manifest: pd.DataFrame
    label_vocabulary: tuple[str, ...] = ()
    target_names: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def validated_manifest(self) -> pd.DataFrame:
        missing = [column for column in REQUIRED_MANIFEST_COLUMNS if column not in self.manifest.columns]
        if missing:
            raise ValueError(
                f"Manifest for {self.dataset_name}/{self.task_name} is missing columns: {missing}"
            )
        return self.manifest.copy()


def _manifest_dir(output_root: Path, dataset_name: str, task_name: str) -> Path:
    return output_root / dataset_name.lower() / task_name.lower()


def write_manifest_bundle(bundle: ManifestBundle, output_root: Path) -> Path:
    output_root = Path(output_root)
    manifest_dir = _manifest_dir(output_root, bundle.dataset_name, bundle.task_name)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest = bundle.validated_manifest().copy()
    manifest = manifest.sort_values(["split", "fold", "sample_id"], kind="stable").reset_index(drop=True)
    manifest.to_csv(manifest_dir / "manifest.csv", index=False)

    payload = {
        "dataset_name": bundle.dataset_name,
        "task_name": bundle.task_name,
        "target_type": bundle.target_type,
        "label_vocabulary": list(bundle.label_vocabulary),
        "target_names": list(bundle.target_names),
        "num_examples": int(len(manifest)),
        "metadata": bundle.metadata,
    }
    (manifest_dir / "task_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_dir


def _normalize_audio_root_text(audio_root: str | None) -> str | None:
    if audio_root is None:
        return None
    normalized = str(audio_root)
    for legacy_root in LEGACY_DATA_ROOTS:
        normalized = normalized.replace(str(legacy_root), str(CURRENT_DATA_ROOT))
    return normalized


def _normalize_manifest_audio_paths(
    manifest: pd.DataFrame,
    *,
    audio_root: str | None,
) -> pd.DataFrame:
    normalized = manifest.copy()
    normalized["audio_path"] = normalized["audio_path"].astype(str)
    for legacy_root in LEGACY_DATA_ROOTS:
        normalized["audio_path"] = normalized["audio_path"].str.replace(
            str(legacy_root),
            str(CURRENT_DATA_ROOT),
            regex=False,
        )

    normalized_root = _normalize_audio_root_text(audio_root)
    if normalized_root:
        missing_mask = ~normalized["audio_path"].map(lambda value: Path(value).exists())
        if missing_mask.any():
            normalized.loc[missing_mask, "audio_path"] = normalized.loc[missing_mask, "audio_relpath"].map(
                lambda relpath: str((Path(normalized_root) / str(relpath)).resolve())
            )
    return normalized


def load_manifest_bundle(manifest_dir: str | Path) -> ManifestBundle:
    manifest_dir = Path(manifest_dir)
    manifest = pd.read_csv(manifest_dir / "manifest.csv", keep_default_na=False)
    payload = json.loads((manifest_dir / "task_metadata.json").read_text(encoding="utf-8"))
    metadata = dict(payload.get("metadata", {}))
    metadata["audio_root"] = _normalize_audio_root_text(metadata.get("audio_root"))
    manifest = _normalize_manifest_audio_paths(manifest, audio_root=metadata.get("audio_root"))
    return ManifestBundle(
        dataset_name=str(payload["dataset_name"]),
        task_name=str(payload["task_name"]),
        target_type=str(payload["target_type"]),
        manifest=manifest,
        label_vocabulary=tuple(payload.get("label_vocabulary", [])),
        target_names=tuple(payload.get("target_names", [])),
        metadata=metadata,
    )


def infer_partition_column(manifest: pd.DataFrame) -> str | None:
    split_values = [value for value in manifest["split"].astype(str).tolist() if value and value != "unspecified"]
    if split_values:
        return "split"
    fold_values = [value for value in manifest["fold"].astype(str).tolist() if value]
    if fold_values:
        return "fold"
    return None


def iter_partitions(
    manifest: pd.DataFrame,
    partition_column: str | None = None,
) -> list[tuple[str, pd.DataFrame]]:
    partition_column = partition_column or infer_partition_column(manifest)
    if partition_column is None:
        return [("all", manifest.reset_index(drop=True))]

    groups: list[tuple[str, pd.DataFrame]] = []
    for partition_name, partition_df in manifest.groupby(partition_column, sort=True, dropna=False):
        partition_text = str(partition_name) if partition_name == partition_name else "all"
        groups.append((partition_text, partition_df.reset_index(drop=True)))
    return groups


def save_partition_embeddings(
    *,
    bundle: ManifestBundle,
    partition_name: str,
    partition_df: pd.DataFrame,
    embeddings: np.ndarray,
    model_name: str,
    representation_name: str,
    output_root: str | Path,
    extractor_metadata: dict[str, Any] | None = None,
) -> Path:
    output_root = Path(output_root)
    representation_dir = (
        output_root
        / bundle.dataset_name.lower()
        / bundle.task_name.lower()
        / model_name.lower()
        / representation_name
    )
    representation_dir.mkdir(parents=True, exist_ok=True)
    output_path = representation_dir / f"{partition_name}.npz"

    if embeddings.shape[0] != len(partition_df):
        raise ValueError(
            f"Embedding row count mismatch for {bundle.dataset_name}/{bundle.task_name} "
            f"{model_name}/{representation_name}: {embeddings.shape[0]} vs {len(partition_df)}"
        )

    np.savez_compressed(
        output_path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        sample_id=partition_df["sample_id"].astype(str).to_numpy(),
        audio_relpath=partition_df["audio_relpath"].astype(str).to_numpy(),
        audio_path=partition_df["audio_path"].astype(str).to_numpy(),
        split=partition_df["split"].astype(str).to_numpy(),
        fold=partition_df["fold"].astype(str).to_numpy(),
        target_json=partition_df["target_json"].astype(str).to_numpy(),
        label_text=partition_df["label_text"].astype(str).to_numpy(),
        dataset_name=np.array(bundle.dataset_name, dtype=object),
        task_name=np.array(bundle.task_name, dtype=object),
        target_type=np.array(bundle.target_type, dtype=object),
        label_vocabulary=np.asarray(bundle.label_vocabulary, dtype=object),
        target_names=np.asarray(bundle.target_names, dtype=object),
        model_name=np.array(model_name, dtype=object),
        representation_name=np.array(representation_name, dtype=object),
        extractor_metadata=np.array(json.dumps(extractor_metadata or {}, sort_keys=True), dtype=object),
    )
    return output_path


def partition_embedding_output_path(
    *,
    bundle: ManifestBundle,
    partition_name: str,
    model_name: str,
    representation_name: str,
    output_root: str | Path,
) -> Path:
    output_root = Path(output_root)
    return (
        output_root
        / bundle.dataset_name.lower()
        / bundle.task_name.lower()
        / model_name.lower()
        / representation_name
        / f"{partition_name}.npz"
    )


def has_complete_partition_embeddings(
    *,
    bundle: ManifestBundle,
    partition_name: str,
    partition_df: pd.DataFrame,
    model_name: str,
    representation_name: str,
    output_root: str | Path,
) -> bool:
    output_path = partition_embedding_output_path(
        bundle=bundle,
        partition_name=partition_name,
        model_name=model_name,
        representation_name=representation_name,
        output_root=output_root,
    )
    if not output_path.exists():
        return False

    try:
        with np.load(output_path, allow_pickle=True) as payload:
            embeddings = payload["embeddings"]
            sample_id = payload["sample_id"]
            saved_model_name = str(payload["model_name"].item())
            saved_representation_name = str(payload["representation_name"].item())
    except Exception:
        return False

    if embeddings.shape[0] != len(partition_df):
        return False
    if len(sample_id) != len(partition_df):
        return False
    if saved_model_name.lower() != model_name.lower():
        return False
    if saved_representation_name != representation_name:
        return False

    expected_sample_ids = partition_df["sample_id"].astype(str).to_numpy()
    saved_sample_ids = np.asarray(sample_id).astype(str)
    return np.array_equal(saved_sample_ids, expected_sample_ids)


def has_complete_partition_job(
    *,
    bundle: ManifestBundle,
    partition_name: str,
    partition_df: pd.DataFrame,
    model_name: str,
    representation_names: list[str],
    output_root: str | Path,
) -> bool:
    return all(
        has_complete_partition_embeddings(
            bundle=bundle,
            partition_name=partition_name,
            partition_df=partition_df,
            model_name=model_name,
            representation_name=representation_name,
            output_root=output_root,
        )
        for representation_name in representation_names
    )
