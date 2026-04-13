from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import urllib.error
import urllib.request
import zipfile
from typing import Any

from datasets import Audio, load_dataset
from huggingface_hub import snapshot_download
import pandas as pd

from .datasets import build_vocalset_manifest
from .storage import CURRENT_DATA_ROOT, LEGACY_DATA_ROOTS, ManifestBundle, write_manifest_bundle


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = CURRENT_DATA_ROOT
DEFAULT_MANIFEST_ROOT = PACKAGE_ROOT / "manifests"
DEFAULT_CACHE_ROOT = PACKAGE_ROOT / ".hf_cache"
DEFAULT_CMI_VOCALSET_SPLITS = PACKAGE_ROOT / "cmi_manifests" / "vocalset"
DEFAULT_MTG_ROOT = DEFAULT_DATA_ROOT / "mtg-jamendo-dataset"
DEFAULT_MTG_AUDIO_ROOT = DEFAULT_DATA_ROOT / "raw_30s" / "audio-low"

MTG_GITHUB_SPLIT_BASE = (
    "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/splits/split-0"
)
MTG_TASK_DEFINITIONS = (
    {
        "dataset_name": "MTGTop50",
        "task_name": "music_tagging",
        "local_dir": "mtg_top50",
        "remote_prefix_candidates": ("autotagging_top50tags", "autotagging_top50"),
    },
    {
        "dataset_name": "MTGMoodTheme",
        "task_name": "emotion_classification",
        "local_dir": "mtg_moodtheme",
        "remote_prefix_candidates": ("autotagging_moodtheme",),
    },
    {
        "dataset_name": "MTGGenre",
        "task_name": "genre_classification",
        "local_dir": "mtg_genre",
        "remote_prefix_candidates": ("autotagging_genre",),
    },
    {
        "dataset_name": "MTGInstrument",
        "task_name": "instrument_classification",
        "local_dir": "mtg_instrument",
        "remote_prefix_candidates": ("autotagging_instrument",),
    },
)

MAGNATAGATUNE_TOP50 = tuple(
    sorted(
        [
            "guitar",
            "classical",
            "slow",
            "techno",
            "strings",
            "drums",
            "electronic",
            "rock",
            "fast",
            "piano",
            "ambient",
            "beat",
            "violin",
            "vocal",
            "synth",
            "female",
            "indian",
            "opera",
            "male",
            "singing",
            "vocals",
            "no vocals",
            "harpsichord",
            "loud",
            "quiet",
            "flute",
            "woman",
            "male vocal",
            "no vocal",
            "pop",
            "soft",
            "sitar",
            "solo",
            "man",
            "classic",
            "choir",
            "voice",
            "new age",
            "dance",
            "male voice",
            "female vocal",
            "beats",
            "harp",
            "cello",
            "no voice",
            "weird",
            "country",
            "metal",
            "female voice",
            "choral",
        ]
    )
)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _normalize_split(split_name: str) -> str:
    split_name = split_name.strip().lower()
    alias_map = {
        "validation": "valid",
        "val": "valid",
        "dev": "valid",
    }
    return alias_map.get(split_name, split_name)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_audio_payload(
    *,
    payload: dict[str, Any] | None,
    destination: Path,
    force_audio_write: bool,
) -> None:
    if destination.exists() and not force_audio_write:
        return
    if payload is None:
        raise ValueError(f"Audio payload is missing for {destination}")

    audio_bytes = payload.get("bytes")
    source_path = payload.get("path")
    _ensure_parent(destination)

    if audio_bytes is not None:
        destination.write_bytes(audio_bytes)
        return

    if source_path:
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Audio source path is missing: {source}")
        shutil.copy2(source, destination)
        return

    raise ValueError(f"Could not materialize audio for {destination}")


def _build_classification_manifest(
    *,
    dataset_name: str,
    task_name: str,
    rows: list[dict[str, Any]],
    output_root: Path,
    label_vocabulary: tuple[str, ...],
    metadata: dict[str, Any],
) -> Path:
    bundle = ManifestBundle(
        dataset_name=dataset_name,
        task_name=task_name,
        target_type="single_label_classification",
        manifest=pd.DataFrame(rows),
        label_vocabulary=label_vocabulary,
        metadata=metadata,
    )
    return write_manifest_bundle(bundle, output_root)


def _build_multilabel_manifest(
    *,
    dataset_name: str,
    task_name: str,
    rows: list[dict[str, Any]],
    output_root: Path,
    label_vocabulary: tuple[str, ...],
    metadata: dict[str, Any],
) -> Path:
    bundle = ManifestBundle(
        dataset_name=dataset_name,
        task_name=task_name,
        target_type="multi_label_classification",
        manifest=pd.DataFrame(rows),
        label_vocabulary=label_vocabulary,
        metadata=metadata,
    )
    return write_manifest_bundle(bundle, output_root)


def _build_regression_manifest(
    *,
    dataset_name: str,
    task_name: str,
    rows: list[dict[str, Any]],
    output_root: Path,
    target_names: tuple[str, ...],
    metadata: dict[str, Any],
) -> Path:
    bundle = ManifestBundle(
        dataset_name=dataset_name,
        task_name=task_name,
        target_type="regression",
        manifest=pd.DataFrame(rows),
        target_names=target_names,
        metadata=metadata,
    )
    return write_manifest_bundle(bundle, output_root)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def _download_url_to_path(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as response:
        destination.write_bytes(response.read())
    return destination


def _try_download_first(urls: list[str], destination: Path) -> tuple[str, Path]:
    last_error: Exception | None = None
    for url in urls:
        try:
            return url, _download_url_to_path(url, destination)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 404:
                raise
        except urllib.error.URLError as exc:
            last_error = exc
            raise
    raise FileNotFoundError(f"Could not download any candidate URL to {destination}: {urls}") from last_error


def _repair_manifest_audio_paths(manifest_root: Path) -> int:
    legacy_roots = [str(root) for root in LEGACY_DATA_ROOTS]
    new_root = str(DEFAULT_DATA_ROOT.resolve())
    updated_files = 0

    for manifest_csv in manifest_root.rglob("manifest.csv"):
        manifest = pd.read_csv(manifest_csv, keep_default_na=False)
        if "audio_path" not in manifest.columns:
            continue
        updated = manifest["audio_path"].astype(str)
        for legacy_root in legacy_roots:
            updated = updated.str.replace(legacy_root, new_root, regex=False)
        if not updated.equals(manifest["audio_path"].astype(str)):
            manifest["audio_path"] = updated
            manifest.to_csv(manifest_csv, index=False)
            updated_files += 1

    for metadata_json in manifest_root.rglob("task_metadata.json"):
        payload = json.loads(metadata_json.read_text(encoding="utf-8"))
        metadata = dict(payload.get("metadata", {}))
        audio_root = metadata.get("audio_root")
        if isinstance(audio_root, str):
            updated_audio_root = audio_root
            for legacy_root in legacy_roots:
                updated_audio_root = updated_audio_root.replace(legacy_root, new_root)
            if updated_audio_root != audio_root:
                metadata["audio_root"] = updated_audio_root
                payload["metadata"] = metadata
                metadata_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                updated_files += 1

    return updated_files


def _build_mtg_multilabel_manifest(
    *,
    dataset_name: str,
    task_name: str,
    split_tsv_paths: dict[str, Path],
    audio_root: Path,
    output_root: Path,
    metadata: dict[str, Any],
) -> Path:
    rows: list[dict[str, Any]] = []
    observed_labels: set[str] = set()

    for split_name, tsv_path in split_tsv_paths.items():
        with tsv_path.open("r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                if line_index == 0:
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 6:
                    continue
                audio_relpath = parts[3].strip()
                active_labels = sorted(
                    {
                        tag.split("---", 1)[-1].strip()
                        for tag in parts[5:]
                        if str(tag).strip()
                    }
                )
                if len(active_labels) == 0:
                    continue
                observed_labels.update(active_labels)
                audio_path = audio_root / audio_relpath
                if not audio_path.exists() and audio_relpath.endswith(".mp3"):
                    audio_relpath = audio_relpath[:-4] + ".low.mp3"
                    audio_path = audio_root / audio_relpath
                if not audio_path.exists():
                    raise FileNotFoundError(f"Missing MTG audio file referenced by manifest: {audio_path}")
                rows.append(
                    {
                        "sample_id": Path(audio_relpath).stem,
                        "audio_relpath": audio_relpath,
                        "audio_path": str(audio_path.resolve()),
                        "split": _normalize_split(split_name),
                        "fold": "",
                        "target_json": _json_dumps(active_labels),
                        "label_text": ", ".join(active_labels),
                    }
                )

    return _build_multilabel_manifest(
        dataset_name=dataset_name,
        task_name=task_name,
        rows=rows,
        output_root=output_root,
        label_vocabulary=tuple(sorted(observed_labels)),
        metadata=metadata,
    )


def prepare_mtg_split_metadata(
    *,
    data_root: Path,
) -> dict[str, dict[str, Path]]:
    split_root = data_root / "mtg-jamendo-dataset" / "data" / "splits" / "split-0"
    split_root.mkdir(parents=True, exist_ok=True)

    prepared: dict[str, dict[str, Path]] = {}
    for task_def in MTG_TASK_DEFINITIONS:
        split_files: dict[str, Path] = {}
        for split_name, official_name in (("train", "train"), ("valid", "validation"), ("test", "test")):
            destination = split_root / f"{task_def['local_dir']}_{split_name}.tsv"
            if not destination.exists():
                existing_source = next(
                    (
                        split_root / f"{prefix}-{official_name}.tsv"
                        for prefix in task_def["remote_prefix_candidates"]
                        if (split_root / f"{prefix}-{official_name}.tsv").exists()
                    ),
                    None,
                )
                if existing_source is not None:
                    shutil.copy2(existing_source, destination)
                else:
                    candidate_urls = [
                        f"{MTG_GITHUB_SPLIT_BASE}/{prefix}-{official_name}.tsv"
                        for prefix in task_def["remote_prefix_candidates"]
                    ]
                    _try_download_first(candidate_urls, destination)
            split_files[split_name] = destination
        prepared[task_def["local_dir"]] = split_files
    return prepared


def prepare_mtg_manifests(
    *,
    data_root: Path,
    manifest_root: Path,
    audio_root: Path | None = None,
) -> list[Path]:
    audio_root = audio_root or DEFAULT_MTG_AUDIO_ROOT
    if not audio_root.exists():
        raise FileNotFoundError(
            "MTG raw_30s/audio-low root not found. "
            f"Expected at: {audio_root}"
        )

    split_files = prepare_mtg_split_metadata(data_root=data_root)
    manifest_dirs: list[Path] = []
    for task_def in MTG_TASK_DEFINITIONS:
        manifest_dirs.append(
            _build_mtg_multilabel_manifest(
                dataset_name=task_def["dataset_name"],
                task_name=task_def["task_name"],
                split_tsv_paths=split_files[task_def["local_dir"]],
                audio_root=audio_root,
                output_root=manifest_root,
                metadata={
                    "source": "MTG/mtg-jamendo-dataset split-0",
                    "audio_root": str(audio_root.resolve()),
                    "partitioning": "official_train_valid_test",
                    "split_files": {
                        split_name: str(path.resolve())
                        for split_name, path in split_files[task_def["local_dir"]].items()
                    },
                },
            )
        )
    return manifest_dirs


def _build_vocalset_fold_manifest(
    *,
    audio_root: Path,
    output_root: Path,
    num_folds: int = 5,
) -> Path:
    rows: list[dict[str, Any]] = []
    for technique_dir in sorted(path for path in audio_root.iterdir() if path.is_dir()):
        audio_paths = sorted(technique_dir.glob("*.wav"))
        for index, audio_path in enumerate(audio_paths):
            technique = technique_dir.name
            rows.append(
                {
                    "sample_id": audio_path.stem,
                    "audio_relpath": audio_path.relative_to(audio_root).as_posix(),
                    "audio_path": str(audio_path.resolve()),
                    "split": "",
                    "fold": f"{index % num_folds:02d}",
                    "target_json": _json_dumps(technique),
                    "label_text": technique,
                    "technique": technique,
                }
            )

    return _build_classification_manifest(
        dataset_name="VocalSet",
        task_name="singing_technique",
        rows=rows,
        output_root=output_root,
        label_vocabulary=tuple(sorted({row["label_text"] for row in rows})),
        metadata={
            "audio_root": str(audio_root.resolve()),
            "partitioning": "deterministic_stratified_kfold",
            "num_folds": num_folds,
        },
    )


def _relative_audio_path(source_path: str, prefix_to_strip: str) -> str:
    normalized = source_path.strip().replace("\\", "/")
    if normalized.startswith(prefix_to_strip):
        return normalized[len(prefix_to_strip) :].lstrip("/")
    return normalized


def download_gtzan_hf(
    *,
    data_root: Path,
    manifest_root: Path,
    cache_root: Path,
    force_audio_write: bool = False,
) -> tuple[Path, Path]:
    dataset = load_dataset("confit/gtzan-parquet", cache_dir=str(cache_root / "gtzan"))
    dataset = dataset.cast_column("audio", Audio(decode=False))

    audio_root = data_root / "GTZAN" / "audio"
    rows: list[dict[str, Any]] = []
    observed_labels: set[str] = set()

    for split_name, split_dataset in dataset.items():
        normalized_split = _normalize_split(split_name)
        for row in split_dataset:
            genre = str(row["genre"])
            observed_labels.add(genre)
            payload = row["audio"]
            source_name = str((payload or {}).get("path") or f"{genre}_{len(rows):06d}.au")
            destination = audio_root / genre / Path(source_name).name
            _write_audio_payload(payload=payload, destination=destination, force_audio_write=force_audio_write)
            rows.append(
                {
                    "sample_id": destination.stem,
                    "audio_relpath": destination.relative_to(audio_root).as_posix(),
                    "audio_path": str(destination.resolve()),
                    "split": normalized_split,
                    "fold": "",
                    "target_json": _json_dumps(genre),
                    "label_text": genre,
                    "genre": genre,
                }
            )

    manifest_dir = _build_classification_manifest(
        dataset_name="GTZAN",
        task_name="genre_classification",
        rows=rows,
        output_root=manifest_root,
        label_vocabulary=tuple(sorted(observed_labels)),
        metadata={
            "source": "confit/gtzan-parquet",
            "audio_root": str(audio_root.resolve()),
            "partitioning": "huggingface_splits",
        },
    )
    return audio_root, manifest_dir


def download_vocalset_hf(
    *,
    data_root: Path,
    manifest_root: Path,
    cache_root: Path,
    force_audio_write: bool = False,
    split_jsonl_dir: Path = DEFAULT_CMI_VOCALSET_SPLITS,
) -> tuple[Path, Path]:
    dataset = load_dataset("Bill13579/vocalset-mirror", cache_dir=str(cache_root / "vocalset"))
    dataset = dataset.cast_column("audio", Audio(decode=False))

    label_feature = dataset["train"].features["label"]
    label_names = getattr(label_feature, "names", None)
    if not label_names:
        raise ValueError("VocalSet label names are unavailable in the Hugging Face mirror.")

    audio_root = data_root / "VocalSet" / "audio"
    for row in dataset["train"]:
        label_id = int(row["label"])
        technique = str(label_names[label_id])
        payload = row["audio"]
        file_name = str((payload or {}).get("path") or f"{technique}_{label_id:03d}_{hash(str(row)) & 0xFFFFFFFF:08x}.wav")
        destination = audio_root / technique / Path(file_name).name
        _write_audio_payload(payload=payload, destination=destination, force_audio_write=force_audio_write)

    manifest_dir = build_vocalset_manifest(
        audio_root=audio_root,
        output_root=manifest_root,
        split_jsonl_dir=split_jsonl_dir if split_jsonl_dir.exists() else None,
    )
    manifest = pd.read_csv(manifest_dir / "manifest.csv", keep_default_na=False)
    specified_ratio = float((manifest["split"] != "unspecified").mean())
    if specified_ratio < 0.5:
        manifest_dir = _build_vocalset_fold_manifest(audio_root=audio_root, output_root=manifest_root)
    return audio_root, manifest_dir


def download_magnatagatune_hf(
    *,
    data_root: Path,
    manifest_root: Path,
    cache_root: Path,
    force_audio_write: bool = False,
) -> tuple[Path, Path]:
    local_dir = data_root / "MagnaTagATune"
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="confit/magnatagatune",
        repo_type="dataset",
        allow_patterns=[
            "annotations_final.csv",
            "train_gt_mtt.tsv",
            "val_gt_mtt.tsv",
            "test_gt_mtt.tsv",
            "mp3.zip",
        ],
        local_dir=str(local_dir),
        cache_dir=str(cache_root / "magnatagatune"),
        local_dir_use_symlinks=False,
    )

    archive_path = local_dir / "mp3.zip"
    audio_root = local_dir / "mp3"
    if force_audio_write or not audio_root.exists():
        audio_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(audio_root)

    annotations = pd.read_csv(local_dir / "annotations_final.csv", sep="\t")
    annotations = annotations[annotations[list(MAGNATAGATUNE_TOP50)].sum(axis=1) > 0].reset_index(drop=True)

    split_tables = {
        "train": pd.read_csv(local_dir / "train_gt_mtt.tsv", sep="\t", header=None),
        "valid": pd.read_csv(local_dir / "val_gt_mtt.tsv", sep="\t", header=None),
        "test": pd.read_csv(local_dir / "test_gt_mtt.tsv", sep="\t", header=None),
    }

    rows: list[dict[str, Any]] = []
    for split_name, split_table in split_tables.items():
        split_ids = set(split_table[0].astype(int).tolist())
        split_df = annotations[annotations["clip_id"].astype(int).isin(split_ids)].reset_index(drop=True)
        for _, row in split_df.iterrows():
            active_labels = [label for label in MAGNATAGATUNE_TOP50 if int(row[label]) == 1]
            audio_path = audio_root / str(row["mp3_path"])
            if not audio_path.exists():
                raise FileNotFoundError(f"MagnaTagATune audio file missing after extraction: {audio_path}")
            rows.append(
                {
                    "sample_id": str(int(row["clip_id"])),
                    "audio_relpath": audio_path.relative_to(audio_root).as_posix(),
                    "audio_path": str(audio_path.resolve()),
                    "split": split_name,
                    "fold": "",
                    "target_json": _json_dumps(active_labels),
                    "label_text": ", ".join(active_labels),
                }
            )

    manifest_dir = _build_multilabel_manifest(
        dataset_name="MagnaTagATune",
        task_name="music_tagging",
        rows=rows,
        output_root=manifest_root,
        label_vocabulary=MAGNATAGATUNE_TOP50,
        metadata={
            "source": "confit/magnatagatune",
            "audio_root": str(audio_root.resolve()),
            "partitioning": "official_train_valid_test",
            "num_tags": len(MAGNATAGATUNE_TOP50),
        },
    )
    return audio_root, manifest_dir


def download_emomusic_map(
    *,
    data_root: Path,
    manifest_root: Path,
    cache_root: Path,
) -> tuple[Path, Path]:
    local_dir = data_root / "EMO"
    snapshot_download(
        repo_id="m-a-p/EMO",
        repo_type="dataset",
        allow_patterns=[
            "EMO.train.jsonl",
            "EMO.val.jsonl",
            "EMO.test.jsonl",
            "emomusic/wav/*",
        ],
        local_dir=str(local_dir),
        cache_dir=str(cache_root / "emo"),
        local_dir_use_symlinks=False,
    )

    audio_root = local_dir / "emomusic" / "wav"
    split_map = {
        "train": local_dir / "EMO.train.jsonl",
        "valid": local_dir / "EMO.val.jsonl",
        "test": local_dir / "EMO.test.jsonl",
    }
    rows: list[dict[str, Any]] = []
    for split_name, jsonl_path in split_map.items():
        for payload in _read_jsonl(jsonl_path):
            audio_relpath = _relative_audio_path(str(payload["audio_path"]), "data/EMO/")
            audio_path = local_dir / audio_relpath
            if not audio_path.exists():
                raise FileNotFoundError(f"EMO audio file missing: {audio_path}")
            label = payload["label"]
            valence = float(label[0])
            arousal = float(label[1])
            rows.append(
                {
                    "sample_id": audio_path.stem,
                    "audio_relpath": audio_path.relative_to(audio_root).as_posix(),
                    "audio_path": str(audio_path.resolve()),
                    "split": split_name,
                    "fold": "",
                    "target_json": _json_dumps([valence, arousal]),
                    "label_text": f"valence={valence:.6f}, arousal={arousal:.6f}",
                    "valence": valence,
                    "arousal": arousal,
                }
            )

    manifest_dir = _build_regression_manifest(
        dataset_name="EMO",
        task_name="emotion_regression",
        rows=rows,
        output_root=manifest_root,
        target_names=("valence", "arousal"),
        metadata={
            "source": "m-a-p/EMO",
            "audio_root": str(audio_root.resolve()),
            "partitioning": "official_train_valid_test",
        },
    )
    return audio_root, manifest_dir


def download_giantsteps_map(
    *,
    data_root: Path,
    manifest_root: Path,
    cache_root: Path,
) -> tuple[Path, Path]:
    local_dir = data_root / "GiantSteps"
    snapshot_download(
        repo_id="m-a-p/GS",
        repo_type="dataset",
        allow_patterns=[
            "GS.train.jsonl",
            "GS.val.jsonl",
            "GS.test.jsonl",
            "giantsteps_clips/wav/*",
        ],
        local_dir=str(local_dir),
        cache_dir=str(cache_root / "giantsteps"),
        local_dir_use_symlinks=False,
    )

    audio_root = local_dir / "giantsteps_clips" / "wav"
    split_map = {
        "train": local_dir / "GS.train.jsonl",
        "valid": local_dir / "GS.val.jsonl",
        "test": local_dir / "GS.test.jsonl",
    }
    rows: list[dict[str, Any]] = []
    observed_labels: set[str] = set()
    for split_name, jsonl_path in split_map.items():
        for payload in _read_jsonl(jsonl_path):
            key_label = str(payload["label"])
            observed_labels.add(key_label)
            audio_relpath = _relative_audio_path(str(payload["audio_path"]), "data/GS/")
            audio_path = local_dir / audio_relpath
            if not audio_path.exists():
                raise FileNotFoundError(f"GiantSteps audio file missing: {audio_path}")
            rows.append(
                {
                    "sample_id": audio_path.stem,
                    "audio_relpath": audio_path.relative_to(audio_root).as_posix(),
                    "audio_path": str(audio_path.resolve()),
                    "split": split_name,
                    "fold": "",
                    "target_json": _json_dumps(key_label),
                    "label_text": key_label,
                    "key": key_label,
                    "ori_uid": str(payload.get("ori_uid", "")),
                }
            )

    manifest_dir = _build_classification_manifest(
        dataset_name="GiantSteps",
        task_name="key_detection",
        rows=rows,
        output_root=manifest_root,
        label_vocabulary=tuple(sorted(observed_labels)),
        metadata={
            "source": "m-a-p/GS",
            "audio_root": str(audio_root.resolve()),
            "partitioning": "official_train_valid_test",
        },
    )
    return audio_root, manifest_dir


def prepare_nsynth_pitch_local(
    *,
    nsynth_root: Path,
    manifest_root: Path,
) -> tuple[Path, Path]:
    audio_root = nsynth_root / "48000"
    if not audio_root.exists():
        raise FileNotFoundError(f"NSynth pitch audio root not found: {audio_root}")

    split_files = {
        "train": nsynth_root / "train.json",
        "valid": nsynth_root / "valid.json",
        "test": nsynth_root / "test.json",
    }
    rows: list[dict[str, Any]] = []
    for split_name, json_path in split_files.items():
        mapping = json.loads(json_path.read_text(encoding="utf-8"))
        for file_name, label_values in mapping.items():
            pitch_midi = int(label_values[0])
            audio_path = audio_root / split_name / file_name
            if not audio_path.exists():
                raise FileNotFoundError(f"NSynth pitch audio file missing: {audio_path}")
            rows.append(
                {
                    "sample_id": audio_path.stem,
                    "audio_relpath": audio_path.relative_to(audio_root).as_posix(),
                    "audio_path": str(audio_path.resolve()),
                    "split": split_name,
                    "fold": "",
                    "target_json": _json_dumps(pitch_midi),
                    "label_text": str(pitch_midi),
                    "pitch_midi": pitch_midi,
                }
            )

    manifest_dir = _build_regression_manifest(
        dataset_name="NSynthPitch5h",
        task_name="pitch_estimation",
        rows=rows,
        output_root=manifest_root,
        target_names=("pitch_midi",),
        metadata={
            "source": str(nsynth_root.resolve()),
            "audio_root": str(audio_root.resolve()),
            "partitioning": "official_train_valid_test",
        },
    )
    return audio_root, manifest_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download easy downstream datasets and prepare manifests.")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=[
            "gtzan",
            "vocalset",
            "magnatagatune",
            "emomusic",
            "giantsteps",
            "nsynth_pitch",
            "mtg_metadata",
            "mtg_moodtheme",
            "mtg_top50",
            "mtg_genre",
            "mtg_instrument",
        ],
        required=True,
        help="Dataset(s) to download and prepare.",
    )
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--manifest-root", type=str, default=str(DEFAULT_MANIFEST_ROOT))
    parser.add_argument("--cache-root", type=str, default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--force-audio-write", action="store_true")
    parser.add_argument(
        "--mtg-audio-root",
        type=str,
        default=str(DEFAULT_MTG_AUDIO_ROOT),
        help="Official MTG-Jamendo raw_30s/audio-low root used for MTG manifests.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_root = Path(args.data_root)
    manifest_root = Path(args.manifest_root)
    cache_root = Path(args.cache_root)
    mtg_audio_root = Path(args.mtg_audio_root)
    data_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    repaired_count = _repair_manifest_audio_paths(manifest_root)
    if repaired_count:
        print(f"repaired_manifest_paths={repaired_count}")

    for dataset_name in args.dataset:
        if dataset_name == "gtzan":
            audio_root, manifest_dir = download_gtzan_hf(
                data_root=data_root,
                manifest_root=manifest_root,
                cache_root=cache_root,
                force_audio_write=args.force_audio_write,
            )
        elif dataset_name == "vocalset":
            audio_root, manifest_dir = download_vocalset_hf(
                data_root=data_root,
                manifest_root=manifest_root,
                cache_root=cache_root,
                force_audio_write=args.force_audio_write,
            )
        elif dataset_name == "magnatagatune":
            audio_root, manifest_dir = download_magnatagatune_hf(
                data_root=data_root,
                manifest_root=manifest_root,
                cache_root=cache_root,
                force_audio_write=args.force_audio_write,
            )
        elif dataset_name == "emomusic":
            audio_root, manifest_dir = download_emomusic_map(
                data_root=data_root,
                manifest_root=manifest_root,
                cache_root=cache_root,
            )
        elif dataset_name == "giantsteps":
            audio_root, manifest_dir = download_giantsteps_map(
                data_root=data_root,
                manifest_root=manifest_root,
                cache_root=cache_root,
            )
        elif dataset_name == "nsynth_pitch":
            audio_root, manifest_dir = prepare_nsynth_pitch_local(
                nsynth_root=data_root / "nsynth_pitch-v2.2.3-5h",
                manifest_root=manifest_root,
            )
        elif dataset_name == "mtg_metadata":
            split_files = prepare_mtg_split_metadata(data_root=data_root)
            print(json.dumps({key: {split: str(path) for split, path in value.items()} for key, value in split_files.items()}, indent=2))
            continue
        elif dataset_name in {"mtg_moodtheme", "mtg_top50", "mtg_genre", "mtg_instrument"}:
            manifest_dirs = prepare_mtg_manifests(
                data_root=data_root,
                manifest_root=manifest_root,
                audio_root=mtg_audio_root,
            )
            wanted = {
                "mtg_moodtheme": "MTGMoodTheme",
                "mtg_top50": "MTGTop50",
                "mtg_genre": "MTGGenre",
                "mtg_instrument": "MTGInstrument",
            }[dataset_name]
            manifest_dir = next(path for path in manifest_dirs if path.parent.name == wanted.lower())
            audio_root = mtg_audio_root
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        print(f"{dataset_name}: audio_root={audio_root}")
        print(f"{dataset_name}: manifest_dir={manifest_dir}")


if __name__ == "__main__":
    main()
