from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
import pickle
import tarfile
from typing import Any

import numpy as np
import pandas as pd
import scipy.io as sio


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"

# Norman 2015 stimulus metadata + audio
NORMAN_ROOT = DATASET_ROOT / "norman2015"
NORMAN_DATA_MATRIX_PATH = NORMAN_ROOT / "norman2015_data_matrix.mat"
NORMAN_COMPONENTS_PATH = NORMAN_ROOT / "norman2015_components.mat"
NORMAN_WAV_DIR = NORMAN_ROOT / "stimuli" / "wav"

# Tuckute 2023 release (NH2015/B2021)
TUCKUTE_DATA_TAR_PATH = DATASET_ROOT / "tuckute2023" / "tuckute2023_data.tar"
TUCKUTE_SOUND_CATEGORY_ORDER = (
    "Music",
    "Song",
    "EngSpeech",
    "ForSpeech",
    "HumVoc",
    "AniVoc",
    "HumNonVoc",
    "AniNonVoc",
    "Nature",
    "Mechanical",
    "EnvSound",
)


@dataclass(frozen=True)
class BrainDataBundle:
    dataset_name: str
    responses: np.ndarray
    metadata: pd.DataFrame
    voxel_ids: np.ndarray
    repetition_labels: tuple[str, ...]
    valid_voxel_mask: np.ndarray
    voxel_metadata: pd.DataFrame


@dataclass(frozen=True)
class ComponentBundle:
    responses: np.ndarray
    metadata: pd.DataFrame
    component_names: tuple[str, ...]


def normalize_dataset_name(dataset_name: str) -> str:
    normalized = dataset_name.strip().upper()
    alias_map = {
        "N2015": "NH2015",
        "NH2015": "NH2015",
        "B2021": "B2021",
    }
    if normalized not in alias_map:
        raise ValueError(f"Unsupported dataset {dataset_name!r}. Expected one of: {sorted(alias_map)}")
    return alias_map[normalized]


def _unwrap_matlab_string(value: Any) -> str:
    while isinstance(value, np.ndarray):
        if value.size != 1:
            break
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _matlab_string_list(array: np.ndarray) -> list[str]:
    return [_unwrap_matlab_string(item) for item in np.ravel(array)]


def _read_tar_bytes(member_name: str) -> bytes:
    with tarfile.open(TUCKUTE_DATA_TAR_PATH, "r") as tar_file:
        extracted = tar_file.extractfile(member_name)
        if extracted is None:
            raise FileNotFoundError(f"Member {member_name!r} not found in {TUCKUTE_DATA_TAR_PATH}")
        return extracted.read()


def _load_tar_npy(member_name: str) -> np.ndarray:
    return np.load(io.BytesIO(_read_tar_bytes(member_name)), allow_pickle=True)


def _load_tar_pickle(member_name: str) -> object:
    return pickle.loads(_read_tar_bytes(member_name))


def _load_tar_mat(member_name: str) -> dict[str, Any]:
    return sio.loadmat(io.BytesIO(_read_tar_bytes(member_name)), squeeze_me=True)


def load_local_norman_metadata(
    data_mat_path: Path | None = None,
    wav_dir: Path | None = None,
) -> pd.DataFrame:
    data_mat_path = data_mat_path or NORMAN_DATA_MATRIX_PATH
    wav_dir = wav_dir or NORMAN_WAV_DIR

    mat = sio.loadmat(data_mat_path, squeeze_me=True)
    stim_names = _matlab_string_list(mat["stim_names"])
    category_labels = _matlab_string_list(mat["category_labels"])

    rows: list[dict[str, object]] = []
    for stimulus_index, (stim_name, category_label) in enumerate(
        zip(stim_names, category_labels, strict=True)
    ):
        wav_path = wav_dir / f"{stim_name}.wav"
        rows.append(
            {
                "stimulus_index": stimulus_index,
                "stim_name": stim_name,
                "category_label": category_label,
                "wav_path": str(wav_path.resolve()),
            }
        )

    return pd.DataFrame(rows)


def load_norman_metadata() -> pd.DataFrame:
    return load_local_norman_metadata()


def load_tuckute_stimulus_metadata(dataset_name: str = "NH2015") -> pd.DataFrame:
    dataset_name = normalize_dataset_name(dataset_name)
    stimulus_df = _load_tar_pickle("data/stimuli/df_stimuli_meta.pkl")
    if not isinstance(stimulus_df, pd.DataFrame):
        raise TypeError("Expected data/stimuli/df_stimuli_meta.pkl to contain a pandas DataFrame.")

    if dataset_name == "NH2015":
        neural_stim_meta = _load_tar_npy("data/neural/NH2015/neural_stim_meta.npy")
        stimulus_ids = [row[0][:-4].decode("utf-8") for row in neural_stim_meta]
    else:
        stim_info = sio.loadmat(io.BytesIO(_read_tar_bytes("data/neural/B2021/stim_info_v4.mat")))["stim_info"]
        stimulus_ids = [item[0][0].astype(str) for item in stim_info["stim_names"][0][0][:165]]

    rows: list[dict[str, object]] = []
    for stimulus_index, stim_name in enumerate(stimulus_ids):
        wav_path = WAV_DIR / f"{stim_name}.wav"
        rows.append(
            {
                "stimulus_index": stimulus_index,
                "stim_name": str(stim_name),
                "category_label": str(stimulus_df.loc[str(stim_name), "category_label"]),
                "wav_path": str(wav_path.resolve()),
            }
        )
    return pd.DataFrame(rows)


def category_order(metadata: pd.DataFrame) -> list[str]:
    labels = metadata["category_label"].astype(str).tolist()
    ordered = [label for label in TUCKUTE_SOUND_CATEGORY_ORDER if label in labels]
    for label in labels:
        if label not in ordered:
            ordered.append(label)
    return ordered


def category_sort_index(metadata: pd.DataFrame) -> np.ndarray:
    order = category_order(metadata)
    category_to_rank = {label: index for index, label in enumerate(order)}
    pairs = [
        (category_to_rank[row.category_label], row.stimulus_index, int(row.Index))
        for row in metadata.reset_index().itertuples()
    ]
    pairs.sort()
    return np.asarray([index for _, _, index in pairs], dtype=int)


def load_brain_voxel_data(
    dataset_name: str = "NH2015",
    require_complete_repetitions: bool = True,
) -> BrainDataBundle:
    dataset_name = normalize_dataset_name(dataset_name)
    metadata = load_tuckute_stimulus_metadata(dataset_name)

    if dataset_name == "NH2015":
        responses = np.asarray(_load_tar_npy("data/neural/NH2015/voxel_features_array.npy"), dtype=np.float64)
        voxel_meta_struct = _load_tar_npy("data/neural/NH2015/voxel_features_meta.npy")
        voxel_metadata = _load_tar_pickle("data/neural/NH2015/df_roi_meta.pkl")
        valid_voxel_mask = np.asarray(voxel_meta_struct["n_reps"] == 3, dtype=bool)
        if require_complete_repetitions:
            responses = responses[:, valid_voxel_mask, :]
            voxel_meta_struct = voxel_meta_struct[valid_voxel_mask]
            voxel_metadata = voxel_metadata.copy().reset_index(drop=True)
        voxel_ids = np.asarray(voxel_meta_struct["voxel_id"], dtype=int)
    else:
        responses = np.asarray(_load_tar_npy("data/neural/B2021/voxel_features_array.npy"), dtype=np.float64)[:165, :, :]
        voxel_ids = np.asarray(_load_tar_npy("data/neural/B2021/voxel_features_meta.npy"), dtype=int)
        voxel_metadata = _load_tar_pickle("data/neural/B2021/df_roi_meta.pkl").copy().reset_index(drop=True)
        valid_voxel_mask = np.ones(responses.shape[1], dtype=bool)

    repetition_labels = tuple(f"rep_{index}" for index in range(responses.shape[2]))
    return BrainDataBundle(
        dataset_name=dataset_name,
        responses=responses,
        metadata=metadata,
        voxel_ids=voxel_ids,
        repetition_labels=repetition_labels,
        valid_voxel_mask=valid_voxel_mask,
        voxel_metadata=voxel_metadata,
    )


def load_component_responses(
    components_mat_path: Path | None = None,
) -> ComponentBundle:
    if components_mat_path is not None:
        mat = sio.loadmat(components_mat_path, squeeze_me=True)
    else:
        mat = _load_tar_mat("data/neural/NH2015comp/components.mat")

    metadata = load_tuckute_stimulus_metadata("NH2015")
    responses = np.asarray(mat["R"], dtype=np.float64)
    component_names = tuple(_matlab_string_list(mat["component_names"]))
    stim_names = _matlab_string_list(mat["stim_names"])

    if stim_names != metadata["stim_name"].tolist():
        raise ValueError("Component stimulus order does not match Tuckute stimulus metadata.")

    return ComponentBundle(
        responses=responses,
        metadata=metadata,
        component_names=component_names,
    )
