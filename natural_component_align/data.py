from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from natural_sound_align.data import ComponentBundle, NORMAN_COMPONENTS_PATH, load_tuckute_stimulus_metadata


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

DEFAULT_COMPONENT_NAMES = (
    "lowfreq",
    "highfreq",
    "envsounds",
    "pitch",
    "speech",
    "music",
)


def load_component_bundle() -> ComponentBundle:
    mat = sio.loadmat(NORMAN_COMPONENTS_PATH, squeeze_me=True)

    def _unwrap_matlab_string(value):
        while isinstance(value, np.ndarray):
            if value.size != 1:
                break
            value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def _matlab_string_list(array: np.ndarray) -> list[str]:
        return [_unwrap_matlab_string(item) for item in np.ravel(array)]

    metadata = load_tuckute_stimulus_metadata("NH2015")
    component_names = tuple(_matlab_string_list(mat["component_names"]))
    stim_names = _matlab_string_list(mat["stim_names"])
    responses = np.asarray(mat["R"], dtype=np.float64)

    reorder_index = {stim_name: index for index, stim_name in enumerate(stim_names)}
    order = [reorder_index[stim_name] for stim_name in metadata["stim_name"].tolist()]
    bundle = ComponentBundle(
        responses=responses[order],
        metadata=metadata,
        component_names=component_names,
    )

    if component_names != DEFAULT_COMPONENT_NAMES:
        raise ValueError(
            "Unexpected Norman component ordering. "
            f"Expected {DEFAULT_COMPONENT_NAMES}, got {component_names}."
        )
    return bundle


def select_component_indices(
    component_names: tuple[str, ...],
    requested_components: tuple[str, ...] | list[str] | None = None,
) -> np.ndarray:
    if requested_components is None:
        return np.arange(len(component_names), dtype=int)

    lookup = {name.lower(): index for index, name in enumerate(component_names)}
    indices: list[int] = []
    for component_name in requested_components:
        normalized = component_name.strip().lower()
        if normalized not in lookup:
            raise ValueError(
                f"Unsupported component {component_name!r}. "
                f"Expected one of {sorted(component_names)}."
            )
        indices.append(lookup[normalized])
    return np.asarray(indices, dtype=int)


def component_metadata_frame(bundle: ComponentBundle, component_indices: np.ndarray | None = None) -> pd.DataFrame:
    component_indices = (
        np.arange(bundle.responses.shape[1], dtype=int)
        if component_indices is None
        else np.asarray(component_indices, dtype=int)
    )
    return pd.DataFrame(
        {
            "component_index": component_indices,
            "component_name": [bundle.component_names[index] for index in component_indices],
        }
    )
