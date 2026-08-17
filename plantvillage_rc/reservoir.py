from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np


Variant = Literal["full", "tonic", "event"]
Nonlinearity = Literal["saturating", "linear"]
DeviceProfile = Literal["uniform", "row-spectrum"]


def variant_channels(variant: Variant) -> tuple[int, ...]:
    channels = {
        "full": tuple(range(9)),
        "tonic": (0, 1, 2),
        "event": (3, 4, 5, 6, 7, 8),
    }
    try:
        return channels[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown input variant: {variant}") from exc


def generate_mask(
    seed: int = 42,
    *,
    rows: int = 12,
    channels: int = 9,
    kind: Literal["continuous", "binary"] = "continuous",
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "continuous":
        return rng.uniform(0.5, 1.5, size=(rows, channels)).astype(np.float32)
    if kind == "binary":
        return rng.integers(0, 2, size=(rows, channels)).astype(np.float32)
    raise ValueError(f"Unknown mask kind: {kind}")


def row_spectrum(
    alpha_min: float = 0.05,
    alpha_max: float = 0.98,
    *,
    rows: int = 12,
    channels: int = 9,
) -> np.ndarray:
    """Build a deterministic bank of decay constants across physical rows."""
    if not 0.0 <= alpha_min < alpha_max < 1.0:
        raise ValueError("alpha bounds must satisfy 0 <= alpha_min < alpha_max < 1")
    values = np.linspace(alpha_min, alpha_max, rows, dtype=np.float32)
    return np.repeat(values[:, None], channels, axis=1)


def dynamics_parameters(
    profile: DeviceProfile,
    *,
    alpha: float = 0.75,
    alpha_min: float = 0.05,
    alpha_max: float = 0.98,
    gamma: float = 1.0,
) -> tuple[float | np.ndarray, float]:
    if profile == "uniform":
        return alpha, gamma
    if profile == "row-spectrum":
        return row_spectrum(alpha_min, alpha_max), gamma
    raise ValueError(f"Unknown device profile: {profile}")


def _node_parameter(
    value: float | np.ndarray,
    *,
    name: str,
    lower: float,
    upper: float | None,
) -> np.ndarray:
    parameter = np.asarray(value, dtype=np.float32)
    if parameter.ndim == 0:
        parameter = np.full((12, 9), float(parameter), dtype=np.float32)
    elif parameter.shape != (12, 9):
        raise ValueError(f"{name} must be a scalar or have shape [12, 9], got {parameter.shape}")
    if np.any(parameter < lower) or (upper is not None and np.any(parameter >= upper)):
        relation = f"{lower} <= {name} < {upper}" if upper is not None else f"{name} >= {lower}"
        raise ValueError(f"All node values must satisfy {relation}")
    return parameter


def reservoir_states(
    inputs: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float | np.ndarray = 0.75,
    gamma: float | np.ndarray = 1.0,
    variant: Variant = "full",
    nonlinearity: Nonlinearity = "saturating",
    batch_size: int = 4096,
    initial_state: np.ndarray | None = None,
    initial_decay_steps: int = 0,
    return_state_matrix: bool = False,
) -> np.ndarray:
    inputs = np.asarray(inputs, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    if inputs.ndim != 3 or inputs.shape[1:] != (12, 9):
        raise ValueError(f"Expected inputs shape [N, 12, 9], got {inputs.shape}")
    if mask.shape != (12, 9):
        raise ValueError(f"Expected mask shape [12, 9], got {mask.shape}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if initial_decay_steps < 0:
        raise ValueError("initial_decay_steps must be non-negative")

    channels = np.asarray(variant_channels(variant))
    selected_mask = mask[:, channels]
    alpha_nodes = _node_parameter(
        alpha, name="alpha", lower=0.0, upper=1.0
    )[:, channels]
    gamma_nodes = _node_parameter(
        gamma, name="gamma", lower=np.finfo(np.float32).tiny, upper=None
    )[:, channels]
    expected_initial_shape = (len(inputs), 12, len(channels))
    if initial_state is not None:
        initial_state = np.asarray(initial_state, dtype=np.float32)
        if initial_state.shape != expected_initial_shape:
            raise ValueError(
                f"Expected initial_state shape {expected_initial_shape}, got {initial_state.shape}"
            )
    output_shape = expected_initial_shape if return_state_matrix else (
        len(inputs),
        12 * len(channels),
    )
    states = np.empty(output_shape, dtype=np.float32)
    for start in range(0, len(inputs), batch_size):
        stop = min(start + batch_size, len(inputs))
        batch = inputs[start:stop, :, channels]
        if initial_state is None:
            state = np.zeros((stop - start, 12, len(channels)), dtype=np.float32)
        else:
            state = initial_state[start:stop].copy()
            if initial_decay_steps:
                state *= alpha_nodes[None, :, :] ** initial_decay_steps
        for time_step in range(12):
            voltage = batch[:, time_step, None, :] * selected_mask[None, :, :]
            if nonlinearity == "saturating":
                response = -np.expm1(-gamma_nodes[None, :, :] * voltage)
            elif nonlinearity == "linear":
                response = voltage
            else:
                raise ValueError(f"Unknown nonlinearity: {nonlinearity}")
            state = (
                alpha_nodes[None, :, :] * state
                + (1.0 - alpha_nodes[None, :, :]) * response
            )
        states[start:stop] = state if return_state_matrix else state.reshape(stop - start, -1)
    return states


def dual_pass_states(
    rgb_inputs: np.ndarray,
    texture_inputs: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float | np.ndarray,
    gamma: float | np.ndarray,
    gap_steps: int = 0,
    carry_state: bool = True,
    nonlinearity: Nonlinearity = "saturating",
    batch_size: int = 4096,
) -> dict[str, np.ndarray]:
    if np.shape(rgb_inputs) != np.shape(texture_inputs):
        raise ValueError("RGB and texture inputs must have identical shapes")
    rgb_matrix = reservoir_states(
        rgb_inputs,
        mask,
        alpha=alpha,
        gamma=gamma,
        nonlinearity=nonlinearity,
        batch_size=batch_size,
        return_state_matrix=True,
    )
    texture_matrix = reservoir_states(
        texture_inputs,
        mask,
        alpha=alpha,
        gamma=gamma,
        nonlinearity=nonlinearity,
        batch_size=batch_size,
        initial_state=rgb_matrix if carry_state else None,
        initial_decay_steps=gap_steps if carry_state else 0,
        return_state_matrix=True,
    )
    rgb_states = rgb_matrix.reshape(len(rgb_matrix), -1)
    texture_states = texture_matrix.reshape(len(texture_matrix), -1)
    return {
        "rgb_states": rgb_states,
        "texture_states": texture_states,
        "combined_states": np.concatenate([rgb_states, texture_states], axis=1),
    }


def save_reservoir_states(
    input_dir: Path,
    output_dir: Path,
    *,
    quantile_tag: str,
    alpha: float,
    gamma: float,
    variant: Variant,
    nonlinearity: Nonlinearity,
    mask_seed: int = 42,
    mask_kind: Literal["continuous", "binary"] = "continuous",
    device_profile: DeviceProfile = "uniform",
    alpha_min: float = 0.05,
    alpha_max: float = 0.98,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mask = generate_mask(mask_seed, kind=mask_kind)
    np.save(output_dir / "mask.npy", mask)
    alpha_parameter, gamma_parameter = dynamics_parameters(
        device_profile,
        alpha=alpha,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        gamma=gamma,
    )
    alpha_matrix = _node_parameter(
        alpha_parameter, name="alpha", lower=0.0, upper=1.0
    )
    gamma_matrix = _node_parameter(
        gamma_parameter, name="gamma", lower=np.finfo(np.float32).tiny, upper=None
    )
    np.save(output_dir / "alpha_nodes.npy", alpha_matrix)
    np.save(output_dir / "gamma_nodes.npy", gamma_matrix)
    sizes: dict[str, int] = {}
    for split in ("train", "val", "test"):
        inputs = np.load(input_dir / f"{split}_inputs_{quantile_tag}.npy", mmap_mode="r")
        states = reservoir_states(
            inputs,
            mask,
            alpha=alpha_parameter,
            gamma=gamma_parameter,
            variant=variant,
            nonlinearity=nonlinearity,
        )
        np.save(output_dir / f"{split}_states.npy", states)
        sizes[split] = len(states)
    return {
        "quantile_tag": quantile_tag,
        "alpha": alpha,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "gamma": gamma,
        "device_profile": device_profile,
        "alpha_nodes": alpha_matrix.tolist(),
        "gamma_nodes": gamma_matrix.tolist(),
        "variant": variant,
        "nonlinearity": nonlinearity,
        "mask_seed": mask_seed,
        "mask_kind": mask_kind,
        "feature_dimension": int(12 * len(variant_channels(variant))),
        "split_sizes": sizes,
    }
