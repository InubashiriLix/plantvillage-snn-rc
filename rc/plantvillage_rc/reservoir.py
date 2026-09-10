from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np


Variant = Literal["full", "tonic", "event"]
Nonlinearity = Literal["saturating", "linear"]
DeviceProfile = Literal["uniform", "row-spectrum"]


def variant_channels(variant: Variant, input_channels: int = 9) -> tuple[int, ...]:
    channels = {
        "full": tuple(range(input_channels)),
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


def time_constant_spectrum(
    tau_min: float = 1.0,
    tau_max: float = 128.0,
    *,
    rows: int = 12,
    channels: int = 12,
) -> np.ndarray:
    """Create logarithmically spaced memory constants and return alpha=exp(-1/tau)."""
    if tau_min <= 0 or tau_max <= tau_min:
        raise ValueError("tau bounds must satisfy 0 < tau_min < tau_max")
    tau = np.geomspace(tau_min, tau_max, rows).astype(np.float32)
    alpha = np.exp(-1.0 / tau).astype(np.float32)
    return np.repeat(alpha[:, None], channels, axis=1)


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
    rows: int = 12,
    channels: int = 9,
) -> np.ndarray:
    parameter = np.asarray(value, dtype=np.float32)
    if parameter.ndim == 0:
        parameter = np.full((rows, channels), float(parameter), dtype=np.float32)
    elif parameter.shape != (rows, channels):
        raise ValueError(
            f"{name} must be a scalar or have shape [{rows}, {channels}], "
            f"got {parameter.shape}"
        )
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
    checkpoint_steps: tuple[int, ...] | None = None,
) -> np.ndarray:
    inputs = np.asarray(inputs, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    if inputs.ndim != 3:
        raise ValueError(f"Expected inputs shape [N, T, C], got {inputs.shape}")
    reservoir_rows, input_channels = mask.shape if mask.ndim == 2 else (-1, -1)
    if mask.ndim != 2 or input_channels != inputs.shape[2]:
        raise ValueError(
            f"Mask must have shape [rows, {inputs.shape[2]}], got {mask.shape}"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if initial_decay_steps < 0:
        raise ValueError("initial_decay_steps must be non-negative")
    checkpoints = tuple(checkpoint_steps or ())
    if checkpoints:
        if return_state_matrix:
            raise ValueError("return_state_matrix and checkpoint_steps cannot be combined")
        if tuple(sorted(set(checkpoints))) != checkpoints:
            raise ValueError("checkpoint_steps must be sorted and unique")
        if checkpoints[0] < 1 or checkpoints[-1] > inputs.shape[1]:
            raise ValueError("checkpoint_steps must be within the input sequence")

    channels = np.asarray(variant_channels(variant, input_channels))
    if np.any(channels >= input_channels):
        raise ValueError(f"Variant {variant} requires channels absent from the input")
    selected_mask = mask[:, channels]
    alpha_nodes = _node_parameter(
        alpha,
        name="alpha",
        lower=0.0,
        upper=1.0,
        rows=reservoir_rows,
        channels=input_channels,
    )[:, channels]
    gamma_nodes = _node_parameter(
        gamma,
        name="gamma",
        lower=np.finfo(np.float32).tiny,
        upper=None,
        rows=reservoir_rows,
        channels=input_channels,
    )[:, channels]
    expected_initial_shape = (len(inputs), reservoir_rows, len(channels))
    if initial_state is not None:
        initial_state = np.asarray(initial_state, dtype=np.float32)
        if initial_state.shape != expected_initial_shape:
            raise ValueError(
                f"Expected initial_state shape {expected_initial_shape}, got {initial_state.shape}"
            )
    if checkpoints:
        output_shape = (len(inputs), len(checkpoints) * reservoir_rows * len(channels))
    elif return_state_matrix:
        output_shape = expected_initial_shape
    else:
        output_shape = (len(inputs), reservoir_rows * len(channels))
    states = np.empty(output_shape, dtype=np.float32)
    for start in range(0, len(inputs), batch_size):
        stop = min(start + batch_size, len(inputs))
        batch = inputs[start:stop, :, channels]
        if initial_state is None:
            state = np.zeros(
                (stop - start, reservoir_rows, len(channels)), dtype=np.float32
            )
        else:
            state = initial_state[start:stop].copy()
            if initial_decay_steps:
                state *= alpha_nodes[None, :, :] ** initial_decay_steps
        captured = []
        for time_step in range(inputs.shape[1]):
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
            if time_step + 1 in checkpoints:
                captured.append(state.copy())
        if checkpoints:
            states[start:stop] = np.stack(captured, axis=1).reshape(stop - start, -1)
        else:
            states[start:stop] = (
                state if return_state_matrix else state.reshape(stop - start, -1)
            )
    return states


def reservoir_states_to_memmap(
    inputs: np.ndarray,
    output_path: Path | str,
    mask: np.ndarray,
    *,
    checkpoint_steps: tuple[int, ...],
    alpha: float | np.ndarray,
    gamma: float | np.ndarray,
    batch_size: int = 512,
) -> np.memmap:
    """Generate checkpoint features in bounded memory and persist as a .npy memmap."""
    inputs = np.asarray(inputs)
    if inputs.ndim != 3:
        raise ValueError(f"Expected inputs shape [N, T, C], got {inputs.shape}")
    if not checkpoint_steps:
        raise ValueError("checkpoint_steps cannot be empty")
    mask = np.asarray(mask)
    feature_dimension = len(checkpoint_steps) * mask.shape[0] * mask.shape[1]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(inputs), feature_dimension),
    )
    for start in range(0, len(inputs), batch_size):
        stop = min(start + batch_size, len(inputs))
        output[start:stop] = reservoir_states(
            inputs[start:stop],
            mask,
            alpha=alpha,
            gamma=gamma,
            checkpoint_steps=checkpoint_steps,
            batch_size=batch_size,
        )
    output.flush()
    return output


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
