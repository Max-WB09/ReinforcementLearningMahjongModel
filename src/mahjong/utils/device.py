"""JAX device inspection with no GPU requirement."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    """Portable summary of JAX runtime device configuration."""

    jax_version: str
    devices: tuple[str, ...]
    default_backend: str
    gpu_available: bool


def get_device_info() -> DeviceInfo:
    """Inspect JAX devices, returning CPU information when CUDA is unavailable."""
    import jax

    devices = tuple(str(device) for device in jax.devices())
    return DeviceInfo(
        jax_version=jax.__version__,
        devices=devices,
        default_backend=jax.default_backend(),
        gpu_available=any(device.platform == "gpu" for device in jax.devices()),
    )


def format_device_info() -> str:
    """Format device inspection for command-line use."""
    info = get_device_info()
    return "\n".join(
        (
            f"JAX version: {info.jax_version}",
            f"Available devices: {', '.join(info.devices)}",
            f"Default backend: {info.default_backend}",
            f"GPU available: {info.gpu_available}",
        )
    )
