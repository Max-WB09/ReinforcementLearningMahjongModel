# Environment Setup

## Supported versions

This project requires Python 3.10--3.13. It pins MahJax to 0.1.6 and accepts JAX versions from 0.4.28 up to, but not including, 0.7. MahJax itself declares Python 3.9+ support; the narrower project range is the range represented in this project's metadata.

## CPU installation

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m mahjong.simulation --episodes 10 --seed 42
pytest
```

The normal JAX package installation is CPU-capable. To inspect the active runtime:

```bash
python -c "from mahjong.utils.device import format_device_info; print(format_device_info())"
```

## NVIDIA GPU installation

Install an NVIDIA driver compatible with the CUDA runtime selected by the current official [JAX installation guide](https://docs.jax.dev/en/latest/installation.html), then install JAX using that guide's CUDA-specific command before installing this project. CUDA wheels, driver support, and supported Python versions change; do not hardcode a CUDA release from this document. Afterward, install the project with `python -m pip install -e '.[dev]'` and run the device-inspection command above. It should report `Default backend: gpu` and `GPU available: True` when JAX detects the GPU.

Start with the sequential runner. Any future `vmap` batch size should be benchmarked and sized to fit the available 8 GB VRAM, including model and compilation overhead.

## Common problems

- `ModuleNotFoundError`: activate the virtual environment and install `-e '.[dev]'`.
- CPU reported instead of GPU: verify the NVIDIA driver and reinstall JAX following its current CUDA instructions.
- First run is slow: JAX compilation occurs on initial environment calls.
- MahJax API failures after an upgrade: retain the pinned version or update the adapter and tests together.
