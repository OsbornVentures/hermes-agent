"""Carved-out AMD UMA memory must not require large pinned Windows host buffers."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.local_runtime import binaries, catalog, hardware, presets
from hermes_cli.local_runtime.devices import GGML_DEVICE_GPU, GGML_DEVICE_IGPU
from hermes_cli.local_runtime.estimator import HardwareBudget, ModelProfile
from hermes_platform.host import facts


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("backend, device_type, vendor, visible, installed, expected", [
    ("vulkan", GGML_DEVICE_IGPU, "amd", 8, 32, True),
    ("vulkan", GGML_DEVICE_IGPU, "amd", 24, 32, False),
    ("vulkan", GGML_DEVICE_IGPU, "intel", 8, 32, False),
    ("vulkan", GGML_DEVICE_GPU, "amd", 8, 32, False),
    ("cpu", GGML_DEVICE_IGPU, "amd", 8, 32, False),
    ("hip", GGML_DEVICE_IGPU, "amd", 8, 32, False),
    ("vulkan", GGML_DEVICE_IGPU, "amd", 8, None, False),
])
def test_host_buffer_workaround_requires_the_affected_memory_topology(
        monkeypatch, backend, device_type, vendor, visible, installed, expected):
    monkeypatch.setattr(hardware, "_configured_engine", lambda: binaries.Engine(backend, "b12144", Path("engine")))
    monkeypatch.setattr(hardware, "_accelerator_device", lambda: {"type": device_type})
    monkeypatch.setattr(hardware, "gpu_class", lambda: vendor)
    monkeypatch.setattr(facts, "ram_total_bytes", lambda: visible << 30)
    monkeypatch.setattr(facts, "ram_installed_bytes", lambda: installed << 30 if installed else None)

    assert hardware.vulkan_uma_host_limited() is expected


@pytest.mark.platforms("windows")
def test_uma_preset_keeps_vision_and_mtp_with_bounded_host_allocations(tmp_path, monkeypatch):
    entry = next(e for e in catalog.CATALOG if e.mtp and e.mmproj)
    model = tmp_path / f"{entry.variants[0].model_id}.gguf"
    projector = tmp_path / "vision.gguf"
    monkeypatch.setattr(presets, "read_gguf_header", lambda p: SimpleNamespace(sampling_defaults={}))
    monkeypatch.setattr(presets, "profile_from_gguf", lambda h: ModelProfile("target", 15 << 30, 0, 65536, []))
    monkeypatch.setattr(presets, "_companions", lambda e: presets._Companions(mmproj=projector, nbytes=1 << 30))
    monkeypatch.setattr(hardware, "_configured_engine", lambda: binaries.Engine("vulkan", "b12144", Path("engine")))
    monkeypatch.setattr(hardware, "_accelerator_device", lambda: {"type": GGML_DEVICE_IGPU})
    monkeypatch.setattr(hardware, "gpu_class", lambda: "amd")
    monkeypatch.setattr(facts, "ram_total_bytes", lambda: 8 << 30)
    monkeypatch.setattr(facts, "ram_installed_bytes", lambda: 32 << 30)
    budget = HardwareBudget(22 << 30, 28 << 30, 0, uma=True, platform="win32")

    plan = presets.preset_for_model(model, budget, set(), requested_window=65536, engine_tag="b12144")

    assert plan.refusal is None and not plan.spilled
    assert plan.window == 16384
    assert plan.keys["no-host"] == "on"
    assert plan.keys["parallel"] == "1"
    assert plan.keys["mmproj-offload"] == "off"
    assert plan.keys["mmproj"] == str(projector)
    assert plan.keys["spec-type"] == "draft-mtp"

    monkeypatch.setattr(facts, "ram_total_bytes", lambda: 24 << 30)
    ordinary = presets.preset_for_model(model, budget, set(), engine_tag="b12144")
    assert "no-host" not in ordinary.keys
    assert "mmproj-offload" not in ordinary.keys
