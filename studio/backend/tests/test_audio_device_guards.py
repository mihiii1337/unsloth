# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Guards that must not charge a CPU-placed audio load for VRAM it never takes.

A load held in system RAM still passed the training coexistence check, the GPU
arbiter and the memory preflight, so it could be refused on a full card, evict an
image or video pipeline, or be reported already-loaded while sitting on the GPU.
"""

import asyncio
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import HTTPException  # noqa: E402

import routes.inference as ri  # noqa: E402
from routes.training_vram import _stt_sidecar_holds_no_vram  # noqa: E402


def _audio(audio_type = "higgs_tts2", **kwargs):
    return types.SimpleNamespace(audio_type = audio_type, is_lora = False, identifier = "x/y", **kwargs)


def _request(audio_device = None):
    return types.SimpleNamespace(audio_device = audio_device)


# --- the gate itself -------------------------------------------------------


def test_only_a_native_audio_model_counts_as_a_cpu_audio_load():
    assert ri._native_audio_cpu_load(_audio(), _request("cpu"))
    assert not ri._native_audio_cpu_load(_audio(), _request("auto"))
    assert not ri._native_audio_cpu_load(_audio(), _request(None))


def test_a_chat_model_cannot_skip_the_guards_by_sending_audio_device():
    """audio_device is documented as ignored off the audio path. If it were not
    gated here, any load could set it and walk past the training guard."""
    assert not ri._native_audio_cpu_load(_audio(audio_type = None), _request("cpu"))
    assert not ri._native_audio_cpu_load(_audio(audio_type = "whisper"), _request("cpu"))


# --- the training coexistence guard ----------------------------------------


def test_a_cpu_audio_load_is_not_refused_while_training_runs(monkeypatch):
    """The guard 409s an unsized load, and refuses everything outright during
    diffusion training. Neither applies to weights that never reach the card."""
    monkeypatch.setattr(ri, "_diffusion_training_active", lambda: True)

    assert (
        ri._guard_chat_load_against_training(
            _audio(is_gguf = False),
            types.SimpleNamespace(
                audio_device = "cpu",
                gpu_memory_mode = "auto",
                gpu_layers = -1,
                tensor_parallel = False,
            ),
            load_in_4bit = False,
            placement = types.SimpleNamespace(
                requested_gpu_ids = None,
                gpu_ids_are_vulkan_ordinals = False,
                diffusion_kind = None,
            ),
        )
        is None
    )


def test_a_gpu_audio_load_is_still_refused_during_diffusion_training(monkeypatch):
    """The exemption must be the CPU placement, not the audio type."""
    monkeypatch.setattr(ri, "_diffusion_training_active", lambda: True)

    with pytest.raises(HTTPException) as excinfo:
        ri._guard_chat_load_against_training(
            _audio(is_gguf = False),
            types.SimpleNamespace(
                audio_device = "auto",
                gpu_memory_mode = "auto",
                gpu_layers = -1,
                tensor_parallel = False,
            ),
            load_in_4bit = False,
            placement = types.SimpleNamespace(
                requested_gpu_ids = None,
                gpu_ids_are_vulkan_ordinals = False,
                diffusion_kind = None,
            ),
        )
    assert excinfo.value.status_code == 409


# --- the memory preflight --------------------------------------------------


def test_a_cpu_load_skips_the_vram_preflight_entirely(monkeypatch):
    """Sizing it would refuse the load on a full GPU, which is the case the
    option exists for. The probe must not even be reached."""

    def _never():
        raise AssertionError("a CPU load must not size GPU memory")

    monkeypatch.setattr(ri, "_native_audio_post_handoff_free_gb", _never)
    placement = types.SimpleNamespace(requested_gpu_ids = None)

    result = asyncio.run(ri._preflight_native_audio_placement(_audio(), _request("cpu"), placement))
    assert result is placement


def test_minimax_on_cpu_is_refused_before_the_resident_model_is_evicted():
    """Its runtime needs CUDA. Failing later in the worker would cost the user
    the model they already had, since the switch evicts before the load runs."""
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            ri._preflight_native_audio_placement(
                _audio(audio_type = "minimax_music3"),
                _request("cpu"),
                types.SimpleNamespace(requested_gpu_ids = None),
            )
        )
    assert excinfo.value.status_code == 400
    assert "CPU RAM" in excinfo.value.detail


# --- the already-loaded shortcut -------------------------------------------


def _backend(audio_cpu):
    entry = {"is_audio": True}
    if audio_cpu is not None:
        entry["audio_cpu"] = audio_cpu
    return types.SimpleNamespace(active_model_name = "x/y", models = {"x/y": entry})


def test_a_resident_gpu_audio_model_does_not_satisfy_a_cpu_request():
    assert not ri._resident_audio_placement_matches(
        _backend(audio_cpu = False), _audio(), _request("cpu")
    )


def test_a_resident_cpu_audio_model_satisfies_the_same_request_again():
    assert ri._resident_audio_placement_matches(_backend(audio_cpu = True), _audio(), _request("cpu"))


def test_a_model_loaded_before_this_existed_is_read_as_gpu():
    """No recorded key means the load predates the option, which placed on GPU."""
    assert ri._resident_audio_placement_matches(
        _backend(audio_cpu = None), _audio(), _request("auto")
    )
    assert not ri._resident_audio_placement_matches(
        _backend(audio_cpu = None), _audio(), _request("cpu")
    )


def test_a_non_audio_model_keeps_the_shortcut():
    assert ri._resident_audio_placement_matches(
        _backend(audio_cpu = None), _audio(audio_type = None), _request("cpu")
    )


# --- training eviction -----------------------------------------------------


def test_a_cpu_placed_sidecar_is_left_alone_when_training_claims_vram():
    assert _stt_sidecar_holds_no_vram(types.SimpleNamespace(device = "cpu"))
    assert _stt_sidecar_holds_no_vram(types.SimpleNamespace(device = "whisper.cpp", _forced_cpu = True))
    assert _stt_sidecar_holds_no_vram(types.SimpleNamespace(device = "llama.cpp", _gpu_disabled = True))


def test_anything_that_might_hold_vram_is_still_evicted():
    """Default-deny: starving the run this makes room for is the worse failure."""
    assert not _stt_sidecar_holds_no_vram(types.SimpleNamespace(device = "cuda"))
    assert not _stt_sidecar_holds_no_vram(types.SimpleNamespace(device = "mps"))
    assert not _stt_sidecar_holds_no_vram(
        types.SimpleNamespace(device = "whisper.cpp", _forced_cpu = False)
    )
    assert not _stt_sidecar_holds_no_vram(types.SimpleNamespace())

    class _Raises:
        @property
        def device(self):
            raise RuntimeError("unreadable")

    assert not _stt_sidecar_holds_no_vram(_Raises())


# --- training eviction of the chat backend ---------------------------------


def _inference_backend(active, entry, loading = ()):
    return types.SimpleNamespace(
        active_model_name = active,
        models = {active: entry} if active else {},
        loading_models = set(loading),
    )


def test_a_cpu_placed_audio_model_survives_training_starting():
    """It holds no VRAM, so tearing it down cannot help the run. Mirrors the
    exemption the GGUF branch already makes for a CPU-only server."""
    from routes.training_vram import _resident_audio_holds_no_vram

    assert _resident_audio_holds_no_vram(
        _inference_backend("x/y", {"is_audio": True, "audio_cpu": True})
    )


def test_a_gpu_audio_model_is_still_torn_down_for_training():
    from routes.training_vram import _resident_audio_holds_no_vram

    assert not _resident_audio_holds_no_vram(
        _inference_backend("x/y", {"is_audio": True, "audio_cpu": False})
    )
    # No marker means the load predates the option, which placed on the GPU.
    assert not _resident_audio_holds_no_vram(_inference_backend("x/y", {"is_audio": True}))
    assert not _resident_audio_holds_no_vram(_inference_backend(None, {}))


def test_a_load_in_flight_is_never_exempted():
    """Its placement is not recorded yet, and it is already taking memory."""
    from routes.training_vram import _resident_audio_holds_no_vram

    assert not _resident_audio_holds_no_vram(
        _inference_backend("x/y", {"audio_cpu": True}, loading = ("a/b",))
    )
