"""Where an audio model's weights go: the accelerator, or plain CPU RAM.

Audio loads take an accelerator whenever one exists. That is right until the GPU
is the scarce resource (a resident chat model, a training run, a card too small
for the checkpoint), and Whisper and the smaller TTS models run fine on CPU.

Values match ``RAG_EMBED_DEVICE`` (``core/rag/config.py``):

``auto``  detect as before.
``cpu``   force CPU RAM, even with a working accelerator.
``gpu``   prefer the accelerator. The existing CPU retry after a failed load
          still applies, so this is a preference and not a guarantee.

``UNSLOTH_AUDIO_DEVICE`` supplies the default for a request that names none.
"""

from __future__ import annotations

import os
from typing import Optional

__all__ = [
    "AUDIO_DEVICE_CHOICES",
    "audio_device_default",
    "audio_device_forces_cpu",
    "normalize_audio_device",
]

AUDIO_DEVICE_CHOICES = ("auto", "cpu", "gpu")

# Spellings other Studio surfaces already use. The device names arrive from a
# status string being echoed back at us.
_CPU_ALIASES = frozenset({"cpu", "ram", "cpu_ram", "system", "system_ram"})
_GPU_ALIASES = frozenset(
    {"gpu", "cuda", "rocm", "hip", "xpu", "mps", "metal", "accelerator", "accel"}
)


def normalize_audio_device(value: Optional[str]) -> str:
    """Map any accepted spelling onto ``auto``/``cpu``/``gpu``.

    Anything unrecognised becomes ``auto``: an unknown preference must not fail
    a load, and detection is what the caller would have done regardless.
    """
    text = str(value or "").strip().lower()
    if not text:
        return "auto"
    if text in _CPU_ALIASES:
        return "cpu"
    if text in _GPU_ALIASES:
        return "gpu"
    if text == "auto":
        return "auto"
    return "auto"


def audio_device_default() -> str:
    """The preference for a request that carries none (``UNSLOTH_AUDIO_DEVICE``)."""
    return normalize_audio_device(os.environ.get("UNSLOTH_AUDIO_DEVICE"))


def audio_device_forces_cpu(value: Optional[str]) -> bool:
    """True when this preference means "load into CPU RAM".

    ``None`` falls back to the environment default, so an older caller still
    honours a server-wide setting.
    """
    if value is None:
        return audio_device_default() == "cpu"
    return normalize_audio_device(value) == "cpu"
