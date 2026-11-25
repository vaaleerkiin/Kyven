"""Validated settings shared by the CLI, scheduler and Nuke adapter."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from kyven.errors import ErrorCode, KyvenError
from kyven.inpaint.models import InpaintRequest

PROVIDER_ID = "flux2-klein-4b-q4-lanpaint"
MOSS_PROVIDER_ID = "moss-sft-int8-edit-lanpaint"
FILL_PROVIDERS = (PROVIDER_ID, MOSS_PROVIDER_ID)
from kyven.generative_fill.prompts import DEFAULT_SYSTEM_PROMPT


def _invalid(message: str) -> None:
    raise KyvenError(ErrorCode.INVALID_REQUEST, message)


@dataclass(frozen=True, slots=True)
class FillSettings:
    prompt: str = ""
    seed: int = 42
    steps: int = 4
    working_size: int = 768
    thinking_steps: int = 2
    step_size: float = 0.2
    friction: float = 15.0
    strength: float = 8.0
    beta: float = 1.0
    early_stop: int = 1
    vae_tile_size: int = 256
    timeout_seconds: int = 1800
    oom_retry: bool = True
    color_match: float = 1.0
    color_match_mode: str = "auto"
    mask_overlap: int = 32
    seamless_blend: str = "auto"
    moss_guidance: float = 4.0

    def validate(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip() or len(self.prompt) > 16000:
            _invalid("Enter a prompt of 1 to 16000 characters.")
        for name, low, high in (
            ("mask_overlap", 0, 128),
            ("seed", 0, 2**32 - 1), ("steps", 2, 30), ("working_size", 256, 1024),
            ("thinking_steps", 1, 8), ("early_stop", 0, 30),
            ("vae_tile_size", 128, 512), ("timeout_seconds", 30, 7200),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                _invalid(f"{name} must be an integer between {low} and {high}.")
        for name, low, high in (
            ("moss_guidance", 1.0, 10.0),
            ("color_match", 0.0, 1.0),
            ("step_size", 0.001, 1.0), ("friction", 0.1, 50.0),
            ("strength", 0.1, 32.0), ("beta", 0.01, 4.0),
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not low <= value <= high:
                _invalid(f"{name} must be between {low} and {high}.")
        if self.early_stop >= self.steps:
            _invalid("LanPaint early stop must be smaller than the diffusion step count.")
        if type(self.oom_retry) is not bool:
            _invalid("OOM retry must be a boolean.")
        if self.color_match_mode not in {"auto", "context", "background", "off"}:
            _invalid("Color matching must be auto, context, background or off.")
        if self.seamless_blend not in {"auto", "on", "off"}:
            _invalid("Seamless blending must be auto, on or off.")

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> FillSettings:
        allowed = cls.__dataclass_fields__
        unknown = set(payload) - set(allowed)
        if unknown:
            _invalid(f"Unknown generative fill settings: {', '.join(sorted(unknown))}")
        result = cls(**payload)
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class GenerativeFillRequest(InpaintRequest):
    provider_id: str = PROVIDER_ID
    context_padding: int = 256
    fill: FillSettings = field(default_factory=FillSettings)
    preserve_input_mask: bool = True

    def validate(self) -> None:
        InpaintRequest.validate(self)
        self.fill.validate()
        if self.provider_id not in FILL_PROVIDERS or not self.preserve_input_mask:
            _invalid("Generative Fill requires a supported fill model and strict mask preservation.")
        if self.quality_mode != "standard":
            _invalid("LaMa refinement is not available for Generative Fill.")
        if self.mask_grow < 0:
            _invalid("Generative Fill model mask grow cannot be negative.")
        if not self.preprocess_mask:
            _invalid("Generative Fill requires mask preprocessing; grow remains independent of compositing.")

    def canonical(self) -> dict[str, Any]:
        settings = asdict(self.fill)
        if self.provider_id == PROVIDER_ID:
            settings.pop("moss_guidance")
        return {**InpaintRequest.canonical(self), "fill": settings,
                "preserve_input_mask": True}


@dataclass(frozen=True, slots=True)
class EnhancePromptRequest:
    prompt: str
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_new_tokens: int = 192
    temperature: float = 0.6
    top_p: float = 0.9
    seed: int = 42
    timeout_seconds: int = 600
    provider_id: str = PROVIDER_ID

    def validate(self) -> None:
        if self.provider_id not in FILL_PROVIDERS:
            _invalid("Choose a supported prompt enhancement model.")
        if not isinstance(self.prompt, str) or not self.prompt.strip() or len(self.prompt) > 16000:
            _invalid("Enter a prompt of 1 to 16000 characters.")
        if not isinstance(self.system_prompt, str) or len(self.system_prompt) > 16000:
            _invalid("The Qwen system prompt must contain at most 16000 characters.")
        for name in ("max_new_tokens", "seed", "timeout_seconds"):
            if type(getattr(self, name)) is not int:
                _invalid(f"Qwen {name} must be an integer.")
        if not 16 <= self.max_new_tokens <= 512:
            _invalid("Qwen output length must be between 16 and 512 tokens.")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            _invalid("Qwen temperature must be between 0 and 2.")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            _invalid("Qwen top-p must be greater than 0 and at most 1.")
        if not 0 <= self.seed < 2**32 or not 30 <= self.timeout_seconds <= 1800:
            _invalid("Qwen seed or timeout is out of range.")

    def canonical(self) -> dict[str, Any]:
        return asdict(self)


def working_dimensions(width: int, height: int, maximum: int) -> tuple[int, int]:
    """Fit a crop to a bounded canvas, aligned to Klein's 16-pixel grid."""
    scale = min(1.0, maximum / max(width, height))
    return tuple(max(16, math.ceil(value * scale / 16) * 16) for value in (width, height))
