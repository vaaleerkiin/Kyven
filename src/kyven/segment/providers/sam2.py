"""Lazy SAM 2.1 image-segmentation provider."""

from __future__ import annotations

import hashlib
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from kyven.cancellation import CancellationToken
from kyven.errors import ErrorCode, KyvenError
from kyven.segment.models import (
    ExecutionProfile,
    PointLabel,
    ProviderCapabilities,
    SegmentPrediction,
    SegmentRequest,
)
from kyven.segment.providers.base import SegmentationProvider


class Sam2Provider(SegmentationProvider):
    """SAM 2.1 adapter with no import or VRAM use until prediction begins."""

    def __init__(
        self,
        checkpoint: str,
        model_config: str,
        device: str = "auto",
        expected_checksum: str | None = None,
        provider_id: str = "sam2",
        display_name: str = "SAM 2.1",
    ) -> None:
        self._checkpoint = Path(checkpoint)
        self._model_config = model_config
        self._requested_device = device
        self._expected_checksum = expected_checksum
        self._provider_id = provider_id
        self._display_name = display_name
        self._resolved_device = "unresolved"
        self._predictor: Any | None = None
        self._checkpoint_checksum: str | None = None

    @property
    def capabilities(self) -> ProviderCapabilities:
        checksum = self._checkpoint_checksum or self._calculate_checksum_if_present()
        return ProviderCapabilities(
            provider_id=self._provider_id,
            display_name=self._display_name,
            provider_version="1",
            model_family="sam2",
            model_variant=self._model_config,
            model_checksum=checksum,
            license_name="Apache-2.0",
            license_url="https://github.com/facebookresearch/sam2/blob/main/LICENSE",
            supports_cpu=True,
            supports_points=True,
            supports_boxes=True,
            minimum_vram_mb=None,
            supported_profiles=(
                ExecutionProfile.LOW_MEMORY,
                ExecutionProfile.BALANCED,
                ExecutionProfile.QUALITY,
            ),
        )

    def _calculate_checksum_if_present(self) -> str:
        if self._checkpoint_checksum is not None:
            return self._checkpoint_checksum
        if not self._checkpoint.is_file():
            return "missing"
        digest = hashlib.sha256()
        with self._checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        self._checkpoint_checksum = digest.hexdigest()
        return self._checkpoint_checksum

    def _load(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        if not self._checkpoint.is_file():
            raise KyvenError(
                code=ErrorCode.MODEL_NOT_FOUND,
                message=f"SAM 2 checkpoint was not found: {self._checkpoint}",
                recoverable=True,
                suggested_action="Download a licensed SAM 2.1 checkpoint and pass its path.",
            )
        actual_checksum = self._calculate_checksum_if_present()
        if self._expected_checksum and actual_checksum.lower() != self._expected_checksum.lower():
            raise KyvenError(
                code=ErrorCode.MODEL_NOT_FOUND,
                message="The SAM 2 checkpoint checksum does not match the trusted manifest.",
                technical_detail=(
                    f"Expected {self._expected_checksum.lower()}, received {actual_checksum.lower()}."
                ),
                suggested_action="Delete the checkpoint and download it again from the official source.",
            )
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise KyvenError(
                code=ErrorCode.DEPENDENCY_MISSING,
                message="The optional SAM 2 runtime is not installed.",
                technical_detail=str(exc),
                recoverable=True,
                suggested_action="Install PyTorch and the official SAM 2 package.",
            ) from exc

        if self._requested_device == "auto":
            self._resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._resolved_device = self._requested_device

        try:
            model = build_sam2(
                self._model_config,
                str(self._checkpoint),
                device=self._resolved_device,
            )
            self._predictor = SAM2ImagePredictor(model)
            self._calculate_checksum_if_present()
            return self._predictor
        except Exception as exc:
            raise KyvenError(
                code=ErrorCode.INFERENCE_FAILED,
                message="SAM 2 could not be loaded.",
                technical_detail=str(exc),
                recoverable=True,
                suggested_action="Check the checkpoint, model config, device, and runtime versions.",
            ) from exc

    def predict(
        self,
        request: SegmentRequest,
        cancellation: CancellationToken,
    ) -> SegmentPrediction:
        request.validate()
        cancellation.raise_if_cancelled()
        predictor = self._load()

        image = np.array(Image.open(request.source).convert("RGB"), copy=True)
        point_coords = None
        point_labels = None
        if request.points:
            point_coords = np.asarray([(point.x, point.y) for point in request.points])
            point_labels = np.asarray(
                [1 if point.label is PointLabel.POSITIVE else 0 for point in request.points]
            )
        box = None
        if request.box is not None:
            box = np.asarray(
                [request.box.x0, request.box.y0, request.box.x1, request.box.y1]
            )

        try:
            import torch

            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if self._resolved_device == "cuda"
                else nullcontext()
            )
            with torch.inference_mode(), autocast:
                predictor.set_image(image)
                cancellation.raise_if_cancelled()
                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                    multimask_output=request.multimask_output,
                )
        except KyvenError:
            raise
        except Exception as exc:
            raise KyvenError(
                code=ErrorCode.INFERENCE_FAILED,
                message="SAM 2 segmentation failed.",
                technical_detail=str(exc),
                recoverable=True,
                suggested_action="Try the Low Memory profile or CPU device.",
            ) from exc

        best = int(np.argmax(scores))
        return SegmentPrediction(
            mask=np.asarray(masks[best], dtype=np.bool_),
            score=float(scores[best]),
            metadata={
                "provider": self._provider_id,
                "model_config": self._model_config,
                "device": self._resolved_device,
                "profile": request.profile.value,
            },
        )

    def unload(self) -> None:
        """Drop model references and release cached CUDA allocations when possible."""

        self._predictor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            return
