from __future__ import annotations

from typing import Any

from .base import DenoisingMethod
from .fft_threshold import FFTThresholdMethod
from .l1_norm import L1NormMethod
from .least_squares import LeastSquaresMethod


METHODS: dict[str, type[DenoisingMethod]] = {
    FFTThresholdMethod.name: FFTThresholdMethod,
    L1NormMethod.name: L1NormMethod,
    LeastSquaresMethod.name: LeastSquaresMethod,
}


def available_methods() -> tuple[str, ...]:
    return tuple(sorted(METHODS))


def create_method(
    method_name: str,
    config: dict[str, Any],
    dataset_name: str,
    overrides: dict[str, Any] | None = None,
) -> DenoisingMethod:
    try:
        method_cls = METHODS[method_name]
    except KeyError as exc:
        choices = ", ".join(available_methods())
        raise ValueError(f"Unknown method '{method_name}'. Available methods: {choices}") from exc
    return method_cls.from_config(config, dataset_name, overrides=overrides)
