"""Shared helpers for staged water-plant scenario definitions."""

from __future__ import annotations

import struct

from .models import HsTransaction

COMMON_LIMITATIONS = (
    "Scenario traffic is synthetic and deterministic.",
    "Scenario labels are review prompts, not operational conclusions.",
    "HydraSim does not claim incident, safety, commissioning, or certification evidence.",
)


def float32_words(value: float) -> tuple[int, int]:
    return struct.unpack(">HH", struct.pack(">f", value))


def tx(
    ordinal: int,
    ts: int,
    actor: str,
    target: str,
    stage: str,
    area: str,
    function_code: int,
    operation: str,
    address: int,
    quantity: int,
    value: str,
    response: str,
    label: str,
    hint: str = "",
    wire: tuple[int, ...] = (),
) -> HsTransaction:
    return HsTransaction(
        ordinal,
        ts,
        actor,
        target,
        stage,
        area,
        function_code,
        operation,
        address,
        quantity,
        value,
        response,
        label,
        hint,
        wire,
    )
