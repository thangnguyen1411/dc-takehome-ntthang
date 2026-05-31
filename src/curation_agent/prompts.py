"""Shared prompt fragments used across the LLM-facing components.

Defining these once keeps the verifier, reflector, and refiner consistent — they
all read the same EVIDENCE, so they must interpret its structure the same way.
"""

from __future__ import annotations

# How to treat evidence that retrieval has split into labeled sections. Appended
# to every system prompt that reads EVIDENCE so the precedence is a standing rule
# rather than an instruction repeated inline in each evidence blob.
EVIDENCE_PRECEDENCE_RULE = (
    "The EVIDENCE is presented in one of two forms. (1) A single 'EVIDENCE' "
    "section — treat it as the sole authoritative evidence. (2) Two sections, "
    "'PRIMARY EVIDENCE' and 'SUPPLEMENTARY RETRIEVED CONTEXT' — treat PRIMARY "
    "EVIDENCE as authoritative and judge against it first, and use the "
    "supplementary retrieved context only as supporting background that must "
    "never override or contradict the primary evidence."
)
