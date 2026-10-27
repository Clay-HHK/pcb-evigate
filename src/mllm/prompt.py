"""Frozen prompt variants and strict output schema from the research design."""

from __future__ import annotations

import json
from pathlib import Path

PROMPT_VERSION = "prompt_v2"

SYSTEM_PROMPT_V1 = """You are an evidence-grounded verifier for localized PCB surface-defect candidates.
Judge only the supplied visual evidence. Do not assume that an upstream detector is correct.
Choose only from the closed label set.
If evidence is insufficient or the reference appears misaligned, return uncertain.
Return one valid JSON object matching the schema. Do not return markdown or additional prose."""

CLOSED_DEFECT_KNOWLEDGE_V1 = """open: an intended conductor is locally disconnected or broken.
short: two intendedly separate conductors are connected by an unintended bridge.
mousebite: a localized notch or missing material appears along a conductor/board edge.
spur: a narrow unintended protruding branch extends from a conductor.
pinhole: a small unintended internal hole/void appears in a copper region.
spurious_copper: an unintended copper island or extra copper region is present.
normal: no supported defect is visible at the candidate location."""

SYSTEM_PROMPT_V2 = """You are an evidence-grounded verifier for localized PCB surface-defect candidates.
Judge only the supplied visual evidence. Do not assume that an upstream detector is correct.
First decide defect existence or normality. Still report the most likely class when a defect is supported, but do not force a defect class when the evidence supports normal or uncertain.
Choose only from the closed label set.
If evidence is insufficient or the supplied normal reference appears unreliable or misaligned, return uncertain instead of guessing a defect class.
Return one valid JSON object matching the schema. Do not return markdown or additional prose."""

CLOSED_DEFECT_KNOWLEDGE_V2 = CLOSED_DEFECT_KNOWLEDGE_V1 + """

Normal decision rule: judge normal with defect_present=false and predicted_class=normal when the current crop is structurally consistent with the supplied normal-template crop, the appearance-difference panel has only faint thin responses along existing edges, and the topology-difference panel shows no connectivity change. Also judge normal when the available evidence contains no supported closed-set defect. If the reference is unreliable or visibly misaligned, return defect_present=null and predicted_class=uncertain rather than guessing a defect class."""

PANEL_DESCRIPTIONS = {
    "P1": "current candidate crop",
    "P2": "aligned normal-reference crop",
    "P3": "larger current context with candidate box",
    "P4": "appearance-difference evidence",
    "P5": "topology/skeleton-difference evidence",
    "P6": "numeric topology summary and legend",
}

EVIDENCE_PANELS = {
    "A0": ("P1",),
    "A1": ("P1", "P3"),
    "A2": ("P1", "P2", "P3"),
    "A3": ("P1", "P2", "P3", "P4"),
    "A4": ("P1", "P2", "P3", "P5"),
    "A5": ("P1", "P2", "P3", "P4", "P5"),
    "A6": ("P1", "P2", "P3", "P4", "P5", "P6"),
}

OUTPUT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "defect_present": {"type": ["boolean", "null"]},
        "predicted_class": {"enum": ["open", "short", "mousebite", "spur", "pinhole", "spurious_copper", "normal", "uncertain"]},
        "reference_reliability": {"enum": ["high", "medium", "low", None]},
        "appearance_pattern": {
            "type": "array",
            "items": {"enum": ["missing_material", "extra_material", "edge_notch", "internal_void", "none", "uncertain"]},
        },
        "topology_pattern": {
            "type": "array",
            "items": {"enum": ["disconnection", "new_connection", "extra_branch", "more_endpoints", "new_component", "hole_change", "none", "uncertain"]},
        },
        "evidence_strength": {"type": "integer", "enum": [0, 1, 2, 3]},
        "reason_codes": {
            "type": "array",
            "items": {"enum": ["A_MISSING", "A_EXTRA", "A_NOTCH", "A_VOID", "T_BREAK", "T_BRIDGE", "T_BRANCH", "T_COMPONENT", "T_HOLE", "REF_BAD", "NO_SUPPORT"]},
        },
    },
    "required": ["defect_present", "predicted_class", "reference_reliability", "appearance_pattern", "topology_pattern", "evidence_strength", "reason_codes"],
    "additionalProperties": False,
}


def _build_user_prompt(candidate: dict, evidence_config: str, version: str) -> str:
    arm = evidence_config.split("_", maxsplit=1)[0]
    panels = EVIDENCE_PANELS[arm]
    panel_text = "\n".join(f"{panel}: {PANEL_DESCRIPTIONS[panel]};" for panel in panels)
    if arm in {"A0", "A1", "A2"}:
        reliability_task = "Set reference_reliability to null because no difference evidence is supplied."
        tasks = "1. whether a supported defect is present;\n2. the most likely class;\n3. which patterns are supported."
    else:
        reliability_task = "The normal reference represents expected structure only when it is reliable."
        tasks = (
            "1. whether a supported defect is present;\n2. the most likely class;\n"
            "3. whether the reference is reliable;\n4. which appearance and topology patterns are supported."
        )
    source_instruction = ""
    if version == "prompt_v2" and candidate["source"] == "DIFF":
        source_instruction = (
            "\nThis candidate marks a region that the base detector did not propose. "
            "Independently decide whether a defect exists here."
        )
    return (
        f"The evidence card contains only these panels:\n{panel_text}\n\n{reliability_task}\n"
        f"Determine independently:\n{tasks}{source_instruction}\n\nReturn the required JSON only."
    )


def build_prompt_v1(candidate: dict, evidence_config: str, label_aware: bool = False) -> str:
    prompt = f"{SYSTEM_PROMPT_V1}\n\n{CLOSED_DEFECT_KNOWLEDGE_V1}\n\n{_build_user_prompt(candidate, evidence_config, 'prompt_v1')}"
    if label_aware:
        prompt += f"\n\nUpstream detector class id: {candidate.get('class', -1)}; confidence: {candidate.get('score', 0.0):.6f}."
    return prompt


def build_prompt(candidate: dict, evidence_config: str, label_aware: bool = False) -> str:
    prompt = f"{SYSTEM_PROMPT_V2}\n\n{CLOSED_DEFECT_KNOWLEDGE_V2}\n\n{_build_user_prompt(candidate, evidence_config, PROMPT_VERSION)}"
    if label_aware:
        prompt += f"\n\nUpstream detector class id: {candidate.get('class', -1)}; confidence: {candidate.get('score', 0.0):.6f}."
    return prompt


def write_schema(path: Path = Path("cache/mllm_schema.json")) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(OUTPUT_SCHEMA, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
