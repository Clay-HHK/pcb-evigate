"""Codex CLI, deterministic stub, and OpenAI-compatible MLLM clients."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

from src.mllm.prompt import PROMPT_VERSION, build_prompt, write_schema


DEFAULT_CONCLUSION = {
    "defect_present": None,
    "predicted_class": "uncertain",
    "reference_reliability": None,
    "appearance_pattern": ["uncertain"],
    "topology_pattern": ["uncertain"],
    "evidence_strength": 0,
    "reason_codes": ["NO_SUPPORT"],
}


def _parse_conclusion(text: str) -> tuple[dict, bool]:
    parsed = subprocess.run(
        [sys.executable, "-m", "json.tool"],
        input=text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return (json.loads(parsed.stdout), True) if parsed.returncode == 0 else (dict(DEFAULT_CONCLUSION), False)


def cache_key(candidate: dict, evidence_config: str, model: str, temperature: float) -> str:
    return f"{candidate['pair_id']}_{candidate['candidate_id']}__{evidence_config}__{PROMPT_VERSION}__{model}__t{temperature:g}"


def _stub(candidate: dict, config: dict) -> dict:
    residual = candidate.get("appearance_residual", {})
    appearance = float(residual.get("mean", candidate.get("diff_score", 0.0)))
    topology = list(candidate.get("topology_vector", [0.0] * 8))
    q_reg = float(candidate.get("q_reg", 1.0))
    structural = sum(abs(float(value)) for value in topology[:5])
    weak_residual = appearance < float(config["weak_residual_threshold"])
    topology_changed = structural > float(config["topology_change_threshold"])
    if q_reg < float(config["low_q_reg_threshold"]):
        return {
            "defect_present": None,
            "predicted_class": "uncertain",
            "reference_reliability": "low",
            "appearance_pattern": ["uncertain"],
            "topology_pattern": ["uncertain"],
            "evidence_strength": 0,
            "reason_codes": ["REF_BAD"],
        }
    if weak_residual and not topology_changed:
        return {
            "defect_present": False,
            "predicted_class": "normal",
            "reference_reliability": "high" if q_reg >= 0.7 else "medium",
            "appearance_pattern": ["none"],
            "topology_pattern": ["none"],
            "evidence_strength": 1,
            "reason_codes": ["NO_SUPPORT"],
        }
    if weak_residual or not topology_changed:
        return {
            "defect_present": None,
            "predicted_class": "uncertain",
            "reference_reliability": "high" if q_reg >= 0.7 else "medium",
            "appearance_pattern": ["weak" if weak_residual else "strong"],
            "topology_pattern": ["changed" if topology_changed else "none"],
            "evidence_strength": 1,
            "reason_codes": ["EVIDENCE_CONFLICT"],
        }
    if topology[2] >= 1:
        predicted_class, topology_pattern, reason = "spur", "extra_branch", "T_BRANCH"
        appearance_pattern = "extra_material"
    elif topology[0] >= 1 and topology[1] >= 2:
        predicted_class, topology_pattern, reason = "short", "new_connection", "T_CONNECT"
        appearance_pattern = "extra_material"
    elif topology[1] >= 2:
        predicted_class, topology_pattern, reason = "mousebite", "more_endpoints", "T_ENDPOINT"
        appearance_pattern = "missing_material"
    elif topology[3] >= 1:
        predicted_class, topology_pattern, reason = "pinhole", "hole_change", "T_HOLE"
        appearance_pattern = "internal_void"
    elif topology[0] >= 1:
        predicted_class, topology_pattern, reason = "open", "disconnection", "T_BREAK"
        appearance_pattern = "missing_material"
    else:
        predicted_class, topology_pattern, reason = "spurious_copper", "new_component", "A_EXTRA"
        appearance_pattern = "extra_material"
    strength = min(3, 1 + int(appearance >= float(config["strong_residual_threshold"])) + int(structural >= 4.0))
    return {
        "defect_present": True,
        "predicted_class": predicted_class,
        "reference_reliability": "high" if q_reg >= 0.7 else "medium",
        "appearance_pattern": [appearance_pattern],
        "topology_pattern": [topology_pattern],
        "evidence_strength": strength,
        "reason_codes": [reason],
    }


def _codex_cli(config: dict, image_path: Path, schema_path: Path, prompt: str, output_path: Path) -> tuple[dict, bool]:
    environment = os.environ.copy()
    if config.get("proxy"):
        environment["https_proxy"] = config["proxy"]
        environment["http_proxy"] = config["proxy"]
    command = [
        config["codex_path"],
        "exec",
        "-m",
        config["model"],
        "-c",
        f"model_reasoning_effort={config['reasoning_effort']}",
        "-c",
        f"model_temperature={config['temperature']}",
        "-i",
        str(image_path),
        "--output-schema",
        str(schema_path),
        "-o",
        str(output_path),
        "--skip-git-repo-check",
        prompt,
    ]
    subprocess.run(command, env=environment, check=True, stdout=subprocess.DEVNULL)
    return _parse_conclusion(output_path.read_text(encoding="utf-8"))


def _openai_compatible(config: dict, image_path: Path, prompt: str) -> tuple[dict, bool]:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "model": config["model"],
        "temperature": config["temperature"],
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                ],
            }
        ],
    }
    request = urllib.request.Request(
        config["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {os.environ[config['api_key_env']]}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        body = json.loads(response.read().decode("utf-8"))
    return _parse_conclusion(body["choices"][0]["message"]["content"])


def _run_one(task: tuple[dict, dict, str, str, str, str]) -> dict:
    config, candidate, image_path_text, schema_path_text, prompt, key = task
    start = time.perf_counter()
    image_path = Path(image_path_text)
    if config["backend"] == "stub":
        result = _stub(candidate, config)
        json_parseable = True
    elif config["backend"] == "codex_cli":
        temporary = Path("cache/mllm_outputs/tmp") / f"{key}.json"
        temporary.parent.mkdir(parents=True, exist_ok=True)
        result, json_parseable = _codex_cli(config, image_path, Path(schema_path_text), prompt, temporary)
        temporary.unlink()
    else:
        result, json_parseable = _openai_compatible(config, image_path, prompt)
    return {
        "key": key,
        "result": result,
        "json_parseable": json_parseable,
        "latency_seconds": time.perf_counter() - start,
    }


def review_candidates(
    candidates: list[dict],
    image_paths: list[Path],
    evidence_config: str,
    config_path: Path,
    label_aware: bool = False,
    progress_path: Path | None = None,
) -> list[dict]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    schema_path = write_schema()
    cache_path = Path(config["cache_file"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = {
        item["key"]: item
        for item in map(json.loads, cache_path.read_text(encoding="utf-8").splitlines())
    } if cache_path.exists() else {}
    keys = [cache_key(candidate, evidence_config, config["model"], float(config["temperature"])) for candidate in candidates]
    tasks = [
        (
            config,
            candidate,
            str(image_path),
            str(schema_path),
            build_prompt(candidate, evidence_config, label_aware),
            key,
        )
        for candidate, image_path, key in zip(candidates, image_paths, keys)
        if key not in cached
    ]
    if tasks:
        with cache_path.open("a", encoding="utf-8") as handle:
            with ProcessPoolExecutor(max_workers=int(config["workers"])) as executor:
                futures = [executor.submit(_run_one, task) for task in tasks]
                for completed, future in enumerate(as_completed(futures), start=1):
                    item = future.result()
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                    handle.flush()
                    cached[item["key"]] = item
                    if progress_path is not None:
                        with progress_path.open("a", encoding="utf-8") as progress:
                            progress.write(
                                f"{datetime.now().astimezone().isoformat(timespec='seconds')},mllm,{completed},{len(tasks)},{item['key']}\n"
                            )
    return [cached[key] for key in keys]
