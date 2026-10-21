"""Bounded prior phase three-tier poison construction; no downstream feedback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.common import file_sha256
from adversarial.prepare import build_poison
from .backend import Backend
from .protocol import seal, verify
from .serialization import (
    assess, assess_passage, messages, passage_schema, schema,
    select_pool_answer, tier3_messages,
)


def request(protocol_fingerprint: str, target: dict[str, str], candidate: dict,
            attempt_index: int, history: list[dict[str, Any]], *, tier: int = 1,
            selected_answer: str | None = None) -> dict[str, Any]:
    if tier == 3:
        seed = int(candidate.get("seed", 46))
        request_messages = tier3_messages(
            question=target["question"], reference_answer=target["reference_answer"],
            selected_answer=selected_answer or "",
        )
        request_schema = passage_schema()
    else:
        seed = int(candidate["attempt_seeds"][attempt_index - 1])
        count = int(candidate.get("candidate_count", 6 if candidate.get("explicit_ban") else 4))
        request_messages = messages(
            question=target["question"], reference_answer=target["reference_answer"],
            history=history, candidate_count=count,
            explicit_ban=bool(candidate.get("explicit_ban")),
        )
        request_schema = schema(count)
    return {
        "protocol_fingerprint": protocol_fingerprint,
        "sample_id": target["sample_id"],
        "question": target["question"],
        "reference_answer": target["reference_answer"],
        "candidate_protocol": candidate["id"],
        "candidate": candidate,
        "tier": tier,
        "attempt_index": attempt_index,
        "seed": seed,
        "selected_answer": selected_answer,
        "schema": request_schema,
        "messages": request_messages,
    }


def _assess(record_text: str | None, target: dict[str, str], request_record: dict[str, Any]) -> dict[str, Any]:
    if request_record["tier"] == 3:
        return assess_passage(record_text, target["reference_answer"], request_record["selected_answer"])
    count = int(request_record["candidate"].get("candidate_count", 6 if request_record["candidate"].get("explicit_ban") else 4))
    return assess(record_text, target["reference_answer"], count)


def call_once(backend: Backend | None, target: dict[str, str], candidate: dict,
              request_record: dict[str, Any], history: list[dict[str, Any]],
              directory: Path) -> dict[str, Any]:
    tier = request_record["tier"]
    prefix = "attempt" if tier == 1 else f"tier{tier}_attempt"
    slot = directory / f"{prefix}{request_record['attempt_index']}"
    request_path = slot.with_suffix(".request.json")
    raw_path = slot.with_suffix(".raw.json")
    directory.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        cached = verify(json.loads(raw_path.read_text(encoding="utf-8")))
        if cached["request"] != request_record:
            raise ValueError("Cached prior phase request mismatch")
        return cached
    if request_path.exists():
        reserved = verify(json.loads(request_path.read_text(encoding="utf-8")))
        if reserved["request"] != request_record:
            raise ValueError("Reserved prior phase request mismatch")
        result = {
            "status": "interrupted_slot_consumed", "raw_text": None,
            **_assess(None, target, request_record),
            "cost_unknown": True, "prompt_tokens": 0,
            "generated_tokens": 0, "runtime_seconds": 0.0,
        }
    else:
        request_path.write_text(json.dumps(seal({"request": request_record}), ensure_ascii=False) + "\n")
        if backend is None:
            raise ValueError("Backend is required for an unused generation slot")
        output = backend.generate(
            sample_id=target["sample_id"], question=target["question"],
            reference_answer=target["reference_answer"], candidate=candidate,
            attempt_index=request_record["attempt_index"], history=history,
            tier=tier, selected_answer=request_record.get("selected_answer"),
        )
        result = {**output, "raw_text": output["text"],
                  **_assess(output["text"], target, request_record),
                  "status": "completed", "cost_unknown": False}
    record = seal({"request": request_record, **target,
                   "candidate_protocol": candidate["id"],
                   "tier": tier,
                   "attempt_index": request_record["attempt_index"], **result})
    raw_path.write_text(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def _finalize(target: dict[str, str], candidate: dict, protocol_fingerprint: str,
              attempts: list[dict[str, Any]], *, tier: int,
              selected_candidate_index: int | None = None,
              pool_category: str | None = None,
              pool_answer: str | None = None) -> dict[str, Any]:
    last = attempts[-1]
    accepted = bool(last["validator_result"]["valid"])
    result: dict[str, Any] = {
        "protocol_fingerprint": protocol_fingerprint,
        "sample_id": target["sample_id"],
        "question": target["question"],
        "reference_answer": target["reference_answer"],
        "candidate_protocol": candidate["id"],
        "accepted": accepted,
        "construction_tier": tier,
        "attempt_count": len(attempts),
        "tier_attempt_count": len([item for item in attempts if item.get("tier") == tier]),
        "selected_target_answer": last.get("selected_target_answer") if accepted else None,
        "selected_candidate_index": selected_candidate_index if accepted else None,
        "candidate_pool_category": pool_category,
        "candidate_pool_answer": pool_answer,
        "constructed_passage": last.get("constructed_passage") if accepted else None,
        "passage_sentences": last.get("passage_sentences") if accepted else None,
        "failure_reason": None if accepted else last.get("error_message"),
        "parse_trace": [item.get("parse_success", False) for item in attempts],
        "schema_trace": [item.get("schema_success", False) for item in attempts],
        "validator_trace": [item["validator_result"] for item in attempts],
        "runtime_seconds": sum(float(item.get("runtime_seconds", 0.0)) for item in attempts),
        "input_tokens": sum(int(item.get("prompt_tokens", 0)) for item in attempts),
        "output_tokens": sum(int(item.get("generated_tokens", 0)) for item in attempts),
        "attempt_trace": [
            {
                "tier": item.get("tier"), "attempt_index": item.get("attempt_index"),
                "seed": item.get("seed"), "temperature": item.get("decoding_config", {}).get("temperature"),
                "target_answer_candidates": item.get("target_answer_candidates"),
                "selected_target_answer": item.get("selected_target_answer"),
                "selected_candidate_index": item.get("selected_candidate_index"),
                "error_message": item.get("error_message"),
            }
            for item in attempts
        ],
    }
    if accepted:
        result["poison"] = build_poison(
            target["sample_id"], target["question"],
            result["selected_target_answer"], result["constructed_passage"],
            candidate["document_title"],
        )
    return seal(result)


def construct_one(*, backend: Backend, protocol_fingerprint: str,
                  target: dict[str, str], candidate: dict, directory: Path,
                  tier3_queue_index: int = 1) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    # Tier 1: existing three fixed attempts.
    for local_index in range(1, int(candidate["max_attempts"]) + 1):
        req = request(protocol_fingerprint, target, candidate, local_index, attempts, tier=1)
        record = call_once(backend, target, candidate, req, attempts, directory)
        attempts.append(record)
        if record["validator_result"]["valid"]:
            return _finalize(target, candidate, protocol_fingerprint, attempts,
                             tier=1, selected_candidate_index=record["selected_candidate_index"])
    # Tier 2: one explicit-ban sampling attempt with six candidates.
    tier2 = dict(candidate.get("tier2", {}))
    tier2.setdefault("id", "tier2_explicit_ban")
    tier2.setdefault("max_new_tokens", candidate["max_new_tokens"])
    tier2.setdefault("attempt_seeds", [45])
    tier2.setdefault("candidate_count", 6)
    tier2.setdefault("max_attempts", 1)
    tier2.setdefault("do_sample", True)
    tier2.setdefault("temperature", 0.9)
    tier2.setdefault("top_p", 0.9)
    tier2["explicit_ban"] = True
    req = request(protocol_fingerprint, target, tier2, 1, attempts, tier=2)
    record = call_once(backend, target, tier2, req, attempts, directory)
    attempts.append(record)
    if record["validator_result"]["valid"]:
        return _finalize(target, tier2, protocol_fingerprint, attempts,
                         tier=2, selected_candidate_index=record["selected_candidate_index"])
    # Tier 3: deterministic answer-pool selection followed by passage-only LM call.
    pool_answer, category, pool_position = select_pool_answer(target["reference_answer"], tier3_queue_index)
    tier3 = dict(candidate.get("tier3", {}))
    tier3.setdefault("id", "tier3_deterministic_answer_pool")
    tier3.setdefault("document_title", candidate["document_title"])
    tier3.setdefault("max_new_tokens", candidate["max_new_tokens"])
    tier3.setdefault("seed", 46)
    tier3.setdefault("max_attempts", 1)
    tier3.setdefault("do_sample", False)
    req = request(protocol_fingerprint, target, tier3, 1, attempts, tier=3,
                  selected_answer=pool_answer)
    record = call_once(backend, target, tier3, req, attempts, directory)
    attempts.append(record)
    return _finalize(target, tier3, protocol_fingerprint, attempts,
                     tier=3, selected_candidate_index=pool_position,
                     pool_category=category, pool_answer=pool_answer)
