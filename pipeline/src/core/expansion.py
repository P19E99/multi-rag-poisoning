"""Deterministic, resumable query rewriting with an offline cache interface.

Validation establishes the output schema and uniqueness, not semantic equivalence.
The prompt asks the model to preserve meaning; researchers should inspect rewrites.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Protocol

from .common import file_sha256, fingerprint, read_jsonl, seed_everything, write_jsonl

LOGGER = logging.getLogger(__name__)
PROMPT_VERSION = "meaning-preserving-four-rewrites-v5"
SYSTEM_PROMPT = r"""You rewrite questions for document retrieval.
Produce exactly FOUR distinct, complete, meaning-preserving rewrites of the question.
Make conservative changes to wording or word order; semantic fidelity matters more than novelty.

Meaning rules for EVERY rewrite:
- Ask for exactly the same unknown as the original. Do not answer, insert an answer,
  infer missing facts, add another question, or turn the question into a subquestion.
- Keep every named entity, number, nationality, location, time, comparison, negation,
  conjunction, qualifier, and constraint. Do not add or remove constraints.
- Preserve who does what to whom: the subject and object of every relation must keep
  their roles, including in awkwardly worded questions. For example, "Which person
  is A the manager of?" asks whom A manages, NOT who manages A. A remains the manager.
- Keep specific actions specific: do not replace a named action with a broader
  category. Preserve the original action word when a synonym could weaken it.
- Preserve ambiguity and repeated names in the input. Do not assume distinct
  versions, invent disambiguating descriptions, fix factual errors, or resolve a
  reference using knowledge outside the question.
- Recheck each rewrite against the original for reversed relations, changed answer
  type, broadened actions, and omitted details before returning it.
- The four rewrites must differ from each other and from the original question.

Output rules:
Return ONLY one valid JSON array containing exactly four nonempty strings.
Use double quotes as JSON string delimiters and a comma between strings.
Any double quotation mark inside a string must be escaped as \". You may instead
leave titles or quoted phrases unquoted to avoid embedded quotation marks.
Write the array on one line and include its closing square bracket.
No Markdown fences, commentary, explanations, analysis, or text outside the array."""


class ExpansionValidationError(ValueError):
    """A cached or generated rewrite record violates the experiment contract."""


def _normalized_query(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", text).strip().rstrip("?.! ")


def validate_rewrites(original_query: str, rewrites: Any, num_rewrites: int = 4) -> list[str]:
    """Check strict schema and uniqueness without claiming semantic validation."""
    if not isinstance(rewrites, list) or len(rewrites) != num_rewrites:
        raise ExpansionValidationError(f"Expected a JSON list of exactly {num_rewrites} rewrites")
    if any(not isinstance(value, str) or not value.strip() for value in rewrites):
        raise ExpansionValidationError("Every rewrite must be a nonempty string")
    normalized = [_normalized_query(value) for value in rewrites]
    if any(not value for value in normalized):
        raise ExpansionValidationError("Rewrites cannot consist only of punctuation")
    if len(set(normalized)) != num_rewrites:
        raise ExpansionValidationError("The four rewrites must be distinct")
    if _normalized_query(original_query) in normalized:
        raise ExpansionValidationError("The original query does not count as a rewrite")
    return [value.strip() for value in rewrites]


def validate_review_checks(sample_id: str, rewrites: list[str], options: dict[str, Any]) -> None:
    """Optional reviewer-specified lexical checks; not a semantic equivalence proof."""
    checks = options.get("review_checks", {}).get(sample_id, {})
    for index, rewrite in enumerate(rewrites, 1):
        for pattern in checks.get("forbidden_regex", []):
            match = re.search(pattern, rewrite, flags=re.IGNORECASE)
            if match:
                raise ExpansionValidationError(
                    f"Rewrite {index} contains forbidden wording {match.group()!r}; "
                    "remove this semantic error and preserve the reviewed original relation"
                )
        for pattern in checks.get("required_regex", []):
            if not re.search(pattern, rewrite, flags=re.IGNORECASE):
                raise ExpansionValidationError(
                    f"Rewrite {index} omits required wording matching {pattern!r}; "
                    "preserve the original constraints and the semantic review feedback"
                )


def parse_rewrites(text: str, original_query: str, num_rewrites: int = 4) -> list[str]:
    """Parse only a JSON list; do not repair fences or extract a list from prose."""
    try:
        values = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ExpansionValidationError(
            f"Model output must be only a valid JSON list: {exc.msg} "
            f"at line {exc.lineno}, column {exc.colno}"
        ) from exc
    except AttributeError as exc:
        raise ExpansionValidationError("Model output must be only a valid JSON list") from exc
    return validate_rewrites(original_query, values, num_rewrites)


def _targets_by_id(targets: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for target in targets:
        sample_id, question = target.get("sample_id"), target.get("question")
        if not isinstance(sample_id, str) or not sample_id:
            raise ExpansionValidationError("Every target needs a nonempty string sample_id")
        if not isinstance(question, str) or not question.strip():
            raise ExpansionValidationError(f"Target {sample_id} has no question")
        if sample_id in mapping:
            raise ExpansionValidationError(f"Duplicate target sample_id: {sample_id}")
        mapping[sample_id] = target
    if not mapping:
        raise ExpansionValidationError("Targets cannot be empty")
    return mapping


def _validate_record(target: dict[str, Any], row: dict[str, Any], num_rewrites: int) -> None:
    if row.get("original_query") != target["question"]:
        raise ExpansionValidationError(f"Original question mismatch for sample {target['sample_id']}")
    validate_rewrites(row["original_query"], row.get("rewrites"), num_rewrites)


def validate_expansions(
    targets: list[dict[str, Any]], rows: list[dict[str, Any]], num_rewrites: int = 4
) -> dict[str, dict[str, Any]]:
    """Return complete expansions keyed by target ID; usable with imported JSONL.

Imported cache records need only sample_id, original_query, and rewrites. Generated
records additionally retain provenance. Unknown, duplicate, and missing IDs fail.
"""
    target_map = _targets_by_id(targets)
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in target_map:
            raise ExpansionValidationError(f"Unknown expansion sample_id: {sample_id!r}")
        if sample_id in records:
            raise ExpansionValidationError(f"Duplicate expansion sample_id: {sample_id}")
        _validate_record(target_map[sample_id], row, num_rewrites)
        records[sample_id] = row
    missing = sorted(set(target_map) - set(records))
    if missing:
        raise ExpansionValidationError(f"Missing expansions for {len(missing)} targets: {missing[:5]}")
    return records


class ExpansionBackend(Protocol):
    """Backend boundary; retrieval only depends on validated JSONL records."""

    def generate(self, *, sample_id: str, question: str, attempt: int = 1, previous_error: str = "") -> list[str]:
        ...


class CachedExpansionBackend:
    """Offline backend: reads previously generated expansions, never loads a model."""

    def __init__(self, targets: list[dict[str, Any]], path: Path, num_rewrites: int = 4):
        self.records = validate_expansions(targets, list(read_jsonl(path)), num_rewrites)

    def generate(self, *, sample_id: str, question: str, attempt: int = 1, previous_error: str = "") -> list[str]:
        row = self.records[sample_id]
        _validate_record({"sample_id": sample_id, "question": question}, row, len(row["rewrites"]))
        return list(row["rewrites"])


class QwenExpansionBackend:
    """Lazy-loaded Qwen backend using greedy decoding and Qwen3 thinking disabled."""

    def __init__(self, options: dict[str, Any]):
        # Keeping these imports here makes imported cache use independent of GPUs
        # and the transformer model-loading stack.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        seed_everything(int(options.get("seed", 42)))
        self.options = options
        self.torch = torch
        model_source = options.get("model_path") or options["model_name"]
        device = options.get("device", "cuda")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Set expansion.device to cpu or supply generated "
                "query_expansions.jsonl and use --backend cached."
            )
        dtype_name = options.get("dtype", "bfloat16")
        dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        if dtype_name not in dtypes and dtype_name != "auto":
            raise ValueError(f"Unsupported expansion dtype: {dtype_name}")
        common_kwargs = {"local_files_only": options.get("local_files_only", False)}
        if options.get("revision"):
            common_kwargs["revision"] = options["revision"]
        LOGGER.info("Loading expansion model %s on %s", model_source, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_source, **common_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=dtypes.get(dtype_name, "auto"),
            **common_kwargs,
        ).to(device)
        self.model.eval()
        self.device = device
        self.last_raw_output = ""
        self.json_constraint = None
        if options.get("constrain_json", True):
            from .json_constraints import JsonListConstraint
            self.json_constraint = JsonListConstraint(self.tokenizer, int(options.get("num_rewrites", 4)))

    def generate(self, *, sample_id: str, question: str, attempt: int = 1, previous_error: str = "") -> list[str]:
        user_prompt = "Question: " + json.dumps(question, ensure_ascii=False)
        guidance = self.options.get("review_guidance", {}).get(sample_id)
        if guidance:
            user_prompt += (
                "\nSemantic review feedback for this question (preserve these roles and constraints): "
                + guidance + "\nRewrite conservatively. Do not repeat the semantic errors described above."
            )
        if attempt > 1:
            user_prompt += (
                f"\nThis is correction attempt {attempt}. The preceding output failed validation: "
                f"{previous_error}. Return four distinct, complete paraphrases as a strict JSON array."
            )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": 'Question: "Which person is Alex the manager of?"'},
            {"role": "assistant", "content": json.dumps([
                "For which person does Alex act as manager?",
                "Who is managed by Alex?",
                "Alex serves as the manager of which person?",
                "Which person has Alex as their manager?",
            ])},
            {"role": "user", "content": user_prompt},
        ]
        if attempt > 1 and self.last_raw_output:
            messages.extend([
                {"role": "assistant", "content": self.last_raw_output},
                {"role": "user", "content": (
                    "Your previous response above was rejected: " + previous_error +
                    '. Resubmit the complete corrected JSON array. Put a comma BETWEEN every '
                    'pair of strings and close the array with ]. Keep exactly four faithful rewrites.'
                )},
            ])
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        generation_kwargs = {}
        if self.json_constraint is not None:
            generation_kwargs["prefix_allowed_tokens_fn"] = self.json_constraint.prefix_function()
        with self.torch.inference_mode():
            tokens = self.model.generate(
                **inputs,
                **generation_kwargs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=int(self.options.get("max_new_tokens", 512)),
                pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id),
            )
        text = self.tokenizer.decode(tokens[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        self.last_raw_output = text
        try:
            return parse_rewrites(text, question, int(self.options.get("num_rewrites", 4)))
        except ExpansionValidationError:
            LOGGER.warning(
                "Rejected raw Qwen output for %s (attempt %d, generated_tokens=%d): %.4000r",
                sample_id, attempt, tokens.shape[1] - inputs["input_ids"].shape[1], text,
            )
            raise


def _generation_spec(config: dict[str, Any]) -> dict[str, Any]:
    options = config["expansion"]
    spec = {
        "backend": "qwen",
        "implementation_sha256": file_sha256(__file__),
        "model_name": options.get("model_name"),
        "model_path": options.get("model_path"),
        "revision": options.get("revision"),
        "seed": config.get("seed", 42),
        "prompt_version": PROMPT_VERSION,
        "review_guidance_sha256": fingerprint(options.get("review_guidance", {})),
        "review_checks_sha256": fingerprint(options.get("review_checks", {})),
        "constrain_json": options.get("constrain_json", True),
        "json_constraint_version": "lm-format-enforcer-array-v1",
        "system_prompt_sha256": fingerprint(SYSTEM_PROMPT),
        "num_rewrites": int(options.get("num_rewrites", 4)),
        "temperature": 0,
        "do_sample": False,
        "num_beams": 1,
        "enable_thinking": False,
        "dtype": options.get("dtype", "bfloat16"),
        "device": options.get("device", "cuda"),
        "max_new_tokens": int(options.get("max_new_tokens", 512)),
        "max_attempts": int(options.get("max_attempts", 3)),
    }
    model_path = options.get("model_path")
    if model_path and Path(model_path).is_dir():
        spec["local_model_metadata_sha256"] = {
            name: file_sha256(Path(model_path) / name)
            for name in (
                "config.json", "generation_config.json", "tokenizer_config.json",
                "tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt",
                "model.safetensors.index.json", "pytorch_model.bin.index.json",
            )
            if (Path(model_path) / name).is_file()
        }
        spec["local_weight_file_stats"] = {
            path.name: {"size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in sorted(Path(model_path).iterdir())
            if path.is_file() and path.suffix in (".safetensors", ".bin")
        }
    return spec


def _target_generation_spec(config: dict[str, Any], sample_id: str) -> dict[str, Any]:
    options = dict(config["expansion"])
    for field in ("review_guidance", "review_checks"):
        mapping = options.get(field, {})
        options[field] = {sample_id: mapping[sample_id]} if sample_id in mapping else {}
    return _generation_spec({**config, "expansion": options})


def run_expansion(
    config: dict[str, Any], *, force: bool = False, backend_override: str | None = None
) -> list[dict[str, Any]]:
    """Generate or validate expansions; checkpoint each successful target atomically."""
    from .contracts import guard_writable, require_stage, record_stage, strict
    guard_writable(config)
    if strict(config):
        require_stage(config, "corpus", targets_only=True)
        if config["expansion"].get("review_guidance") or config["expansion"].get("review_checks"):
            raise ValueError("Formal protocol forbids per-target guidance and review checks")
        if force and Path(config["paths"]["expansions"]).exists():
            raise ValueError("Formal rewrites are immutable; use a new experiment directory")
    if strict(config) and Path(config["paths"]["cache_dir"], "expansion.contract.json").exists():
        require_stage(config, "expansion")
    options = config["expansion"]
    num_rewrites = int(options.get("num_rewrites", 4))
    if num_rewrites != 4:
        raise ValueError("This pipeline requires exactly four rewrites")
    if float(options.get("temperature", 0)) != 0:
        raise ValueError("This pipeline requires temperature=0 (greedy do_sample=False)")
    max_attempts = int(options.get("max_attempts", 3))
    if max_attempts < 1:
        raise ValueError("expansion.max_attempts must be at least 1")
    # Project immediately: no answer/gold/context object reaches backend or prompt code.
    targets = [{"sample_id": row["sample_id"], "question": row["question"]}
               for row in read_jsonl(Path(config["paths"]["targets"]))]
    target_map = _targets_by_id(targets)
    path = Path(config["paths"]["expansions"])
    backend_name = backend_override or options.get("backend", "qwen")
    if backend_name == "cached":
        if strict(config):
            require_stage(config, "expansion")
        if force:
            raise ValueError("--force cannot regenerate expansions with the cached backend")
        backend = CachedExpansionBackend(targets, path, num_rewrites)
        for sample_id, row in backend.records.items():
            validate_review_checks(sample_id, row["rewrites"], options)
        LOGGER.info("Validated %d cached expansions without loading a model", len(backend.records))
        return [backend.records[target["sample_id"]] for target in targets]
    if backend_name != "qwen":
        raise ValueError(f"Unknown expansion backend: {backend_name}")

    records: dict[str, dict[str, Any]] = {}
    metadata_migrated = False
    if path.exists() and not force:
        for row in read_jsonl(path):
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or sample_id not in target_map or sample_id in records:
                raise ExpansionValidationError(
                    f"Unexpected or duplicate cached sample_id {sample_id!r}; use --force to rebuild"
                )
            _validate_record(target_map[sample_id], row, num_rewrites)
            spec = _target_generation_spec(config, sample_id)
            signature = fingerprint(spec)
            generated = row.get("generation", {})
            review_keys = {"review_guidance_sha256", "review_checks_sha256"}
            if any(generated.get(key) != value for key, value in spec.items() if key not in review_keys):
                raise ExpansionValidationError(
                    f"Expansion provenance mismatch for {sample_id}; use --force to rebuild, "
                    "or --backend cached to explicitly consume an imported cache"
                )
            guidance = options.get("review_guidance", {}).get(sample_id)
            checks = options.get("review_checks", {}).get(sample_id)
            if generated.get("semantic_review_guidance") != guidance or generated.get("semantic_review_checks") != checks:
                LOGGER.info("Review feedback changed for %s; regenerating only this target", sample_id)
                continue
            validate_review_checks(sample_id, row["rewrites"], options)
            if generated.get("fingerprint") != signature:
                # Earlier cache identities included other targets' guidance even
                # though only this target's guidance was passed to the model.
                row["generation"] = {**generated, **spec, "fingerprint": signature}
                metadata_migrated = True
            records[sample_id] = row
    missing = [target for target in targets if target["sample_id"] not in records]
    if not missing:
        if metadata_migrated:
            write_jsonl(path, [records[target["sample_id"]] for target in targets])
        LOGGER.info("Reusing all %d cached expansions", len(records))
        record_stage(config, "expansion")
        return [records[target["sample_id"]] for target in targets]

    seed_everything(int(config.get("seed", 42)))
    generator: ExpansionBackend = QwenExpansionBackend({**options, "seed": int(config.get("seed", 42))})
    LOGGER.info("Generating %d expansions; reusing %d", len(missing), len(records))
    for target in missing:
        spec = _target_generation_spec(config, target["sample_id"])
        signature = fingerprint(spec)
        previous_error = ""
        for attempt in range(1, max_attempts + 1):
            try:
                rewrites = validate_rewrites(
                    target["question"], generator.generate(sample_id=target["sample_id"], question=target["question"],
                                                         attempt=attempt, previous_error=previous_error), num_rewrites
                )
                validate_review_checks(target["sample_id"], rewrites, options)
                break
            except ExpansionValidationError as exc:
                previous_error = str(exc)
                LOGGER.warning("Rewrite validation failed for %s (attempt %d): %s", target["sample_id"], attempt, exc)
        else:
            raise ExpansionValidationError(
                f"Could not generate valid rewrites for {target['sample_id']} after {max_attempts} "
                f"attempts: {previous_error}. Earlier successful samples remain cached."
            )
        records[target["sample_id"]] = {
            "sample_id": target["sample_id"],
            "original_query": target["question"],
            "rewrites": rewrites,
            "generation": {**spec, "fingerprint": signature, "attempt": attempt,
                           "semantic_review_guidance": options.get("review_guidance", {}).get(target["sample_id"]),
                           "semantic_review_checks": options.get("review_checks", {}).get(target["sample_id"])},
        }
        ordered = [records[item["sample_id"]] for item in targets if item["sample_id"] in records]
        write_jsonl(path, ordered)
        LOGGER.info("Cached valid rewrites for %s (%d/%d)", target["sample_id"], len(records), len(targets))
    ordered = [records[target["sample_id"]] for target in targets]
    validate_expansions(targets, ordered, num_rewrites)
    record_stage(config, "expansion")
    return ordered
