"""Qwen backend with fixed greedy-then-sampling construction fallback."""

from __future__ import annotations

import copy
import random
import time

from core.common import fingerprint, seed_everything
from core.json_constraints import JsonListConstraint
from .serialization import messages, passage_schema, schema, tier3_messages


class Backend:
    def __init__(self, model_options: dict):
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.numpy = np
        self.options = model_options
        torch.set_num_threads(8)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_options["model_path"], local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_options["model_path"], local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation=model_options.get("attention_implementation", "eager"),
        ).to(model_options.get("device", "cuda")).eval()
        self.constraint = JsonListConstraint(self.tokenizer, 4)

    def generate(self, *, sample_id: str, question: str, reference_answer: str,
                 candidate: dict, attempt_index: int, history: list[dict],
                 tier: int = 1, selected_answer: str | None = None) -> dict:
        seeds = candidate.get("attempt_seeds", [42, 43, 44])
        if attempt_index < 1 or attempt_index > candidate["max_attempts"]:
            raise ValueError("attempt index exceeds the fixed construction budget")
        if tier == 1:
            if attempt_index != len(history) + 1:
                raise ValueError("attempt history is not contiguous")
            seed = int(seeds[attempt_index - 1])
        else:
            # Tier 2 and Tier 3 each have a fresh local one-call budget after
            # Tier 1. Their attempt index is intentionally reset to one.
            if attempt_index != 1:
                raise ValueError("fallback tier attempt index must be one")
            seed = int(candidate.get("seed", seeds[0]))
        seed_everything(seed)
        random.seed(seed)
        self.numpy.random.seed(seed)
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)
        if tier == 3:
            if not selected_answer:
                raise ValueError("Tier 3 requires a deterministic selected answer")
            msg = tier3_messages(question=question, reference_answer=reference_answer, selected_answer=selected_answer)
            output_schema = passage_schema()
        else:
            candidate_count = int(candidate.get("candidate_count", 6 if candidate.get("explicit_ban") else 4))
            msg = messages(question=question, reference_answer=reference_answer, history=history,
                           candidate_count=candidate_count, explicit_ban=bool(candidate.get("explicit_ban")))
            output_schema = schema(candidate_count)
        prompt = self.tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False, truncation=False,
        ).to(self.options.get("device", "cuda"))
        prompt_tokens = int(inputs["input_ids"].shape[1])
        if prompt_tokens > self.options["max_input_tokens"]:
            raise ValueError("Poison prompt exceeds max_input_tokens")
        if prompt_tokens + candidate["max_new_tokens"] > self.model.config.max_position_embeddings:
            raise ValueError("Poison generation exceeds model context window")
        self.constraint.schema = output_schema
        do_sample = bool(candidate.get("do_sample", attempt_index > 1))
        decoding = {
            "do_sample": do_sample,
            "num_beams": 1,
            "max_new_tokens": candidate["max_new_tokens"],
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            temperatures = candidate.get("attempt_temperatures", [0.0, 0.7, 0.9])
            temperature = candidate.get("temperature")
            if temperature is None:
                temperature = temperatures[attempt_index - 1]
            decoding.update(temperature=temperature, top_p=candidate.get("top_p", 0.9), top_k=candidate.get("top_k", 0))
        else:
            decoding.update(temperature=None, top_p=None, top_k=None)
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with self.torch.inference_mode():
            tokens = self.model.generate(
                **inputs, **decoding,
                prefix_allowed_tokens_fn=self.constraint.prefix_function(),
            )
        self.torch.cuda.synchronize()
        generated = tokens[0, prompt_tokens:].tolist()
        return {
            "text": self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
            "messages": msg,
            "actual_prompt": prompt,
            "chat_template": self.tokenizer.chat_template,
            "input_token_ids": inputs["input_ids"][0].tolist(),
            "output_token_ids": generated,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": len(generated),
            "input_truncated": False,
            "seed": seed,
            "decoding_config": copy.deepcopy(decoding),
            "candidate_protocol": copy.deepcopy(candidate),
            "attempt_index": attempt_index,
            "tier": tier,
            "enable_thinking": False,
            "prompt_sha256": fingerprint(prompt),
            "runtime_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": self.torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": self.torch.cuda.max_memory_reserved(),
        }

