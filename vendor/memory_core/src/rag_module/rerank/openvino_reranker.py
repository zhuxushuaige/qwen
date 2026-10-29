"""OpenVINO INT8 implementation of the Qwen3 reranker."""

from __future__ import annotations

import time

from src.rag_module.retrieval.models import RagCandidate

DEFAULT_OPENVINO_RERANKER_MODEL_ID = "OpenVINO/Qwen3-Reranker-0.6B-int8-ov"
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 2
PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based "
    'on the Query and the Instruct provided. Note that the answer can only be "yes" '
    'or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class OpenVINOReranker:
    """Rerank candidates with the official INT8 OpenVINO Qwen3 export."""

    def __init__(
        self,
        model_id: str = DEFAULT_OPENVINO_RERANKER_MODEL_ID,
        device: str | None = "CPU",
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = MAX_LENGTH,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self._model_id = model_id
        self._device = _openvino_device(device)
        self._batch_size = batch_size
        self._max_length = max_length
        self._model = None
        self._tokenizer = None
        self._prefix_tokens: list[int] = []
        self._suffix_tokens: list[int] = []
        self._token_false_id: int | None = None
        self._token_true_id: int | None = None
        self._pair_logits_only = False
        self.last_timings_ms = {"tokenization": 0.0, "inference": 0.0}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from optimum.intel import OVModelForCausalLM
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "OpenVINOReranker requires OpenVINO and Optimum Intel. "
                "Install with: pip install -e '.[openvino]'"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(
            self._model_id,
            padding_side="left",
            trust_remote_code=True,
        )
        model = OVModelForCausalLM.from_pretrained(
            self._model_id,
            device=self._device,
            use_cache=False,
            export=False,
            compile=False,
        )
        token_false_id = tokenizer.convert_tokens_to_ids("no")
        token_true_id = tokenizer.convert_tokens_to_ids("yes")
        _optimize_yes_no_lm_head(
            model,
            false_token_id=token_false_id,
            true_token_id=token_true_id,
        )
        model.compile()
        prefix_tokens = tokenizer.encode(PREFIX, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)
        if self._max_length <= len(prefix_tokens) + len(suffix_tokens):
            raise ValueError("max_length is too small for reranker prompt tokens")
        self._tokenizer = tokenizer
        self._model = model
        self._token_false_id = token_false_id
        self._token_true_id = token_true_id
        self._prefix_tokens = prefix_tokens
        self._suffix_tokens = suffix_tokens
        self._pair_logits_only = True

    @property
    def runtime_config(self) -> dict[str, object]:
        return {
            "provider": "openvino",
            "model_id": self._model_id,
            "device": self._device,
            "batch_size": self._batch_size,
            "max_length": self._max_length,
            "weight_compression": "INT8_ASYM",
            "use_cache": False,
            "lm_head_token_pruning": True,
            "logits_shape": "[batch, 1, 2]",
        }

    def rerank(
        self,
        query: str,
        candidates: list[RagCandidate],
        top_k: int,
    ) -> list[RagCandidate]:
        if top_k <= 0 or not candidates:
            return []

        self._ensure_loaded()
        self.last_timings_ms = {"tokenization": 0.0, "inference": 0.0}
        pairs = [
            _format_instruction(DEFAULT_INSTRUCTION, query, candidate.content)
            for candidate in candidates
        ]
        scores: list[float] = []
        for start in range(0, len(pairs), self._batch_size):
            scores.extend(self._score_batch(pairs[start : start + self._batch_size]))

        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = score
        candidates.sort(key=lambda candidate: candidate.rerank_score or 0.0, reverse=True)
        return candidates[:top_k]

    def _score_batch(self, pairs: list[str]) -> list[float]:
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._token_false_id is not None
        assert self._token_true_id is not None

        import torch

        tokenization_started = time.perf_counter()
        encoded = self._tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=(self._max_length - len(self._prefix_tokens) - len(self._suffix_tokens)),
        )
        for index, input_ids in enumerate(encoded["input_ids"]):
            encoded["input_ids"][index] = self._prefix_tokens + input_ids + self._suffix_tokens
        encoded = self._tokenizer.pad(
            encoded,
            padding=True,
            return_tensors="pt",
        )
        model_device = getattr(self._model, "device", None)
        if model_device is not None:
            encoded = {key: value.to(model_device) for key, value in encoded.items()}
        self.last_timings_ms["tokenization"] += (
            time.perf_counter() - tokenization_started
        ) * 1000

        inference_started = time.perf_counter()
        logits = self._model(**encoded).logits[:, -1, :]
        self.last_timings_ms["inference"] += (
            time.perf_counter() - inference_started
        ) * 1000
        if self._pair_logits_only:
            pair_logits = logits
        else:
            true_logits = logits[:, self._token_true_id]
            false_logits = logits[:, self._token_false_id]
            pair_logits = torch.stack((false_logits, true_logits), dim=1)
        return pair_logits.log_softmax(dim=1)[:, 1].exp().float().cpu().tolist()


def _optimize_yes_no_lm_head(model, *, false_token_id: int, true_token_id: int) -> None:
    """Prune the causal LM head to the last position and the no/yes token rows.

    The official OpenVINO export produces ``[batch, sequence, vocabulary]`` logits.
    Slicing that tensor in Python is too late: the full LM-head MatMul has already
    run. This graph rewrite changes the MatMul inputs so the IR computes exactly
    the two logits consumed by the reranker and returns ``[batch, 1, 2]``.
    """
    try:
        import numpy as np
        from openvino import opset13 as ops
    except ImportError as exc:
        raise ImportError(
            "OpenVINOReranker requires OpenVINO. Install with: "
            "pip install -e '.[openvino]'"
        ) from exc

    ov_model = getattr(model, "model", None)
    if ov_model is None or len(ov_model.outputs) != 1:
        raise RuntimeError("unsupported OpenVINO reranker graph: expected one logits output")

    result = ov_model.output(0).get_node()
    lm_head = result.input_value(0).get_node()
    if lm_head.get_type_name() != "MatMul" or "lm_head" not in lm_head.get_friendly_name():
        raise RuntimeError(
            "unsupported OpenVINO reranker graph: logits output is not the Qwen LM head"
        )

    attributes = lm_head.get_attributes()
    if attributes.get("transpose_a") or not attributes.get("transpose_b"):
        raise RuntimeError(
            "unsupported OpenVINO reranker graph: unexpected LM-head MatMul layout"
        )

    hidden_states = lm_head.input_value(0)
    lm_head_weights = lm_head.input_value(1)
    if hidden_states.get_partial_shape().rank.get_length() != 3:
        raise RuntimeError(
            "unsupported OpenVINO reranker graph: expected rank-3 hidden states"
        )
    if lm_head_weights.get_partial_shape().rank.get_length() != 2:
        raise RuntimeError(
            "unsupported OpenVINO reranker graph: expected rank-2 LM-head weights"
        )

    vocabulary = lm_head_weights.get_partial_shape()[0]
    largest_token_id = max(false_token_id, true_token_id)
    if vocabulary.is_static and largest_token_id >= vocabulary.get_length():
        raise RuntimeError("reranker yes/no token id is outside the LM-head vocabulary")

    last_hidden_state = ops.gather(
        hidden_states,
        ops.constant(np.array([-1], dtype=np.int64)),
        ops.constant(np.array(1, dtype=np.int64)),
    )
    last_hidden_state.set_friendly_name("reranker/last_hidden_state")
    yes_no_weights = ops.gather(
        lm_head_weights,
        ops.constant(np.array([false_token_id, true_token_id], dtype=np.int64)),
        ops.constant(np.array(0, dtype=np.int64)),
    )
    yes_no_weights.set_friendly_name("reranker/no_yes_lm_head_weights")

    lm_head.input(0).replace_source_output(last_hidden_state.output(0))
    lm_head.input(1).replace_source_output(yes_no_weights.output(0))
    ov_model.validate_nodes_and_infer_types()

    output_shape = ov_model.output(0).get_partial_shape()
    if (
        output_shape.rank.get_length() != 3
        or not output_shape[1].is_static
        or output_shape[1].get_length() != 1
    ):
        raise RuntimeError("failed to prune OpenVINO reranker sequence logits")
    if not output_shape[2].is_static or output_shape[2].get_length() != 2:
        raise RuntimeError("failed to prune OpenVINO reranker vocabulary logits")


def _openvino_device(device: str | None) -> str:
    if device is None:
        return "CPU"
    normalized = device.strip().upper()
    if normalized.startswith("CUDA"):
        raise ValueError("OpenVINO does not use CUDA device names; use CPU, GPU, or AUTO")
    return normalized


def _format_instruction(instruction: str, query: str, document: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
