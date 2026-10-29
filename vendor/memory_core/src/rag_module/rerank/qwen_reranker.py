"""Qwen3-Reranker-0.6B reranker implementation."""

from __future__ import annotations

import time

from src.rag_module.retrieval.models import RagCandidate

DEFAULT_RERANKER_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 2
SUPPORTED_DTYPES = ("auto", "float32", "float16", "bfloat16")
PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based "
    'on the Query and the Instruct provided. Note that the answer can only be "yes" '
    'or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class QwenReranker:
    """Rerank candidates using Qwen3-Reranker-0.6B."""

    def __init__(
        self,
        model_id: str = DEFAULT_RERANKER_MODEL_ID,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = MAX_LENGTH,
        dtype: str | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if dtype is not None and dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported reranker dtype: {dtype}")
        self._model_id = model_id
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._dtype = dtype
        self._model = None
        self._tokenizer = None
        self._prefix_tokens: list[int] = []
        self._suffix_tokens: list[int] = []
        self._token_false_id: int | None = None
        self._token_true_id: int | None = None
        self.last_timings_ms = {"tokenization": 0.0, "inference": 0.0}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for QwenReranker. "
                "Install with: pip install transformers torch"
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id,
            padding_side="left",
            trust_remote_code=True,
        )
        model_options = {"trust_remote_code": True}
        load_dtype = self._load_dtype()
        if load_dtype is not None:
            model_options["dtype"] = load_dtype
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            **model_options,
        )
        if self._device:
            self._model = self._model.to(self._device)
        self._model.eval()
        self._token_false_id = self._tokenizer.convert_tokens_to_ids("no")
        self._token_true_id = self._tokenizer.convert_tokens_to_ids("yes")
        self._prefix_tokens = self._tokenizer.encode(PREFIX, add_special_tokens=False)
        self._suffix_tokens = self._tokenizer.encode(SUFFIX, add_special_tokens=False)
        if self._max_length <= len(self._prefix_tokens) + len(self._suffix_tokens):
            raise ValueError("max_length is too small for reranker prompt tokens")

    def _load_dtype(self):
        if self._dtype is None:
            return None
        if self._dtype == "auto":
            if self._device is None or not self._device.startswith("cuda"):
                return None
            return "auto"
        import torch

        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self._dtype]

    @property
    def runtime_config(self) -> dict[str, object]:
        actual_dtype = None
        if self._model is not None:
            parameters = getattr(self._model, "parameters", None)
            if parameters is not None:
                actual_dtype = str(next(parameters()).dtype)
        return {
            "model_id": self._model_id,
            "device": self._device,
            "batch_size": self._batch_size,
            "max_length": self._max_length,
            "configured_dtype": self._dtype,
            "actual_dtype": actual_dtype,
            "use_cache": False,
            "logits_to_keep": 1,
        }

    def rerank(
        self,
        query: str,
        candidates: list[RagCandidate],
        top_k: int,
    ) -> list[RagCandidate]:
        if not candidates:
            return []

        self._ensure_loaded()
        self.last_timings_ms = {"tokenization": 0.0, "inference": 0.0}
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._token_false_id is not None
        assert self._token_true_id is not None

        pairs = [
            _format_instruction(DEFAULT_INSTRUCTION, query, candidate.content)
            for candidate in candidates
        ]
        scores: list[float] = []
        for start in range(0, len(pairs), self._batch_size):
            scores.extend(self._score_batch(pairs[start : start + self._batch_size]))

        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = score

        candidates.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
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
        if self._device:
            encoded = {k: v.to(self._device) for k, v in encoded.items()}
        self.last_timings_ms["tokenization"] += (
            time.perf_counter() - tokenization_started
        ) * 1000

        inference_started = time.perf_counter()
        with torch.inference_mode():
            logits = self._model(
                **encoded,
                use_cache=False,
                logits_to_keep=1,
            ).logits[:, -1, :]
        self.last_timings_ms["inference"] += (
            time.perf_counter() - inference_started
        ) * 1000

        true_logits = logits[:, self._token_true_id]
        false_logits = logits[:, self._token_false_id]
        pair_logits = torch.stack((false_logits, true_logits), dim=1)
        return pair_logits.log_softmax(dim=1)[:, 1].exp().float().cpu().tolist()


def _format_instruction(instruction: str, query: str, document: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
