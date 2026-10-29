"""OpenVINO INT8 implementation of the multilingual GTE cross-encoder."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from src.rag_module.retrieval.models import RagCandidate

DEFAULT_OPENVINO_GTE_RERANKER_MODEL_ID = "onnx-community/gte-multilingual-reranker-base"
DEFAULT_ONNX_FILENAME = "onnx/model_int8.onnx"
MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_STREAMS = 2
DEFAULT_INFER_REQUESTS = 2
PERFORMANCE_HINT = "THROUGHPUT"


class OpenVINOGTEReranker:
    """Rerank query-document pairs with the encoder-only multilingual GTE model."""

    def __init__(
        self,
        model_id: str = DEFAULT_OPENVINO_GTE_RERANKER_MODEL_ID,
        device: str | None = "CPU",
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = MAX_LENGTH,
        num_streams: int = DEFAULT_NUM_STREAMS,
        infer_requests: int = DEFAULT_INFER_REQUESTS,
    ) -> None:
        if batch_size != 1:
            raise ValueError(
                "OpenVINO GTE INT8 currently requires batch_size=1 because "
                "the community ONNX export produces batch-dependent scores"
            )
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if num_streams <= 0 or infer_requests <= 0:
            raise ValueError("num_streams and infer_requests must be positive")
        self._model_id = model_id
        self._device = _openvino_device(device)
        self._batch_size = batch_size
        self._max_length = max_length
        self._num_streams = num_streams
        self._infer_requests = infer_requests
        self._model = None
        self._infer_queue = None
        self._tokenizer = None
        self._input_names: tuple[str, ...] = ()
        self._rerank_lock = threading.RLock()
        self.last_timings_ms = {"tokenization": 0.0, "inference": 0.0}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from huggingface_hub import hf_hub_download
            from openvino import AsyncInferQueue, Core
            from transformers import XLMRobertaTokenizerFast
        except ImportError as exc:
            raise ImportError(
                "OpenVINOGTEReranker requires OpenVINO, Transformers, and "
                "huggingface-hub. Install with: pip install -e '.[openvino]'"
            ) from exc

        model_root = Path(self._model_id).expanduser()
        if model_root.is_dir():
            tokenizer_id = str(model_root.resolve())
            model_path = self._local_model_path(model_root)
        else:
            tokenizer_id = self._model_id
            model_path = Path(
                hf_hub_download(
                    repo_id=self._model_id,
                    filename=DEFAULT_ONNX_FILENAME,
                )
            )

        # The model config uses remote custom modeling code, but reranking only
        # needs the standard XLM-R tokenizer. Loading it explicitly keeps an
        # offline deployment from prompting for unrelated remote code.
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(tokenizer_id)
        core = Core()
        ov_model = core.read_model(model_path)
        compiled_model = core.compile_model(
            ov_model,
            self._device,
            {
                "PERFORMANCE_HINT": PERFORMANCE_HINT,
                "NUM_STREAMS": self._num_streams,
            },
        )
        infer_queue = AsyncInferQueue(compiled_model, self._infer_requests)
        infer_queue.set_callback(_store_async_score)

        self._tokenizer = tokenizer
        self._model = compiled_model
        self._infer_queue = infer_queue
        self._input_names = tuple(
            model_input.get_any_name() for model_input in compiled_model.inputs
        )

    def _local_model_path(self, model_root: Path) -> Path:
        return _local_onnx_path(model_root)

    @property
    def runtime_config(self) -> dict[str, object]:
        return {
            "provider": "openvino-gte",
            "model_id": self._model_id,
            "device": self._device,
            "batch_size": self._batch_size,
            "max_length": self._max_length,
            "architecture": "encoder-sequence-classification",
            "weight_compression": "INT8",
            "performance_hint": PERFORMANCE_HINT,
            "num_streams": self._num_streams,
            "infer_requests": self._infer_requests,
            "logits_shape": "[batch, 1]",
        }

    def rerank(
        self,
        query: str,
        candidates: list[RagCandidate],
        top_k: int,
    ) -> list[RagCandidate]:
        if top_k <= 0 or not candidates:
            return []

        with self._rerank_lock:
            self._ensure_loaded()
            self.last_timings_ms = {"tokenization": 0.0, "inference": 0.0}
            if self._infer_queue is None:
                scores: list[float] = []
                for start in range(0, len(candidates), self._batch_size):
                    scores.extend(
                        self._score_batch(query, candidates[start : start + self._batch_size])
                    )
            else:
                scores = self._score_async(query, candidates)

            for candidate, score in zip(candidates, scores):
                candidate.rerank_score = score
            candidates.sort(key=lambda candidate: candidate.rerank_score or 0.0, reverse=True)
            return candidates[:top_k]

    def _score_async(
        self,
        query: str,
        candidates: list[RagCandidate],
    ) -> list[float]:
        assert self._infer_queue is not None

        tokenization_started = time.perf_counter()
        prepared_inputs = [self._encode_batch(query, [candidate]) for candidate in candidates]
        self.last_timings_ms["tokenization"] += (time.perf_counter() - tokenization_started) * 1000

        scores: list[float | None] = [None] * len(prepared_inputs)
        inference_started = time.perf_counter()
        for index, model_inputs in enumerate(prepared_inputs):
            self._infer_queue.start_async(
                model_inputs,
                userdata=(scores, index),
            )
        self._infer_queue.wait_all()
        self.last_timings_ms["inference"] += (time.perf_counter() - inference_started) * 1000
        if any(score is None for score in scores):
            raise RuntimeError("OpenVINO GTE asynchronous inference did not return every score")
        return [float(score) for score in scores]

    def _score_batch(
        self,
        query: str,
        candidates: list[RagCandidate],
    ) -> list[float]:
        assert self._tokenizer is not None
        assert self._model is not None

        tokenization_started = time.perf_counter()
        model_inputs = self._encode_batch(query, candidates)
        self.last_timings_ms["tokenization"] += (time.perf_counter() - tokenization_started) * 1000

        inference_started = time.perf_counter()
        outputs = self._model(model_inputs)
        first_output = next(iter(outputs.values()))
        self.last_timings_ms["inference"] += (time.perf_counter() - inference_started) * 1000
        return [_score_from_logit(value) for value in first_output]

    def _encode_batch(
        self,
        query: str,
        candidates: list[RagCandidate],
    ) -> dict[str, object]:
        assert self._tokenizer is not None

        import numpy as np

        encoded = self._tokenizer(
            [query] * len(candidates),
            [candidate.content for candidate in candidates],
            padding=True,
            truncation="longest_first",
            return_tensors="np",
            max_length=self._max_length,
        )
        model_inputs = {
            name: np.asarray(encoded[name]) for name in self._input_names if name in encoded
        }
        # The community ONNX export keeps the conventional segment-id input.
        # XLM-R has a single segment and therefore omits it; all zeros are the
        # exact semantic equivalent expected by type_vocab_size=1.
        if "token_type_ids" in self._input_names and "token_type_ids" not in model_inputs:
            model_inputs["token_type_ids"] = np.zeros_like(encoded["input_ids"])
        missing_inputs = set(self._input_names) - set(model_inputs)
        if missing_inputs:
            raise RuntimeError(
                "GTE tokenizer did not produce required ONNX inputs: "
                + ", ".join(sorted(missing_inputs))
            )
        return model_inputs


def _store_async_score(request, userdata: tuple[list[float | None], int]) -> None:
    scores, index = userdata
    scores[index] = _score_from_logit(request.get_output_tensor(0).data)


def _score_from_logit(value: object) -> float:
    import numpy as np

    logit = float(np.asarray(value, dtype=np.float32).reshape(-1)[0])
    # The official model exposes one raw relevance logit. Sigmoid keeps the
    # project-wide score contract in [0, 1] without changing ranking order.
    return float(1.0 / (1.0 + np.exp(-np.clip(logit, -60.0, 60.0))))


def _local_onnx_path(model_root: Path) -> Path:
    candidates = (
        model_root / DEFAULT_ONNX_FILENAME,
        model_root / "model_int8.onnx",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(
        f"GTE ONNX reranker model is incomplete at {model_root.resolve()}; "
        f"missing: {DEFAULT_ONNX_FILENAME}"
    )


def _openvino_device(device: str | None) -> str:
    if device is None:
        return "CPU"
    normalized = device.strip().upper()
    if normalized.startswith("CUDA"):
        raise ValueError("OpenVINO does not use CUDA device names; use CPU, GPU, or AUTO")
    return normalized
