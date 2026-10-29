"""OpenVINO QInt8 adapter for the multilingual mMiniLM cross-encoder."""

from __future__ import annotations

from pathlib import Path

from src.rag_module.rerank.openvino_gte_reranker import (
    DEFAULT_INFER_REQUESTS,
    DEFAULT_NUM_STREAMS,
    MAX_LENGTH,
    OpenVINOGTEReranker,
)

DEFAULT_OPENVINO_MMINILM_RERANKER_MODEL_ID = (
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)
DEFAULT_OPENVINO_MMINILM_RERANKER_MODEL_PATH = (
    "models/mmarco-mMiniLMv2-L12-H384-v1-int8-ov"
)
DEFAULT_OPENVINO_IR = "openvino/openvino_model_qint8_quantized.xml"
DEFAULT_OPENVINO_WEIGHTS = "openvino/openvino_model_qint8_quantized.bin"


class OpenVINOMMiniLMReranker(OpenVINOGTEReranker):
    """Run the official multilingual mMiniLM QInt8 IR with OpenVINO."""

    def __init__(
        self,
        model_id: str = DEFAULT_OPENVINO_MMINILM_RERANKER_MODEL_PATH,
        device: str | None = "CPU",
        batch_size: int = 1,
        max_length: int = MAX_LENGTH,
        num_streams: int = DEFAULT_NUM_STREAMS,
        infer_requests: int = DEFAULT_INFER_REQUESTS,
    ) -> None:
        if batch_size != 1:
            raise ValueError(
                "openvino-mminilm currently uses the validated independent "
                "batch=1 InferRequest protocol"
            )
        super().__init__(
            model_id=model_id,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            num_streams=num_streams,
            infer_requests=infer_requests,
        )

    def _local_model_path(self, model_root: Path) -> Path:
        model_path = model_root / DEFAULT_OPENVINO_IR
        weights_path = model_root / DEFAULT_OPENVINO_WEIGHTS
        if model_path.is_file() and weights_path.is_file():
            return model_path.resolve()
        missing = [
            path
            for path, exists in (
                (DEFAULT_OPENVINO_IR, model_path.is_file()),
                (DEFAULT_OPENVINO_WEIGHTS, weights_path.is_file()),
            )
            if not exists
        ]
        raise ValueError(
            f"mMiniLM OpenVINO reranker is incomplete at {model_root.resolve()}; "
            f"missing: {', '.join(missing)}"
        )

    @property
    def runtime_config(self) -> dict[str, object]:
        config = super().runtime_config
        config.update(
            {
                "provider": "openvino-mminilm",
                "model_family": "mmarco-mMiniLMv2-L12-H384-v1",
                "license": "Apache-2.0",
                "hidden_size": 384,
                "num_hidden_layers": 12,
            }
        )
        return config
