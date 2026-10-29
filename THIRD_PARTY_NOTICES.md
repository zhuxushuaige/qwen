# Third-party source notices

## hust-kylin-os-memory memory components

Source: https://github.com/ichor-001/hust-kylin-os-memory

The Python sources under `vendor/memory_core/src/` are the local source dependency closure of the reused memory writer, memory retriever, SQLite memory store and related data structures. These files retain upstream module documentation and were copied from the development checkout used by the desktop assistant. They are not the complete upstream application and may include locally modified development code.

The source checkout's `pyproject.toml` declares the project license as MIT. No standalone LICENSE file was present in the source checkout at packaging time. This notice records that metadata and provenance; it does not assign new ownership or a new license to upstream code. File checksums are in `vendor/memory_core/SOURCE_MANIFEST.json`.

## Qwen-Agent

https://github.com/QwenLM/Qwen-Agent — Apache-2.0. Installed as a dependency; not vendored here. The tool adapter follows the documented `BaseTool` custom-tool interface.
