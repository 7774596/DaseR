# DaseR

RAG-native KV cache service for LLM inference. Integrates with vLLM via `KVConnectorBase_V1`; stores KV tensors directly to NVMe using NVIDIA cuFile (GDS) or io_uring as a fallback.

## Install

```bash
source <venv>/bin/activate
pip install -e .
```

## Docs

- [Architecture](docs/architecture.md) — system overview, components, design decisions
- [Development guide](docs/development.md) — server setup, tests, lint, benchmarks, vLLM integration
- [Contributing](CONTRIBUTING.md) — branch conventions, commit format, PR process
