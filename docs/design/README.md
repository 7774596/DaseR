# DaseR 系统设计文档

DaseR 是面向 LLM 推理的 RAG-native KV cache 服务。它通过 vLLM
`KVConnectorBase_V1` 接入推理流程，在 vLLM worker 进程内执行 KV 数据面
IO，在 DaseR server 进程内维护 HTTP server、IPC server 和控制面状态。

## 目录

| 文档 | 内容 |
|------|------|
| [整体架构](architecture.md) | 进程拓扑、HTTP/IPC server 边界、启动流程和关键设计决策 |
| [组件详解](components.md) | 组件职责、存储布局、可插拔接口和 IPC 协议 |
| [数据流程](flows.md) | 文档上传、推理、KV store 和 KV load 的完整流程 |
| [可观测性指标优化](metrics-optimization.md) | Prometheus 指标精简、L1/vLLM 集成、Dashboard 重设计 |
| [在途 load 合并](inflight-load-coalescing.md) | 同前缀在途外部 load 时后到请求等待并复用本地前缀命中（`coalesce_loads`，默认关） |
| [槽位分配、叶子优先淘汰与 L1/L2 驻留](tier-residency.md) | 空闲区间分配器、前缀链叶子优先 LRU 淘汰树、compressed-online 的 packed arena、L2 预算与 L1 held 驻留 |
