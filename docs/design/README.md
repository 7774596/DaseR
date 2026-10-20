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
| [HBM 常驻压缩 KV](hbm-resident-kv.md) | AF1 attention backend：KV cache 以无损 AF1 记录常驻 HBM（block ×4/3）、store 状态机、融合解压 decode、prefill gather、CUDA graph 约束 |
| [AF1 记录的外部层存储](af1-external-tiers.md) | `--storage-format af1`：已满 block 的 AF1 记录以 block blob 原样存入 L1/L2、命中时装回（不重编码）、overflow 位图与并发安全、约束 |
| [独立 L1/L2 分层与叶子优先淘汰](tier-residency.md) | server 索引的逐层 chunk 副本、TLSF 分配器、每层叶子优先 LRU 与对称候选规则、write-through 与 L1 接纳、promotion/readahead、prefetch lease、NOCOW store 文件 |
