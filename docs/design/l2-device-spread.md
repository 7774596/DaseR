# L2 设备感知的链分散放置

L2 store 文件所在文件系统可能横跨多块 NVMe（btrfs `single`/多设备、LVM 线性
拼接等）。这类布局下文件的一段连续区间只落在一块盘上，单块盘约 7 GB/s；而一次
前缀 load 读的是同一条 chunk 链。TLSF 为一批新 chunk 做连续预留、稳态下又从
同一条旧链淘汰出的洞里分配，于是一条链的 chunk 几乎都在同一块盘上，单个读任务
被单盘带宽封顶，其余盘空闲。

本设计让 server 在启动时（无特权）探测 L2 文件区间到物理盘的映射，并在放置
新 chunk 时让它避开链上父 chunk 所在的盘，使一次前缀读自然分散到多块盘。

## 实测动机

btrfs `single` 数据 profile、4 块 PCIe Gen4 x4 NVMe，64 GiB NOCOW 文件，读任务
为 64 个 9 MiB O_DIRECT extent（与生产读任务同形），目标为 cudaHostAlloc 内存：

| 并发读任务 | 同一块盘 | 随机分布 | 轮转 4 盘 |
|---|---|---|---|
| 1 | 5.7 GB/s | 16.6 GB/s | 20.3 GB/s |
| 2 | 10.2 | 19.8 | 22.3 |
| 4 | 12.7 | 18.8 | 21.0 |
| 8 | 13.3 | 16.9 | 19.6 |

生产（cmp 8 600）中单并发读任务实测 4.45 GB/s，与"同一块盘"一列一致；旧 ring
代码的大段连续读同样只落单盘。读路径本身（SQE 数、内存类型）不是瓶颈。

## 设备区间探测（`daser/server/device_zones.py`）

`probe_device_zones(path, length, grain)` 在 store 文件就位后、transfer layer
打开文件前调用一次，返回覆盖 `[0, length)` 的 `(start, end, zone)` 区间：

1. 叶子块设备：`/sys/block` 中 `slaves/` 为空、大小非零的设备（排除 loop/ram/
   zram 等）。少于 2 个直接返回空（单区间）。
2. `FS_IOC_FIEMAP` 取文件 extent。btrfs 一个 extent 不跨 chunk，`single`
   profile 下一个 chunk 只在一块盘上，所以每个 extent 只属于一块盘。
3. 逐 extent 发一个 64 KiB O_DIRECT 探针，比较前后 `/sys/block/<dev>/stat` 的
   扇区计数：未写（`FIEMAP_EXTENT_UNWRITTEN`）extent 写 0（未写区间本就读出 0，
   内容不变，且未写区间的读不落盘），已写 extent 只读。恰有一块盘增量 ≥ 探针
   大小即归属该盘；否则重试，多次结果取交集，仍不唯一则标为未知。后台 IO 只会
   让探针重试，不会写坏数据。
4. 未知 extent 与文件空洞继承前一个已知区间；边界按粒度 `G` 向下取整；相邻同盘
   区间合并，短于 16 MiB 的区间并入前一个区间。只剩一块盘时返回空。

`--l2-device-zones {auto,off}`（默认 `auto`）。每次启动重新探测：物理位置只在
文件系统 balance/迁移后改变，且映射错误只影响性能，不影响正确性。

## 分区分配器（`ZonedAllocator`，`daser/server/allocator.py`）

每个 zone 一个 `TlsfAllocator`，覆盖整条 lane、只把本 zone 的区间作为初始空闲
（`TlsfAllocator(size, grain, free_ranges=...)`），因此任何预留都不跨 zone，也
不会与别的 zone 的空闲块合并。`reserve(length, avoid=())` 按空闲字节从多到少
尝试不在 `avoid` 中的 zone；`release` / `take` 按区间边界切开分派（持久化恢复
的 extent 可能跨 zone）。`free_bytes`、`holes`、`largest_free` 为各 zone 汇总，
`zone_of(offset)` 二分查找。单 zone 时行为与原 `TlsfAllocator` 相同。

## 放置策略（`TierIndex._reserve_l2`）

- 只有一个 zone：保持原逻辑（整批连续预留，失败再逐 chunk）。
- 多个 zone：逐 chunk 预留，`avoid` 为该 chunk 最近一个有 L2 副本的祖先（最多
  向上 4 级，含同批刚预留的祖先）所在 zone。空间不足时照常按叶子优先 LRU 淘汰
  L2 候选，直到能在允许的 zone 中放下；为此最多多淘汰 `2 × zone 数` 个受害者，
  之后或无候选可淘汰时放开约束（`daser_tier_l2_spread_relaxed_total` 计数）。

只避开父 chunk 一个 zone（而不是严格轮转全部 zone）是有意的：允许集合占
`(Z-1)/Z`，LRU 受害者绝大多数直接落在允许的 zone，额外淘汰很少；空闲 slack
留在其他 zone，会被之后允许在那里的 chunk 用掉，不形成永久容量损失。严格轮转
会让容量较小的盘按 `1/Z` 的写入份额被反复淘汰，等效缓存容量随盘容量差异缩水。
相邻 chunk 必在不同盘，一次前缀读的盘负载接近"随机分布"一列。

## 已知限制

- TP > 1 时 zone 按 rank 0 的 lane（文件 `[0, l2_lane_bytes)`）探测，其他 rank
  的 lane 偏移不同，只得到近似分散。
- RAID0/条带化设备上一个探针会落到多块盘，extent 标为未知，退化为单区间。
- 探测依赖可读的 `/sys/block/*/stat` 与 FIEMAP；不可用时退化为单区间。
