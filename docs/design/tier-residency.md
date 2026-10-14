# 压缩在线存储的 packed arena 与 L1/L2 驻留

本文描述 `--storage-format compressed-online` 下 server 如何放置 packed
record，以及 L1（pinned host）和 L2（SSD）如何共同决定一个 chunk 是否留在
索引里。raw 格式按固定 slot 寻址，不使用本文机制。

## packed arena

每个 rank lane 是一段连续字节地址空间，由 `OnlinePackedLayout` 管理：

- 地址空间 = `--l2-size + --l1-size`（`DaserConfig.total_store_bytes`），
  `daser.store` 按此长度 truncate 成稀疏文件；只有写入 SSD 的 record 占用
  磁盘块。
- 空闲区间是按地址排序、相邻合并的 free extent 列表，first-fit 放置。
  同一批 source 连续的 record 优先放成一个连续区间，以便合并成一次 DMA。
- 每个 record 在整个索引生命周期内地址不变（placement）。chunk 离开索引时
  释放其 placement；仍有在途写（write hold）的 placement 先 retired，最后一个
  写结束后才归还字节，避免迟到写覆盖新 record。
- 逻辑 ring slot 按最小 packed 长度超额供给，ring 只负责 slot 分配和
  元数据身份；物理容量由 arena 和下面的 L2 预算决定。

## 两种驻留状态

每个 placement 处于以下状态之一：

| 状态 | 含义 | 计入 L2 预算 |
|------|------|--------------|
| charged | SSD 上有（或将有）有效副本，位于 L2 FIFO 中 | 是 |
| held | 只保证在 L1 中驻留；L1 条目 dirty（尚未写 SSD）或 clean | 否 |

chunk 留在索引中的条件是它的每个 record 要么 charged，要么 held。L2 FIFO
只回收 L2 预算，不直接决定索引驻留，因此 SSD 的 FIFO 不会淘汰 L1 中的热数据。

### 回收

每次 store 放置后，server 在本 lane 的 charged 字节超过 `--l2-size` 时
循环回收（`ServerCore.reclaim_packed`）：

1. 取 L2 FIFO 中最老且不属于本次 store 的 charged record。
2. 若该 record 整段驻留在一个 L1 条目内，`TransferLayer.hold_resident`
   把它切成独立 L1 条目并标记为 held-clean：零拷贝 subslice，保留原
   allocation 的 LRU/BIP 位置和容量计费，不产生 IO。record 转为 held。
3. 否则删除整个 chunk（`reason=capacity`）。

arena 地址空间不足（`PackedArenaFull`）时用同一循环，并在 FIFO 耗尽时删除
最老的 held chunk 作为兜底；正常配置下 held 字节受 L1 容量约束，不会走到
兜底。

### L1 淘汰 held 条目

L1 按自身 LRU/BIP 淘汰。held 条目离开 L1 时：

- dirty：先发起 L2 写回（保留同一地址），然后通知 server。
- server 把该 record 重新 charge：exclusive 策略放到 FIFO 头（最新），
  inclusive 策略放到 FIFO 尾（最老，下次超预算时首先淘汰）。

重新 charge 可能使 charged 字节暂时超出预算，超出量受一次 L1 淘汰批次约束，
在下一次 store 放置时回收。chunk 删除时，layout 对其 held placement 调用
`discard_held`，L1 中的对应条目被直接丢弃、不写回，因此地址复用前不会残留
别名条目。

## inclusive 与 exclusive

两者只在 store 策略上不同：

- inclusive（`DASER_EXCLUSIVE_TIERS=0`）：record 写入 L1 并立即异步写 SSD，放置即 charged。
  L1 与 L2 内容可以重叠；行为对应 LMCache 的 L1/L2 write-through。
- exclusive（默认）：新 record 以 held-dirty 放置，只写
  L1、不计 L2 预算；L1 淘汰时才写 SSD 并进入 FIFO 头。L1 中驻留的数据不占
  L2 预算，两层合计容量接近 `L1 + L2`。

| | inclusive | exclusive |
|---|---|---|
| store 写 SSD | 立即 | L1 淘汰时 |
| 新 record 状态 | charged | held-dirty |
| L2 FIFO 淘汰的 L1 驻留 record | 转 held-clean，保留在索引中 | 同左 |
| held 条目被 L1 淘汰 | 回 FIFO 尾 | 写回（若 dirty）后回 FIFO 头 |

`--skip-l2` 下没有 L2 预算，arena 只寻址 L1，lookup 仍只返回 L1 驻留的前缀；
GDS backend 没有 L1，`hold_resident` 总是失败，退化为纯 L2 FIFO。

## 已知限制

- 一个 record 跨两个 L1 条目时（例如被 readahead 分片切开）不能 hold，按
  非驻留处理。
- ring 仍按 slot 分配顺序回绕；held record 存活时间可以长于 ring 一圈，回绕
  到它时会被 ring 淘汰。ring 按最小 packed 长度超额供给，实际负载下很少发生。
- packed placement 只存在于 server 内存，该格式的索引是易失的，启动冷启动。
