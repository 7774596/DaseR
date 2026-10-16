# 槽位分配、叶子优先淘汰与 L1/L2 驻留

本文描述 server 如何分配存储位置、选择淘汰对象，以及在
`--storage-format compressed-online` 下 L1（pinned host）和 L2（SSD）如何共同
决定一个 chunk 是否留在索引里。raw 与 compressed-online 共用分配器和淘汰树；
L1 驻留（held）只用于 compressed-online。

## 槽位分配

`ChunkManager` 用按地址排序、相邻合并的空闲区间列表（`FreeExtents`）分配
逻辑槽位，first-fit：

- 没有 ring、head/tail 和回绕 SKIP 块；chunk 删除后槽位立即可复用。
- 空间不足时 `alloc` 抛 `SlotsExhausted`，由 `ServerCore` 从淘汰树取一个
  victim 删除后重试；没有可淘汰的 chunk 时分配失败，不会淘汰在写的 chunk。
- 一次 store 期间 IPC server 对其 chunk 持有槽位写保持（write hold）。保持
  期间被删除的 chunk，其槽位在最后一个保持结束后才归还，避免迟到写覆盖
  新 chunk。
- raw 的逻辑槽位数等于 L2 物理槽位数。compressed-online 按最小 packed 长度
  超额供给逻辑槽位，物理容量由下面的 packed arena 和 L2 预算决定。

## 淘汰树

每个已分配 chunk 是 `EvictionTree` 的一个节点，记录 `parent_key`（前缀链中
上一个块的 key，由 connector 的 store intent 携带，链首为空）。

- 只有叶子（没有仍在索引中的子节点）可以被淘汰，按最近访问（alloc、commit、
  lookup 命中）从旧到新。父节点只在最后一个子节点离开后才成为候选，因此
  前缀闭包天然成立：一个会话的链从冷尾部往前淘汰，链首最后离开。
- 孤儿：父 key 不在索引中的节点。lookup 在第一个缺失 key 处停止，孤儿不可
  达，优先级最低。孤儿只在非叶子被删除（held record 被 L1 淘汰、显式淘汰、
  文档删除）或分配时父节点已不在时出现；不做级联删除，只把子树的叶子重新
  入堆，按需淘汰。父 key 再次被存入时子节点自动重新挂上。
- 不可淘汰（跳过）：有未提交写者的 chunk、当前 store 放置阶段固定的 chunk、
  排队保护的 chunk（见下）。
  被跳过的子节点同时让其父节点保持非叶子。
- lookup 命中一段前缀时按 `parent_key` 刷新整条链（祖先比后代旧），因此
  链尾始终是链上最冷的节点。
- 实现为两个惰性最小堆：全部叶子，以及 charged 前沿（见「淘汰触发」3）。
  出堆时校验节点仍存在、仍是候选、优先级未变；单次淘汰均摊 O(log n)，不产生
  IO。

### 回访感知选择（研究开关 `DASER_EVICTION_POLICY=return`，默认 `lru`）

多个 agent 会话轮流回访、总工作集超过容量时，叶子 LRU 淘汰的正是下一个
要回来的会话（循环抖动）。`return` 策略按线程选牺牲者：

- 线程：节点继承父节点的线程；没有在索引中的父节点时，以父 key（链首则用
  自身 key）为线程 id。线程即一条前缀链的根，服务端不需要 session id。
- 回访：`add` 与 lookup 命中时记录线程的到达时间；与上次相隔小于 1 s 视为
  同一次访问。回访周期用相邻到达间隔的 EWMA（α=0.5）估计，未知时用全部
  线程周期的均值。
- 选择：孤儿候选优先（同 LRU）。否则在有候选的线程中，先选已过期的
  线程（距上次到达超过 3 个周期，按超期时长从大到小），再选预测下次到达
  （上次到达 + 周期）最晚的线程；在该线程内淘汰最旧的候选（叶子，或 L2
  预算路径的 charged 前沿节点）。
- 代价：每次淘汰扫描线程和所选线程的叶子，O(线程数 + 线程内节点数)。只在
  开关打开时运行；默认路径仍是上面的两个堆。

### 排队保护（研究开关 `DASER_PROTECT_WAITING=<比例>`，默认 0 关闭）

高并发下请求在 vLLM waiting 队列里可能等上百秒，DaseR 只在调度时
（`get_num_new_matched_tokens`）查找。这段时间里其他请求的 store 会把它的
前缀挤出去：前缀明明在到达时还在，调度时已经没了。保护让到达时已命中的
前缀撑到调度：

- connector 在 `on_new_request`（请求进入 waiting 队列）时发 `protect_prefix`，
  以 request id 为 lease。释放：`update_state_after_alloc` 且没有外部 token；
  有外部 load 时等它落地（`finished_recving`），避免 chunk 在读排队期间被
  淘汰、字节被复用；以及抢占、`request_finished`（含 waiting 中被 abort）。
  layer-wise load 没有 `finished_recving`，到 `request_finished` 才释放。
- server 沿 `RetrievalIndex.chain_keys` 从链首往后保护仍在淘汰树中的 key
  （包括尚未提交的写者），遇到第一个缺失 key 停止。保护的 key 不可淘汰
  （和 pin 一样被三种触发跳过）；held 的被保护叶子被 L1 淘汰时重新计入
  预算而不删除。保护不刷新访问时间，不计 lookup 指标。
- 预算：被保护 key 的去重字节合计不超过比例 × 容量。compressed-online 按
  record 的 stored 字节计（同一预算下能保护更多块），容量为 packed arena
  （L1 + L2）；raw 按槽位计，容量为 L2。字节未知的 key（未提交写者）按 raw
  槽位计。超出预算时只保护链首方向能放下的部分，链首被多个请求共享时只
  计一次。
- 泄漏兜底：lease 超过 10 min（实测排队 p90 的约 2 倍）未释放，在下一次
  `protect_prefix` 时回收。
- 只适用于有固定匹配顺序的索引（前缀链）；chunk 复用索引的 `chain_keys`
  为空，不保护。

与已有工作的关系：CachedAttention/AttentionStore（ATC'24）按调度队列决定外存
淘汰与预取。这里的差别在保护预算按压缩字节计，以及叶子优先树上的链保护。

## packed arena

每个 rank lane 是一段连续字节地址空间，由 `OnlinePackedLayout` 管理：

- 地址空间 = `--l2-size + --l1-size`（`DaserConfig.total_store_bytes`），
  `daser.store` 按此长度 truncate 成稀疏文件。
- 空闲字节同样由 `FreeExtents` 管理。同一批 source 连续的 record 优先放成一
  个连续区间，以便合并成一次 DMA。
- record 在索引生命周期内地址不变；仍有在途写的 placement 先 retired，最后
  一个写结束后才归还字节。

## 驻留状态

每个 packed record 处于以下状态之一：

| 状态 | 含义 | 计入 L2 预算 |
|------|------|--------------|
| charged | SSD 上有有效副本 | 是 |
| held | L1 中有完整干净副本，SSD 副本不再计入预算 | 否 |

chunk 留在索引中的条件是它的每个 record 要么 charged，要么 held。

## 淘汰触发

1. 逻辑槽位耗尽：从淘汰树取 victim 删除。
2. arena 地址空间不足（`PackedArenaFull`）：从淘汰树取 victim 删除（held
   与否都可）；只剩不可淘汰节点时本次 store 失败。
3. 每次 store 放置后本 lane 的 charged 字节超过 `--l2-size`：从 charged 前沿
   取 victim。若它的每个 charged record 都整段驻留在 L1 条目内，
   `TransferLayer.hold_resident` 把它们标记为 held（零拷贝 subslice，保留 L1
   的 LRU/BIP 位置，无 IO），chunk 留在索引中；否则删除 chunk，连同它下面
   （必然全部 held）的子树。

### held 闭包

不变量：held 节点的每个后代都是 held。L2 预算因此只在「charged 前沿」上
回收：未 held、且没有未 held 子节点的节点（叶子是特例）。held 叶子不再挡住
父节点，预算不会因为链尾被 held 而失控。维护方式：

- 在 held 节点下新增子节点（会话回来续写链）时，沿父链把 held 祖先重新计入
  预算（`TransferLayer.release_hold`：L1 条目保留为普通条目，SSD 副本一直
  有效，无 IO），直到遇到未 held 的祖先。超出的预算由下一次回收从前沿收回。
- held record 被 L1 淘汰时：chunk 若还有子节点，只把该 record 重新计入预算，
  chunk 留在索引中；没有子节点才删除 chunk。L1 淘汰因此不会在链中间开洞、
  把后面的块变成孤儿。
- 删除一个前沿节点时，它下面的 held 子树一并删除（不可达，留着只占 L1 和
  arena）。正在写或被固定的后代跳过，成为孤儿按需淘汰。

## L1 淘汰 held 条目

L1 按自身 LRU/BIP 淘汰。held 条目离开 L1 时只通知 server，server 删除该
record 所属的 chunk。通知到删除之间 placement 尚未释放，SSD 副本仍有效，
并发的 L2 load 读到的仍是正确数据。chunk 因其他原因删除时，layout 对其 held
placement 调用 `discard_held`，L1 中的对应条目被直接丢弃，地址复用前不会
残留别名条目。

L1 与 L2 内容可以重叠（write-through），被 L2 预算淘汰的 L1 驻留 record 转为
held 留在索引中，两层合计的不同内容接近 `L1 + L2`。

`--skip-l2` 下没有 L2 预算，arena 只寻址 L1；GDS backend 没有 L1，
`hold_resident` 总是失败，退化为纯 L2 叶子 LRU。

## 已知限制

- held record 的 SSD 字节不打洞，`du` 可超过 `--l2-size`，超出量不超过 held
  字节（受 L1 容量约束）。重新计入预算依赖这一点：held record 的 SSD 副本
  直到 placement 释放前都有效。
- raw 格式没有 held 路径，两层合计容量只有 L2；比较 raw 与压缩时需注意。
- lookup 到 L2 load 之间被淘汰的 chunk 可能被新 chunk 覆盖；lookup 会刷新
  整条链的访问时间，叶子 LRU 下只在容量严重不足时发生。
- 一个 record 跨两个 L1 条目时（例如被 readahead 分片切开）不能 hold，按
  非驻留处理。
- packed placement 只存在于 server 内存，该格式的索引是易失的，启动冷启动。
  raw 的持久化只保存 chunk 索引（含 `parent_key`），分配器和淘汰树在加载时
  重建。
