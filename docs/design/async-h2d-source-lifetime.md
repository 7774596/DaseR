# Asynchronous H2D source ownership

An io_uring load can submit a CUDA copy and then suspend while other
operations run. Submission does not finish the DMA: every source extent must
remain immutable until an event recorded after that copy completes.

Source extents are owned by the server's `TierIndex`. Before a load submits
its moves, the server takes an `io_pin` on the tier copy it reads (L1 extent
or the staging slice of an L2 read). A pinned copy is never an eviction
candidate and its extent is never released, so no other store can reuse the
bytes while a DMA is in flight. The transfer layer polls the copy event
without blocking the server loop and reports completion; the server then
unpins and wakes allocations waiting on `io_released`. No payload is copied
again and no extra pinned or GPU capacity is allocated.

Each load awaits its own copy events before returning or propagating an
error. Completed copies unpin independently while later L2 reads are still
pending. Transfer drain also waits for outstanding copy events. CPU
destinations keep their synchronous copy behavior. CUDA event creation
precedes copy submission; an error after partial submission still records and
drains the completion event before the pins are released.

Stores have the inverse ownership boundary: an executor thread copies the
worker's source into the placed L1 extent or a staging slice. Cancelling the
asyncio await cannot stop that thread. The store therefore shields and drains
the copy task before propagating cancellation or releasing the extent,
including when the caller cancels repeatedly. The L1 extent stays io-pinned
until its L2 write completes. Failed snapshots release their placement and
wake waiters before returning.

Tier extents are never partially overwritten: a chunk copy is placed once and
released whole, so there are no surviving fragments of an older allocation.
Validation covers delayed DMA under eviction pressure, L2 promotion,
cancellation and byte-exact restoration on a real CUDA device.
