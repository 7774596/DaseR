# SPDX-License-Identifier: Apache-2.0

# Standard
import time

import pytest

from daser.server.metadata_store import ChunkMeta, MetadataStore


def make_meta(key: str, start: int, num: int, tokens: int = 16) -> ChunkMeta:
    return ChunkMeta(
        chunk_key=key,
        start_slot=start,
        num_slots=num,
        token_count=tokens,
        pos_offset=0,
        model_id="test-model",
        created_at=time.time(),
    )


def test_insert_and_get() -> None:
    store = MetadataStore()
    meta = make_meta("abc", start=0, num=3)
    store.insert(meta)
    assert store.get("abc") == meta
    assert len(store) == 1


def test_remove_returns_meta() -> None:
    store = MetadataStore()
    meta = make_meta("abc", start=0, num=2)
    store.insert(meta)
    assert store.remove("abc") is meta
    assert store.get("abc") is None
    assert len(store) == 0


def test_removals_count_every_removal() -> None:
    store = MetadataStore()
    store.insert(make_meta("abc", start=0, num=2))
    assert store.removals == 0
    store.remove("abc")
    assert store.removals == 1
    with pytest.raises(KeyError):
        store.remove("abc")
    assert store.removals == 1


def test_remove_nonexistent_raises() -> None:
    store = MetadataStore()
    with pytest.raises(KeyError):
        store.remove("nonexistent")


def test_insert_duplicate_raises() -> None:
    store = MetadataStore()
    store.insert(make_meta("abc", start=0, num=2))
    with pytest.raises(ValueError):
        store.insert(make_meta("abc", start=2, num=2))
