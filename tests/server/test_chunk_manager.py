# SPDX-License-Identifier: Apache-2.0

# Standard
import os

# Third Party
import msgpack
import pytest

# First Party
from daser.server.chunk_manager import ChunkManager, SlotsExhausted
from daser.server.doc_registry import DocEntry, DocRegistry
from daser.server.metadata_store import MetadataStore


def make_manager(total_slots: int = 8) -> ChunkManager:
    store = MetadataStore()
    return ChunkManager(total_slots=total_slots, metadata_store=store)


def alloc(mgr: ChunkManager, key: str, num_slots: int, parent_key: str = "") -> int:
    meta = mgr.alloc(
        key,
        num_slots=num_slots,
        token_count=16 * num_slots,
        model_id="m",
        pos_offset=0,
        parent_key=parent_key,
    )
    return meta.start_slot


def test_alloc_is_first_fit_from_zero():
    mgr = make_manager(8)
    assert alloc(mgr, "key1", 3) == 0
    assert alloc(mgr, "key2", 2) == 3
    assert mgr.free_slots == 3
    assert mgr.store.get("key2").num_slots == 2


def test_alloc_records_parent_key():
    mgr = make_manager(8)
    alloc(mgr, "root", 1)
    alloc(mgr, "child", 1, parent_key="root")
    assert mgr.store.get("child").parent_key == "root"
    assert mgr.store.get("root").parent_key == ""


def test_full_store_raises_without_evicting():
    mgr = make_manager(4)
    alloc(mgr, "a", 2)
    alloc(mgr, "b", 2)
    with pytest.raises(SlotsExhausted):
        alloc(mgr, "c", 1)
    assert mgr.store.get("a") is not None
    assert mgr.store.get("b") is not None
    assert mgr.store.get("c") is None


def test_alloc_rejects_invalid_length():
    mgr = make_manager(4)
    with pytest.raises(ValueError):
        alloc(mgr, "zero", 0)
    with pytest.raises(ValueError):
        alloc(mgr, "huge", 5)


def test_removed_slots_are_reused_and_coalesced():
    mgr = make_manager(6)
    alloc(mgr, "a", 2)
    alloc(mgr, "b", 2)
    alloc(mgr, "c", 2)
    mgr.remove("a")
    mgr.remove("c")
    assert mgr.free_slots == 4
    assert mgr.free_holes == 2
    # Neither hole fits three slots until b leaves and they merge.
    with pytest.raises(SlotsExhausted):
        alloc(mgr, "d", 3)
    mgr.remove("b")
    assert mgr.free_holes == 1
    assert alloc(mgr, "d", 3) == 0


def test_remove_unknown_raises():
    mgr = make_manager(4)
    with pytest.raises(KeyError):
        mgr.remove("missing")


def test_write_hold_defers_slot_reuse():
    mgr = make_manager(2)
    alloc(mgr, "a", 2)
    token = mgr.hold_writes(["a", "missing"])
    mgr.remove("a")
    assert mgr.store.get("a") is None
    with pytest.raises(SlotsExhausted):
        alloc(mgr, "b", 1)
    mgr.end_writes(token)
    assert mgr.free_slots == 2
    assert alloc(mgr, "b", 2) == 0


def test_nested_write_holds_free_after_last():
    mgr = make_manager(2)
    alloc(mgr, "a", 2)
    first = mgr.hold_writes(["a"])
    second = mgr.hold_writes(["a"])
    mgr.remove("a")
    mgr.end_writes(first)
    assert mgr.free_slots == 0
    mgr.end_writes(second)
    assert mgr.free_slots == 2


def test_write_hold_on_live_chunk_frees_nothing():
    mgr = make_manager(2)
    alloc(mgr, "a", 2)
    mgr.end_writes(mgr.hold_writes(["a"]))
    assert mgr.free_slots == 0
    with pytest.raises(KeyError):
        mgr.end_writes(12345)


def test_save_and_load_state(tmp_path):
    mgr = make_manager(8)
    alloc(mgr, "key1", 3)
    alloc(mgr, "key2", 2, parent_key="key1")
    alloc(mgr, "key3", 1)
    mgr.remove("key1")
    path = str(tmp_path / "daser.index")
    mgr.save(path)

    mgr2 = make_manager(8)
    mgr2.load(path)
    assert mgr2.store.get("key1") is None
    assert mgr2.store.get("key2").parent_key == "key1"
    assert mgr2.free_slots == 5
    assert mgr2.free_holes == 2
    assert alloc(mgr2, "key4", 3) == 0


def test_load_rejects_different_slot_capacity(tmp_path):
    mgr = make_manager(8)
    alloc(mgr, "key1", 3)
    path = str(tmp_path / "daser.index")
    mgr.save(path)

    with pytest.raises(ValueError, match="slot capacity"):
        make_manager(16).load(path)


def test_load_rejects_overlapping_chunks(tmp_path):
    mgr = make_manager(8)
    alloc(mgr, "a", 3)
    alloc(mgr, "b", 3)
    path = str(tmp_path / "daser.index")
    mgr.save(path)
    with open(path, "rb") as f:
        payload = msgpack.unpackb(f.read(), raw=False)
    index = msgpack.unpackb(payload["index"], raw=False)
    index["chunk_index"]["b"]["start_slot"] = 1
    payload["index"] = msgpack.packb(index, use_bin_type=True)
    with open(path, "wb") as f:
        f.write(msgpack.packb(payload, use_bin_type=True))

    with pytest.raises(ValueError, match="not free"):
        make_manager(8).load(path)


def test_load_rejects_different_cache_reuse_mode(tmp_path):
    mgr = make_manager(8)
    alloc(mgr, "key1", 3)
    path = str(tmp_path / "daser.index")
    mgr.save(path, cache_reuse_mode="chunk")

    restored = make_manager(8)
    with pytest.raises(ValueError, match="cache reuse mode"):
        restored.load(path, expected_cache_reuse_mode="prefix")


def test_load_rejects_missing_cache_reuse_mode_when_expected(tmp_path):
    mgr = make_manager(8)
    alloc(mgr, "key1", 3)
    path = str(tmp_path / "daser.index")
    mgr.save(path)

    restored = make_manager(8)
    with pytest.raises(ValueError, match="cache reuse mode"):
        restored.load(path, expected_cache_reuse_mode="chunk")


def test_save_and_load_state_includes_doc_registry(tmp_path):
    registry = DocRegistry()
    store = MetadataStore()
    mgr = ChunkManager(total_slots=8, metadata_store=store, doc_registry=registry)
    alloc(mgr, "key1", 3)
    registry.insert(
        DocEntry(
            doc_id="doc1",
            title="Doc 1",
            token_count=48,
            chunk_keys=["key1"],
            cached_mask=[True],
            tokens=[1, 2, 3],
        )
    )
    meta = mgr.store.get("key1")
    assert meta is not None
    meta.doc_ids.append("doc1")
    path = str(tmp_path / "daser.index")

    mgr.save(path)

    registry2 = DocRegistry()
    store2 = MetadataStore()
    mgr2 = ChunkManager(
        total_slots=8,
        metadata_store=store2,
        doc_registry=registry2,
    )
    mgr2.load(path)

    restored_doc = registry2.get("doc1")
    restored_meta = mgr2.store.get("key1")
    assert restored_doc is not None
    assert restored_doc.title == "Doc 1"
    assert restored_doc.tokens == [1, 2, 3]
    assert restored_meta is not None
    assert restored_meta.doc_ids == ["doc1"]


def test_save_failure_keeps_existing_index_readable(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    path = str(tmp_path / "daser.index")
    mgr = make_manager(8)
    alloc(mgr, "old", 2)
    mgr.save(path)

    alloc(mgr, "new", 2)
    original_replace = os.replace

    def fail_replace(src: str, dst: str) -> None:
        if dst == path:
            raise OSError("replace failed")
        original_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        mgr.save(path)

    restored = make_manager(8)
    restored.load(path)
    assert restored.store.get("old") is not None
    assert restored.store.get("new") is None
