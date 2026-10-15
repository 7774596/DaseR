# SPDX-License-Identifier: Apache-2.0

from daser.transfer.base import TransferLayer
from daser.transfer.gds.layer import GDSTransferLayer


def test_backends_without_a_memory_tier_inherit_held_no_ops() -> None:
    """Held-record hooks are concrete on the base class, so GDS keeps working."""
    for name in ("hold_resident", "release_hold", "discard_held"):
        assert name not in TransferLayer.__abstractmethods__
        assert getattr(GDSTransferLayer, name) is getattr(TransferLayer, name)
