# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any

# Third Party
import pytest

# First Party
from daser.transfer.iouring import TieredIOUringTransferLayer
from tests.transfer.store_files import create_store_file


# TODO(independent-tiers phase 3): remove; the rewritten transfer tests create
# their store files explicitly instead of wrapping the layer constructor.
@pytest.fixture(autouse=True)
def tiered_layer_store_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Create the L2 store file before each tiered layer opens it.

    In production the server creates the store file before it builds the
    transfer layer; the layer only opens it. Tests that construct the layer
    directly get the same precondition here. ``L2IoEngine`` tests create their
    files explicitly so the missing-file contract stays covered.
    """
    original = TieredIOUringTransferLayer.__init__

    def init_with_store_file(self: Any, *args: Any, **kwargs: Any) -> None:
        if not kwargs.get("skip_l2", False):
            create_store_file(kwargs["path"], kwargs["l2_bytes"])
        original(self, *args, **kwargs)

    monkeypatch.setattr(TieredIOUringTransferLayer, "__init__", init_with_store_file)
