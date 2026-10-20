# SPDX-License-Identifier: Apache-2.0
"""Per-(layer, kv, head) AF1 lookup tables for the HBM-resident KV store.

The tables are calibrated offline from captured KV pages and saved as one
``.npz`` with ``table`` uint8 ``[L, 2, H, 22]`` (high bytes, most frequent
first). :func:`load_tables` derives the device-side forms used by the kernels.

CLI::

    python -m daser.compression.attn_tables DUMP.npy OUT.npz

where ``DUMP.npy`` is uint16 ``[S, L, 2, 128, H, 128]`` BF16 bit patterns.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import torch

from daser.compression import attn_page as ap
from daser.logging import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class DeviceTables:
    """Kernel lookup tables for every layer.

    Attributes:
        rank_of: uint8 ``[L, 2, H, 256]`` table rank of each high byte, 255 if
            absent (encoder).
        plut: int32 ``[L, 2, H, 2]`` primary ``prmt`` words (decoder).
        slut: uint8 ``[L, 2, H, 16]`` extended table (decoder).
    """

    rank_of: torch.Tensor
    plut: torch.Tensor
    slut: torch.Tensor


def calibrate(dump: NDArray[np.uint16]) -> NDArray[np.uint8]:
    """Calibrate one table per (layer, kv, head) from captured pages.

    Args:
        dump: uint16 ``[S, L, 2, 128, H, 128]`` BF16 bit patterns.

    Returns:
        uint8 ``[L, 2, H, 22]`` tables.

    Async/thread-safety:
        Pure function.
    """
    _, n_layers, kv, _, heads, _ = dump.shape
    out = np.empty((n_layers, kv, heads, ap.TABLE_ENTRIES), dtype=np.uint8)
    for layer in range(n_layers):
        block = np.asarray(dump[:, layer])
        for k in range(kv):
            for h in range(heads):
                out[layer, k, h] = ap.calibrate_table(
                    (block[:, k, :, h, :] >> 8).astype(np.uint8)
                )
    return out


def load_tables(path: str, device: torch.device) -> DeviceTables:
    """Load a calibrated ``.npz`` and build the kernel tables on ``device``.

    Args:
        path: File written by the CLI.
        device: Target CUDA device.

    Returns:
        Device tables.

    Async/thread-safety:
        Blocking file read and host-to-device copies.
    """
    return device_tables(np.load(path)["table"], device)


def device_tables(table: NDArray[np.uint8], device: torch.device) -> DeviceTables:
    """Build the kernel tables from calibrated tables.

    Args:
        table: uint8 ``[L, 2, H, 22]`` from :func:`calibrate`.
        device: Target CUDA device.

    Returns:
        Device tables.

    Async/thread-safety:
        Blocking host-to-device copies.
    """
    n_layers, kv, heads, entries = table.shape
    if entries != ap.TABLE_ENTRIES:
        raise ValueError(f"expected {ap.TABLE_ENTRIES} table entries, got {entries}")
    rank_of = np.full((n_layers, kv, heads, 256), 255, dtype=np.uint8)
    plut = np.empty((n_layers, kv, heads, 2), dtype=np.uint32)
    slut = np.empty((n_layers, kv, heads, 16), dtype=np.uint8)
    for idx in np.ndindex(n_layers, kv, heads):
        t = table[idx]
        rank_of[idx][t] = np.arange(ap.TABLE_ENTRIES, dtype=np.uint8)
        plut[idx] = ap.primary_lut(t)
        slut[idx] = np.frombuffer(ap.secondary_lut(t), dtype=np.uint8)
    return DeviceTables(
        rank_of=torch.from_numpy(rank_of).to(device),
        plut=torch.from_numpy(plut.view(np.int32)).to(device),
        slut=torch.from_numpy(slut).to(device),
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dump", help="uint16 [S, L, 2, 128, H, 128] .npy")
    parser.add_argument("out", help="output .npz")
    args = parser.parse_args()
    dump = np.load(args.dump, mmap_mode="r")
    table = calibrate(dump)
    np.savez(args.out, table=table)
    logger.info("wrote %s tables %s", args.out, table.shape)


if __name__ == "__main__":
    _main()


__all__ = ["DeviceTables", "calibrate", "device_tables", "load_tables"]
