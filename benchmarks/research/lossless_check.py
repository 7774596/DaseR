# SPDX-License-Identifier: Apache-2.0
"""Check that a compressed store decodes exactly what a raw store decodes.

Compares the ``probe.jsonl`` files that ``cache_state_probe`` wrote for two
arms run on the same trace and model (for example ``daser-raw`` and
``daser-compressed``). Requests are greedy with ``ignore_eos``, so a lossless
store must make the candidate's ``external`` pass (KV restored from the
connector) produce the same text as the reference's ``external`` pass: both
restore the same KV bytes into the same single-request batch shape. The
``cold`` and ``gpu`` passes never read the store and serve as controls for
nondeterminism unrelated to the codec.

A row is only compared when, in both arms, its ``external`` pass was served
mostly by the connector (over half the prompt from external hits); other rows
are reported as invalid instead of counted as agreement.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

PASSES = ("cold", "gpu", "external")


def load_probe(path: Path) -> list[dict[str, Any]]:
    """Read the per-pass records of one probe run.

    Args:
        path (Path): Arm directory holding ``probe.jsonl``, or the file.

    Returns:
        list[dict[str, Any]]: Records in file order.
    """
    if path.is_dir():
        path = path / "probe.jsonl"
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def served_externally(row: dict[str, Any]) -> bool:
    """Return whether the connector served over half of a pass's prompt.

    Args:
        row (dict[str, Any]): One probe record.

    Returns:
        bool: True for an error-free pass with external hits above half the
        prompt length.
    """
    return (
        row.get("error") is None
        and 2 * row.get("external_hit_tokens", 0.0) > row["input_length"]
    )


def compare_probes(
    reference: Sequence[dict[str, Any]], candidate: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Compare the generated texts of two probe runs pass by pass.

    Args:
        reference (Sequence[dict[str, Any]]): Records of the reference arm.
        candidate (Sequence[dict[str, Any]]): Records of the arm under test.

    Returns:
        dict[str, Any]: Per pass, the number of request indices present in
        both runs (``paired``), how many were comparable (``compared``; for
        ``external`` only rows served by the connector in both arms, for the
        other passes error-free rows), how many decoded identical text
        (``identical``), the indices that differed (``mismatched``) and the
        indices skipped as not comparable (``invalid``). The top-level
        ``lossless`` is True when every comparable ``external`` row is
        identical and at least one was compared.

    Thread-safety:
        Pure function.
    """
    by_key = {(r["index"], r["pass"]): r for r in candidate}
    result: dict[str, Any] = {}
    for name in PASSES:
        paired = compared = identical = 0
        mismatched: list[int] = []
        invalid: list[int] = []
        for ref in (r for r in reference if r["pass"] == name):
            other = by_key.get((ref["index"], name))
            if other is None:
                continue
            paired += 1
            if name == "external":
                usable = served_externally(ref) and served_externally(other)
            else:
                usable = ref.get("error") is None and other.get("error") is None
            if not usable:
                invalid.append(ref["index"])
                continue
            compared += 1
            if ref["text_sha1"] == other["text_sha1"]:
                identical += 1
            else:
                mismatched.append(ref["index"])
        result[name] = {
            "paired": paired,
            "compared": compared,
            "identical": identical,
            "mismatched": mismatched,
            "invalid": invalid,
        }
    external = result["external"]
    result["lossless"] = external["compared"] > 0 and not external["mismatched"]
    return result


def main(argv: list[str] | None = None) -> None:
    """Compare two probe runs and print (and optionally write) the result.

    Args:
        argv (list[str] | None): Arguments without the program name; ``None``
            reads ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = compare_probes(load_probe(args.reference), load_probe(args.candidate))
    text = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
