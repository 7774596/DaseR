# SPDX-License-Identifier: Apache-2.0
"""Run the DaseR prefill gateway: ``python -m daser.gateway``."""

import argparse

import uvicorn

from daser.gateway.app import (
    ADMIT_MODES,
    DISPATCH_MODES,
    PRIORITY_MODES,
    GatewayConfig,
    build_gateway_app,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse gateway command-line arguments.

    Args:
        argv: Optional argument list; defaults to ``sys.argv``.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-url", required=True)
    parser.add_argument("--socket-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--split",
        action="store_true",
        help="Plan bidirectional prefill; without it requests pass through.",
    )
    parser.add_argument("--compute-tokens-per-s", type=float, default=30000.0)
    parser.add_argument("--load-bytes-per-s", type=float, default=15e9)
    parser.add_argument("--min-compute-tokens", type=int, default=4096)
    parser.add_argument(
        "--priority",
        choices=PRIORITY_MODES,
        default="none",
        help=(
            "Set vLLM request priority: 'length' by prompt tokens, 'residual' "
            "by tokens left after DaseR's reusable prefix. Needs vLLM "
            "--scheduling-policy priority."
        ),
    )
    parser.add_argument(
        "--admit",
        choices=ADMIT_MODES,
        default="none",
        help=(
            "Gate large jobs: 'length' by prompt tokens, 'residual' by tokens "
            "left after DaseR's reusable prefix."
        ),
    )
    parser.add_argument("--admit-tokens", type=int, default=8192)
    parser.add_argument("--admit-slots", type=int, default=1)
    parser.add_argument(
        "--dispatch",
        choices=DISPATCH_MODES,
        default="none",
        help=(
            "Hold completions in the gateway and release them to vLLM in "
            "'fifo', 'length' (prompt tokens) or 'residual' (tokens left after "
            "DaseR's reusable prefix) order; 'residual-fresh' re-estimates the "
            "best-ranked held requests at each release; 'cost-fresh' does the "
            "same but ranks by estimated prefill FLOPs."
        ),
    )
    parser.add_argument("--dispatch-slots", type=int, default=4)
    parser.add_argument("--dispatch-refresh", type=int, default=8)
    parser.add_argument(
        "--dispatch-attn-tokens",
        type=float,
        default=13500.0,
        help="Context length where a token's attention FLOPs equal its linear "
        "FLOPs (cost-fresh); about 2*params / (4*layers*q_heads*head_dim).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Start the gateway with uvicorn.

    Args:
        argv: Optional argument list; defaults to ``sys.argv``.
    """
    args = parse_args(argv)
    cfg = GatewayConfig(
        vllm_url=args.vllm_url,
        socket_path=args.socket_path,
        split=args.split,
        compute_tokens_per_s=args.compute_tokens_per_s,
        load_bytes_per_s=args.load_bytes_per_s,
        min_compute_tokens=args.min_compute_tokens,
        priority=args.priority,
        admit=args.admit,
        admit_tokens=args.admit_tokens,
        admit_slots=args.admit_slots,
        dispatch=args.dispatch,
        dispatch_slots=args.dispatch_slots,
        dispatch_refresh=args.dispatch_refresh,
        dispatch_attn_tokens=args.dispatch_attn_tokens,
    )
    uvicorn.run(
        build_gateway_app(cfg), host=args.host, port=args.port, log_level="warning"
    )


if __name__ == "__main__":
    main()
