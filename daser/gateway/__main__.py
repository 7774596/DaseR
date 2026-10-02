# SPDX-License-Identifier: Apache-2.0
"""Run the DaseR prefill gateway: ``python -m daser.gateway``."""

import argparse

import uvicorn

from daser.gateway.app import GatewayConfig, build_gateway_app


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
    )
    uvicorn.run(
        build_gateway_app(cfg), host=args.host, port=args.port, log_level="warning"
    )


if __name__ == "__main__":
    main()
