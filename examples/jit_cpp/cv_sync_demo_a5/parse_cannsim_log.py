#!/usr/bin/env python3
"""Parse cannsim.log and compute predicted stream bandwidth."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CYCLES_PER_US = 1650
HARDWARE_RE = re.compile(
    r"\[Hardware\] parallel simulation finish\. sim time: ([0-9.]+)s, cycle: (\d+),"
)
SIM_SUMMARY_RE = re.compile(
    r"SIM_SUMMARY kernel=(\S+) num_iters=(\d+) block_dim=(\d+) direct_bytes=(\d+)"
)


def max_cycle(log_text: str) -> int:
    cycles = [int(m.group(2)) for m in HARDWARE_RE.finditer(log_text)]
    if not cycles:
        raise ValueError("no [Hardware] cycle lines found in log")
    return max(cycles)


def parse_summary(log_text: str) -> dict[str, str | int]:
    matches = list(SIM_SUMMARY_RE.finditer(log_text))
    if not matches:
        raise ValueError("no SIM_SUMMARY line found in log")
    m = matches[-1]
    return {
        "kernel": m.group(1),
        "num_iters": int(m.group(2)),
        "block_dim": int(m.group(3)),
        "direct_bytes": int(m.group(4)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=Path)
    parser.add_argument("--cycles-per-us", type=float, default=CYCLES_PER_US)
    args = parser.parse_args()

    text = args.log_file.read_text(errors="replace")
    summary = parse_summary(text)
    cycle = max_cycle(text)
    predicted_us = cycle / args.cycles_per_us
    direct_bytes = int(summary["direct_bytes"])
    predicted_gbps = direct_bytes / predicted_us * 1e-3 if predicted_us > 0 else 0.0

    print(f"kernel={summary['kernel']}")
    print(f"num_iters={summary['num_iters']}")
    print(f"block_dim={summary['block_dim']}")
    print(f"direct_bytes={direct_bytes}")
    print(f"max_cycle={cycle}")
    print(f"predicted_us={predicted_us:.3f}")
    print(f"predicted_GB_s={predicted_gbps:.1f}")


if __name__ == "__main__":
    main()
