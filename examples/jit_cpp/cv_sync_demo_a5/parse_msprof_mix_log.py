#!/usr/bin/env python3
"""Extract mix-kernel simulator ticks from msprof stdout logs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

CYCLES_PER_US = 1650
MIX_START = re.compile(
    r"\[info\] \[(\d+)\] \[block_start\]\s+: AIC_MIX, task_id=\d+, core_id=\d+, main_blk_id=(\d+)"
)
MIX_END_AIC = re.compile(
    r"\[info\] \[(\d+)\] \[block_end\]\s+: AIC, task_id=\d+, core_id=\d+, block_id=(\d+)"
)
SIM_SUMMARY_STREAM = re.compile(
    r"SIM_SUMMARY kernel=(\S+) num_iters=(\d+) block_dim=(\d+) direct_bytes=(\d+)"
)
SIM_SUMMARY_MATMUL = re.compile(
    r"SIM_SUMMARY kernel=(\S+) rounds=(\d+) batch=(\d+) block_dim=(\d+) direct_bytes=(\d+)"
)


def _parse_summary(text: str) -> tuple[str, int, int, int]:
    matmul_matches = list(SIM_SUMMARY_MATMUL.finditer(text))
    if matmul_matches:
        m = matmul_matches[-1]
        return m.group(1), int(m.group(2)), int(m.group(4)), int(m.group(5))
    stream_matches = list(SIM_SUMMARY_STREAM.finditer(text))
    if stream_matches:
        m = stream_matches[-1]
        return m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    raise ValueError("no SIM_SUMMARY line found in log")


def parse_mix_kernel(text: str) -> dict[str, int | float | str]:
    kernel, size_param, block_dim, direct_bytes = _parse_summary(text)

    # Use the last AIC_MIX launch (stream timed kernel).
    starts = [m for m in MIX_START.finditer(text) if m.group(2) == "0"]
    ends = [m for m in MIX_END_AIC.finditer(text) if m.group(2) == "0"]
    if not starts or not ends:
        raise ValueError("could not find AIC_MIX block_start/block_end pair")
    start_tick = int(starts[-1].group(1))
    end_candidates = [int(m.group(1)) for m in ends if int(m.group(1)) > start_tick]
    if not end_candidates:
        raise ValueError("could not find AIC block_end after stream launch")
    end_tick = max(end_candidates)
    cycles = end_tick - start_tick
    predicted_us = cycles / CYCLES_PER_US
    predicted_gbps = direct_bytes / predicted_us * 1e-3 if predicted_us > 0 else 0.0
    return {
        "kernel": kernel,
        "rounds_or_iters": size_param,
        "block_dim": block_dim,
        "direct_bytes": direct_bytes,
        "start_tick": start_tick,
        "end_tick": end_tick,
        "predicted_cycles": cycles,
        "predicted_us": predicted_us,
        "predicted_GB_s": predicted_gbps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=Path)
    args = parser.parse_args()
    result = parse_mix_kernel(args.log_file.read_text(errors="replace"))
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}={value:.3f}" if key.endswith("_us") else f"{key}={value:.1f}")
        else:
            print(f"{key}={value}")


if __name__ == "__main__":
    main()
