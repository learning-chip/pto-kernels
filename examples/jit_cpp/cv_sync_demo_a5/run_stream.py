#!/usr/bin/env python3
"""Smoke and bandwidth sweep for direct A5 Cube/Vector stream kernels."""

from __future__ import annotations

import argparse
import os

import torch
import torch_npu  # noqa: F401

os.environ.setdefault("NPU_DEVICE", "npu:0")
DEVICE = os.environ["NPU_DEVICE"]

DTYPE = torch.float16
ITERS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def configure_torch_npu(*, simulator_safe: bool = False) -> None:
    torch.npu.config.allow_internal_format = False
    if simulator_safe:
        torch_npu.npu.set_compile_mode(jit_compile=False)


def _warmup_repeats(sim_mode: bool) -> tuple[int, int]:
    if sim_mode or os.environ.get("PTO_SIMULATOR") == "1":
        return 0, 1
    return int(os.environ.get("WARMUP", "5")), int(os.environ.get("REPEATS", "20"))


def direct_bytes(block_dim: int, num_iters: int, element_size: int) -> int:
    tile_size = 128
    return block_dim * tile_size * tile_size * element_size * num_iters


def old_equiv_bytes(block_dim: int, num_iters: int) -> int:
    tile_size = 128
    return 2 * block_dim * tile_size * tile_size * 2 * num_iters


def time_kernel(fn, *args, repeats: int) -> float:
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn(*args)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats * 1e3


def run_c2v(
    kernels,
    *,
    block_dim: int,
    iters: list[int],
    warmup: int,
    repeats: int,
    sim_mode: bool,
) -> list[tuple[int, float, float, float, int]]:
    print("=" * 72)
    print("stream_c2v direct A5 (Cube L0C -> Vec UB)")
    print("=" * 72)
    print(f"{'num_iters':>10}  {'host_us':>10}  {'direct_GB/s':>12}  {'old_equiv_GB/s':>15}")

    tile_size = 128
    wave_rows = block_dim * tile_size
    kw = dict(dtype=DTYPE, device=DEVICE)
    A = torch.randn(wave_rows, tile_size, **kw)
    B = torch.randn(tile_size, tile_size, **kw)

    if not sim_mode:
        kernels.stream_c2v(A, B, 4)
        torch.npu.synchronize()
        print("  smoke (num_iters=4): OK")
    else:
        print("  smoke (num_iters=4): skipped in simulator mode", flush=True)

    records = []
    for num_iters in iters:
        for _ in range(warmup):
            kernels.stream_c2v(A, B, num_iters)
        torch.npu.synchronize()
        dur_us = time_kernel(lambda: kernels.stream_c2v(A, B, num_iters), repeats=repeats)
        nbytes = direct_bytes(block_dim, num_iters, 4)
        direct = nbytes / dur_us * 1e-3 if dur_us > 0 else 0.0
        equiv = old_equiv_bytes(block_dim, num_iters) / dur_us * 1e-3 if dur_us > 0 else 0.0
        print(f"{num_iters:>10d}  {dur_us:>10.2f}  {direct:>12.1f}  {equiv:>15.1f}")
        if sim_mode:
            print(
                f"SIM_SUMMARY kernel=stream_c2v num_iters={num_iters} "
                f"block_dim={block_dim} direct_bytes={nbytes} host_us={dur_us:.2f}",
                flush=True,
            )
        records.append((num_iters, dur_us, direct, equiv, nbytes))
    print()
    return records


def run_v2c(
    kernels,
    *,
    block_dim: int,
    iters: list[int],
    warmup: int,
    repeats: int,
    sim_mode: bool,
) -> list[tuple[int, float, float, float, int]]:
    print("=" * 72)
    print("stream_v2c direct A5 (Vec UB -> Cube L1)")
    print("=" * 72)
    print(f"{'num_iters':>10}  {'host_us':>10}  {'direct_GB/s':>12}  {'old_equiv_GB/s':>15}")

    tile_size = 128
    wave_rows = block_dim * tile_size
    kw = dict(dtype=DTYPE, device=DEVICE)
    records = []
    if not sim_mode:
        A_smoke = torch.randn(4 * wave_rows, tile_size, **kw)
        D_smoke = torch.randn(4 * wave_rows, tile_size, **kw)
        kernels.stream_v2c(A_smoke, D_smoke, 4)
        torch.npu.synchronize()
        print("  smoke (num_iters=4): OK")
    else:
        print("  smoke (num_iters=4): skipped in simulator mode", flush=True)

    for num_iters in iters:
        total_rows = num_iters * wave_rows
        A = torch.randn(total_rows, tile_size, **kw)
        D = torch.randn(total_rows, tile_size, **kw)
        for _ in range(warmup):
            kernels.stream_v2c(A, D, num_iters)
        torch.npu.synchronize()
        dur_us = time_kernel(lambda: kernels.stream_v2c(A, D, num_iters), repeats=repeats)
        nbytes = direct_bytes(block_dim, num_iters, 2)
        direct = nbytes / dur_us * 1e-3 if dur_us > 0 else 0.0
        equiv = old_equiv_bytes(block_dim, num_iters) / dur_us * 1e-3 if dur_us > 0 else 0.0
        print(f"{num_iters:>10d}  {dur_us:>10.2f}  {direct:>12.1f}  {equiv:>15.1f}")
        if sim_mode:
            print(
                f"SIM_SUMMARY kernel=stream_v2c num_iters={num_iters} "
                f"block_dim={block_dim} direct_bytes={nbytes} host_us={dur_us:.2f}",
                flush=True,
            )
        records.append((num_iters, dur_us, direct, equiv, nbytes))
    print()
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", help="Single-case simulator mode")
    parser.add_argument("--kernel", choices=("c2v", "v2c", "both"), default="both")
    parser.add_argument("--num-iters", type=int, default=4)
    parser.add_argument("--block-dim", type=int, default=None)
    args = parser.parse_args()

    sim_mode = args.sim or os.environ.get("PTO_SIMULATOR") == "1"
    configure_torch_npu(simulator_safe=sim_mode)
    torch.npu.set_device(DEVICE)

    from jit_util import CvSyncKernels, resolve_block_dim  # noqa: E402

    block_dim = args.block_dim
    if block_dim is None and sim_mode:
        block_dim = int(os.environ.get("PTO_BLOCK_DIM", "1"))
    elif block_dim is None:
        block_dim = resolve_block_dim()

    warmup, repeats = _warmup_repeats(sim_mode)
    iters = [args.num_iters] if sim_mode else ITERS

    print(f"Using device: {DEVICE}")
    print(f"BLOCK_DIM (Cube cores): {block_dim}")
    if sim_mode:
        print(f"Simulator mode: kernel={args.kernel} num_iters={args.num_iters} warmup={warmup} repeats={repeats}")

    kernels = CvSyncKernels(verbose=True, block_dim=block_dim)
    c2v: list[tuple[int, float, float, float, int]] = []
    v2c: list[tuple[int, float, float, float, int]] = []

    if args.kernel in ("c2v", "both"):
        c2v = run_c2v(kernels, block_dim=block_dim, iters=iters, warmup=warmup, repeats=repeats, sim_mode=sim_mode)
    if args.kernel in ("v2c", "both"):
        v2c = run_v2c(kernels, block_dim=block_dim, iters=iters, warmup=warmup, repeats=repeats, sim_mode=sim_mode)

    if c2v:
        print(f"Peak stream_c2v direct (host timing): {max(r[2] for r in c2v):.1f} GB/s")
    if v2c:
        print(f"Peak stream_v2c direct (host timing): {max(r[2] for r in v2c):.1f} GB/s")


if __name__ == "__main__":
    main()
