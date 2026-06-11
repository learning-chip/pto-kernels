#!/usr/bin/env python3
"""Correctness and bandwidth tests for direct A5 matmul/add kernels."""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch_npu  # noqa: F401

os.environ.setdefault("NPU_DEVICE", "npu:0")
DEVICE = os.environ["NPU_DEVICE"]

DTYPE = torch.float16
KW = dict(dtype=DTYPE, device=DEVICE)
RTOL = 1e-3
ATOL = 1e-2


def configure_torch_npu(*, simulator_safe: bool = False) -> None:
    torch.npu.config.allow_internal_format = False
    if simulator_safe:
        torch_npu.npu.set_compile_mode(jit_compile=False)


def _warmup_repeats(sim_mode: bool) -> tuple[int, int]:
    if sim_mode or os.environ.get("PTO_SIMULATOR") == "1":
        return 0, 1
    return int(os.environ.get("WARMUP", "10")), int(os.environ.get("REPEATS", "30"))


def make_batch(rounds: int, block_dim: int, tile_size: int = 128) -> int:
    return rounds * block_dim * tile_size


def benchmark_bytes(batch: int, tile_size: int = 128) -> int:
    return (batch * tile_size * 3 + tile_size * tile_size) * 2


def benchmark_bytes_c2v(batch: int, tile_size: int = 128) -> int:
    return batch * tile_size * (2 + 4 + 4) + tile_size * tile_size * 2


def assert_close(name: str, got: torch.Tensor, expected: torch.Tensor) -> None:
    try:
        torch.testing.assert_close(got, expected, rtol=RTOL, atol=ATOL)
    except AssertionError as exc:
        print(f"{name} FAILED: {exc}")
        sys.exit(1)


def test_matmul_add_c2v(kernels, *, block_dim: int, tile_size: int = 128) -> None:
    passed = 0
    for seed in range(3):
        for rounds in range(1, 11):
            batch = make_batch(rounds, block_dim, tile_size)
            torch.manual_seed(seed)
            A = torch.randn(batch, tile_size, **KW)
            B = torch.randn(tile_size, tile_size, **KW)
            D = torch.randn(batch, tile_size, dtype=torch.float32, device=DEVICE)
            C = torch.empty(batch, tile_size, dtype=torch.float32, device=DEVICE)
            kernels.matmul_add_c2v(A, B, C, D)
            torch.npu.synchronize()
            expected = A @ B + D
            assert_close(f"matmul_add_c2v seed={seed} rounds={rounds}", C, expected)
            passed += 1
    print(f"matmul_add_c2v correctness: {passed}/30 passed")


def test_add_matmul_v2c(kernels, *, block_dim: int, tile_size: int = 128) -> None:
    passed = 0
    for seed in range(3):
        for rounds in range(1, 11):
            batch = make_batch(rounds, block_dim, tile_size)
            torch.manual_seed(seed)
            A = torch.randn(batch, tile_size, **KW)
            B = torch.randn(batch, tile_size, **KW)
            D = torch.randn(tile_size, tile_size, **KW)
            C = torch.empty_like(A)
            kernels.add_matmul_v2c(A, B, C, D)
            torch.npu.synchronize()
            expected = ((A + B) @ D).to(DTYPE)
            assert_close(f"add_matmul_v2c seed={seed} rounds={rounds}", C, expected)
            passed += 1
    print(f"add_matmul_v2c correctness: {passed}/30 passed")


def time_repeated(fn, *args, repeats: int = 30) -> float:
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn(*args)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats * 1e3


def bench_matmul_add_c2v(
    kernels,
    *,
    block_dim: int,
    rounds_list: list[int],
    warmup: int,
    repeats: int,
    sim_mode: bool,
    tile_size: int = 128,
) -> list[tuple[int, int, float, float, int]]:
    print("=" * 72)
    print("matmul_add_c2v direct A5: C = A @ B + D")
    print("=" * 72)
    print(f"{'batch':>10}  {'rounds':>6}  {'dur_us':>10}  {'bw_GB/s':>10}")
    records = []
    for rounds in rounds_list:
        batch = make_batch(rounds, block_dim, tile_size)
        torch.manual_seed(0)
        A = torch.randn(batch, tile_size, **KW)
        B = torch.randn(tile_size, tile_size, **KW)
        D = torch.randn(batch, tile_size, dtype=torch.float32, device=DEVICE)
        C = torch.empty(batch, tile_size, dtype=torch.float32, device=DEVICE)
        for _ in range(warmup):
            kernels.matmul_add_c2v(A, B, C, D)
        torch.npu.synchronize()
        dur_us = time_repeated(kernels.matmul_add_c2v, A, B, C, D, repeats=repeats)
        nbytes = benchmark_bytes_c2v(batch, tile_size)
        bw = nbytes / dur_us * 1e-3 if dur_us > 0 else 0.0
        print(f"{batch:>10d}  {rounds:>6d}  {dur_us:>10.2f}  {bw:>10.1f}")
        if sim_mode:
            print(
                f"SIM_SUMMARY kernel=matmul_add_c2v rounds={rounds} batch={batch} "
                f"block_dim={block_dim} direct_bytes={nbytes} host_us={dur_us:.2f}",
                flush=True,
            )
        records.append((batch, rounds, dur_us, bw, nbytes))
    print()
    return records


def bench_add_matmul_v2c(
    kernels,
    *,
    block_dim: int,
    rounds_list: list[int],
    warmup: int,
    repeats: int,
    sim_mode: bool,
    tile_size: int = 128,
) -> list[tuple[int, int, float, float, int]]:
    print("=" * 72)
    print("add_matmul_v2c direct A5: C = (A + B) @ D")
    print("=" * 72)
    print(f"{'batch':>10}  {'rounds':>6}  {'dur_us':>10}  {'bw_GB/s':>10}")
    records = []
    for rounds in rounds_list:
        batch = make_batch(rounds, block_dim, tile_size)
        torch.manual_seed(0)
        A = torch.randn(batch, tile_size, **KW)
        B = torch.randn(batch, tile_size, **KW)
        D = torch.randn(tile_size, tile_size, **KW)
        C = torch.empty_like(A)
        for _ in range(warmup):
            kernels.add_matmul_v2c(A, B, C, D)
        torch.npu.synchronize()
        dur_us = time_repeated(kernels.add_matmul_v2c, A, B, C, D, repeats=repeats)
        nbytes = benchmark_bytes(batch, tile_size)
        bw = nbytes / dur_us * 1e-3 if dur_us > 0 else 0.0
        print(f"{batch:>10d}  {rounds:>6d}  {dur_us:>10.2f}  {bw:>10.1f}")
        if sim_mode:
            print(
                f"SIM_SUMMARY kernel=add_matmul_v2c rounds={rounds} batch={batch} "
                f"block_dim={block_dim} direct_bytes={nbytes} host_us={dur_us:.2f}",
                flush=True,
            )
        records.append((batch, rounds, dur_us, bw, nbytes))
    print()
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", help="Single-case simulator mode")
    parser.add_argument("--kernel", choices=("c2v", "v2c", "both"), default="both")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--block-dim", type=int, default=None)
    args = parser.parse_args()

    sim_mode = args.sim or os.environ.get("PTO_SIMULATOR") == "1"
    configure_torch_npu(simulator_safe=sim_mode)
    torch.npu.set_device(DEVICE)

    from jit_util import CvSyncKernels, resolve_block_dim, TILE_SIZE  # noqa: E402

    block_dim = args.block_dim
    if block_dim is None and sim_mode:
        block_dim = int(os.environ.get("PTO_BLOCK_DIM", "1"))
    elif block_dim is None:
        block_dim = resolve_block_dim()

    warmup, repeats = _warmup_repeats(sim_mode)
    rounds_list = [args.rounds] if sim_mode else [1, 2, 4, 8, 16, 32, 64]

    print(f"Using device: {DEVICE}")
    print(f"BLOCK_DIM (Cube cores): {block_dim}")
    if sim_mode:
        print(
            f"Simulator mode: kernel={args.kernel} rounds={args.rounds} "
            f"warmup={warmup} repeats={repeats}",
            flush=True,
        )

    kernels = CvSyncKernels(verbose=True, block_dim=block_dim)

    if not sim_mode:
        if args.kernel in ("c2v", "both"):
            test_matmul_add_c2v(kernels, block_dim=block_dim, tile_size=TILE_SIZE)
        if args.kernel in ("v2c", "both"):
            test_add_matmul_v2c(kernels, block_dim=block_dim, tile_size=TILE_SIZE)

    c2v: list[tuple[int, int, float, float, int]] = []
    v2c: list[tuple[int, int, float, float, int]] = []
    if args.kernel in ("c2v", "both"):
        c2v = bench_matmul_add_c2v(
            kernels,
            block_dim=block_dim,
            rounds_list=rounds_list,
            warmup=warmup,
            repeats=repeats,
            sim_mode=sim_mode,
            tile_size=TILE_SIZE,
        )
    if args.kernel in ("v2c", "both"):
        v2c = bench_add_matmul_v2c(
            kernels,
            block_dim=block_dim,
            rounds_list=rounds_list,
            warmup=warmup,
            repeats=repeats,
            sim_mode=sim_mode,
            tile_size=TILE_SIZE,
        )

    if c2v:
        print(f"Peak matmul_add_c2v bandwidth (host timing): {max(r[3] for r in c2v):.1f} GB/s")
    if v2c:
        print(f"Peak add_matmul_v2c bandwidth (host timing): {max(r[3] for r in v2c):.1f} GB/s")


if __name__ == "__main__":
    main()
