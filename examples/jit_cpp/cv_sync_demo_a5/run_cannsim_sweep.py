#!/usr/bin/env python3
"""Run cannsim stream sweeps and collect predicted bandwidth."""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
CYCLES_PER_US = 1650
WALL_CAP_S = 300
ITERS_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
HARDWARE_RE = re.compile(
    r"\[Hardware\] parallel simulation finish\. sim time: ([0-9.]+)s, cycle: (\d+),"
)
SIM_SUMMARY_RE = re.compile(
    r"SIM_SUMMARY kernel=(\S+) num_iters=(\d+) block_dim=(\d+) direct_bytes=(\d+)"
)


@dataclass
class CaseResult:
    kernel: str
    num_iters: int
    block_dim: int
    direct_bytes: int
    max_cycle: int
    predicted_us: float
    predicted_gbps: float
    wall_s: float
    omp: str
    log_file: str


def find_log(out_dir: Path) -> Path:
    logs = sorted(out_dir.glob("**/cannsim.log"))
    if not logs:
        raise FileNotFoundError(f"no cannsim.log under {out_dir}")
    return logs[-1]


def parse_case(log_file: Path, wall_s: float, omp: str) -> CaseResult:
    text = log_file.read_text(errors="replace")
    summary_matches = list(SIM_SUMMARY_RE.finditer(text))
    if not summary_matches:
        raise ValueError(f"no SIM_SUMMARY in {log_file}")
    summary = summary_matches[-1]
    cycles = [int(m.group(2)) for m in HARDWARE_RE.finditer(text)]
    if not cycles:
        raise ValueError(f"no hardware cycles in {log_file}")
    max_cycle = max(cycles)
    direct_bytes = int(summary.group(4))
    predicted_us = max_cycle / CYCLES_PER_US
    predicted_gbps = direct_bytes / predicted_us * 1e-3 if predicted_us > 0 else 0.0
    return CaseResult(
        kernel=summary.group(1),
        num_iters=int(summary.group(2)),
        block_dim=int(summary.group(3)),
        direct_bytes=direct_bytes,
        max_cycle=max_cycle,
        predicted_us=predicted_us,
        predicted_gbps=predicted_gbps,
        wall_s=wall_s,
        omp=omp,
        log_file=str(log_file),
    )


def run_case(
    *,
    kernel: str,
    num_iters: int,
    omp: str,
    out_dir: Path,
    gen_report: bool = False,
    timeout_s: int = WALL_CAP_S,
) -> tuple[int, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "PTO_SIMULATOR": "1",
        "NPU_DEVICE": "npu:0",
        "PTO_PROCESS_TIMEOUT_S": str(timeout_s),
        "PTO_BLOCK_DIM": "1",
    }
    if omp != "default":
        env["OMP_NUM_THREADS"] = omp
    cmd = [
        "bash",
        str(HERE / "run_sim.sh"),
        "cannsim",
        "--output",
        str(out_dir),
        "--kernel",
        kernel,
        "--num-iters",
        str(num_iters),
        "--block-dim",
        "1",
    ]
    if gen_report:
        cmd.insert(2, "--gen-report")
    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=HERE,
        env={**dict(subprocess.os.environ), **env},
        capture_output=True,
        text=True,
    )
    wall_s = time.time() - start
    return proc.returncode, wall_s


def main() -> None:
    results_dir = HERE / "outputs" / "sweep"
    results_dir.mkdir(parents=True, exist_ok=True)

    omp_scores: dict[str, float] = {}
    for omp in ("1", "16", "default"):
        total = 0.0
        for kernel in ("c2v", "v2c"):
            out = results_dir / f"omp_{omp}_{kernel}_it4"
            rc, wall = run_case(kernel=kernel, num_iters=4, omp=omp, out_dir=out)
            total += wall
            print(f"OMP={omp} kernel={kernel} rc={rc} wall={wall:.1f}s")
        omp_scores[omp] = total / 2.0
    best_omp = min(omp_scores, key=omp_scores.get)
    print(f"Best OMP: {best_omp} avg wall={omp_scores[best_omp]:.1f}s")

    all_results: list[CaseResult] = []
    max_iters: dict[str, int] = {}
    for kernel in ("c2v", "v2c"):
        last_ok = None
        for num_iters in ITERS_LIST:
            out = results_dir / f"{kernel}_it{num_iters}"
            rc, wall = run_case(kernel=kernel, num_iters=num_iters, omp=best_omp, out_dir=out)
            print(f"kernel={kernel} num_iters={num_iters} rc={rc} wall={wall:.1f}s")
            if wall >= WALL_CAP_S or rc != 0:
                try:
                    log = find_log(out)
                    if "SIM_SUMMARY" in log.read_text(errors="replace"):
                        case = parse_case(log, wall, best_omp)
                        all_results.append(case)
                        last_ok = num_iters
                except (FileNotFoundError, ValueError):
                    pass
                break
            log = find_log(out)
            case = parse_case(log, wall, best_omp)
            all_results.append(case)
            last_ok = num_iters
        max_iters[kernel] = last_ok or 0

    peaks = {}
    for kernel in ("c2v", "v2c"):
        kernel_rows = [r for r in all_results if r.kernel == f"stream_{kernel}"]
        if kernel_rows:
            peak = max(kernel_rows, key=lambda r: r.predicted_gbps)
            peaks[kernel] = peak

    final_runs = {}
    for kernel in ("c2v", "v2c"):
        n = max_iters.get(kernel, 0)
        if n <= 0:
            continue
        out = HERE / "outputs" / f"cannsim_final_{kernel}"
        rc, wall = run_case(
            kernel=kernel,
            num_iters=n,
            omp=best_omp,
            out_dir=out,
            gen_report=True,
            timeout_s=1800,
        )
        export = sorted(out.glob("cannsim_*"))[-1]
        report_out = HERE / "outputs" / f"report_{kernel}"
        report_out.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["cannsim", "report", "-e", str(export), "-o", str(report_out), "-n", "all"],
            cwd=HERE,
            check=False,
        )
        final_runs[kernel] = {
            "num_iters": n,
            "export_dir": str(export),
            "report_dir": str(report_out),
            "rc": rc,
            "wall_s": wall,
        }

    payload = {
        "omp_scores": omp_scores,
        "best_omp": best_omp,
        "max_iters_under_cap": max_iters,
        "results": [asdict(r) for r in all_results],
        "peaks": {k: asdict(v) for k, v in peaks.items()},
        "final_runs": final_runs,
    }
    out_json = results_dir / "sweep_results.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
