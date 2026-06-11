# CANNSIM / CA-Model Bandwidth Simulation

Simulator-mode bandwidth study for Cube/Vector sync kernels in this demo:
`stream_c2v`, `stream_v2c`, `matmul_add_c2v`, and `add_matmul_v2c`.
No real A5 NPU is required; runs use the Ascend950 CA model (`Ascend950PR_9599`).

## Prerequisites

```bash
source /usr/local/Ascend/cann-9.0.0/bin/setenv.bash
python3 -c "import torch, torch_npu; print('torch_npu OK')"
which cannsim msprof
```

Optional: activate your conda env if you use one. Set `NPU_DEVICE=npu:0` (default).

## One-time build

Compile the kernel `.so` once so per-case wall time excludes bisheng compile:

```bash
cd /workdir/pto-kernels-fork/examples/jit_cpp/cv_sync_demo_a5
python3 common_build.py
```

## Simulator wrappers

| Script | Role |
| --- | --- |
| `run_sim_entry.sh` | Runs `run_stream.py` under `run_with_timeout.sh`. **Do not** `source setenv.bash` here (breaks cannsim `LD_LIBRARY_PATH` injection). |
| `run_sim.sh` | Sources CANN env, sets `PTO_SIMULATOR=1`, prepends simulator libs, invokes `cannsim record`. |

Single-case cannsim smoke:

```bash
export PTO_SIMULATOR=1 PTO_BLOCK_DIM=1 PTO_PROCESS_TIMEOUT_S=300
./run_sim.sh cannsim --output outputs/cannsim_smoke \
  --kernel c2v --num-iters 1 --block-dim 1
```

Equivalent direct invocation:

```bash
cannsim record -s Ascend950 -o outputs/cannsim_c2v_it1 \
  ./run_sim_entry.sh -u "--sim --kernel c2v --num-iters 1 --block-dim 1"
```

## OpenMP tuning

Plan target: compare host wall time at `num_iters=4` under `cannsim record`.
**Observed blocker:** `cannsim record` on AIC_MIX stream kernels does not finish (see [Known issues](#known-issues)).
OMP tuning was therefore run with **`msprof op simulator`** at `num_iters=1`, `block_dim=1`.

```bash
cd /workdir/pto-kernels-fork/examples/jit_cpp/cv_sync_demo_a5
source /usr/local/Ascend/cann-9.0.0/bin/setenv.bash
export PTO_SIMULATOR=1 PTO_BLOCK_DIM=1

# Repeat for OMP_NUM_THREADS=1, OMP_NUM_THREADS=16, and unset (default)
OMP_NUM_THREADS=16 msprof op simulator --soc-version=Ascend950PR_9599 --timeout=900 \
  --output=outputs/omp_sweep/msprof_omp16_c2v \
  python3 run_stream.py --sim --kernel c2v --num-iters 1 --block-dim 1 \
  2>&1 | tee outputs/omp_sweep/omp16_c2v.log
```

### OMP wall-time results (msprof, num_iters=1)

| OMP | stream_c2v wall (s) | stream_v2c wall (s) | Average (s) |
| ---: | ---: | ---: | ---: |
| 1 | 764 | 684 | 724.0 |
| **16** | **612** | **625** | **618.5** |
| default | 646 | 762 | 704.0 |

**Chosen config:** `OMP_NUM_THREADS=16` (lowest average host wall time).

Predicted simulator cycles/bandwidth are identical across OMP settings; OMP only affects host-side parallelism.

## Per-kernel cannsim sweep (300 s cap)

```bash
export PTO_SIMULATOR=1 PTO_BLOCK_DIM=1 PTO_PROCESS_TIMEOUT_S=300 OMP_NUM_THREADS=16
for k in c2v v2c; do
  for n in 1 2 4 8 16 32 64 128 256 512 1024; do
    ./run_sim.sh cannsim --output outputs/sweep/${k}_it${n} \
      --kernel ${k} --num-iters ${n} --block-dim 1 || break
  done
done
```

### cannsim sweep outcome

| Kernel | num_iters tried | Wall (s) | SIM_SUMMARY | Kernel Hardware cycle line |
| --- | ---: | ---: | --- | --- |
| stream_c2v | 1, 2, 4 | ~306 each | no | no (max init cycle = 420) |
| stream_v2c | 1, 2, 4 | ~306 each | no | no (max init cycle = 420) |

**Largest num_iters under 300 s (cannsim):** none — all cases time out in `sendStarsSQE` without completing the mix kernel.

### msprof fallback (largest completed case)

Even `num_iters=1` exceeds the 300 s wall cap under msprof (~612–625 s with OMP=16), but it is the only size that completes:

```bash
OMP_NUM_THREADS=16 msprof op simulator --soc-version=Ascend950PR_9599 --timeout=900 \
  --output=outputs/msprof_c2v \
  python3 run_stream.py --sim --kernel c2v --num-iters 1 --block-dim 1 \
  2>&1 | tee outputs/msprof_c2v_stdout.log
```

Parse predicted bandwidth from mix-kernel ticks:

```bash
python3 parse_msprof_mix_log.py outputs/omp_sweep/omp16_c2v.log
python3 parse_msprof_mix_log.py outputs/omp_sweep/omp16_v2c.log
```

## Cycle → bandwidth formula

A5 clock used by CANN simulator docs: **1650 cycles/μs**.

Direct byte counts (same as real-device script):

- **stream_c2v:** `direct_bytes = block_dim × 128 × 128 × 4 × num_iters`
- **stream_v2c:** `direct_bytes = block_dim × 128 × 128 × 2 × num_iters`

From cannsim.log (when kernel completes):

```
predicted_us = max_cycle / 1650
predicted_GB/s = direct_bytes / predicted_us × 1e-3
```

From msprof stdout (fallback used here):

```
predicted_cycles = AIC block_end_tick − AIC_MIX block_start_tick
predicted_us = predicted_cycles / 1650
predicted_GB/s = direct_bytes / predicted_us × 1e-3
```

**Example (stream_c2v, block_dim=1, num_iters=1):**

```
direct_bytes = 1 × 128 × 128 × 4 × 1 = 65536
predicted_cycles = 5010
predicted_us = 5010 / 1650 = 3.036 μs
predicted_GB/s = 65536 / 3.036 × 1e-3 ≈ 21.6 GB/s
```

## Results tables

### Sweep data (msprof, OMP=16, block_dim=1)

| Kernel | num_iters | direct_bytes | predicted_cycles | predicted_us | predicted GB/s | wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stream_c2v | 1 | 65536 | 5010 | 3.04 | 21.6 | 612 |
| stream_v2c | 1 | 32768 | 4710 | 2.86 | 11.5 | 625 |

### Comparison to README measured peaks

Real-device peaks from [README.md](README.md) (`WARMUP=1 REPEATS=3`, full device `block_dim` sweep):

| Metric | stream_c2v | stream_v2c |
| --- | ---: | ---: |
| Peak predicted GB/s (sim, block_dim=1, num_iters=1) | 21.6 | 11.5 |
| num_iters at peak | 1 | 1 |
| A5 measured peak (README) | 9704.5 GB/s | 8328.2 GB/s |
| Predicted / measured ratio | 0.22% | 0.14% |
| Old DAV_2201 baseline | 1154.2 GB/s | 1102.8 GB/s |
| Predicted / old ratio | 1.9% | 1.0% |

**Interpretation:** Simulator numbers use `block_dim=1` and a single inner loop iteration — not comparable to README peaks which sweep large `num_iters` on all Cube cores. The large gap is expected for this micro-config; the simulator still reports consistent cycle-level timing for the executed tile.

Machine-readable summary: [outputs/sweep/sweep_results.json](outputs/sweep/sweep_results.json).

## Trace generation

### Stream kernels (msprof trace — cannsim blocked)

Open in Chrome: `chrome://tracing`

| Kernel | Trace path |
| --- | --- |
| stream_c2v | `outputs/msprof_c2v_stdout/OPPROF_20260610224722_SCSVCOYBUALFDCSO/simulator/trace.json` |
| stream_v2c | `outputs/msprof_v2c/OPPROF_20260610222715_HDDFSWCBKBBFDCGW/simulator/trace.json` |

### cannsim report workflow (reference add kernel)

Stream mix kernels do not complete under `cannsim record`, so the report pipeline was verified on the reference **add** kernel export:

```bash
# Record (reference directory; teardown segfault may occur after PASS)
cd /workdir/pto-kernels/.skills/testing-pto-kernels/reference/dynamic_multi_core/a5
source /usr/local/Ascend/cann-9.0.0/bin/setenv.bash
cannsim record -s Ascend950 --gen-report -o outputs/cannsim_add_report \
  ./run_sim_entry.sh -u "--n 128 --block-dim 8 --output-json outputs/add.json"

# Report — use inner export dir containing instr.bin
EXPORT=outputs/cannsim_add_report/cannsim_*_run_sim_entry.sh
cannsim report -e "$EXPORT" -o /workdir/pto-kernels-fork/examples/jit_cpp/cv_sync_demo_a5/outputs/report_add_all -n all
```

Generated traces in this demo:

```
outputs/report_add_all/trace_core0.json
outputs/report_add_all/trace_core1.json
outputs/report_add_all/trace_core16.json
outputs/report_add_all/trace_core17.json
```

## Known issues

1. **cannsim hang on AIC_MIX stream kernels:** `cannsim.log` shows three init `[Hardware]` lines (cycles 248/419/420), then endless `[DRVSTUB_LOG] sendStarsSQE` with no kernel-completion cycle line and no `SIM_SUMMARY`. Cases hit the 300 s process timeout.
2. **Teardown segfault:** `cannsim record` may exit non-zero after a successful kernel on simpler kernels (e.g. add). `run_sim.sh` treats non-zero exit as success when `SIM_SUMMARY` appears in captured stdout.
3. **Do not source setenv in `run_sim_entry.sh`:** causes `aclInit 507008` by overriding cannsim-injected simulator libraries.
4. **First run compile:** pre-build with `python3 common_build.py` to keep sweep wall times meaningful.
5. **Simulator vs host time:** msprof/cannsim wall time (minutes) reflects CPU simulation cost; predicted GB/s uses NPU cycle model (`/1650` μs), which is orders of magnitude smaller.

## Helper scripts

| Script | Purpose |
| --- | --- |
| `parse_cannsim_log.py` | Parse `cannsim.log` + `SIM_SUMMARY` → predicted GB/s (when kernel completes) |
| `parse_msprof_mix_log.py` | Parse msprof stdout AIC_MIX ticks → predicted GB/s (stream + matmul) |
| `run_cannsim_sweep.py` | Automated cannsim OMP + iters sweep (requires working cannsim completion) |

---

## Matmul mix kernels (`run_matmul.py`)

`run_matmul.py` supports the same simulator flags as `run_stream.py`:

| Flag | Purpose |
| --- | --- |
| `--sim` | Single-case mode (skip 30-case correctness sweep) |
| `--kernel {c2v,v2c,both}` | `c2v` = `matmul_add_c2v`, `v2c` = `add_matmul_v2c` |
| `--rounds N` | Inner loop count (`batch = rounds × block_dim × 128`) |
| `--block-dim N` | Cube core count (default `1` under sim via `PTO_BLOCK_DIM`) |

### Run commands

Pre-build once, then run under msprof (mix kernels complete via msprof; `cannsim record` hangs on stream mix kernels and was not retried for matmul):

```bash
cd /workdir/pto-kernels-fork/examples/jit_cpp/cv_sync_demo_a5
source /usr/local/Ascend/cann-9.0.0/bin/setenv.bash
export PTO_SIMULATOR=1 PTO_BLOCK_DIM=1 OMP_NUM_THREADS=16

msprof op simulator --soc-version=Ascend950PR_9599 --timeout=900 \
  --output=outputs/matmul_sweep/msprof_c2v_r1 \
  python3 run_matmul.py --sim --kernel c2v --rounds 1 --block-dim 1 \
  2>&1 | tee outputs/matmul_sweep/c2v_r1.log

msprof op simulator --soc-version=Ascend950PR_9599 --timeout=900 \
  --output=outputs/matmul_sweep/msprof_v2c_r1 \
  python3 run_matmul.py --sim --kernel v2c --rounds 1 --block-dim 1 \
  2>&1 | tee outputs/matmul_sweep/v2c_r1.log
```

Parse predicted bandwidth:

```bash
python3 parse_msprof_mix_log.py outputs/matmul_sweep/c2v_r1.log
python3 parse_msprof_mix_log.py outputs/matmul_sweep/v2c_r1.log
```

### Byte formulas (same as real-device script)

- **matmul_add_c2v:** `direct_bytes = batch × 128 × (2 + 4 + 4) + 128 × 128 × 2`
- **add_matmul_v2c:** `direct_bytes = (batch × 128 × 3 + 128 × 128) × 2`

With `block_dim=1`, `rounds=1`: `batch=128`.

### Rounds sweep under 300 s wall cap

Sweep script (stop when wall ≥ 300 s or no `SIM_SUMMARY`):

```bash
WALL_CAP=300
for k in c2v v2c; do
  for r in 1 2 4 8 16 32 64; do
    msprof op simulator ... python3 run_matmul.py --sim --kernel $k --rounds $r --block-dim 1
    # break when wall >= WALL_CAP
  done
done
```

**Result:** even `rounds=1` (minimum data size) exceeds the 300 s cap. No larger rounds were attempted.

| Kernel | rounds | batch | wall (s) | Under 300 s? |
| --- | ---: | ---: | ---: | --- |
| matmul_add_c2v | 1 | 128 | 989 | no |
| add_matmul_v2c | 1 | 128 | 869 | no |

**Largest rounds under 5 min wall time:** none at `block_dim=1`. Minimum config (`rounds=1`) needs ~14–16 min host simulation time.

### Matmul predicted vs README measured peaks

Real-device peaks from [README.md](README.md) (`WARMUP=1 REPEATS=3`, full device `block_dim`, rounds sweep up to 64):

| Metric | matmul_add_c2v | add_matmul_v2c |
| --- | ---: | ---: |
| Sim config | rounds=1, batch=128, block_dim=1 | rounds=1, batch=128, block_dim=1 |
| direct_bytes | 196608 | 131072 |
| predicted_cycles | 10127 | 8139 |
| predicted_us (÷1650) | 6.14 | 4.93 |
| **Predicted GB/s** | **32.0** | **26.6** |
| A5 measured peak (README) | 2039.4 GB/s | 1727.6 GB/s |
| Predicted / measured | 1.57% | 1.54% |
| Old DAV_2201 baseline | 1401.3 GB/s | 1593.8 GB/s |
| Predicted / old | 2.3% | 1.7% |

**Example calculation (matmul_add_c2v):**

```
direct_bytes = 128 × 128 × (2+4+4) + 128×128×2 = 196608
predicted_cycles = 153744 − 143617 = 10127
predicted_us = 10127 / 1650 = 6.14 μs
predicted_GB/s = 196608 / 6.14 × 1e-3 ≈ 32.0 GB/s
```

Machine-readable summary: [outputs/matmul_sweep/matmul_sweep_results.json](outputs/matmul_sweep/matmul_sweep_results.json).

**Caveat:** README peaks use all Cube cores and larger `rounds`; simulator runs here use `block_dim=1` and the smallest `rounds=1` case. Predicted GB/s reflects cycle-model throughput for that micro-config, not full-device peak bandwidth.

## Known issues
