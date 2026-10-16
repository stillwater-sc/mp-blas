# Level-3 accumulator benchmark: what each accumulation configuration costs

`applications/level3/matmul_accumulator_bench` times `C = A B` under the three
accumulation configurations of MTL5's `accumulator_traits`, and measures how each
scales across MTL5's thread pool. It is the cost side of
`matmul_accumulator_study`, which measures the error side.

| Config | Call | Rounding |
|---|---|---|
| default | `mtl::mult(A, B, C)` | MTL5's own dispatch (vectorized i-k-j kernel; BLAS when enabled) |
| 1 plain | `mtl::mult<T>(A, B, C)` | the product rounds, then the sum |
| 2 fma | `mtl::mult<fma_accumulator<T>>(A, B, C)` | one rounding per term |
| 3 quire | `mtl::mult<quire<T>>(A, B, C)` | one rounding per C entry |

Every configuration delivers `C` in the element type `T`. Configuration 3 is the
exact product rounded once, so the benchmark uses it as the reference. Each row
reports the share of `C` entries the other configurations get wrong, and the
normwise error.

## Running it

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target matmul_accumulator_bench -j4

# one run: 128 x 128, all types and configurations, 8 threads
MTL5_NUM_THREADS=8 ./build/applications/level3/matmul_accumulator_bench/matmul_accumulator_bench --n 128

# sweep thread counts and P-core / E-core placements
applications/level3/matmul_accumulator_bench/run_sweep.sh -n 128 -p all,pcore,pcore2,ecore

# the native configurations alone, at a size where they take long enough to time
applications/level3/matmul_accumulator_bench/run_sweep.sh -n 1024 -y double -c default,plain,fma
```

**Thread count:** MTL5's thread pool reads `MTL5_NUM_THREADS` once, so each run
measures a single thread count.

**Placement:** the pool doesn't pin its workers, so `run_sweep.sh` places them
with `taskset`. It reads the core lists that Linux publishes for hybrid Intel
CPUs in `/sys/devices/cpu_core` and `/sys/devices/cpu_atom`:

| Placement | CPUs used |
|---|---|
| `all` | not pinned |
| `pcore` | one hardware thread per P-core |
| `pcore2` | both SMT threads of every P-core |
| `ecore` | E-cores only |

**Requirements:**
- Configuration 3 for `posit` needs Universal v5.7.0 or later.
- Configuration 3 for `float` and `double` needs stillwater-sc/universal#1703.
- With an older Universal, the program reports configuration 3 as skipped. It
  doesn't time a quire whose every product would round.

## Results: Intel Core i7-12700K

The machine has 8 P-cores with SMT (CPUs 0–15, up to 5.0 GHz) and 4 E-cores
(CPUs 16–19, up to 3.8 GHz). The build was gcc 15, Release, against MTL5 `main`
(5.18.0 plus #584) and Universal with #1703. Throughput is in million
multiply-adds per second (MMAC/s), best of 3.

### n = 128, all configurations

Single thread on a P-core:

| Type | default | 1 plain | 2 fma | 3 quire | quire vs plain |
|---|---:|---:|---:|---:|---:|
| `double` | 6150 | 2218 | 1310 | 0.57 | 3900× slower |
| `float` | 15329 | 2419 | 1314 | 2.18 | 1100× slower |
| `posit<32,2>` | 1.53 | 1.46 | 1.24 | 1.50 | same |

C entries that differ from the correctly rounded product:

| Type | 1 plain | 2 fma | 3 quire |
|---|---:|---:|---:|
| `double` | 86 % | 85 % | 0 % |
| `float` | 85 % | 85 % | 0 % |
| `posit<32,2>` | 86 % | 85 % | 0 % |

Quire speedup over one thread:

| Type | 4 P-cores | 8 P-cores | 20 threads, unpinned | 4 E-cores |
|---|---:|---:|---:|---:|
| `double` | 3.9× | 7.6× | 10.6× | 3.8× |
| `float` | 3.8× | 7.5× | 11.9× | 3.8× |
| `posit<32,2>` | 3.9× | 7.5× | 10.8× | 3.8× |

### n = 1024, native configurations

| Config | `double`, 1 P-core | `double`, 8 P-cores | `float`, 1 P-core | `float`, 8 P-cores |
|---|---:|---:|---:|---:|
| default | 7447 | 49591 | 13976 | 87823 |
| 1 plain | 546 | 3862 | 543 | 3852 |
| 2 fma | 537 | 3437 | 512 | 3430 |

### What the numbers say

**Cost of exactness**
- **An IEEE quire costs three orders of magnitude.**
  - `quire<double>` is about 3900× slower than a `double` accumulator.
  - `quire<float>` is about 1100× slower than a `float` accumulator. Its quire
    is 584 bits, against 4226 for `quire<double>`.
  - Every term is a software multiply and a wide fixed-point add, against one
    hardware FMA.
  - In exchange, every C entry is correctly rounded. Configurations 1 and 2 miss
    the correctly rounded value in about 85 % of entries.
- **For posits the quire is free.** `posit<32,2>` arithmetic is already software
  emulated, and an exact quire add costs no more than a rounded posit add. So
  configuration 3 runs at the speed of configuration 1, and slightly faster than
  configuration 2.

**Scaling**
- **The quire scales almost perfectly across P-cores.** With 4 and 8 threads it
  gets 3.9× and 7.5–7.6×.
  - It is compute-bound, and each `C` entry is independent work.
  - The SMT siblings and E-cores add another 40–60 %, reaching 10.6–11.9× on
    all 20 hardware threads.
- **Mixing P-cores and E-cores can make static partitioning worse.**
  - MTL5 splits the rows of `C` into equal chunks per thread. An E-core runs
    configurations 2 and 3 at 30–80 % of a P-core's speed (configuration 2 is
    the slowest), so it finishes its share last.
  - At n = 1024, unpinned, 12 threads are no faster than 8 for the native
    configurations:
    - `double` plain gets 6.9× at 8 threads and 7.0× at 12;
    - `double` fma gets 6.4× at 8 and 6.3× at 12.
  - Pinned to the 16 P-core hardware threads (`pcore2`), the same kernels reach
    12.9× and 8.6×.
- **Configurations 1 and 2 are memory-bound at n = 1024.**
  - The accumulator kernel sums each `C(i,j)` as one inner product, so it walks
    `B` down a column.
  - At n = 128, `B` fits in L2 and configuration 1 runs at 2.2 GMAC/s. At
    n = 1024 it runs at 0.55 GMAC/s, about 13× below the default i-k-j kernel.
  - Blocking or packing `B` for the accumulator path would close most of that
    gap (stillwater-sc/mtl5#587). The quire is compute-bound and wouldn't notice.
- **At n = 128 the native configurations are too short to time.** The default
  and plain kernels finish in under a millisecond, so their thread scaling there
  measures the pool's wake-up cost, not the kernel. Use `-c default,plain,fma`
  at a larger n for those, as in the n = 1024 table.

**Determinism**
- **The thread count doesn't change any result.** Each `C` entry is summed by one
  thread in ascending `k`. Across all 20 runs of the n = 128 sweep and all
  thread counts and placements at n = 1024, every type and configuration produced
  a single checksum.
- This holds for GEMM in every configuration. For a single dot product split
  across threads (MTL5 #584), only the quire keeps its bits.
