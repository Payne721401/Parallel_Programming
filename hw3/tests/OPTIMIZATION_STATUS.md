# HW3-1 optimization evidence and remaining costs (2026-10-26)

## Current result and dispatch

Latest user-provided full judge: 12/12 AC, total 2.261 s versus prior 2.573 s,
12.1% less time (1.138x). Exact revision/environment are not printed in that judge
excerpt; source default Graphs is off. No hidden-case result is available.

| Cases | Current default update | Latest judge s |
| --- | --- | --- |
| 01/02/03 | dense packed R1 | 0.003 / 0.003 / 0.003 |
| 04 | direct R2 von Neumann | 0.011 |
| 05/07 | dense tiled Moore sliding | 0.157 / 0.135 |
| 06 | dense tiled von Neumann row prefixes | 0.216 |
| 08/10 | sparse packed R1 | 0.425 / 0.386 |
| 09 | sparse tiled Moore sliding | 0.471 |
| 11/12 | dense packed R1 | 0.178 / 0.275 |

Dispatch is based on input rule/size and observed activity, not case numbers.
The case mapping documents the public workload, not a hard-coded lookup.
Rounded case times sum to 2.263 s; use the judge's 2.261 s total for total ratios.

## What each change solves

| Change | Problem addressed | Evidence and limit |
| --- | --- | --- |
| GPU double buffering, all steps resident | Serial cell updates; unsafe in-place dependencies; repeated host/device transfer per generation | Foundation of every current path; no claim of a separately isolated speedup here |
| Shared tiles and prefix sums | Repeated O(R^2) Moore neighbour enumeration | Public 05..07 passed after correcting default dispatch; retained SAT reference |
| Sparse worklist, dilation and cleanup | Updating mostly empty large grids; stale ping-pong cells; growth across tiles | CPU-reference motion/decay/growth checks and public 08..10 AC; management is costly on small grids |
| Packed R1 in sparse and dense paths | Scalar counting/addressing and missed data reuse for eight-neighbour rules | Sparse 08/10 profile Compute went from 1.26875/0.994618 to 0.426054/0.390084 s across different jobs; dense 11/12 passed at 0.202853/0.317 s in job 404889; these are not all same-allocation A/B results |
| Moore sliding windows | Vertical SAT construction/query overhead | Same-allocation SAT/sliding: 05 1.424x, 07 1.399x, 09 1.216x; latest judge agrees with these changes |
| CUDA Graph batching | Repeated host submission of short dependent kernels | 152 CPU-reference checks and 24 measured outputs passed in job 408274; no reliable net speedup in tested matrix; remains opt-in |
| Phase timing, feature matrix and traces | Confusing total time, API waits and kernel execution; choosing methods from initial density alone | Identified dense update cost and small-work sparse overhead; diagnostics themselves are not optimizations |
| Sampled evolution diagnostics | Missing evidence about late sparsity, exact state changes and fixed points | Implemented, GPU validation pending; does not yet skip work or accelerate simulation |

Packed R1 stores four byte states in a 32-bit word, not one bit per cell. Only
state 1 contributes to neighbour counts; C>2 decay remains represented exactly.

## Why sliding does not improve 08/10/12

All three have radius 1 and use packed R1 under default dispatch. The sliding
change is for the tiled Moore path (automatic radius >=5). Thus these cases do
not execute the changed kernel. There is no reason to force them onto sliding
without a new comparison: at radius 1, preparing prefix/window tables may cost
more than directly combining the small packed neighbourhood. Case 06 is von
Neumann and also does not use Moore sliding. The latest 10 improvement is a
single-run observation, not evidence that sliding accelerated it.

## Distance to a 0.3 s total

2.261 -> 0.3 s requires about 7.54x overall speedup, or 86.7% less time.
Public 08..10 total 1.282 s (~56.7%), 11..12 total 0.453 s (~20.0%),
05..07 total 0.508 s (~22.5%). Optimizing only one family cannot close the gap.

Existing profiles put most time in the evolution interval: packed sparse 08/10
about 382/346 ms, sliding 09 about 428 ms; dense 11/12 about 191/311 ms in older
profiled runs. These are different runs from the latest judge, not an additive
breakdown of its total. A large dense feature trace had 28.147 ms of GPU updates
out of a previous ~31.6 ms Compute; the small sparse trace exposed prepare and
update work, but did not separate update arithmetic from tracking/atomics.

For the 1-GiB grids, prior full H2D+D2H was around 40 ms per input. Even major
kernel gains would leave this floor in the CURRENT transfer path. It is not a
hardware lower bound for every valid implementation. Reaching the user-supplied
leaders' 24..35 ms totals for those cases would require changing transfer volume,
representation, or other assumptions/costs of this path, not just launch batching.
No inference about the leaders' exact algorithms is established by their scores.

Unknowns to resolve: evolution becoming sparse or fixed, actual memory traffic,
instruction throughput, shared-memory/synchronization stalls, and worklist cost.
Temporal blocking (plan 5d), bit compression (5a), diamond recurrences (5b),
selective D2H (Task 4) and resource tuning (5e) remain candidates, not measured wins.
The evolution experiment targets the first unknown before another large rewrite.
