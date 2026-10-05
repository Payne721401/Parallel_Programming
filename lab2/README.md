# Lab 2: MPI circle counting

Parallelize the supplied `lab2.cpp`. The program takes radius `r` and modulus
`k`, and prints the supplied sequential program's exact integer result.
Only rank 0 should print, with one answer and no other nonempty stdout lines.
Diagnostic output belongs on stderr. Preserve wide integer and long-double
arithmetic near the largest input; narrowing intermediate values can change
answers. Keep a copy of the original program for local comparisons.

## Build and practice

Load the course compiler/MPI modules and run `make`. Run compute work inside a
Slurm allocation, for example:

```sh
srun -n4 -c1 --hint=nomultithread ./lab2 5 100
```

This example prints `88`. Submit `lab2.cpp` and your `Makefile`, plus any extra
sources your Makefile needs. The build target and executable must be `lab2`.
The course submission announcement specifies the collection path/deadline.

The published case directory is `cases/lab2/` (repository: `lab2/cases/`). Each
of the 12 text files contains `nproc`, `r`, `k`, the exact `answer`, and the
whole-program `timelimit` in seconds. Use its rank count to reproduce a case.
The judge checks both exact output and the case's time limit. Aim for useful
MPI scaling; `n/2` is a large-input speedup target, not an additional ratio test.

See `TIMING.md` for measurement commands. Lab 2 measures the slowest rank's
whole-program elapsed time, including MPI initialization/finalization; HW2's
fixed compute timers are different. Your practice QoS may allow fewer ranks
than the judge. After course deployment, `lab2-judge` runs all public cases and
`lab2-judge 12` runs just case 12. Availability is announced separately.
