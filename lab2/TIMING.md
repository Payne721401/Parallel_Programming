# Measuring the running time of your program

Speedup is a ratio of two running times, so it is only as good as the two
measurements behind it. This page explains how to time a program on SEIREN in
a way that means something, which methods to avoid, and, at the end, exactly
how to measure for Lab 2 so that your numbers match what the judge sees.

## 1. What you are measuring

Two different clocks are in play whenever you time a parallel program:

- **Wall-clock time (elapsed time)**: the time from start to finish as a clock
  on the wall would show it. This is what a user waits for, and this is what
  "speedup" is defined on.
- **CPU time**: the time the CPU spent executing your process. With `n`
  processes or threads it adds up across all of them, so a perfectly parallel
  program shows *the same* CPU time as the sequential one, or more.

Always compare wall-clock times. A method that reports CPU time will tell you
that your parallel program got slower, when it did not.

A second trap is *what* the wall clock covers. On a cluster your program runs
inside a Slurm job, and the job may sit in the queue before it starts. Queue
time is not running time. Whatever you measure has to start its clock after
the job has been allocated.

## 2. Correct methods

### 2.1 `/usr/bin/time` inside `srun` (whole program)

```sh
srun -n4 /usr/bin/time -f %e ./hello
```

`srun` allocates the job, then starts four copies of `/usr/bin/time`, each of
which starts one rank and prints how long *that rank* took, in seconds, to
stderr. The queue wait is over before any clock starts.

- You get one line per rank. The program is finished when its last rank is, so
  the running time is the **largest** of them.
- `-f %e` prints elapsed seconds only. Without `-f` GNU time prints a longer
  summary; the `real`/`elapsed` field is the one you want.
- Write `/usr/bin/time`, not `time`. Inside a `bash` command line, `time` is a
  shell keyword with different behaviour (see 3.1).

This measures the whole process: MPI start-up, your computation, and
`MPI_Finalize`. Its scope matches Lab 2
whole-program limits. HW2 uses different fixed timed regions.

### 2.2 `MPI_Wtime()` (a section of the program)

```c
double tik = MPI_Wtime();
/* ... the part you want to time ... */
double tok = MPI_Wtime();
printf("rank %d: %f seconds\n", rank, tok - tik);
```

`MPI_Wtime()` returns wall-clock seconds as a `double`; resolution depends on
the MPI implementation. Use it to find out *where*
the time goes: how long the loop took, how long the reduction took, how uneven
the ranks are. Each rank reports its own value; add an `MPI_Barrier` before
`tik` if you want every rank to start the clock together.

It cannot see anything before `MPI_Init` or after `MPI_Finalize`, so a timer around only the computation
measures a smaller scope than the Lab 2 whole-program limit.

### 2.3 `clock_gettime(CLOCK_MONOTONIC, ...)` and `std::chrono::steady_clock`

```c
struct timespec a, b;
clock_gettime(CLOCK_MONOTONIC, &a);
/* ... */
clock_gettime(CLOCK_MONOTONIC, &b);
double sec = (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
```

```cpp
auto a = std::chrono::steady_clock::now();
/* ... */
auto b = std::chrono::steady_clock::now();
double sec = std::chrono::duration<double>(b - a).count();
```

Both are monotonic wall clocks: they never jump backwards and are not affected
by anyone adjusting the system time. They do the same job as `MPI_Wtime()`
and work in code that has no MPI, such as the sequential program.

## 3. Methods to avoid

### 3.1 `time srun -n4 ./hello`

The shell's `time` starts its clock *before* `srun` asks Slurm for the job, so
the number includes however long the job waited in the queue. On a busy
cluster that can be longer than the program itself, and it changes from one
run to the next for reasons that have nothing to do with your code.

Put the timer inside `srun` (2.1) so it starts on the compute node.

### 3.2 `time(NULL)`

Returns whole seconds. A program that runs for 0.4 s measures as 0 or 1.

### 3.3 `clock()`

Returns CPU time, not wall-clock time (see section 1). Two threads doing the
same work count twice, so a parallel program looks no faster. Time spent
blocked on I/O or waiting for a message is not counted at all, unless the MPI
library spins while it waits, in which case it *is* counted. Either way the
number is not what you want.

### 3.4 `clock_gettime(CLOCK_REALTIME, ...)` and `gettimeofday()`

These read the system's calendar time, which NTP nudges continuously and which
may jump after manual clock adjustments. A jump in the middle of your run corrupts the
measurement. Use `CLOCK_MONOTONIC` (2.3).

### 3.5 `std::chrono::high_resolution_clock`

Despite the name, in GCC's standard library this is an alias for
`system_clock`, so it has the same problem as 3.4. Use `steady_clock`.

## 4. Getting a number you can trust

- **Run more than once.** Take the median, or at least look at the spread. A
  single run can be off by 20% or more.
- **Time one job at a time.** Your own other jobs on the same node compete with
  the one you are timing.
- **Ask for whole cores.** SEIREN's nodes have hyper-threading on: without
  `--hint=nomultithread`, two of your ranks may share one physical core and
  they compete for execution resources. Pass the flag whenever you time something,
  and pass it the same way for the sequential and the parallel run.
- **Compare like with like.** Same input, same compiler flags, same `srun`
  options; change only the number of ranks.

---

## Note: how to measure for Lab 2

The judge measures each rank's whole-program elapsed time inside the launcher
and uses the slowest rank. This includes MPI initialization and finalization,
and excludes allocation queue time and launcher startup. The pinned judge uses
a shell timer with five decimal places; GNU time below is a convenient local
approximation with coarser precision, not the exact timer implementation:

```sh
srun -n4 -c1 --hint=nomultithread /usr/bin/time -f "%e" ./lab2 <r> <k>
```

Take the largest of the four timing lines. The published
`cases/lab2/*.txt` files give each case's rank count and whole-program limit.
In this repository those files are under `lab2/cases/`.
Judge commands below become available when the course announces deployment.

### Step 1: keep the sequential program

Before you change anything, keep a copy of the starting code. It is your
baseline, and it also gives the correct answer for any `r` and `k` you want to
try.

```sh
cp lab2.cpp lab2_seq.cpp
mpicxx -std=c++17 -O3 -o lab2_seq lab2_seq.cpp    # it uses no MPI, but mpicxx builds it fine
```

### Step 2: time the sequential program

```sh
srun -n1 --hint=nomultithread /usr/bin/time -f %e ./lab2_seq 4294967295 1099511627775
```

This is the input of case 12 and takes about 47 seconds; case 10's takes about
24. Both are well inside your 20-minute job limit.

### Step 3: time your MPI program on the same input

```sh
srun -n4 --hint=nomultithread /usr/bin/time -f %e ./lab2 4294967295 1099511627775
```

Four lines come back; take the largest. Speedup is the sequential time divided
by that. Aim for `n/2` speedup on sufficiently large inputs; the judge applies
the published per-case time limits, which include allowance for MPI startup
and run-to-run variation. It does not compute a separate `n/2` ratio. For reference, an even split of the loop with one `MPI_Reduce` measures
about 12–13 s here, a speedup of about 3.7; two ranks give about 25 s, a
speedup of about 1.9.

Run it more than once. One of our own four-rank runs took 19 s while the next
five took 12–13 s; a single measurement can mislead you.

### What you can and cannot reproduce yourself

Your available rank count depends on your account's current Slurm QoS.
Use one physical core per rank and stay within your allocation. On the course
nodes, reserving a physical core can charge both SMT threads against the CPU
quota; check the course's current resource announcement before requesting a
larger run. A request over the quota may queue or be rejected.

Practice with 1, 2 and 4 ranks when your allocation permits. Cases 09–12 use
9–12 ranks; after deployment, use the judge to obtain the grading allocation:

```sh
lab2-judge            # all cases; updates the scoreboard
lab2-judge 12         # one case; does not wait for, or restart, the cooldown
```

Do not time with `-n8` and no `--hint=nomultithread` to "get more ranks": the
number you get is not the judge's, and a program that looks fine there can
still fail the limit.

### Using `MPI_Wtime()` in Lab 2

Feel free to add `MPI_Wtime()` around the loop and around `MPI_Reduce` while
you develop, to see how evenly the work is split. Remove or silence the extra
`printf` before you submit: the judge expects the answer to be the **only**
line on stdout, and only rank 0 should print it. Printing timings to stderr is
fine.

---

*Credit: the survey of timing methods in sections 2 and 3 is adapted from the
Lab 2 slides by jerrychou, 11410CS542200 Parallel Programming.*
