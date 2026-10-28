#!/usr/bin/env python3
"""CPU gates + unprofiled ABBA + separate phase/dispatch checks. No default promotion."""
import argparse
import csv
import hashlib
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from arch_measure import command, numeric, write_table
from run_cases import run_cases
from summarize_bitplane import records, verify_public_dispatch

CASES = {"reduce": ("08", "09", "10"), "r1": ("08", "10", "12"),
         "r1shared": ("08", "10", "12"),
         "large": ("05", "06", "07"), "all": tuple(f"{i:02d}" for i in range(1,13))}
BITS = {"baseline": 0, "reduce": 1, "r1": 2, "large": 4, "all": 7, "r1shared": 8}


def configure(method, **overrides):
    for key in list(os.environ):
        if key.startswith("PP_"): del os.environ[key]
    os.environ.update(PP_ARCH=method, PP_GRAPH="0", PP_EVOLVE="0", PP_PROFILE="0",
                      PP_SUPPORT="bbox", PP_BITPLANE="auto", PP_BIT_LOGIC="compact", PP_BIT_ROWS="4")
    os.environ.update(overrides)


def verify_arch(record, row, header, method):
    bits = BITS[method]
    radius = int(re.search(r"R(\d+)", header[1]).group(1))
    kernel = row["profile_kernel"]
    sparse = row["sparse_selected"] == "1"
    expected = dict(mode=method, merged=str(int(bool(bits & 1) and sparse)),
                    r1_warp=str(int(bool(bits & 2) and kernel in ("dense_r1", "sparse_r1"))),
                    vn_diagonal=str(int(bool(bits & 4) and kernel in ("tiled", "sparse_tiled")
                                        and "NN" in header[1] and radius <= 30)),
                    r1_shared=str(int(bool(bits & 8) and kernel in ("dense_r1", "sparse_r1"))))
    if record != expected:
        raise ValueError("wrong architecture selection: " + str(record) + "; expected " + str(expected))


def gates(root, method):
    specs = [("baseline", "--quick", 22, {}),
             (method, "--quick", 22, {"PP_ACTIVE":"0"}),
             (method, "--sparse", 30, {"PP_ACTIVE":"1"}),
             (method, "--support", 28, {"PP_ACTIVE":"1"}),
             (method, "--sparse", 30, {"PP_ACTIVE":"1", "PP_SUPPORT":"tile"}),
             (method, "--graphs", 38, {"PP_ACTIVE":"1", "PP_GRAPH":"1"})]
    if method in ("reduce", "all"):
        specs += [(method,"--sparse",30,{"PP_ACTIVE":"1","PP_KERNEL":"naive"}),
                  (method,"--sparse",30,{"PP_ACTIVE":"1","PP_KERNEL":"tiled","PP_MOORE":"sat"}),
                  (method,"--sparse",30,{"PP_ACTIVE":"1","PP_KERNEL":"tiled","PP_MOORE":"sliding"})]
    if method in ("r1", "r1shared", "all"):
        specs += [(method,"--r1",64,{"PP_ACTIVE":"1"}),
                  (method,"--dense-r1",90,{"PP_ACTIVE":"0"})]
    if method in ("large", "all"):
        specs += [(method,"--vn-reuse",128,{"PP_ACTIVE":"0","PP_KERNEL":"tiled"}),
                  (method,"--vn-reuse",128,{"PP_ACTIVE":"1","PP_KERNEL":"tiled"}),
                  (method,"--sliding",48,{"PP_ACTIVE":"0","PP_KERNEL":"tiled"})]
    for number, (selected, flag, count, env) in enumerate(specs):
        configure(selected, PP_BITPLANE="0", **env)
        log = root / f"gate-{number:02d}.log"
        command(["./tests/test_hw3-1", flag], log, 120)
        text = log.read_text(encoding="utf-8")
        if not re.search(rf"^PASS {count}/{count} ", text, re.M):
            raise ValueError("missing/incomplete CPU gate: " + str(log))
        print(f"Gate {number}: {selected} {flag} PASS {count}/{count} {env}")


def summarize(samples, cases, method):
    results = []
    for case in cases:
        pair = {}
        for selected in ("baseline", method):
            group = [s for s in samples if s["case_id"] == case and s["method"] == selected]
            if len(group) != 2 or any(s["verdict"] not in ("AC", "TLE") for s in group):
                raise ValueError("incomplete or incorrect ABBA pair")
            values = [numeric(s["compute_s"]) for s in group]
            if min(values) <= 0: raise ValueError("invalid zero duration")
            pair[selected] = values
        a,b = pair["baseline"], pair[method]
        unstable = max(a)/min(a)>1.2 or max(b)/min(b)>1.2
        results.append(dict(case=case, method=method, baseline_mean_s=statistics.mean(a),
                            candidate_mean_s=statistics.mean(b), baseline_min_s=min(a), baseline_max_s=max(a),
                            candidate_min_s=min(b), candidate_max_s=max(b),
                            baseline_over_candidate=statistics.mean(a)/statistics.mean(b),
                            timing_status="unstable_samples" if unstable else "ok"))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method",choices=CASES)
    parser.add_argument("--mode",choices=("compare","check"),default="compare")
    parser.add_argument("--outdir",type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"): raise ValueError("run through Slurm")
    root=args.outdir; root.mkdir(parents=True,exist_ok=True)
    with (root/"started").open("x") as f: f.write(args.method+"\n")
    command(["git","rev-parse","HEAD"],root/"revision.log",10)
    command(["git","diff","--","."],root/"working-diff.log",10)
    print("Revision: "+(root/"revision.log").read_text().strip())
    print("CUDA_MODULE_LOADING="+os.environ.get("CUDA_MODULE_LOADING","unset"))
    subprocess.run(["nvidia-smi","--query-gpu=name,driver_version","--format=csv"],check=True,timeout=15)
    paths=[Path("hw3-1"),Path("tests/test_hw3-1"),*sorted(Path('.').glob('*.cu')),
           *sorted(Path('.').glob('*.cuh')),*sorted(Path('.').glob('*.hpp')),Path(__file__)]
    (root/"sha256.log").write_text('\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p) for p in paths)+'\n')
    gates(root,args.method)
    if args.mode=="check":
        (root/"COMPLETE").write_text("CPU reference gates completed; no public timing run.\n")
        return
    samples=[]
    for number, selected in enumerate(("baseline",args.method,args.method,"baseline",
                                        "baseline",args.method,args.method,"baseline"),1):
        profiled=number>=5
        configure(selected,PP_PROFILE=str(int(profiled)))
        order=CASES[args.method] if number%2 else tuple(reversed(CASES[args.method]))
        folder=root/f"{number}-{selected}"
        print(f"Round {number}: PP_ARCH={selected} PP_PROFILE={int(profiled)}")
        status=run_cases(argparse.Namespace(inputs=[f"cases/{c}.txt" for c in order],binary="./hw3-1",
                         checker="hw3-1-check",wall_timeout=20,limit=None,outdir=str(folder)))
        with (folder/"results.tsv").open(encoding="utf-8",newline="") as f:
            rows=list(csv.DictReader(f,delimiter='\t'))
        if status not in (0,1) or len(rows)!=len(order): raise ValueError("incomplete public run")
        for i,row in enumerate(rows):
            case=Path(row["case"]).stem
            if row["verdict"] not in ("AC","TLE"): raise ValueError("public output not verified")
            numeric(row["compute_s"])
            if profiled:
                text=(folder/f"{i:02d}-{case}.log").read_text(encoding="utf-8")
                arch=records(text,"PP_ARCH"); bit=records(text,"PP_BITPLANE")
                if len(arch)!=1 or len(bit)!=1: raise ValueError("missing dispatch records; rebuild binary")
                header=Path(row["case"]).read_text().splitlines()
                verify_public_dispatch(bit[0],row,header,"auto",4,"compact")
                verify_arch(arch[0],row,header,selected)
                print(f"  {case}: verified architecture dispatch {arch[0]}")
            elif row["profile_status"]!="absent": raise ValueError("unexpected timing instrumentation")
            samples.append(dict(round=number,method=selected,profiled=int(profiled),case_id=case,**row))
        write_table(root/"samples.tsv",samples)
    summary=summarize([s for s in samples if not s['profiled']],CASES[args.method],args.method)
    write_table(root/"summary.tsv",summary)
    # Keep profiled observations separate; expose allocation/free outliers rather
    # than attributing an end-to-end delay to the update kernel.
    phases=[]
    for case in CASES[args.method]:
        for selected in ("baseline",args.method):
            group=[s for s in samples if s['profiled'] and s['case_id']==case and s['method']==selected]
            if len(group)!=2: raise ValueError("incomplete phase pair")
            phase=dict(case=case,method=selected)
            for key in ("kernels_ms","alloc_host_ms","free_host_ms","h2d_ms","d2h_ms","run_host_ms"):
                values=[numeric(s[key]) for s in group]
                phase[key+"_mean"]=statistics.mean(values)
                phase[key+"_min"]=min(values)
                phase[key+"_max"]=max(values)
            phases.append(phase)
            print(f"  Phase {case} {selected}: kernels={phase['kernels_ms_min']:.3f}..{phase['kernels_ms_max']:.3f} ms, "
                  f"alloc={phase['alloc_host_ms_min']:.3f}..{phase['alloc_host_ms_max']:.3f}, "
                  f"free={phase['free_host_ms_min']:.3f}..{phase['free_host_ms_max']:.3f}")
    write_table(root/"phases.tsv",phases)
    for row in summary:
        print("{case}: {baseline_mean_s:.6f} -> {candidate_mean_s:.6f}s, "
              "baseline/candidate={baseline_over_candidate:.3f}x {timing_status}".format(**row))
    print("Two samples/method, all outliers retained. Profile observations excluded from ratios. Defaults unchanged.")
    (root/"COMPLETE").write_text("All requested gates, ABBA and dispatch checks completed.\n")


if __name__=="__main__":
    try: main()
    except (OSError,ValueError,KeyError,subprocess.SubprocessError) as e:
        print("architecture comparison failed: "+str(e),file=sys.stderr); sys.exit(1)
