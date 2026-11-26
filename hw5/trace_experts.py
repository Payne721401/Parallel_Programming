"""Summarize kernel overlap and peer-copy activity from one Nsight Systems SQLite export."""
import argparse
import sqlite3
from collections import defaultdict
from itertools import groupby
from pathlib import Path


def summarize(path):
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    kernels = db.execute("SELECT start, end, deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL")
    events = []
    for start, end, gpu in kernels:
        events.extend(((start, gpu, 1), (end, gpu, -1)))
    ranges = db.execute("""
        SELECT n.start, n.end, coalesce(n.text, s.value)
        FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId
        WHERE coalesce(n.text, s.value) IN ('prefill','decode') AND n.end IS NOT NULL
    """)
    for start, end, phase in ranges:
        events.extend(((start, phase, 1), (end, phase, -1)))

    active = defaultdict(int)
    durations = defaultdict(lambda: [0] * 4)
    previous = 0
    for stamp, changes in groupby(sorted(events, key=lambda e: e[0]), key=lambda e: e[0]):
        state = int(active[0] > 0) + 2 * int(active[1] > 0)
        for phase in ("prefill", "decode"):
            if active[phase]:
                durations[phase][state] += stamp - previous
        for _, key, delta in changes:
            active[key] += delta
        previous = stamp
    print("phase       total_s   gpu0_only   gpu1_only     both_s   neither_s")
    for phase, (neither, gpu0, gpu1, both) in durations.items():
        times = (neither + gpu0 + gpu1 + both, gpu0, gpu1, both, neither)
        print(f"{phase:8s}", *(f"{t / 1e9:10.4f}" for t in times))
    print("peer API / correlated memory activity / count / bytes")
    for row in db.execute("""
        SELECT s.value, op.label, count(*), sum(m.bytes)
        FROM CUPTI_ACTIVITY_KIND_MEMCPY m
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r USING(correlationId)
        JOIN StringIds s ON r.nameId=s.id
        JOIN ENUM_CUDA_MEMCPY_OPER op ON op.id=m.copyKind
        WHERE s.value LIKE 'cudaMemcpyPeer%'
        GROUP BY s.value, op.label
    """):
        print(*row)
    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    summarize(parser.parse_args().sqlite)
