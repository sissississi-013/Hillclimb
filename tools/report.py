"""Summarize harness logs into per-subtask x checkpoint success tables (markdown), for model selection.

    python tools/report.py logs/harness                 # every run under logs/harness
    python tools/report.py logs/harness/20261107-*      # some runs
    python tools/report.py logs/harness --since 13:00   # only attempts started after 13:00 today

Prints:
- per (subtask, checkpoint): attempts, successes, success rate with a Wilson 95% interval, median time to success, and
  the most common failure ends. Pick the per-subtask checkpoint from this, by real runs and not by loss.
- per run: whether every subtask succeeded (a full-task success), and the attempts used.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import math
import pathlib
import statistics


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load(paths: list[pathlib.Path]) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {}
    for root in paths:
        for f in sorted(root.rglob("attempts.jsonl")) if root.is_dir() else []:
            runs[f.parent.name] = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return runs


def _after(a: dict, since: str | None) -> bool:
    if not since or not a.get("started_at"):
        return True
    started = datetime.datetime.fromtimestamp(float(a["started_at"])) if isinstance(a["started_at"], (int, float)) \
        else datetime.datetime.fromisoformat(str(a["started_at"]).replace("Z", "+00:00"))
    if started.tzinfo is not None:
        started = started.astimezone()  # UTC in the log -> local clock for --since
    return started.strftime("%H:%M") >= since


def report(runs: dict[str, list[dict]], since: str | None = None) -> str:
    groups: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for attempts in runs.values():
        for a in attempts:
            if _after(a, since):
                groups[(a["subtask"], str(a.get("checkpoint") or "station-default"))].append(a)
    out = ["| subtask | checkpoint | n | ok | rate (95% CI) | median s to ok | top failure ends |", "|---|---|---|---|---|---|---|"]
    for (sub, ckpt), items in sorted(groups.items()):
        n, k = len(items), sum(bool(a["success"]) for a in items)
        lo, hi = wilson(k, n)
        ok_t = [a["elapsed_s"] for a in items if a["success"]]
        med = f"{statistics.median(ok_t):.1f}" if ok_t else "-"
        fails = collections.Counter(a["end"] for a in items if not a["success"]).most_common(2)
        out.append(f"| {sub} | {ckpt} | {n} | {k} | {k / n:.0%} ({lo:.0%}–{hi:.0%}) | {med} | "
                   f"{', '.join(f'{e}×{c}' for e, c in fails) or '-'} |")
    out += ["", "| run | full success | attempts used |", "|---|---|---|"]
    for name, attempts in sorted(runs.items()):
        attempts = [a for a in attempts if _after(a, since)]
        if not attempts:
            continue
        subs = list(dict.fromkeys(a["subtask"] for a in attempts))
        full = all(any(a["success"] for a in attempts if a["subtask"] == s) for s in subs)
        out.append(f"| {name} | {'YES' if full else 'no'} ({len(subs)} subtasks) | {len(attempts)} |")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", type=pathlib.Path)
    p.add_argument("--since", help="HH:MM, local time")
    args = p.parse_args()
    print(report(load(args.paths), args.since))


if __name__ == "__main__":
    main()
