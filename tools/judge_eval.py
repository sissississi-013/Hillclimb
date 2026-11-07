"""Score the success judge on a folder of labelled frames: accuracy, false-positive rate, latency.

It calls hillclimb.judge.VLMJudge itself (the harness's code path) with the frozen rubric from config.yaml, so a result
here is what the harness would decide. Layout (one folder per sample; any number of camera jpgs):

    <root>/<subtask>/pos/<sample>/<cam>.jpg          the step DID succeed in this state
    <root>/<subtask>/neg/<sample>/<cam>.jpg          it did not
    <root>/<subtask>/<pos|neg>/<sample>/before/<cam>.jpg   optional: the attempt's first frame (for --before)

<subtask> must be a subtask name in config.yaml (unplug_usb, plug_usb, ...). To build it at the event, copy
logs/<run>/<NN-subtask>-a<k>/inputs/<cam>/<newest>.jpg into a sample folder and file it under pos/ or neg/ by eye.
Optional <root>/rubric.yaml merges per-subtask keys over config.yaml's rubric, e.g. {plug_usb: {scene: "..."}}, to try
rubric wording (or a foreign dataset's scene notes) without editing config.yaml.

    set -a; . ./.env; set +a     # FIREWORKS_API_KEY
    .venv/bin/python tools/judge_eval.py /path/to/eval                      # config.yaml's judge model
    .venv/bin/python tools/judge_eval.py /path/to/eval --model accounts/fireworks/models/kimi-k3 --before
    .venv/bin/python tools/judge_eval.py /path/to/eval --dry-run            # list samples, no API calls

A false positive (judge says success on a neg) is the costly error: the harness moves on instead of retrying.
`unknown` (judge error or unparsable reply) counts as "not success", which is what the harness does with it.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import sys
import time

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hillclimb.judge import VLMJudge  # noqa: E402

IMG = (".jpg", ".jpeg")


def load_samples(root: pathlib.Path, cams: list[str] | None = None) -> list[dict]:
    out = []
    for sub_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for label in ("pos", "neg"):
            for s in sorted(p for p in (sub_dir / label).glob("*") if p.is_dir()):
                frames = {p.stem: p for p in sorted(s.iterdir()) if p.suffix.lower() in IMG}
                before = {p.stem: p for p in sorted((s / "before").glob("*")) if p.suffix.lower() in IMG}
                if cams:
                    frames = {k: v for k, v in frames.items() if k in cams}
                    before = {k: v for k, v in before.items() if k in cams}
                if frames:
                    out.append({"id": f"{sub_dir.name}/{label}/{s.name}", "subtask": sub_dir.name,
                                "label": label == "pos", "frames": frames, "before": before})
    return out


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))] if xs else float("nan")


def metrics(rows: list[dict]) -> dict:
    tp = sum(r["label"] and r["pred"] for r in rows)
    fn = sum(r["label"] and not r["pred"] for r in rows)
    fp = sum(not r["label"] and r["pred"] for r in rows)
    tn = sum(not r["label"] and not r["pred"] for r in rows)
    lat = [r["latency_s"] for r in rows]
    n = len(rows)
    return {"n": n, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "accuracy": round((tp + tn) / n, 3) if n else None,
            "fpr": round(fp / (fp + tn), 3) if fp + tn else None,
            "fnr": round(fn / (tp + fn), 3) if tp + fn else None,
            "unknown": sum(r["success"] is None for r in rows),
            "latency_p50_s": round(pct(lat, 0.5), 2), "latency_p90_s": round(pct(lat, 0.9), 2),
            "latency_max_s": round(max(lat), 2) if lat else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=pathlib.Path)
    ap.add_argument("--config", type=pathlib.Path, default=ROOT / "hillclimb" / "config.yaml")
    ap.add_argument("--model", help="default: config.yaml judge.model")
    ap.add_argument("--threshold", type=float, help="default: config.yaml judge.threshold")
    ap.add_argument("--reasoning-effort", help="default: config.yaml judge.reasoning_effort ('none'); 'omit' drops it")
    ap.add_argument("--max-tokens", type=int, help="default: config.yaml judge.max_tokens (800)")
    ap.add_argument("--no-json-mode", action="store_true", help="do not send response_format=json_object")
    ap.add_argument("--rubric", type=pathlib.Path, help="rubric overrides yaml (default: <root>/rubric.yaml if present)")
    ap.add_argument("--before", action="store_true", help="also send each sample's before/ frames")
    ap.add_argument("--cams", help="comma list of camera stems to send (default: all in the sample folder)")
    ap.add_argument("--limit", type=int, default=0, help="first N samples per subtask/label (0 = all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=pathlib.Path, help="write per-sample results (jsonl) here")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(a.config.read_text())
    subtasks = {s["name"]: s for s in cfg["subtasks"]}
    rub_path = a.rubric or (a.root / "rubric.yaml")
    overrides = yaml.safe_load(rub_path.read_text()) if rub_path.exists() else {}
    for name, extra in (overrides or {}).items():
        subtasks[name] = dict(subtasks[name], rubric={**subtasks[name].get("rubric", {}), **extra})
    jcfg = cfg.get("judge", {})
    json_mode = not a.no_json_mode and jcfg.get("json_mode", True)
    model = a.model or jcfg.get("model") or VLMJudge().model
    threshold = a.threshold if a.threshold is not None else jcfg.get("threshold", 0.7)
    effort = a.reasoning_effort or jcfg.get("reasoning_effort", "none")
    effort = None if effort == "omit" else effort
    max_tokens = a.max_tokens or jcfg.get("max_tokens", 800)
    samples = load_samples(a.root, a.cams.split(",") if a.cams else None)
    unknown_sub = {s["subtask"] for s in samples} - set(subtasks)
    if unknown_sub:
        sys.exit(f"sample folders name subtasks not in {a.config}: {sorted(unknown_sub)}")
    if a.limit:
        seen: dict[tuple, int] = {}
        keep = []
        for s in samples:
            k = (s["subtask"], s["label"])
            seen[k] = seen.get(k, 0) + 1
            if seen[k] <= a.limit:
                keep.append(s)
        samples = keep
    if a.before and any(not s["before"] for s in samples):
        print("warning: some samples have no before/ frames; they are judged on the after frames alone",
              file=sys.stderr)
    print(f"{len(samples)} samples, model={model}, threshold={threshold}, "
          f"reasoning_effort={effort}, max_tokens={max_tokens}, json_mode={json_mode}, before={a.before}, "
          f"rubric_overrides={sorted(overrides or {})}", file=sys.stderr)
    if a.dry_run:
        for s in samples:
            print(s["id"], s["label"], sorted(s["frames"]), sorted(s["before"]))
        return

    judge = VLMJudge(model, threshold, reasoning_effort=effort, max_tokens=max_tokens, json_mode=json_mode)
    if not judge.api_key:
        sys.exit("FIREWORKS_API_KEY is not set")

    def run(s: dict) -> dict:
        t0 = time.monotonic()
        v = judge(subtasks[s["subtask"]], s["frames"], {}, before=s["before"] if a.before else None)
        dt = time.monotonic() - t0
        return {"id": s["id"], "subtask": s["subtask"], "label": s["label"], "pred": bool(v.success),
                "success": v.success, "score": v.score, "latency_s": round(dt, 3), "reason": v.reason[:400]}

    with cf.ThreadPoolExecutor(max(1, a.workers)) as ex:
        rows = list(ex.map(run, samples))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    for r in rows:
        if r["pred"] != r["label"] or r["success"] is None:
            kind = "UNKNOWN" if r["success"] is None else ("FALSE-POS" if r["pred"] else "false-neg")
            print(f"{kind:9} {r['id']} score={r['score']:.2f} {r['reason'][:160]}", file=sys.stderr)
    report = {"model": model, "threshold": threshold, "reasoning_effort": effort, "max_tokens": max_tokens,
              "json_mode": json_mode, "rubric_overrides": overrides or {}, "before": a.before, "overall": metrics(rows),
              "by_subtask": {k: metrics([r for r in rows if r["subtask"] == k])
                             for k in sorted({r["subtask"] for r in rows})}}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
