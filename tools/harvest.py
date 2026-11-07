"""Filtered self-rollout harvest (expert iteration / filtered BC): judged-successful policy runs become training data.

With `rollouts: {enabled: true}` in the station file, every live run (dry runs save none) is saved as a one-episode
LeRobot v2.1 dataset `~/.dreamlabs/rollouts/<time>_run<N>/` plus `rollout.json` (task, run, started_at_utc, status;
robots/rollouts.py:245-290, rollout_episode.py:118-170). This pairs each harness attempt with its episode, keeps the
successes (or a human-confirmed list), and writes one new dataset per subtask in dhk-collect's Metal layout:

    ~/dream-env/bin/python tools/harvest.py logs/harness/2026* --out ~/hackathon_data/self --retime-fps 30 \
        [--confirmed ok.txt] [--trim]

The --out root is NOT a fine-tune parent (it holds harvest.jsonl, which Dream refuses beside sessions). Next:
`tools/rehearse_pipeline.sh --parent ~/hackathon_data/hc-ft2 --self` adds each hc-*-self folder's add_eef output to
the FT#2 parent beside the demos (docs/research/09-pipeline-rehearsal.md).

Videos are hard-linked, or remuxed for --trim/--retime-fps (packet copy, no re-encode). See docs/research/06-harvest.md.
"""

from __future__ import annotations

import argparse
import datetime
import fractions
import itertools
import json
import os
import pathlib
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

CONFIG = pathlib.Path(__file__).resolve().parents[1] / "hillclimb" / "config.yaml"
USABLE_FPS = (30, 15, 10, 5)  # dream shared/datasets/content_checks.py:47; validation refuses any other rate
SAMPLE_FRAMES = 48  # validation.py: an episode needs 48 * fps // 30 + 1 frames
KEYFRAME_EVERY = 10  # rollouts.py VIDEO_OPTIONS "g": a head cut must land on a key frame
KIT_FRAME = [360, 640, 3]  # dhk-collect record_height x record_width (configs/metal.yaml)
GRIPPERS = [6, 13]  # left_gripper, right_gripper in the 14-wide [left j0-5, grip, right j0-5, grip] vector
# Station camera name (a serve role, 05 §2) -> the kit embodiment's video key (embodiments/makermods.json).
VIDEO_KEYS = {r: "observation.images.scene" for r in ("scene", "overhead", "head", "top")}
VIDEO_KEYS |= {r: "observation.images.left_wrist" for r in ("wrist_l", "left_wrist", "left")}
VIDEO_KEYS |= {r: "observation.images.right_wrist" for r in ("wrist_r", "right_wrist", "right")}


def _utc(text: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(text).astimezone(datetime.timezone.utc)


def load_attempts(run_dirs: list[pathlib.Path]) -> list[dict]:
    rows = [(run, json.loads(ln)) for run in run_dirs for ln in (run / "attempts.jsonl").read_text().splitlines()
            if ln.strip()]
    return [a | {"id": f"{run.name}/{pathlib.Path(a['dir']).name}"} for run, a in rows]  # id: <run>/<NN-sub-aK>


def load_rollouts(root: pathlib.Path) -> list[dict]:
    """Complete episodes only: `partial` ones stopped recording early (encoder behind, crash), maybe before the
    success the judge saw; `recording` ones are still being finished; `empty` ones hold no frames."""
    out = []
    for info_file in sorted(root.glob("*/rollout.json")):
        info = json.loads(info_file.read_text())
        if info.get("status") == "complete" and (info_file.parent / "meta" / "info.json").exists():
            out.append(info | {"path": info_file.parent, "t0": _utc(info["started_at_utc"])})
    return out


def pair(attempts: list[dict], rollouts: list[dict], slack_s: float = 2.0) -> list[tuple[dict, dict | None, str]]:
    """Each attempt's episode: same task, started inside [started_at - slack, ended_at + slack] (started_at_utc is
    truncated to whole seconds), the same station run number when both know it, the earliest such; each used once.
    TODO(event): the window assumes harness and station share a clock (run the harness on the station box)."""
    used: set[pathlib.Path] = set()
    out = []
    for a in attempts:
        if not a.get("started_at"):
            out.append((a, None, "no wall clock in attempts.jsonl (logged before harvest support)"))
            continue
        lo = _utc(a["started_at"]) - datetime.timedelta(seconds=slack_s)
        hi = _utc(a["ended_at"]) + datetime.timedelta(seconds=slack_s)
        found = [r for r in rollouts if r["path"] not in used and r["task"] == a["prompt"] and lo <= r["t0"] <= hi
                 and (a.get("station_run") is None or r.get("run") == a["station_run"])]
        if found:
            best = min(found, key=lambda r: r["t0"])
            used.add(best["path"])
            out.append((a, best, f"{len(found)} candidate(s)"))
        else:
            out.append((a, None, "no rollout episode with this task in the attempt's time window"))
    return out


def trim_bounds(action: np.ndarray, eps: float = 0.01, margin: int = 5) -> tuple[int, int]:
    """[start, stop) without the idle head and tail: frames whose command stays within eps of the first / last one.
    The head is floored to a key frame so the video can be cut without re-encoding."""
    moving = np.flatnonzero(np.abs(action - action[0]).max(axis=1) > eps)
    settled = np.flatnonzero(np.abs(action - action[-1]).max(axis=1) > eps)
    if not len(moving) or not len(settled):
        return 0, 0
    start = int(max(0, moving[0] - margin) // KEYFRAME_EVERY * KEYFRAME_EVERY)
    return start, int(min(len(action), settled[-1] + 1 + margin))


def remux(src: pathlib.Path, dst: pathlib.Path, start: int, stop: int, fps: int) -> int:
    """Copies packets [start, stop) into a new MP4, re-timed to one frame every 1/fps s; no decode, no encode.
    Rollout videos are VP9 with no B-frames (rollouts.py VIDEO_OPTIONS), so any cut after a key frame is valid."""
    import av  # only needed for --trim / --retime-fps

    tb = fractions.Fraction(1, fps)
    with av.open(str(src)) as inp, av.open(str(dst), "w", options={"movflags": "faststart"}) as out:
        stream = out.add_stream_from_template(inp.streams.video[0], opaque=True)
        n = 0
        packets = (p for p in inp.demux(inp.streams.video[0]) if p.dts is not None)
        for packet in itertools.islice(packets, start, stop):
            if n == 0 and not packet.is_keyframe:
                raise ValueError(f"{src}: frame {start} is not a key frame; cannot cut here without re-encoding")
            packet.stream, packet.time_base = stream, tb
            packet.pts = packet.dts = n
            packet.duration = 1
            out.mux(packet)
            n += 1
    return n


def _stats(values: np.ndarray) -> dict:
    v = values.astype(np.float64).reshape(len(values), -1)
    return {"min": v.min(0).tolist(), "max": v.max(0).tolist(), "mean": v.mean(0).tolist(),
            "std": v.std(0).tolist(), "count": [len(v)]}


def build(prompt: str, episodes: list[dict], out: pathlib.Path, *, grip_scale: float = 100.0, trim: bool = False,
          retime_fps: int | None = None) -> list[tuple[dict, int | None, str]]:
    """Writes one LeRobot v2.1 dataset of `episodes` (rollout records) under task `prompt`. Returns per episode its
    new index (None = skipped) and why. Episodes must share the first one's fps, cameras and frame sizes."""
    if not episodes:
        return []
    ref = json.loads((episodes[0]["path"] / "meta" / "info.json").read_text())
    fps_in = float(ref["fps"])
    fps = int(retime_fps or fps_in)
    if fps not in USABLE_FPS or (retime_fps is None and fps != fps_in):
        raise SystemExit(f"rollouts are {fps_in:g} fps; Dream training takes only {USABLE_FPS}. Record with "
                         "control_hz: 30 in the station file, or pass --retime-fps 30 (see 06-harvest.md)")
    cams = {k: v for k, v in ref["features"].items() if v.get("dtype") == "video"}
    # TODO(event): read the station file's camera names; an unknown one keeps its key and validation names it.
    rename = {k: VIDEO_KEYS.get(k.removeprefix("observation.images."), k) for k in cams}
    if odd := {k: v["shape"] for k, v in cams.items() if v["shape"] != KIT_FRAME}:
        print(f"WARNING {out.name}: rollout frames {odd} differ from the demos' {KIT_FRAME}; Dream accepts it but "
              "the model sees two aspect ratios. Set the station cameras to 16:9 (06-harvest.md)")
    out.mkdir(parents=True)
    report, lengths, ep_stats, total = [], [], [], 0
    for ep in episodes:
        info = json.loads((ep["path"] / "meta" / "info.json").read_text())
        shapes = {k: v["shape"] for k, v in info["features"].items() if v.get("dtype") == "video"}
        if float(info["fps"]) != fps_in or shapes != {k: v["shape"] for k, v in cams.items()}:
            report.append((ep, None, f"fps/cameras differ from {episodes[0]['path'].name}"))
            continue
        table = pq.read_table(ep["path"] / "data/chunk-000/episode_000000.parquet")
        state, action = (np.asarray(table.column(k).to_pylist(), np.float32) for k in ("observation.state", "action"))
        if state.shape[1] != 14 or action.shape[1] != 14:
            report.append((ep, None, f"vectors are {state.shape[1]}/{action.shape[1]} wide, not 14 (6+1 per arm)"))
            continue
        start, stop = trim_bounds(action) if trim else (0, len(action))
        if stop - start < SAMPLE_FRAMES * fps // 30 + 1:
            report.append((ep, None, f"too short after trim: {stop - start} frames at {fps} fps"))
            continue
        state, action = state[start:stop].copy(), action[start:stop].copy()
        state[:, GRIPPERS] *= grip_scale  # robots records the gripper 0..1; dhk-collect the Metal jaw stroke in mm
        action[:, GRIPPERS] *= grip_scale
        i, n = len(lengths), stop - start
        frame = np.arange(n, dtype=np.int64)
        cols = {"timestamp": (frame / fps).astype(np.float32), "frame_index": frame,
                "episode_index": np.full(n, i, np.int64), "index": frame + total, "task_index": np.zeros(n, np.int64)}
        data = out / f"data/chunk-000/episode_{i:06d}.parquet"
        data.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({
            "observation.state": pa.FixedSizeListArray.from_arrays(pa.array(state.reshape(-1)), 14),
            "action": pa.FixedSizeListArray.from_arrays(pa.array(action.reshape(-1)), 14),
            **{k: pa.array(v) for k, v in cols.items()}}), data)
        src_stats = json.loads((ep["path"] / "meta/episodes_stats.jsonl").read_text().splitlines()[0])["stats"]
        stats = {"observation.state": _stats(state), "action": _stats(action)} | {k: _stats(v) for k, v in cols.items()}
        for key, new_key in rename.items():
            src = ep["path"] / f"videos/chunk-000/{key}/episode_000000.mp4"
            dst = out / f"videos/chunk-000/{new_key}/episode_{i:06d}.mp4"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if trim or retime_fps:
                if remux(src, dst, start, stop, fps) != n:
                    raise SystemExit(f"{src}: fewer video frames than parquet rows")
            else:
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
            stats[new_key] = src_stats[key]  # TODO(event): image stats of the untrimmed episode; fine for normalising
        lengths.append(n)
        ep_stats.append({"episode_index": i, "stats": stats})
        total += n
        report.append((ep, i, f"frames {start}:{stop} of {ep['frames']}"))
    if not lengths:  # an empty dataset would only fail validation
        shutil.rmtree(out)
        return report
    features = {k: v for k, v in ref["features"].items() if not k.startswith("rollout.") and k not in cams}
    for key in ("observation.state", "action"):
        features[key] = {"dtype": "float32", "shape": [14], "names": ref["features"][key]["names"]}
    for key, new_key in rename.items():
        features[new_key] = cams[key] | {"info": cams[key]["info"] | {"video.fps": fps}}
    meta = out / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps(ref | {
        "fps": fps, "total_episodes": len(lengths), "total_frames": total, "total_tasks": 1,
        "total_videos": len(lengths) * len(cams), "total_chunks": 1, "splits": {"train": f"0:{len(lengths)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features}, indent=2) + "\n")
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": prompt}) + "\n")
    (meta / "episodes.jsonl").write_text("".join(
        json.dumps({"episode_index": i, "tasks": [prompt], "length": n}) + "\n" for i, n in enumerate(lengths)))
    (meta / "episodes_stats.jsonl").write_text("".join(json.dumps(s) + "\n" for s in ep_stats))
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("runs", nargs="+", type=pathlib.Path, help="harness run folders holding attempts.jsonl")
    p.add_argument("--rollouts", type=pathlib.Path, default=pathlib.Path("~/.dreamlabs/rollouts"))
    p.add_argument("--out", type=lambda x: pathlib.Path(x).expanduser(), required=True, help="<out>/hc-<sub>-self")
    p.add_argument("--config", type=pathlib.Path, default=CONFIG, help="frozen subtask prompts")
    p.add_argument("--confirmed", type=pathlib.Path,
                   help="human-confirmed attempt ids (<run>/<NN-subtask-aK>), one per line: keep ONLY these")
    p.add_argument("--trim", action="store_true", help="cut idle head/tail (remux, no re-encode)")
    p.add_argument("--retime-fps", type=int, choices=USABLE_FPS, help="relabel one tick per frame at this rate")
    p.add_argument("--grip-scale", type=float, default=100.0, help="gripper 0..1 -> kit units (Metal: 100 mm)")
    p.add_argument("--slack-s", type=float, default=2.0)
    p.add_argument("--replace", action="store_true", help="rebuild output folders this tool wrote before")
    args = p.parse_args(argv)
    prompts = {s["name"]: s["prompt"] for s in yaml.safe_load(args.config.read_text())["subtasks"]}
    confirmed = {ln.strip() for ln in args.confirmed.read_text().splitlines() if ln.strip() and ln[0] != "#"} \
        if args.confirmed else None
    pairs = pair(load_attempts(args.runs), load_rollouts(args.rollouts.expanduser()), args.slack_s)
    args.out.mkdir(parents=True, exist_ok=True)
    log = []
    for name, prompt in prompts.items():
        keep = []
        for a, r, why in (x for x in pairs if x[0]["subtask"] == name):
            ok = a["id"] in confirmed if confirmed is not None else a["success"]
            if r is not None and ok:
                keep.append(r)
            log.append({"attempt": a["id"], "subtask": name, "success": a["success"], "keep": bool(r and ok),
                        "rollout": str(r["path"]) if r else None, "pairing": why})
        target = args.out / f"hc-{name.replace('_', '-')}-self"
        if target.exists():
            if not (args.replace and (target / "meta" / "info.json").exists()):
                raise SystemExit(f"{target} exists; pass --replace to rebuild it")
            shutil.rmtree(target)
        built = build(prompt, keep, target, grip_scale=args.grip_scale, trim=args.trim, retime_fps=args.retime_fps)
        log += [{"rollout": str(ep["path"]), "subtask": name, "episode": i, "build": why} for ep, i, why in built]
        print(f"{name}: {sum(i is not None for _, i, _ in built)} episodes -> {target if target.exists() else '-'}")
        if not target.exists() and args.replace:
            print(f"  (no episodes kept: remove any stale {target.name}-eef from the FT parent)")
    (args.out / "harvest.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
