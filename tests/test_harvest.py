"""tools/harvest.py on tiny synthetic rollout folders laid out like robots/rollout_episode.py finish() writes them."""

import datetime
import fractions
import importlib.util
import json
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("harvest", ROOT / "tools" / "harvest.py")
harvest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harvest)

T0 = datetime.datetime(2026, 11, 7, 18, 0, 0, tzinfo=datetime.timezone.utc)
PLUG, OPEN = "plug the new usb cable into the rack", "open the rack back panel"
CAMS = ("scene", "wrist_l", "wrist_r")
EMBODIMENT_VIDEO = {"observation.images.scene", "observation.images.left_wrist", "observation.images.right_wrist"}


def iso(s):
    return (T0 + datetime.timedelta(seconds=s)).isoformat(timespec="milliseconds")


def names():
    return [f"{a}_{p}" for a in ("left", "right") for p in [*(f"joint_{i}" for i in range(6)), "gripper"]]


def make_rollout(root, name, task, start_s, run, frames=60, fps=30.0, status="complete", video=None):
    """One episode folder as the robots package writes it: list<float> vectors, rollout.* columns, VP9 videos."""
    ep = root / name
    (ep / "data/chunk-000").mkdir(parents=True)
    t = np.arange(frames)
    vec = np.zeros((frames, 14), np.float32)
    vec[:, 0] = np.clip((t - 15) * 0.02, 0, None)  # idle for 15 ticks, then moves
    vec[frames - 12:, 0] = vec[frames - 13, 0]  # idle tail
    vec[:, 6] = vec[:, 13] = 0.5  # gripper half open, in 0..1
    cols = {"observation.state": vec, "action": vec, "timestamp": (t / fps).astype(np.float32), "frame_index": t,
            "episode_index": np.zeros(frames, np.int64), "index": t, "task_index": np.zeros(frames, np.int64),
            "rollout.t_monotonic_s": t * 0.05, "rollout.chunk": t // 30, "rollout.row": t % 30}
    pq.write_table(pa.table({k: pa.array(list(v)) if v.ndim > 1 else pa.array(v) for k, v in cols.items()}),
                   ep / "data/chunk-000/episode_000000.parquet")
    features = {k: {"dtype": str(np.asarray(v).dtype), "shape": [14 if np.ndim(v) > 1 else 1],
                    "names": names() if np.ndim(v) > 1 else None} for k, v in cols.items()}
    stats = {}
    for cam in CAMS:
        key = f"observation.images.{cam}"
        path = ep / f"videos/chunk-000/{key}/episode_000000.mp4"
        path.parent.mkdir(parents=True)
        video(path, frames, fps) if video else path.write_bytes(b"mp4:" + cam.encode())
        features[key] = {"dtype": "video", "shape": [64, 64, 3], "names": ["height", "width", "channels"],
                         "info": {"video.fps": fps, "video.codec": "vp9", "video.height": 64, "video.width": 64}}
        stats[key] = {"mean": [[[0.5]]] * 3, "count": [1]}
    (ep / "meta").mkdir()
    (ep / "meta/info.json").write_text(json.dumps({"codebase_version": "v2.1", "robot_type": "metal", "fps": fps,
                                                   "total_episodes": 1, "features": features}))
    (ep / "meta/episodes_stats.jsonl").write_text(json.dumps({"episode_index": 0, "stats": stats}) + "\n")
    (ep / "rollout.json").write_text(json.dumps({
        "status": status, "task": task, "run": run, "fps": fps, "cameras": list(CAMS), "frames": frames,
        "started_at_utc": (T0 + datetime.timedelta(seconds=start_s)).isoformat(timespec="seconds")}))
    return ep


def make_run(logs, name, attempts):
    run = logs / name
    run.mkdir(parents=True)
    rows = [{"subtask": s, "attempt": k, "success": ok, "dir": str(run / f"{i:02d}-{s}-a{k}"), "prompt": p,
             "started_at": iso(a), "ended_at": iso(b), "station_run": r} for i, (s, k, ok, p, a, b, r) in
            enumerate(attempts)]
    (run / "attempts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run


@pytest.fixture
def station(tmp_path):
    ro = tmp_path / "rollouts"
    make_rollout(ro, "a", PLUG, 0.6, run=1)  # pairs with attempt 1 (start truncated to whole second, in slack)
    make_rollout(ro, "b", PLUG, 40, run=2)  # attempt 2: failed verdict
    make_rollout(ro, "c", PLUG, 80, run=3, frames=70)  # attempt 3
    make_rollout(ro, "d", OPEN, 120, run=4)  # attempt 4, other subtask
    make_rollout(ro, "e", PLUG, 200, run=5, status="recording")  # not finished: ignored
    make_rollout(ro, "f", PLUG, 201, run=5, status="partial")  # stopped recording early: ignored
    run = make_run(tmp_path / "logs", "r1", [
        ("plug_usb", 1, True, PLUG, 0.9, 30, 1), ("plug_usb", 2, False, PLUG, 40, 70, 2),
        ("plug_usb", 3, True, PLUG, 79.5, 110, 3), ("open_backing", 1, True, OPEN, 119, 150, 4),
        ("plug_usb", 4, True, PLUG, 199, 230, 5)])
    return tmp_path, ro, run


def test_pair_by_task_window_and_run(tmp_path):
    ro = tmp_path / "ro"
    make_rollout(ro, "x", PLUG, 10, run=1)
    make_rollout(ro, "y", OPEN, 10, run=1)
    make_rollout(ro, "z", PLUG, 100, run=9)
    rollouts = harvest.load_rollouts(ro)
    a = {"prompt": PLUG, "started_at": iso(9), "ended_at": iso(40), "station_run": 1}
    pairs = harvest.pair([a, dict(a, station_run=2), {"prompt": PLUG}, dict(a, started_at=iso(99), ended_at=iso(130))],
                         rollouts)
    assert pairs[0][1]["path"].name == "x"
    assert pairs[1][1] is None  # x already used, and the run number would not match anyway
    assert pairs[2][1] is None and "wall clock" in pairs[2][2]
    assert pairs[3][1] is None  # z is in the window but from another station run number


def test_harvest_successes_into_one_dataset_per_subtask(station):
    tmp, ro, run = station
    out = tmp / "out"
    assert harvest.main([str(run), "--rollouts", str(ro), "--out", str(out)]) == 0
    ds = out / "hc-plug-usb-self"
    info = json.loads((ds / "meta/info.json").read_text())
    eps = [json.loads(x) for x in (ds / "meta/episodes.jsonl").read_text().splitlines()]
    assert [e["episode_index"] for e in eps] == [0, 1] and [e["length"] for e in eps] == [60, 70]
    assert all(e["tasks"] == [PLUG] for e in eps)
    assert (ds / "meta/tasks.jsonl").read_text() == json.dumps({"task_index": 0, "task": PLUG}) + "\n"
    assert info["total_episodes"] == 2 and info["total_frames"] == 130 and info["fps"] == 30
    # add_eef.py preconditions: v2.1 and 14-wide state/action; dream validation: the embodiment's video keys
    assert info["codebase_version"] == "v2.1"
    assert {k: info["features"][k]["shape"] for k in ("observation.state", "action")} == \
        {"observation.state": [14], "action": [14]}
    assert {k for k, v in info["features"].items() if v["dtype"] == "video"} == EMBODIMENT_VIDEO
    assert not any(k.startswith("rollout.") for k in info["features"])
    t = pq.read_table(ds / "data/chunk-000/episode_000001.parquet")
    assert not any(c.startswith("rollout.") for c in t.column_names)
    assert pa.types.is_fixed_size_list(t.schema.field("action").type)
    assert t.column("episode_index").to_pylist() == [1] * 70 and t.column("index").to_pylist()[0] == 60
    act = np.asarray(t.column("action").to_pylist())
    assert act[0, 6] == pytest.approx(50.0) and act[0, 13] == pytest.approx(50.0)  # gripper 0..1 -> mm
    assert (ds / "videos/chunk-000/observation.images.left_wrist/episode_000001.mp4").read_bytes() == b"mp4:wrist_l"
    stats = [json.loads(x) for x in (ds / "meta/episodes_stats.jsonl").read_text().splitlines()]
    assert [s["episode_index"] for s in stats] == [0, 1] and "observation.images.scene" in stats[0]["stats"]
    assert json.loads((out / "hc-open-backing-self/meta/info.json").read_text())["total_episodes"] == 1
    assert not (out / "hc-unplug-usb-self").exists()
    log = [json.loads(x) for x in (out / "harvest.jsonl").read_text().splitlines()]
    assert {x["attempt"] for x in log if "attempt" in x and x["rollout"] is None} == {"r1/04-plug_usb-a4"}
    with pytest.raises(SystemExit):  # never overwrites without --replace
        harvest.main([str(run), "--rollouts", str(ro), "--out", str(out)])


def test_confirmed_list_is_authoritative(station):
    tmp, ro, run = station
    ok = tmp / "ok.txt"
    ok.write_text("# curator playback\nr1/01-plug_usb-a2\n")
    harvest.main([str(run), "--rollouts", str(ro), "--out", str(tmp / "o"), "--confirmed", str(ok)])
    eps = (tmp / "o/hc-plug-usb-self/meta/episodes.jsonl").read_text().splitlines()
    assert len(eps) == 1 and json.loads(eps[0])["length"] == 60
    assert not (tmp / "o/hc-open-backing-self").exists()


def test_refuses_20fps_without_retime(tmp_path):
    ro = tmp_path / "ro"
    make_rollout(ro, "a", PLUG, 0, run=1, fps=20.0)
    run = make_run(tmp_path / "logs", "r", [("plug_usb", 1, True, PLUG, 0, 30, 1)])
    with pytest.raises(SystemExit, match="20 fps"):
        harvest.main([str(run), "--rollouts", str(ro), "--out", str(tmp_path / "o")])


def test_trim_bounds():
    a = np.zeros((100, 14))
    a[30:70, 0] = np.linspace(0, 1, 40)
    a[70:, 0] = 1
    assert harvest.trim_bounds(a) == (20, 74)  # head 31-5=26 floored to a key frame; tail 68+1+5
    assert harvest.trim_bounds(np.zeros((50, 14))) == (0, 0)


def vp9(path, frames, fps):
    """A video written the way robots/rollouts.py writes one (VP9, fragmented MP4, key frame every 10), finished."""
    av = pytest.importorskip("av")
    with av.open(str(path), "w", format="mp4") as c:
        s = c.add_stream("libvpx-vp9", rate=fractions.Fraction(fps).limit_denominator(1001),
                         options={"deadline": "realtime", "lag-in-frames": "0", "auto-alt-ref": "0", "g": "10"})
        s.width = s.height = 64
        s.pix_fmt = "yuv420p"
        for i in range(frames):
            f = av.VideoFrame.from_ndarray(np.full((64, 64, 3), i * 3 % 255, np.uint8), format="rgb24")
            f.pts = i
            c.mux(s.encode(f))
        c.mux(s.encode(None))


def test_retime_and_trim_remux_without_reencode(tmp_path):
    av = pytest.importorskip("av")
    ro = tmp_path / "ro"
    make_rollout(ro, "a", PLUG, 0, run=1, fps=20.0, frames=80, video=vp9)
    run = make_run(tmp_path / "logs", "r", [("plug_usb", 1, True, PLUG, 0, 30, 1)])
    harvest.main([str(run), "--rollouts", str(ro), "--out", str(tmp_path / "o"), "--trim", "--retime-fps", "30"])
    ds = tmp_path / "o/hc-plug-usb-self"
    info = json.loads((ds / "meta/info.json").read_text())
    n = json.loads((ds / "meta/episodes.jsonl").read_text())["length"]
    assert info["fps"] == 30 and info["features"]["observation.images.scene"]["info"]["video.fps"] == 30
    assert n == 72 - 10  # head: motion at 16, minus margin 5, floored to key frame 10; tail: last motion 66 + 1 + 5
    with av.open(str(ds / "videos/chunk-000/observation.images.scene/episode_000000.mp4")) as c:
        st = c.streams.video[0]
        times = [float(p.pts * p.time_base) for p in c.demux(st) if p.dts is not None]
    assert len(times) == n and np.allclose(np.diff(times), 1 / 30, atol=1e-3) and times[0] == 0
