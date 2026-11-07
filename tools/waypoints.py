"""Make replay primitives (hillclimb/primitives.py) from a recorded LeRobot v2.1 episode or the arms' current pose.

    python tools/waypoints.py extract ~/hackathon_data/hc-plug --episode 3 --frames 120:180 --out prims/plug_pre.yaml
    python tools/waypoints.py extract ~/hackathon_data/hc-plug --episode 3 --frame 150 --out prims/plug_hover.yaml
    HILLCLIMB_ALLOW_ARMS=1 python tools/waypoints.py record --out prims/plug_hover.yaml   # arms opened read only

dhk-collect Metal layout (dream_hackathon/layouts.py): 14-wide [left j0-5, grip, right j0-5, grip], joints in rad,
gripper in mm (100 = open). Rollouts record the gripper 0..1: pass --grip-scale 1. Extract before add_eef.py.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from hillclimb import primitives  # noqa: E402


def extract(dataset: pathlib.Path, episode: int, start: int, stop: int | None, *, column: str = "action",
            grip_scale: float = 100.0, every: int = 1) -> primitives.Primitive:
    """Frames [start, stop) of one episode (stop=None: the single frame `start`), every `every`-th row."""
    dataset = pathlib.Path(dataset).expanduser()
    info = json.loads((dataset / "meta" / "info.json").read_text())
    rel = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    chunk = episode // int(info.get("chunks_size", 1000))
    table = pq.read_table(dataset / rel.format(episode_chunk=chunk, episode_index=episode))
    rows = np.asarray(table.column(column).to_pylist(), float)
    if rows.ndim != 2 or rows.shape[1] != 14:
        raise SystemExit(f"{column} is {rows.shape[1:]} wide, not 14 (6 joints + gripper per arm)")
    stop = start + 1 if stop is None else stop
    if not 0 <= start < stop <= len(rows):
        raise SystemExit(f"frames {start}:{stop} outside the episode's {len(rows)} frames")
    seg = rows[start:stop:every].copy()
    seg[:, [6, 13]] /= grip_scale
    fps = float(info["fps"]) / every
    return primitives.Primitive(fps, {"left": seg[:, 0:6], "right": seg[:, 7:13]},
                                {"left": seg[:, 6], "right": seg[:, 13]},
                                {"source": str(dataset), "episode": episode, "frames": [start, stop],
                                 "column": column, "every": every, "grip_scale": grip_scale})


def record(station_file: str) -> primitives.Primitive:
    """The arms' measured pose now, through the station's own classes, the arms opened read only (never powered)."""
    if os.environ.get("HILLCLIMB_ALLOW_ARMS") != "1":
        raise SystemExit("record opens the real arms (read only): set HILLCLIMB_ALLOW_ARMS=1")
    from robots import arms_lock, bodies  # station only: ~/dream-env

    from hillclimb.client_loop import RobotsStation
    st = RobotsStation(station_file)
    lock = arms_lock.acquire(st.path.parent / arms_lock.LOCK_NAME, arms_lock.ROBOT_RUN)  # refuses while `up` runs
    arms = {}
    try:
        body = bodies.for_config(st.app)
        if body.open_arm_read_only is None:
            raise SystemExit("this body has no read-only open; TODO(event): record via a dry --backend client run")
        for name, cfg in (("left", st.app.follower_l), ("right", st.app.follower_r)):
            arms[name] = body.open_arm_read_only(cfg, name)
        states = {s: arms[s].read() for s in primitives.SIDES}
    finally:
        for arm in arms.values():
            arm.close()  # read-only arms were never switched on (robots/run.py:285-291)
        lock.close()
    return primitives.Primitive(1.0, {s: np.asarray([states[s].positions_rad], float) for s in states},
                                {s: np.asarray([min(max(states[s].gripper, 0.0), 1.0)]) for s in states},
                                {"source": "record", "station_file": str(station_file)})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("dataset", type=pathlib.Path)
    e.add_argument("--episode", type=int, required=True)
    g = e.add_mutually_exclusive_group(required=True)
    g.add_argument("--frames", help="start:stop (stop exclusive)")
    g.add_argument("--frame", type=int, help="one waypoint")
    e.add_argument("--column", choices=["action", "observation.state"], default="action",
                   help="action = the commanded targets (replay); observation.state = where the arm really was")
    e.add_argument("--grip-scale", type=float, default=100.0, help="gripper -> 0..1 (Metal mm: 100; rollouts: 1)")
    e.add_argument("--every", type=int, default=1, help="keep every N-th frame")
    r = sub.add_parser("record")
    r.add_argument("--station-file", default="~/.dreamlabs/station.yaml")
    for q in (e, r):
        q.add_argument("--out", type=pathlib.Path, required=True, help="primitive .yaml")
    a = p.parse_args(argv)
    if a.cmd == "extract":
        lo, hi = (int(x) for x in a.frames.split(":")) if a.frames else (a.frame, None)
        prim = extract(a.dataset, a.episode, lo, hi, column=a.column, grip_scale=a.grip_scale, every=a.every)
    else:
        prim = record(a.station_file)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    primitives.save(prim, a.out)
    primitives.load(a.out)  # validates: gripper in [0, 1], same rows per arm
    print(f"{a.out}: {len(prim.joints['left'])} rows at {prim.fps:g} fps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
