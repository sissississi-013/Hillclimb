"""Replay primitives: deterministic joint motions to a canonical pre-contact pose, before (or instead of) Dream.

A primitive is a joint trajectory per arm plus gripper (0 closed .. 1 open) at `fps` (tools/waypoints.py makes one).
play() writes FollowerArm targets at a fixed rate and holds at the end, as robots/home.py:69-120 does: a straight
joint-space blend from the measured pose to the first target, then the trajectory time-scaled by speed_scale <= 1 and
stretched so no joint exceeds the cap, then a per-tick rate limit + optional `limits` clamp as a backstop. The cap
is OUR conservative clamp, min(max_rad_s, 0.5 rad/s = the station's ramp_max_rad_s default, robots/config.py:587);
the drivers still clamp to the Metal soft joint limits (robots/motor_arm.py:345-350).
Refuses without HILLCLIMB_ALLOW_ARMS=1; dry by default (reads the arms and plans, never writes).
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import time
from typing import Any, Callable

import numpy as np
import yaml

SIDES = ("left", "right")
HARD_MAX_RAD_S = 0.5  # never faster than this, whatever the caller asks
MAX_GRIP_PER_S = 1.0  # full open<->close in >= 1 s
MIN_APPROACH_S = 1.0
MAX_APPROACH_RAD = 1.0  # TODO(event): refuse a first target this far from the measured pose (wrong file / arm swap?)


@dataclasses.dataclass
class Primitive:
    fps: float
    joints: dict[str, np.ndarray]  # side -> [T, 6] radians
    gripper: dict[str, np.ndarray]  # side -> [T] in [0, 1]
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)

    def vector(self) -> np.ndarray:
        """[T, 14]: left j0-5, left grip, right j0-5, right grip (the MakerMods order, layouts.py SIDE_ORDER)."""
        return np.concatenate([np.column_stack([self.joints[s], self.gripper[s]]) for s in SIDES], axis=1)


def save(p: Primitive, path: str | os.PathLike[str]) -> None:  # YAML: small, diffable, hand-editable
    doc = {"fps": float(p.fps), "meta": p.meta,
           **{s: {"joints": np.round(p.joints[s], 5).tolist(), "gripper": np.round(p.gripper[s], 4).tolist()}
              for s in SIDES}}
    pathlib.Path(path).write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=None))


def load(path: str | os.PathLike[str]) -> Primitive:
    path = pathlib.Path(path).expanduser()
    d = yaml.safe_load(path.read_text())
    p = Primitive(float(d["fps"]), {s: np.asarray(d[s]["joints"], float).reshape(-1, 6) for s in SIDES},
                  {s: np.asarray(d[s]["gripper"], float).reshape(-1) for s in SIDES}, d.get("meta") or {})
    n = {len(p.joints[s]) for s in SIDES} | {len(p.gripper[s]) for s in SIDES}
    v = p.vector()
    if len(n) != 1 or not len(v) or not np.isfinite(v).all() or p.fps <= 0:
        raise ValueError(f"{path}: needs fps > 0 and the same number (>= 1) of finite rows per arm and gripper")
    if (v[:, [6, 13]] < 0).any() or (v[:, [6, 13]] > 1).any():
        raise ValueError(f"{path}: gripper outside [0, 1]; extract with the right --grip-scale")
    return p


def plan(p: Primitive, start: np.ndarray, *, hz: float = 40.0, speed_scale: float = 1.0,
         max_rad_s: float = HARD_MAX_RAD_S, limits: dict[str, list] | None = None,
         max_approach_rad: float = MAX_APPROACH_RAD) -> np.ndarray:
    """The [N, 14] commands, one per 1/hz tick, from `start` [14] through the primitive. Raises on anything unsafe."""
    if not 0 < speed_scale <= 1:
        raise ValueError(f"speed_scale {speed_scale} must be in (0, 1]")
    cap = min(float(max_rad_s), HARD_MAX_RAD_S)
    if cap <= 0:
        raise ValueError("max_rad_s must be > 0")
    lo, hi = np.full(14, -np.inf), np.full(14, np.inf)
    lo[[6, 13]], hi[[6, 13]] = 0.0, 1.0
    for i, s in enumerate(SIDES):
        for j, (a, b) in enumerate((limits or {}).get(s) or []):
            lo[7 * i + j], hi[7 * i + j] = a, b
    joints = np.r_[0:6, 7:13]
    traj = np.clip(p.vector(), lo, hi)
    start = np.asarray(start, float)
    bad = np.flatnonzero((start[joints] < lo[joints] - 1e-6) | (start[joints] > hi[joints] + 1e-6))
    if bad.size:  # clamping back inside would be one unrate-limited step (PR #2 review)
        raise ValueError(f"measured pose outside the primitive's joint limits at indices {joints[bad].tolist()}; refusing")
    gap = float(np.abs(traj[0, joints] - start[joints]).max())
    if gap > max_approach_rad:
        raise ValueError(f"first target is {gap:.2f} rad from the measured pose (> {max_approach_rad}); refusing")
    # 1) straight blend to the first target
    n = max(1, int(np.ceil(hz * max(MIN_APPROACH_S, gap / cap,
                                    float(np.abs(traj[0, [6, 13]] - start[[6, 13]]).max()) / MAX_GRIP_PER_S))))
    alpha = (np.arange(1, n + 1) / n)[:, None]
    out = [start + alpha * (traj[0] - start)]
    # 2) the trajectory, time-scaled, stretched so its peak joint speed stays under the cap
    if len(traj) > 1:
        rate = p.fps * speed_scale
        peak = float(np.abs(np.diff(traj[:, joints], axis=0)).max()) * rate
        grip_peak = float(np.abs(np.diff(traj[:, [6, 13]], axis=0)).max()) * rate  # intermediate grasps (PR #2 review)
        rate /= max(1.0, peak / cap * 1.05, grip_peak / MAX_GRIP_PER_S * 1.05)
        t_src = np.arange(len(traj)) / rate
        t = np.arange(1, int(np.ceil(t_src[-1] * hz)) + 1) / hz
        out.append(np.column_stack([np.interp(t, t_src, traj[:, k]) for k in range(14)]))
    # 3) backstop: per-tick rate limit and clamp, whatever the steps above produced
    step = np.full(14, cap / hz)
    step[[6, 13]] = MAX_GRIP_PER_S / hz
    cmds, cur = [], start.copy()
    for goal in np.concatenate(out):
        cur = np.clip(cur + np.clip(goal - cur, -step, step), lo, hi)
        cmds.append(cur.copy())
    # 4) settle: the rate limit can leave the end short (e.g. a fast gripper close); keep stepping to the final target
    #    so a grasp/release actually completes (PR #2 review). Bounded: at most the time a full-range move needs.
    final = np.clip(traj[-1], lo, hi)
    for _ in range(int(np.ceil(hz * max(1.0 / MAX_GRIP_PER_S, max_approach_rad / cap))) + 1):
        if np.allclose(cur, final, atol=1e-6):
            break
        cur = np.clip(cur + np.clip(final - cur, -step, step), lo, hi)
        cmds.append(cur.copy())
    return np.asarray(cmds)


def _read(arms: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    st = {s: arms[s].read() for s in SIDES}
    if any(np.shape(x.positions_rad) != (6,) for x in st.values()):
        raise ValueError("primitives are for 6-joint arms")
    return np.concatenate([np.r_[st[s].positions_rad, min(max(st[s].gripper, 0.0), 1.0)] for s in SIDES]), st


def play(p: Primitive, arms: dict[str, Any], *, dry_run: bool = True, hz: float = 40.0,
         command: Callable[..., Any] | None = None, clock: Callable[[], float] = time.monotonic,
         sleep: Callable[[float], Any] = time.sleep, **plan_kw: Any) -> dict[str, Any]:
    """Plays `p` on `arms` ({"left","right"} -> robots FollowerArm). Dry: reads and plans only. `plan_kw`: plan()'s."""
    if os.environ.get("HILLCLIMB_ALLOW_ARMS") != "1":
        raise RuntimeError("primitives read the real arms: set HILLCLIMB_ALLOW_ARMS=1 with a hand on the e-stop")
    start, _ = _read(arms)
    cmds = plan(p, start, hz=hz, **plan_kw)
    peak = float(np.abs(np.diff(np.vstack([start, cmds])[:, np.r_[0:6, 7:13]], axis=0)).max()) * hz
    if not dry_run:
        if command is None:
            from robots.types import ArmCommand as command  # station only: ~/dream-env
        t0 = clock()
        try:
            for i, row in enumerate(cmds):
                for k, s in enumerate(SIDES):
                    arms[s].write(command(positions_rad=row[7 * k:7 * k + 6].copy(), gripper=float(row[7 * k + 6])))
                sleep(max(0.0, t0 + (i + 1) / hz - clock()))
        finally:
            for s in SIDES:  # hold the last accepted target, as home.py does, even after a fault
                try:
                    arms[s].hold()
                except Exception:  # noqa: BLE001 - the other arm must still hold
                    pass
    _, st = _read(arms)
    last = cmds[-1]
    proprio = {s[0]: {"grip": float(st[s].gripper), "cmd_grip": float(last[7 * k + 6]),
                      "max_track_err_rad": round(float(np.abs(last[7 * k:7 * k + 6] - st[s].positions_rad).max()), 4)}
               for k, s in enumerate(SIDES)}
    return {"ticks": len(cmds), "duration_s": round(len(cmds) / hz, 3), "dry_run": dry_run,
            "peak_rad_s": round(peak, 4), "proprio": proprio}
