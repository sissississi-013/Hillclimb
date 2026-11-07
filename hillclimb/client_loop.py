"""ClientLoopBackend: our own episode loop on `dream.DreamClient`, for the knobs the station hides (05 doc §4, §7).

Ours: the per-subtask policy config (`execution_horizon`, `ik` keep_out/region/solve, `sync_above_ms`), a speed scale,
the tick loop (one `ControlLoop.step()` per tick, a stop condition checked after each), and PrefixPacer, which keeps
contact subtasks re-planning every `execution_horizon` rows even when the round trip is slow (the client's own pacer
goes sync and runs all 48 rows then; dream/pacing.py:24-29).
The station's, imported only on the station (~/dream-env): config loading, arm drivers with their soft joint limits
(robots/motor_arm.py:345-350), cameras, IK warm-up, and `ControlLoop.step()` itself. step() carries the ramp, the
EE-speed clamp, the gripper step, the stall and first-chunk limits and SafetyStop, and the loop's exit holds or
disengages the arms (robots/control.py:860-911, 1000-1078). Like `dreamcli robot run`, a dry run opens the arms read
only and wraps them in `robots.run.DryRun` so no target reaches a motor, and nothing homes after an arm fault
(robots/run.py:118-178, 241-243).
Every hardware call goes through `Station`, so tests pass a fake station and a fake client.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import math
import os
import pathlib
import threading
import time
from collections import deque
from typing import Any, Callable, Protocol

from hillclimb import backends, primitives

STATION_FILE = pathlib.Path("~/.dreamlabs/station.yaml").expanduser()  # robots/cli.py:40, 107
TRACE = pathlib.Path("~/.dreamlabs/hillclimb-client.csv").expanduser()  # our own, so the station's trace is kept
DEFAULTS: dict[str, Any] = {"execution_horizon": 30, "always_async": False, "speed_scale": 1.0,
                            "sync_above_ms": None, "ik": {}}
MIN_SPEED_SCALE = 0.5  # below this a tick outlasts the command thread's 0.1 s interpolation (control.py:82)


@dataclasses.dataclass
class Tick:
    """What the stop condition sees after each step: the loop's own snapshots (control.py:530-582)."""
    n: int
    t_s: float
    state: Any  # robots.types.RobotState: per arm positions_rad, gripper
    video: dict[str, Any]  # camera name -> robots.types.Frame (with .jpeg)
    snapshot: Any  # robots.control.TickSnapshot or None (a hold tick)


class Station(Protocol):
    def open(self, dry_run: bool, dream_config: dict[str, Any]) -> None: ...  # cameras, arms; live: home first
    def policy_config(self, checkpoint: str | None) -> dict[str, Any]: ...  # robots.run.policy_config
    def loop_settings(self) -> dict[str, Any]: ...  # robots.run.loop_settings
    def open_client(self, dream_config: dict[str, Any]) -> Any: ...  # DreamClient with its policy open
    def make_loop(self, client: Any, task: str, settings: dict[str, Any]) -> Any: ...  # async ctx, step(now_s)
    def home(self) -> bool: ...
    def close(self) -> None: ...
    arms: dict[str, Any]  # after open(): "left"/"right" -> FollowerArm (DryRun-wrapped when dry)
    readers: list[Any]  # after open(): the camera readers


def client_settings(cfg: dict[str, Any], sub: dict[str, Any] | None) -> dict[str, Any]:
    """DEFAULTS <- defaults.client <- subtask.client. `ik` merges one level deep."""
    out = copy.deepcopy(DEFAULTS)
    for layer in ((cfg.get("defaults") or {}).get("client") or {}, ((sub or {}).get("client") or {})):
        for k, v in layer.items():
            out[k] = {**out[k], **(v or {})} if k == "ik" else v
    return out


def apply_settings(dream_cfg: dict[str, Any], loop_kw: dict[str, Any], s: dict[str, Any],
                   trace: pathlib.Path = TRACE) -> tuple[dict[str, Any], dict[str, Any]]:
    """The station's policy config and loop kwargs with our knobs on top. Caps can only go down, never up."""
    d, kw = copy.deepcopy(dream_cfg), dict(loop_kw)
    n = int(s["execution_horizon"])
    if not 1 <= n <= kw["action_horizon"]:
        raise ValueError(f"execution_horizon {n} must be in [1, {kw['action_horizon']}] (the chunk length)")
    d["execution_horizon"] = n
    if s.get("sync_above_ms"):
        d["sync_above_ms"] = float(s["sync_above_ms"])
    scale = min(1.0, max(MIN_SPEED_SCALE, float(s.get("speed_scale") or 1.0)))
    kw["control_hz"] = kw["control_hz"] * scale  # rows play slower; the client paces at the same rate
    d["action_hz"] = kw["control_hz"]
    kw["max_ee_speed_m_s"] = kw["max_ee_speed_m_s"] * scale
    ik = s.get("ik") or {}
    if ik and "ik" not in d:
        raise ValueError("the station's policy config has no `ik`; keep_out/region/solve need an eef run")
    if ik.get("solve"):
        d["ik"]["solve"] = ik["solve"]
    for side, boxes in (ik.get("keep_out") or {}).items():  # added to the station file's, never replacing them
        d["ik"].setdefault("keep_out", {})[side] = list(d["ik"].get("keep_out", {}).get(side, [])) + list(boxes or [])
    for side, box in (ik.get("region") or {}).items():
        if box:
            d["ik"].setdefault("region", {})[side] = box
    kw["trace_path"] = str(trace)
    return d, kw


class PrefixPacer:
    """Wraps the client. With `always_async`, refill_due() is the client's async rule with the sync switch removed:
    ask once executed + round-trip rows reach `horizon`, or when the runway is about to run out (pacing.py:123-144).
    So a contact subtask never plays more than ~max(horizon, round trip) rows of a stale chunk."""

    def __init__(self, inner: Any, horizon: int, action_hz: float, always_async: bool,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._inner, self.horizon, self.hz, self.always_async, self._clock = inner, horizon, action_hz, always_async, clock
        self.rtts: deque[float] = deque(maxlen=5)

    def get_action(self, **kw: Any) -> Any:
        t0 = self._clock()
        out = self._inner.get_action(**kw)
        self.rtts.append(self._clock() - t0)
        return out

    def refill_due(self, executed: int, remaining: int) -> bool:
        if not self.always_async:
            return bool(self._inner.refill_due(executed, remaining))
        if remaining <= 0:
            return True
        rtt = min(self.rtts, default=0.0)
        return executed + math.ceil(rtt * self.hz) >= self.horizon or remaining / self.hz <= rtt + 0.25

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class ClientLoopBackend:
    """Backend over our own loop. start() runs one episode on a thread; stop() ends it and the arms hold.
    Dry by default: the model is called for real, the motors are never written."""

    def __init__(self, cfg: dict[str, Any], station: Station | None = None, *, dry_run: bool = True,
                 stop_when: Callable[[Tick], bool] | None = None, trace: pathlib.Path = TRACE) -> None:
        if os.environ.get("HILLCLIMB_ALLOW_ARMS") != "1":
            raise RuntimeError("ClientLoopBackend opens the real arms (even dry): set HILLCLIMB_ALLOW_ARMS=1 with a "
                               "hand on the e-stop")
        self.cfg, self.dry_run, self.stop_when, self.trace = cfg, dry_run, stop_when, trace
        self.station = station if station is not None else RobotsStation(cfg.get("station_file", STATION_FILE))
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._state, self.last_error, self.end, self.ticks = "idle", "", "", 0
        self._arm_fault = False

    def _sub(self, prompt: str) -> dict[str, Any] | None:
        return next((s for s in self.cfg.get("subtasks", []) if s.get("prompt") == prompt), None)

    def start(self, prompt: str, checkpoint: str | None) -> None:
        if self._arm_fault:  # retrying over faulted arms would build a new loop on them (PR #2 review)
            raise RuntimeError("an arm faulted earlier in this session; check the arms, then restart the harness")
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("an episode is still running; stop() it first")
        s = client_settings(self.cfg, self._sub(prompt))
        dream_cfg, loop_kw = apply_settings(self.station.policy_config(checkpoint), self.station.loop_settings(), s,
                                            self.trace)
        self._stop.clear()
        self._state, self.last_error, self.end, self.ticks = "starting", "", "", 0
        self._thread = threading.Thread(target=self._thread_main, args=(prompt, dream_cfg, loop_kw, s), daemon=True)
        self._thread.start()

    def _thread_main(self, prompt: str, dream_cfg: dict[str, Any], loop_kw: dict[str, Any], s: dict[str, Any]) -> None:
        try:
            asyncio.run(self._episode(prompt, dream_cfg, loop_kw, s))
            self._state = "idle"
        except BaseException as exc:  # the loop's __aexit__ has already held or disengaged the arms
            self._arm_fault |= any(c.__name__ == "ArmError" for c in type(exc).__mro__)
            self.last_error, self.end, self._state = f"{type(exc).__name__}: {exc}", "error", "error"

    async def _episode(self, prompt: str, dream_cfg: dict[str, Any], loop_kw: dict[str, Any], s: dict[str, Any]) -> None:
        await asyncio.to_thread(self.station.open, self.dry_run, dream_cfg)
        raw = await asyncio.to_thread(self.station.open_client, dream_cfg)
        if self._stop.is_set():  # stopped while the client opened: nothing moves (run.py:152-155)
            raw.close()
            self.end = "stopped"
            return
        client = PrefixPacer(raw, dream_cfg["execution_horizon"], loop_kw["control_hz"], bool(s.get("always_async")))
        loop = self.station.make_loop(client, prompt, loop_kw)
        period = 1.0 / loop_kw["control_hz"]
        async with loop:
            self._state = "running"
            deadline = time.monotonic()
            while not self._stop.is_set():
                await loop.step(deadline)
                self.ticks += 1
                tick = Tick(self.ticks, deadline, loop.state, dict(loop.video), loop.tick)
                if self.stop_when is not None and self.stop_when(tick):
                    self.end = "stop_condition"
                    return
                deadline += period
                now = time.monotonic()
                deadline = max(deadline, now)  # an overrun or a refill wait restarts the schedule (control.py:986-999)
                await asyncio.sleep(deadline - now)
            self.end = "stopped"

    def play(self, path: str, dest: pathlib.Path | None = None, opts: dict[str, Any] | None = None) -> dict[str, Any]:
        """Plays a replay primitive between episodes (never during one), with the same dry/live gate as the loop.
        The joint speed cap is min(opts.max_rad_s, the station's ramp_max_rad_s, 0.5 rad/s). With `dest`, the newest
        policy JPEG per camera goes to dest/inputs/ for the judge. TODO(event): untested on the station."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("an episode is running; stop() it before playing a primitive")
        if self._arm_fault:
            raise RuntimeError("an arm faulted earlier; not moving it again")
        dream_cfg, kw = apply_settings(self.station.policy_config(self.cfg.get("checkpoint")),
                                       self.station.loop_settings(), client_settings(self.cfg, None), self.trace)
        o = dict(opts or {})
        o["max_rad_s"] = min(float(o.get("max_rad_s", 0.5)), float(kw.get("ramp_max_rad_s", 0.5)))
        try:
            self.station.open(self.dry_run, dream_cfg)  # inside the fault boundary: open() homes (PR #2 review)
            out = primitives.play(primitives.load(path), self.station.arms, dry_run=self.dry_run, **o)
        except Exception as exc:
            self._arm_fault |= any(c.__name__ == "ArmError" for c in type(exc).__mro__)
            raise
        if dest is not None:  # TODO(event): the newest encoded JPEG may predate the final hold by <= obs_period_s
            import base64
            for r in (r for r in self.station.readers if getattr(getattr(r, "frame", None), "jpeg", "")):
                (dest / "inputs" / r.name).mkdir(parents=True, exist_ok=True)
                (dest / "inputs" / r.name / "0.000.jpg").write_bytes(base64.b64decode(r.frame.jpeg))
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "primitive.json").write_text(json.dumps({"path": str(path), **out}, indent=1))
        return out

    def state(self) -> str:  # "starting" | "running" | "idle" | "error"
        return self._state

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)

    def home(self) -> None:
        if self.dry_run or self._arm_fault:  # a dry run never homes; after an arm fault nothing homes (run.py:241)
            return
        if self._thread is not None and self._thread.is_alive():  # stop() joins 30 s at most: the loop may still write
            raise RuntimeError("the episode thread is still running; not homing over it (two writers)")
        if not self.station.home():
            self._arm_fault = True
            raise RuntimeError("an arm did not reach home; see the station log")

    def collect(self, dest: pathlib.Path) -> dict[str, Any]:
        summary = backends._copy_artifacts(self.trace, dest)
        (dest / "client_loop.json").write_text(
            json.dumps({"end": self.end, "ticks": self.ticks, "error": self.last_error}, indent=1))
        return summary

    def latest_frames(self) -> dict[str, pathlib.Path]:
        return backends.latest_frames_in(backends._trace_files(self.trace)[1])  # control.py:1787-1803, once a second

    def close(self) -> None:
        """End of the session: home (live, no arm fault) then switch the arms off, as dreamcli does on exit."""
        self.stop()
        if self._thread is not None and self._thread.is_alive():
            # The worker may still be opening/driving the arms: switching them off under it would release the lock
            # mid-write (PR #2 review). Leave the station to the operator: e-stop, then kill the process.
            print("client loop: episode thread still alive after stop(); NOT closing the station. PRESS THE E-STOP, "
                  "then end this process.")
            return
        try:
            self.home()
        finally:
            self.station.close()


class RobotsStation:
    """The station's own classes (robots 1.0.37), used the way robots/run.py and robots/station.py use them."""

    def __init__(self, path: str | os.PathLike[str] = STATION_FILE) -> None:
        from robots import config, run  # station only: ~/dream-env

        self.path = pathlib.Path(path).expanduser()
        self.app = config.load(self.path)
        run.check_tool_point(self.app, skip=False)
        self.arms: dict[str, Any] = {}
        self.readers: list[Any] = []
        self.kin: dict[str, Any] | None = None
        self._lock: Any = None

    def policy_config(self, checkpoint: str | None) -> dict[str, Any]:
        from robots import errors, run
        d = run.policy_config(self.app, checkpoint)
        if "checkpoint" not in d:
            raise errors.ConfigError("no checkpoint: set `checkpoint` in hillclimb/config.yaml or the station file")
        return d

    def loop_settings(self) -> dict[str, Any]:
        from robots import run
        return run.loop_settings(self.app)

    def open(self, dry_run: bool, dream_config: dict[str, Any]) -> None:
        from robots import arms_lock, bodies, errors, followers, home, run, station

        if not self.readers or any(not r.is_alive() for r in self.readers):  # station.py:383-397
            for r in self.readers:
                r.stop()
            self.readers = run.make_cameras(self.app)
            for r in self.readers:
                r.start()
            if any(r.exposure is not None for r in self.readers):
                time.sleep(run.EXPOSURE_SETTLE_S)
            for r in self.readers:
                r.set_exposure_hold(True)
        if self.arms:
            return
        # One holder of the arms: refuses while `dreamcli robot up` or a robot run holds them (arms_lock.py).
        self._lock = arms_lock.acquire(self.path.parent / arms_lock.LOCK_NAME, arms_lock.ROBOT_RUN)
        self.kin = run.make_kinematics(self.app)
        station.warm_client_ik(dream_config)  # before the arms power: IK compile starves the CAN thread (0xD)
        body = bodies.for_config(self.app)
        opener = body.open_arm_read_only if dry_run and body.open_arm_read_only is not None else body.open_arm
        for name, arm_cfg in (("left", self.app.follower_l), ("right", self.app.follower_r)):
            arm = opener(arm_cfg, name)
            self.arms[name] = followers.FollowerArm(name, run.DryRun(arm), home_rad=arm.home_rad) if dry_run else arm
        if not dry_run and not home.return_to_home(self.arms):
            raise errors.ArmError("an arm did not reach home before the first run")

    def open_client(self, dream_config: dict[str, Any]) -> Any:
        from robots import run

        client = run.open_client(dream_config)
        client.wait_for_ik(run.TRAJ_WARMUP_S)
        return client

    def make_loop(self, client: Any, task: str, settings: dict[str, Any]) -> Any:
        from robots import control
        return control.ControlLoop(followers=self.arms, cameras=self.readers, client=client, task=task,
                                   kinematics=self.kin, inference_log_path=self.app.inference_log_path,
                                   camera_owner=False, recorder=None, **settings)

    def home(self) -> bool:
        from robots import home
        return bool(home.return_to_home(self.arms))

    def close(self) -> None:
        from robots import control

        for r in self.readers:
            r.stop()
        for name, arm in self.arms.items():
            control.run_teardown(f"switch {name} arm off", arm.torque_off)  # run.py:165-167
        self.arms, self.readers = {}, []
        if self._lock is not None:
            self._lock.close()
