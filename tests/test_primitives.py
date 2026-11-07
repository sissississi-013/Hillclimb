"""Replay primitives with fakes: a synthetic dhk-collect parquet, a fake arm, the fake backend. No hardware."""

import dataclasses
import json
import pathlib
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from hillclimb import backends, harness, primitives
from hillclimb.judge import Verdict

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "tools"))
import waypoints  # noqa: E402

CONFIG = pathlib.Path(__file__).parents[1] / "hillclimb" / "config.yaml"


def make_dataset(root, frames=60, fps=30):
    """Episode 1 of a dhk-collect Metal dataset: joint j of arm k ramps as 0.01*i + 0.1*j (+1 right); grip in mm."""
    i = np.arange(frames)[:, None]
    arm = 0.01 * i + 0.1 * np.arange(6)
    vec = np.concatenate([arm, np.full((frames, 1), 80.0), arm + 1, np.full((frames, 1), 20.0)], 1).astype(np.float32)
    (root / "meta").mkdir(parents=True)
    rel = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    (root / "meta/info.json").write_text(json.dumps({"fps": fps, "chunks_size": 1000, "data_path": rel}))
    (root / "data/chunk-000").mkdir(parents=True)
    col = pa.FixedSizeListArray.from_arrays(pa.array(vec.reshape(-1)), 14)
    pq.write_table(pa.table({"observation.state": col, "action": col, "frame_index": pa.array(np.arange(frames))}),
                   root / "data/chunk-000/episode_000001.parquet")
    return vec


@dataclasses.dataclass
class State:
    positions_rad: np.ndarray
    gripper: float


@dataclasses.dataclass
class Cmd:
    positions_rad: np.ndarray
    gripper: float


class FakeArm:
    def __init__(self, q, grip=1.0):
        self.q, self.grip, self.writes, self.holds = np.asarray(q, float), grip, [], 0

    def read(self):
        return State(self.q.copy(), self.grip)

    def write(self, cmd):
        self.writes.append(cmd)
        self.q, self.grip = np.asarray(cmd.positions_rad, float), cmd.gripper

    def hold(self):
        self.holds += 1


@pytest.fixture
def arms_ok(monkeypatch):
    monkeypatch.setenv("HILLCLIMB_ALLOW_ARMS", "1")


def test_extract_trajectory_and_waypoint_roundtrip(tmp_path):
    vec = make_dataset(tmp_path / "ds")
    p = waypoints.extract(tmp_path / "ds", 1, 10, 40, every=2)
    assert p.fps == 15 and p.joints["left"].shape == (15, 6)
    np.testing.assert_allclose(p.joints["right"][0], vec[10, 7:13], atol=1e-6)
    np.testing.assert_allclose(p.gripper["left"], 0.8)
    np.testing.assert_allclose(p.gripper["right"], 0.2)
    primitives.save(p, tmp_path / "p.yaml")
    np.testing.assert_allclose(primitives.load(tmp_path / "p.yaml").vector(), p.vector(), atol=1e-4)
    one = waypoints.extract(tmp_path / "ds", 1, 59, None, column="observation.state")
    assert one.joints["left"].shape == (1, 6)
    with pytest.raises(SystemExit):
        waypoints.extract(tmp_path / "ds", 1, 50, 70)
    primitives.save(waypoints.extract(tmp_path / "ds", 1, 0, 5, grip_scale=1.0), tmp_path / "bad.yaml")
    with pytest.raises(ValueError, match="grip-scale"):  # mm read as 0..1
        primitives.load(tmp_path / "bad.yaml")


def test_cli_extract(tmp_path):
    make_dataset(tmp_path / "ds")
    assert waypoints.main(["extract", str(tmp_path / "ds"), "--episode", "1", "--frame", "5",
                           "--out", str(tmp_path / "w.yaml")]) == 0
    assert yaml.safe_load((tmp_path / "w.yaml").read_text())["meta"]["frames"] == [5, 6]


def waypoint(q, grip=0.5):
    return primitives.Primitive(30.0, {"left": np.asarray([q], float), "right": np.asarray([q], float)},
                                {"left": np.asarray([grip]), "right": np.asarray([grip])})


def test_plan_interpolates_and_caps_velocity():
    start = np.zeros(14)
    start[[6, 13]] = 1.0
    cmds = primitives.plan(waypoint([0.4] * 6), start, hz=40, max_rad_s=2.0)  # asks 2 rad/s: capped to 0.5
    speed = np.abs(np.diff(np.vstack([start, cmds]), axis=0))[:, :6].max() * 40
    assert speed <= 0.5 + 1e-9
    np.testing.assert_allclose(cmds[-1][:6], 0.4)
    np.testing.assert_allclose(cmds[-1][6], 0.5)
    assert len(cmds) >= 0.4 / 0.5 * 40
    mid = cmds[len(cmds) // 2][:6]
    assert np.allclose(mid, mid[0])  # a straight joint-space blend: all joints move together
    # a fast demo (1 rad/s at 30 fps) is stretched, never followed faster than the cap
    fast = primitives.Primitive(30.0, {s: np.linspace(0, 1, 31)[:, None] * np.ones(6) for s in ("left", "right")},
                                {s: np.full(31, 1.0) for s in ("left", "right")})
    out = primitives.plan(fast, np.r_[[0] * 6, 1, [0] * 6, 1], hz=40, speed_scale=1.0)
    assert np.abs(np.diff(out, axis=0)).max() * 40 <= 0.5 + 1e-9 and np.allclose(out[-1][:6], 1.0)
    limited = primitives.plan(waypoint([0.4] * 6), start, limits={"left": [[-0.1, 0.2]] * 6})
    assert limited[:, :6].max() <= 0.2 and np.isclose(limited[-1][7], 0.4)
    with pytest.raises(ValueError, match="speed_scale"):
        primitives.plan(waypoint([0.1] * 6), start, speed_scale=1.5)
    with pytest.raises(ValueError, match="refusing"):
        primitives.plan(waypoint([1.5] * 6), start)


def test_play_refuses_without_env(monkeypatch):
    monkeypatch.delenv("HILLCLIMB_ALLOW_ARMS", raising=False)
    arms = {"left": FakeArm([0] * 6), "right": FakeArm([0] * 6)}
    with pytest.raises(RuntimeError, match="HILLCLIMB_ALLOW_ARMS"):
        primitives.play(waypoint([0.1] * 6), arms, dry_run=False, command=Cmd)
    assert not arms["left"].writes


def test_play_dry_by_default_then_live(arms_ok):
    arms = {"left": FakeArm([0] * 6), "right": FakeArm([0] * 6)}
    dry = primitives.play(waypoint([0.2] * 6), arms, command=Cmd, sleep=lambda s: None)
    assert dry["dry_run"] and not arms["left"].writes and dry["ticks"] > 0
    live = primitives.play(waypoint([0.2] * 6), arms, dry_run=False, command=Cmd, sleep=lambda s: None)
    assert len(arms["right"].writes) == live["ticks"] and arms["left"].holds == 1
    np.testing.assert_allclose(arms["left"].q, 0.2)
    assert live["peak_rad_s"] <= 0.5 + 1e-9
    assert live["proprio"]["l"]["cmd_grip"] == pytest.approx(0.5)


def test_play_holds_both_arms_after_a_fault(arms_ok):
    arms = {"left": FakeArm([0] * 6), "right": FakeArm([0] * 6)}
    arms["right"].write = lambda cmd: (_ for _ in ()).throw(OSError("bus"))
    with pytest.raises(OSError):
        primitives.play(waypoint([0.2] * 6), arms, dry_run=False, command=Cmd, sleep=lambda s: None)
    assert arms["left"].holds == 1 and len(arms["left"].writes) == 1


class Scripted:
    def __init__(self, results):
        self.results, self.seen = list(results), []

    def __call__(self, sub, frames, proprio):
        if not proprio:
            return Verdict(None)
        self.seen.append((sub["name"], sorted(frames)))
        return Verdict(self.results.pop(0), 1.0, "scripted", "test")


def run(tmp_path, sub_extra, results, backend=None):
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["subtasks"] = [s for s in cfg["subtasks"] if s["name"] == "open_backing"]
    cfg["subtasks"][0].update(sub_extra)
    t = [0.0]
    be = backend or backends.FakeBackend(clock=lambda: t[0], run_s=5.0)
    h = harness.Harness(cfg, be, Scripted(results), tmp_path, clock=lambda: t[0],
                        sleep=lambda s: t.__setitem__(0, t[0] + s))
    return h, be, h.run()


def test_harness_pre_plays_before_every_attempt(tmp_path):
    h, be, ok = run(tmp_path, {"pre": "prims/open_pre.yaml"}, [False, True])
    assert ok
    seq = [c[0] for c in be.calls]
    assert seq == ["play", "start", "stop", "home", "play", "start", "stop"]  # recovery unchanged, then pre again


def test_harness_replay_mode_is_judged_and_retried(tmp_path):
    h, be, ok = run(tmp_path, {"pre": "prims/open.yaml", "mode": "replay"}, [False, True])
    assert ok and not any(c[0] == "start" for c in be.calls)
    assert [c[0] for c in be.calls] == ["play", "home", "play"]
    assert [a.end for a in h.attempts] == ["replay", "replay"] and h.attempts[0].prompt == ""
    assert h.judge.seen[0][1] == ["overhead"]


def test_harness_refuses_pre_on_a_backend_that_cannot_play(tmp_path):
    class NoPlay(backends.DreamcliDryRunBackend):
        pass
    with pytest.raises(ValueError, match="--backend client"):
        run(tmp_path, {"pre": "prims/x.yaml"}, [True], backend=NoPlay())
    with pytest.raises(ValueError, match="replay needs"):
        run(tmp_path, {"mode": "replay"}, [True])


def test_client_backend_play_uses_station_arms_and_caps(tmp_path, arms_ok):
    from tests.test_client_loop import FakeStation
    from hillclimb.client_loop import ClientLoopBackend

    st = FakeStation()
    st.arms = {"left": FakeArm([0] * 6), "right": FakeArm([0] * 6)}
    st.readers = []
    st.loop_settings = lambda: {"control_hz": 30.0, "action_horizon": 48, "max_ee_speed_m_s": 0.6,
                                "ramp_max_rad_s": 0.3}
    primitives.save(waypoint([0.3] * 6), tmp_path / "w.yaml")
    cfg = yaml.safe_load(CONFIG.read_text())
    out = ClientLoopBackend(cfg, st).play(str(tmp_path / "w.yaml"), tmp_path / "a", {"max_rad_s": 0.5})
    assert out["dry_run"] and st.opened == [True] and not st.arms["left"].writes  # dry by default
    assert out["peak_rad_s"] <= 0.3 + 1e-9  # the station's ramp cap wins
    assert json.loads((tmp_path / "a" / "primitive.json").read_text())["ticks"] == out["ticks"]
