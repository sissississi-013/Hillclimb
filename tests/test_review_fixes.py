"""Regression tests for the 2026-11-07 code review (docs/research/08-code-review.md)."""

import json
import os
import pathlib
import time

import pytest
import yaml

from hillclimb import backends, harness
from hillclimb.client_loop import ClientLoopBackend
from hillclimb.judge import CombinedJudge, ProprioJudge, Verdict, VLMJudge

CONFIG = pathlib.Path(__file__).parents[1] / "hillclimb" / "config.yaml"


def cfg_one(name="unplug_usb", **over):
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["subtasks"] = [dict(next(s for s in cfg["subtasks"] if s["name"] == name), **over)]
    return cfg


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_exception_mid_run_still_stops_the_run(tmp_path):
    """A /status error, a judge crash or Ctrl-C used to leave the station running the policy (it runs until Stop)."""
    clock = Clock()
    be = backends.FakeBackend(clock=clock, run_s=100.0)
    real_state = be.state

    def flaky_state():
        if clock.t > 2:
            raise KeyboardInterrupt
        return real_state()

    be.state = flaky_state
    h = harness.Harness(cfg_one(), be, lambda s, f, p: Verdict(None), tmp_path, clock=clock, sleep=clock.sleep)
    with pytest.raises(KeyboardInterrupt):
        h.run()
    assert be.calls[-1] == ("stop",)


def test_stop_failure_after_an_error_does_not_mask_it(tmp_path, capsys):
    clock = Clock()
    be = backends.FakeBackend(clock=clock)

    def bad_start(prompt, checkpoint):
        raise ConnectionError("station down")

    def bad_stop():
        raise ConnectionError("still down")

    be.start, be.stop = bad_start, bad_stop
    h = harness.Harness(cfg_one(), be, lambda s, f, p: Verdict(None), tmp_path, clock=clock, sleep=clock.sleep)
    with pytest.raises(ConnectionError, match="station down"):
        h.run()
    assert "E-STOP" in capsys.readouterr().out


@pytest.mark.parametrize("name,held", [("close_backing", False), ("unplug_usb", True)])
def test_vlm_unknown_never_passes_on_proprio_alone(tmp_path, name, held):
    sub = cfg_one(name)["subtasks"][0]
    sub["proprio"] = {"expect_holding": held}  # pin the proprio rule; config values may change
    proprio = {"l": {"grip": 0.3 if held else 1.0, "cmd_grip": 0.0 if held else 1.0, "max_track_err_rad": 0.0}}
    assert ProprioJudge()(sub, {}, proprio).success is True  # proprio alone would say success
    no_key = VLMJudge(api_key=None)
    no_key.api_key = None
    v = CombinedJudge(no_key, ProprioJudge())(sub, {"overhead": tmp_path / "x.jpg"}, proprio)
    assert v.success is None
    assert CombinedJudge(no_key, ProprioJudge())(sub, {}, proprio).success is None  # missing frames too


def test_vlm_string_false_is_not_success(tmp_path):
    img = tmp_path / "f.jpg"
    img.write_bytes(b"x")
    reply = {"choices": [{"message": {"content": '{"description": "d", "score": 0.9, "success": "false"}'}}]}
    assert VLMJudge(post=lambda b: reply)({"name": "x"}, {"o": img}, {}).success is False


def test_vlm_missing_frame_file_is_unknown_not_a_crash(tmp_path):
    v = VLMJudge(post=lambda b: {})({"name": "x"}, {"o": tmp_path / "gone.jpg"}, {})
    assert v.success is None and "vlm error" in v.reason


def test_stale_frames_from_the_previous_run_are_not_judged(tmp_path):
    """Until the loop opens its trace (seconds after a Start), the inputs folder still holds the LAST run's frames."""
    stale = tmp_path / "trace_inputs" / "overhead"
    stale.mkdir(parents=True)
    old = stale / "12.000.jpg"
    old.write_bytes(b"old")
    os.utime(old, (time.time() - 60, time.time() - 60))
    clock = Clock()

    class Be(backends.FakeBackend):
        def latest_frames(self):
            return backends.latest_frames_in(stale.parent)

        def collect(self, dest):  # stopped while still starting: only the old frame gets copied
            backends._copy_artifacts(tmp_path / "trace.csv", dest)
            return {}

    seen = []

    def judge(sub, frames, proprio):
        seen.append(dict(frames))
        return Verdict(bool(frames), 1.0, "", "test")  # would say success on ANY frame

    be = Be(clock=clock, run_s=100.0)
    h = harness.Harness(cfg_one(max_time_s=10, max_attempts=1), be, judge, tmp_path / "logs",
                        clock=clock, sleep=clock.sleep)
    assert not h.run()
    assert h.attempts[0].end == "timeout" and not h.attempts[0].success
    assert all(not f for f in seen)


def test_logged_checkpoint_is_the_one_the_station_ran(tmp_path):
    """The station keeps the last checkpoint a Start named; `checkpoint: null` does not mean the station file's."""
    clock = Clock()
    be = backends.FakeBackend(clock=clock)
    collect = be.collect

    def collect_with_status(dest):
        out = collect(dest)
        (dest / "status.json").write_text(json.dumps({"run": 3, "checkpoint": "dream-1-ft:abcd-hc-ft1"}))
        return out

    be.collect = collect_with_status
    h = harness.Harness(cfg_one(), be, lambda s, f, p: Verdict(True, 1.0), tmp_path, clock=clock, sleep=clock.sleep)
    assert h.run()
    rec = json.loads((h.dir / "attempts.jsonl").read_text().splitlines()[0])
    assert rec["checkpoint"] == "dream-1-ft:abcd-hc-ft1" and rec["station_run"] == 3


def test_load_env_sets_missing_keys_only(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("# c\nFIREWORKS_API_KEY=fw_abc\nexport OTHER='x y'\n")
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.setenv("OTHER", "kept")
    harness.load_env(tmp_path / ".env")
    assert os.environ["FIREWORKS_API_KEY"] == "fw_abc" and os.environ["OTHER"] == "kept"
    harness.load_env(tmp_path / "missing")  # no file: no error


def test_client_home_refuses_while_episode_thread_alive(monkeypatch):
    monkeypatch.setenv("HILLCLIMB_ALLOW_ARMS", "1")

    class St:
        homes = 0

        def home(self):
            St.homes += 1
            return True

    class Alive:
        def is_alive(self):
            return True

    be = ClientLoopBackend({}, St(), dry_run=False)
    be._thread = Alive()
    with pytest.raises(RuntimeError, match="two writers"):
        be.home()
    assert St.homes == 0


def test_trace_path_read_from_station_file(tmp_path):
    from hillclimb import backends
    f = tmp_path / "station.yaml"
    f.write_text("trace_path: /tmp/hc-trace/run.csv\n")
    assert str(backends.trace_from_station_file(str(f))) == "/tmp/hc-trace/run.csv"
    assert backends.trace_from_station_file(str(tmp_path / "missing.yaml")) == backends.DEFAULT_TRACE
    assert backends.trace_from_station_file(None) == backends.DEFAULT_TRACE


def test_recover_skips_home_on_409(tmp_path):
    import urllib.error
    from hillclimb import harness

    class Be:
        def home(self):
            raise urllib.error.HTTPError("http://x/home", 409, "conflict", {}, None)

    h = harness.Harness({"subtasks": []}, Be(), None, tmp_path, sleep=lambda s: None)
    h.recover({"recovery": "home"})  # must not raise

    class Be500(Be):
        def home(self):
            raise urllib.error.HTTPError("http://x/home", 500, "boom", {}, None)

    h.backend = Be500()
    import pytest
    with pytest.raises(urllib.error.HTTPError):
        h.recover({"recovery": "home"})


def test_primitive_fast_gripper_close_completes():
    import numpy as np
    from hillclimb import primitives
    z = np.zeros((2, 6))
    p = primitives.Primitive(fps=30, joints={"left": z, "right": z},
                             gripper={"left": np.array([1.0, 0.0]), "right": np.array([1.0, 1.0])}, meta={})
    start = np.r_[np.zeros(6), 1.0, np.zeros(6), 1.0]
    cmds = primitives.plan(p, start, hz=40)
    assert abs(cmds[-1][6] - 0.0) < 1e-6 and abs(cmds[-1][13] - 1.0) < 1e-6


def test_client_loop_refuses_start_after_arm_fault(tmp_path, monkeypatch):
    import pytest
    from hillclimb import client_loop
    be = client_loop.ClientLoopBackend.__new__(client_loop.ClientLoopBackend)
    be._arm_fault, be._thread = True, None
    with pytest.raises(RuntimeError, match="faulted"):
        be.start("unplug the usb cable from the rack", None)


def test_human_and_confirm_judges(tmp_path):
    from hillclimb.judge import ConfirmJudge, HumanJudge, Verdict
    sub = {"name": "plug_usb"}
    answers = iter(["maybe", "y"])
    assert HumanJudge(ask=lambda q: next(answers))(sub, {}, {}).success is True
    auto = lambda s, f, p: Verdict(True, 0.9, "looks seated", source="vlm")
    cj = ConfirmJudge(auto, ask=lambda q: "")
    assert cj(sub, {}, {}).success is True and not cj.disagreements          # Enter accepts
    cj = ConfirmJudge(auto, ask=lambda q: "n")
    v = cj(sub, {"overhead": tmp_path / "a.jpg"}, {})
    assert v.success is False and v.source == "human" and cj.disagreements[0]["auto"] is True
    assert cj.midrun_judge is auto                                             # mid-run never prompts


def test_harness_judge_override_flag(tmp_path, monkeypatch):
    from hillclimb import harness
    from hillclimb.judge import HumanJudge
    seen = {}

    class Spy(harness.Harness):
        def run(self):
            seen["judge"] = self.judge
            return True

    monkeypatch.setattr(harness, "Harness", Spy)
    assert harness.main(["--judge", "human", "--logs", str(tmp_path)]) == 0
    assert isinstance(seen["judge"], HumanJudge)


def test_primitive_intermediate_grasp_preserved():
    import numpy as np
    from hillclimb import primitives
    z = np.zeros((3, 6))
    p = primitives.Primitive(fps=30, joints={"left": z, "right": z},
                             gripper={"left": np.array([1.0, 0.0, 1.0]), "right": np.ones(3)}, meta={})
    cmds = primitives.plan(p, np.r_[np.zeros(6), 1.0, np.zeros(6), 1.0], hz=40)
    assert cmds[:, 6].min() < 0.05  # the close actually happens before the reopen


def test_replay_open_failure_latches_arm_fault(tmp_path, monkeypatch):
    import pytest
    from hillclimb import client_loop
    monkeypatch.setattr(client_loop, "apply_settings", lambda *a: ({}, {}))

    class ArmError(Exception):
        pass

    class St:
        def policy_config(self, c): return {}
        def loop_settings(self): return {}
        def open(self, dry, cfg): raise ArmError("homing failed")

    be = client_loop.ClientLoopBackend.__new__(client_loop.ClientLoopBackend)
    be._thread, be._arm_fault, be.station, be.cfg, be.dry_run = None, False, St(), {}, False
    be.trace = tmp_path / "t.csv"
    with pytest.raises(ArmError):
        be.play(str(tmp_path / "p.yaml"))
    assert be._arm_fault is True


def test_primitive_rejects_start_outside_limits():
    import numpy as np
    import pytest
    from hillclimb import primitives
    z = np.zeros((2, 6)); z[:, 0] = 0.2
    p = primitives.Primitive(fps=30, joints={"left": z, "right": np.zeros((2, 6))},
                             gripper={"left": np.ones(2), "right": np.ones(2)}, meta={})
    start = np.zeros(14); start[0] = 0.4
    with pytest.raises(ValueError, match="outside"):
        primitives.plan(p, start, hz=40, limits={"left": [[-0.1, 0.2]]})


def test_replay_overrun_does_not_burst(monkeypatch):
    import numpy as np
    from hillclimb import primitives
    monkeypatch.setenv("HILLCLIMB_ALLOW_ARMS", "1")
    t = [0.0]; writes = []
    state = np.zeros(7); state[6] = 1.0

    class Arm:
        def read(self):
            class S: positions_rad = state[:6]; gripper = 1.0
            return S()
        def write(self, c): writes.append(t[0])
        def hold(self): pass

    over = [0.25]  # the first sleep oversleeps by 250 ms
    def sleep(d):
        t[0] += max(0.0, d) + (over.pop() if over else 0.0)
    monkeypatch.setattr(primitives, "_read", lambda arms: (np.r_[state, state], {}))
    z = np.zeros((10, 6)); z[:, 0] = np.linspace(0, 0.05, 10)
    p = primitives.Primitive(fps=30, joints={"left": z, "right": np.zeros((10, 6))},
                             gripper={"left": np.ones(10), "right": np.ones(10)}, meta={})
    try:
        primitives.play(p, {"left": Arm(), "right": Arm()}, dry_run=False, hz=40,
                        command=lambda **k: k, clock=lambda: t[0], sleep=sleep)
    except Exception:
        pass  # post-play state checks may complain about the fake arms; spacing is what we test
    left = writes[::2]
    gaps = np.diff(left)
    assert len(gaps) > 5 and gaps.min() >= 1 / 40 - 1e-9
