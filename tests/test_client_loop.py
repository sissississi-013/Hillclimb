"""ClientLoopBackend with a fake station and a fake DreamClient: no `dream`/`robots` import, no hardware."""

import pathlib
import time

import pytest
import yaml

from hillclimb import client_loop
from hillclimb.client_loop import ClientLoopBackend, PrefixPacer, apply_settings, client_settings

CONFIG = pathlib.Path(__file__).parents[1] / "hillclimb" / "config.yaml"
CFG = yaml.safe_load(CONFIG.read_text())
SUB = {s["name"]: s for s in CFG["subtasks"]}


class FakeClient:
    def __init__(self):
        self.calls, self.closed = 0, False

    def get_action(self, **kw):
        self.calls += 1
        return {"q_r_chunk": [[0.0] * 7] * 48}

    def refill_due(self, executed, remaining):  # stands in for the client's pacer in sync mode
        return remaining <= 0

    def close(self):
        self.closed = True


class FakeLoop:
    """Just the ControlLoop surface we use: async with, step(now_s), state/video/tick. Records what it saw."""

    def __init__(self, client, task, settings):
        self.client, self.task, self.settings, self.steps = client, task, settings, 0
        self.state, self.video, self.tick, self.exited = None, {}, None, False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.exited = True
        self.client.close()

    async def step(self, now_s):
        self.steps += 1
        self.state = {"grip_l": 1.0 - 0.1 * self.steps}
        self.video = {"overhead": f"frame{self.steps}"}


class FakeStation:
    def __init__(self):
        self.opened, self.homes, self.closed, self.loops, self.clients = [], 0, False, [], []

    def policy_config(self, checkpoint):
        return {"checkpoint": checkpoint or "dream-1-ft:test", "action_hz": 20,
                "ik": {"arm": "metal", "head": "eef", "solve": "traj",
                       "keep_out": {"left": [{"kind": "capsule", "center": [0, 0, 0], "radius": 0.05, "height": 1}]}}}

    def loop_settings(self):
        return {"control_hz": 400.0, "action_horizon": 48, "max_ee_speed_m_s": 0.6, "trace_path": "/station.csv"}

    def open(self, dry_run, dream_config):
        self.opened.append(dry_run)

    def open_client(self, dream_config):
        self.clients.append(FakeClient())
        return self.clients[-1]

    def make_loop(self, client, task, settings):
        self.loops.append(FakeLoop(client, task, settings))
        return self.loops[-1]

    def home(self):
        self.homes += 1
        return True

    def close(self):
        self.closed = True


@pytest.fixture
def arms_ok(monkeypatch):
    monkeypatch.setenv("HILLCLIMB_ALLOW_ARMS", "1")


def wait_for(pred, timeout=5.0):
    t0 = time.monotonic()
    while not pred():
        assert time.monotonic() - t0 < timeout, "timed out"
        time.sleep(0.005)


def test_refuses_without_env(monkeypatch):
    monkeypatch.delenv("HILLCLIMB_ALLOW_ARMS", raising=False)
    station = FakeStation()
    with pytest.raises(RuntimeError, match="HILLCLIMB_ALLOW_ARMS"):
        ClientLoopBackend(CFG, station)
    assert station.opened == []


def test_per_subtask_settings_and_caps_only_go_down(tmp_path):
    st = FakeStation()
    plug = client_settings(CFG, SUB["plug_usb"])
    assert plug["execution_horizon"] == 8 and plug["always_async"] and plug["ik"]["solve"] == "traj"
    assert client_settings(CFG, SUB["close_backing"])["execution_horizon"] == 20
    box = {"kind": "box", "center": [0.3, 0, 0.1], "size": [0.2, 0.4, 0.3]}
    plug["ik"] = {"keep_out": {"left": [box]}, "region": {"right": {"lo": [0, -0.3, 0], "hi": [0.5, 0.3, None]}}}
    d, kw = apply_settings(st.policy_config(None), st.loop_settings(), plug, tmp_path / "t.csv")
    assert d["execution_horizon"] == 8
    assert d["ik"]["keep_out"]["left"][0]["kind"] == "capsule" and d["ik"]["keep_out"]["left"][1] == box  # added
    assert d["ik"]["region"]["right"]["hi"][2] is None
    assert kw["control_hz"] == pytest.approx(400 * 0.7) and d["action_hz"] == kw["control_hz"]
    assert kw["max_ee_speed_m_s"] == pytest.approx(0.6 * 0.7) and kw["trace_path"] == str(tmp_path / "t.csv")
    fast = {**plug, "speed_scale": 3.0, "ik": {}}
    assert apply_settings(st.policy_config(None), st.loop_settings(), fast)[1]["max_ee_speed_m_s"] == 0.6
    with pytest.raises(ValueError):
        apply_settings(st.policy_config(None), st.loop_settings(), {**plug, "execution_horizon": 49})


def test_horizon_slicing_prefix_pacer():
    t = [0.0]
    inner = FakeClient()

    def slow_action(**kw):
        t[0] += 0.2  # a 200 ms round trip = 4 rows at 20 Hz
        return {}

    inner.get_action = slow_action
    p = PrefixPacer(inner, horizon=8, action_hz=20.0, always_async=True, clock=lambda: t[0])
    assert not p.refill_due(0, 48)  # no round trip measured yet: lead 0, and 0 < 8 rows executed
    p.get_action()
    # ask once executed + 4 lead rows reach the 8-row horizon, i.e. after 4 rows, not after 30 or 48
    assert [p.refill_due(e, 48 - e) for e in range(6)] == [False, False, False, False, True, True]
    assert p.refill_due(0, 0)  # empty buffer always refills
    assert p.refill_due(0, 9)  # 9 rows = 0.45 s <= rtt + 0.25 s: runway guard
    off = PrefixPacer(inner, horizon=8, action_hz=20.0, always_async=False)
    assert not off.refill_due(20, 28)  # defers to the client's own (here: sync) pacer
    off.close()
    assert inner.closed  # everything else passes through to the client


def test_stop_condition_ends_episode_dry_by_default(arms_ok, tmp_path):
    st = FakeStation()
    seen = []

    def stop_when(tick):
        seen.append(tick)
        return tick.n >= 5

    be = ClientLoopBackend(CFG, st, stop_when=stop_when, trace=tmp_path / "t.csv")
    be.start(SUB["plug_usb"]["prompt"], None)
    wait_for(lambda: be.state() == "idle" and be.end)
    assert st.opened == [True]  # dry
    loop = st.loops[0]
    assert loop.steps == 5 and loop.exited and st.clients[0].closed and be.end == "stop_condition"
    assert seen[-1].video == {"overhead": "frame5"} and seen[-1].state["grip_l"] == pytest.approx(0.5)
    assert loop.task == SUB["plug_usb"]["prompt"] and loop.client.horizon == 8 and loop.client.always_async
    be.home()
    assert st.homes == 0  # a dry run never homes
    summary = be.collect(tmp_path / "a1")
    assert summary == {} and (tmp_path / "a1" / "client_loop.json").exists()


def test_stop_and_error(arms_ok, tmp_path):
    st = FakeStation()
    be = ClientLoopBackend(CFG, st, dry_run=False, trace=tmp_path / "t.csv")
    be.start(SUB["open_backing"]["prompt"], "dream-1-ft:x")
    wait_for(lambda: be.state() == "running")
    be.stop()
    assert be.state() == "idle" and be.end == "stopped" and st.loops[0].exited
    assert st.loops[0].client.horizon == 20 and not st.loops[0].client.always_async

    class ArmError(Exception):
        pass

    async def boom(now_s):
        raise ArmError("can bus gone")

    st.make_loop = lambda c, t, s: setattr(loop := FakeLoop(c, t, s), "step", boom) or loop
    be.start(SUB["close_backing"]["prompt"], None)
    wait_for(lambda: be.state() == "error")
    assert "can bus gone" in be.last_error
    be.home()
    assert st.homes == 0  # after an arm fault nothing homes
    be.close()
    assert st.closed


def test_unknown_prompt_uses_defaults(arms_ok, tmp_path):
    st = FakeStation()
    be = ClientLoopBackend(CFG, st, stop_when=lambda t: True, trace=tmp_path / "t.csv")
    be.start("wave hello", None)
    wait_for(lambda: be.end)
    assert st.loops[0].client.horizon == client_loop.DEFAULTS["execution_horizon"]
