import json
import pathlib

import pytest
import yaml

from hillclimb import backends, harness
from hillclimb.judge import CombinedJudge, ProprioJudge, Verdict, VLMJudge

CONFIG = pathlib.Path(__file__).parents[1] / "hillclimb" / "config.yaml"


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class Scripted:
    """Judge whose final verdicts per subtask come from a list. Mid-run checks (empty proprio) say 'unknown'."""

    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}

    def __call__(self, sub, frames, proprio):
        if not proprio:
            return Verdict(None)
        return Verdict(self.script[sub["name"]].pop(0), 1.0, "scripted", "test")


def make(tmp_path, script, cfg=None):
    clock = Clock()
    cfg = cfg or yaml.safe_load(CONFIG.read_text())
    be = backends.FakeBackend(clock=clock, run_s=5.0)
    h = harness.Harness(cfg, be, Scripted(script), tmp_path, clock=clock, sleep=clock.sleep)
    return h, be


def test_all_succeed_first_try(tmp_path):
    h, be = make(tmp_path, {n: [True] for n in ["unplug_usb", "open_backing", "close_backing", "plug_usb"]})
    assert h.run()
    assert [c[1] for c in be.calls if c[0] == "start"] == [s["prompt"] for s in h.cfg["subtasks"]]
    assert ("home",) not in be.calls
    lines = (h.dir / "attempts.jsonl").read_text().splitlines()
    assert len(lines) == 4 and json.loads(lines[0])["end"] == "run_ended"
    assert (pathlib.Path(json.loads(lines[0])["dir"]) / "verdict.json").exists()


def test_retry_with_home_recovery_then_success(tmp_path):
    h, be = make(tmp_path, {"unplug_usb": [False, True], "open_backing": [True], "close_backing": [True],
                            "plug_usb": [False, False, True]})
    assert h.run()
    starts = [c for c in be.calls if c[0] == "start"]
    assert len(starts) == 7
    # unplug failed once -> home; plug_usb has recovery: none -> no home
    assert be.calls.count(("home",)) == 1


def test_abort_after_max_attempts(tmp_path):
    h, be = make(tmp_path, {"unplug_usb": [False, False, False]})
    assert not h.run()
    assert len([c for c in be.calls if c[0] == "start"]) == 3
    assert all(a.subtask == "unplug_usb" for a in h.attempts)


def test_timeout_stops_run(tmp_path):
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["subtasks"] = [dict(cfg["subtasks"][0], max_time_s=2)]
    h, be = make(tmp_path, {"unplug_usb": [True]}, cfg)
    assert h.run()
    assert h.attempts[0].end == "timeout" and h.attempts[0].elapsed_s <= 2.5
    assert ("stop",) in be.calls


def test_total_budget(tmp_path):
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["total_time_s"] = 1
    h, be = make(tmp_path, {"unplug_usb": [False, False, False]}, cfg)
    assert not h.run()
    assert len(h.attempts) == 1


def test_station_backend_refuses_without_opt_in(monkeypatch):
    monkeypatch.delenv("HILLCLIMB_ALLOW_ARMS", raising=False)
    with pytest.raises(RuntimeError):
        backends.StationBackend()


def test_proprio_summary_and_veto(tmp_path):
    cols = ["t_s"] + [f"measured_l{i}" for i in range(2)] + [f"measured_r{i}" for i in range(2)] + \
        ["gripper_l", "gripper_r"] + [f"commanded_l{i}" for i in range(2)] + [f"commanded_r{i}" for i in range(2)] + \
        ["cmd_gripper_l", "cmd_gripper_r"]
    row = ["0", "0", "0", "0", "0", "0.3", "1.0", "0.1", "0", "0", "0", "0.0", "1.0"]
    (tmp_path / "t.csv").write_text(",".join(cols) + "\n" + ",".join(row) + "\n")
    p = backends.proprio_summary(tmp_path / "t.csv")
    assert p["l"]["max_track_err_rad"] == 0.1
    sub = {"name": "unplug_usb", "proprio": {"expect_holding": True}}
    assert ProprioJudge()(sub, {}, p).success is True  # left closed on something: grip 0.3 vs cmd 0.0
    p["l"]["grip"] = 0.0
    calls = []
    j = CombinedJudge(VLMJudge(post=lambda b: calls.append(b)), ProprioJudge())
    assert j(sub, {"overhead": tmp_path / "t.csv"}, p).success is False and not calls  # vetoed, VLM not called


def test_vlm_judge_parses_reply(tmp_path):
    img = tmp_path / "f.jpg"
    img.write_bytes(b"x")
    reply = {"choices": [{"message": {"content": 'ok {"description": "plug out", "score": 0.9, "success": true}'}}]}
    seen = {}
    j = VLMJudge(post=lambda b: seen.setdefault("b", b) and reply or reply)
    v = j({"name": "unplug_usb", "rubric": {"success": "out"}}, {"overhead": img}, {})
    assert v.success and v.score == 0.9 and v.reason == "plug out"
    assert seen["b"]["messages"][0]["content"][0]["text"].index("DESCRIBE") > 0
    assert VLMJudge(post=lambda b: {"bad": 1})({"name": "x"}, {"o": img}, {}).success is None


def test_prompts_match_demo_protocol():
    import pathlib, re
    import yaml
    root = pathlib.Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(open(root / "hillclimb" / "config.yaml"))
    protocol = (root / "docs" / "DEMO-PROTOCOL.md").read_text()
    frozen = set(re.findall(r"\| `([a-z][^`]+)` \|", protocol))
    for sub in cfg["subtasks"]:
        assert sub["prompt"] in frozen, sub["prompt"]


def test_attempt_records_wall_clock_prompt_and_station_run(tmp_path):
    clock = Clock()
    walls = iter([1_700_000_000.0, 1_700_000_012.5])
    be = backends.FakeBackend(clock=clock, run_s=5.0)
    collect = be.collect

    def collect_with_status(dest):
        out = collect(dest)
        (dest / "status.json").write_text(json.dumps({"run": 7}))
        return out

    be.collect = collect_with_status
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["subtasks"] = cfg["subtasks"][:1]
    h = harness.Harness(cfg, be, Scripted({"unplug_usb": [True]}), tmp_path, clock=clock, sleep=clock.sleep,
                        wall=lambda: next(walls))
    assert h.run()
    rec = json.loads((h.dir / "attempts.jsonl").read_text().splitlines()[0])
    assert rec["started_at"] == "2023-11-14T22:13:20.000+00:00"
    assert rec["ended_at"] == "2023-11-14T22:13:32.500+00:00"
    assert rec["prompt"] == cfg["subtasks"][0]["prompt"] and rec["station_run"] == 7


def test_report_tables(tmp_path):
    import json, sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
    import report
    run = tmp_path / "r1"
    run.mkdir()
    rows = [dict(subtask="plug_usb", success=s, elapsed_s=10.0, end="timeout", checkpoint=None,
                 started_at="2026-11-07T20:00:00.000Z") for s in (True, False, True)]
    (run / "attempts.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    out = report.report(report.load([tmp_path]))
    assert "| plug_usb | station-default | 3 | 2 | 67%" in out and "YES" in out
