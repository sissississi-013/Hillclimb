"""tools/ft.py inspect_parent(): the pre-upload checks Dream's validate_dataset does not make (09-pipeline-rehearsal)."""

import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ft", ROOT / "tools" / "ft.py")
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)  # must import without the `dream` package (lazy imports)

PLUG = "plug the new usb cable into the rack"


def session(path, episodes=3, width=(34, 32), task=PLUG, frame=(360, 640, 3)):
    (path / "meta").mkdir(parents=True)
    feats = {"observation.state": {"shape": [width[0]]}, "action": {"shape": [width[1]]},
             "observation.images.scene": {"dtype": "video", "shape": list(frame)}}
    (path / "meta/info.json").write_text(json.dumps({"total_episodes": episodes, "total_frames": 60 * episodes,
                                                     "fps": 30, "features": feats}))
    (path / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": task}) + "\n")
    return path


def test_clean_parent_passes(tmp_path):
    session(tmp_path / "hc-plug", width=(14, 14))
    session(tmp_path / "ft1/hc-plug-eef")
    info, warn, err = ft.inspect_parent(tmp_path / "ft1", prompts=ft.known_prompts())
    assert len(info) == 1 and not warn and not err


def test_stale_eef_after_more_demos(tmp_path):
    """The RUNBOOK's old `mv` flow: add_eef --replace after the move rebuilds beside the source, not in the parent."""
    session(tmp_path / "hc-plug", episodes=4, width=(14, 14))
    session(tmp_path / "ft1/hc-plug-eef", episodes=3)
    _, _, err = ft.inspect_parent(tmp_path / "ft1")
    assert len(err) == 1 and "STALE" in err[0] and "--replace" in err[0]
    assert ft._stale_ok(err, allow=True) == []


def test_harvest_source_found_under_self(tmp_path):
    session(tmp_path / "self/hc-plug-usb-self", episodes=2, width=(14, 14))
    session(tmp_path / "ft2/hc-plug-usb-self-eef", episodes=1)
    assert "STALE" in ft.inspect_parent(tmp_path / "ft2")[2][0]


def test_raw_session_unknown_task_and_mixed_frames(tmp_path):
    session(tmp_path / "p/hc-plug", width=(14, 14))
    session(tmp_path / "p/hc-plug-usb-self-eef", task="plug the usb", frame=(480, 640, 3))
    _, warn, err = ft.inspect_parent(tmp_path / "p", prompts=ft.known_prompts())
    assert any("raw" in e for e in err)
    assert any("not a frozen prompt" in w for w in warn) and any("frame sizes differ" in w for w in warn)


def test_harvest_root_is_not_a_parent(tmp_path):
    session(tmp_path / "self/hc-plug-usb-self", width=(14, 14))
    (tmp_path / "self/harvest.jsonl").write_text("{}\n")
    (tmp_path / "self/README.md").write_text("notes are fine\n")
    _, _, err = ft.inspect_parent(tmp_path / "self")
    assert len(err) == 1 and "harvest.jsonl" in err[0]


def test_frozen_prompts_include_every_subtask_and_full_run():
    prompts = ft.known_prompts()
    assert PLUG in prompts and ft.FULL_TASK in prompts
    assert ft.FULL_TASK in (ROOT / "docs" / "DEMO-PROTOCOL.md").read_text()
