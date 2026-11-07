# 08 — Code review before the 08:15 station bring-up (2026-11-07)

Scope: `hillclimb/*.py`, `tools/*.py`, `hillclimb/config.yaml`, checked against `docs/RUNBOOK.md` / `PLAYBOOK.md` and the vendor code
(`robots/station.py`, `station_api.py`, `control.py`, `run.py`, `config.py`, `dream/facade.py`, `physical-ai-hack-2026`; dreamlabs 1.0.37).
Line numbers are from **before** the fixes. `tools/api_check.sh` against `~/hc-vendor/dream-env-test`: 0 attribute and 0 signature problems.
Tests: 36 → 46 passing (`tests/test_review_fixes.py`, plus `tests/conftest.py` so tests never load the real `.env`).

## Findings, fixed

| # | Sev | Where | Failure scenario | Fix |
|---|---|---|---|---|
| 1 | **HIGH (safety)** | `harness.py:99-116` | Any exception between `/start` and `/stop` skipped `backend.stop()`. A `/status` 5xx or socket error, a judge crash (see #3), or **the operator pressing Ctrl-C on the harness** left the station running the policy. A station run has `steps=0` and runs until it is told to stop (`station.py:908-913`). The arms kept moving with nobody driving the harness. | The start and poll loop are wrapped in `try/except BaseException`, which does a best-effort `stop()` and then re-raises. If that stop fails too, the harness prints "PRESS THE E-STOP". |
| 2 | **HIGH (false success)** | `judge.py:118-123` | `CombinedJudge` fell back to the **proprio verdict as a positive** whenever the VLM was unknown (no key, network down, no frames, unparsable reply). `close_backing` (`expect_holding: false`) then passed on nearly every attempt, because the gripper is not holding anything. `unplug_usb` passed whenever a gripper was clamped on a cable that was still in the port. The harness moved on, and `harvest.py` would train on those runs. `judge_eval.py` says unknown counts as not-success, so the harness disagreed with it. | Proprio can now only veto. When the VLM is unknown, the verdict stays unknown, which the harness logs as a failure and retries. |
| 3 | MED | `judge.py:93-97` | Reading the frame bytes sat outside the `try`. If the loop `rmtree`s the inputs folder (`control.py:1758-1760`) between the listing and the read, the result was `FileNotFoundError`. Together with #1, that crashed the harness mid-run with the arms moving. | The read is now inside the `try`, so a missing file gives an unknown verdict. |
| 4 | MED | `harness.py:109,119` | Until the loop opens its trace, which happens after the client build (seconds), the `_inputs` folder and the CSV still hold the **previous run's** frames. A mid-run check could judge the last run's final frame against this subtask's rubric. A run stopped while still `starting` (a slow build or a timeout) had stale frames copied and judged as its own. | `_fresh()` only judges frames whose mtime is at or after the attempt's Start. `copy2` keeps mtimes, so the filter also applies to the copied final frames. |
| 5 | MED | `judge.py:108` | `bool(data.get("success"))` turned a reply of `"success": "false"` (a string) into True. | The check is now `data.get("success") is True`. |
| 6 | MED | `harness.py:93,124` + `station.py:478-479` | The station keeps the last checkpoint any Start named (`self.checkpoint = …`, "for this run and the ones after it"). After one `--checkpoint ft1` sweep, a later run with `checkpoint: null` still runs ft1, but the log said `None` / "station-default". The A/B table in `report.py` then credits ft1's results to the base model. | The logged `checkpoint` is now the one `/status` reports after the Stop, when there is one. A RUNBOOK note says the same. |
| 7 | MED (runbook) | `RUNBOOK.md:20-30`, `judge.py:74` | The RUNBOOK writes `FIREWORKS_API_KEY` to `.env` but never sources it, and nothing loads `.env`. Run as written, every VLM verdict is unknown. Before fix #2, that meant `close_backing` and `unplug_usb` passed on proprio alone. After it, every attempt fails. | `harness.main()` loads the repo `.env` without overriding variables that are already set, and prints a loud warning when no key is set. RUNBOOK updated. |
| 8 | MED (safety) | `client_loop.py:225-230` | `stop()` waits at most 30 s for the episode thread. If that join timed out, `home()` (from `recover()` or `close()`) ran `return_to_home` while the loop thread could still be writing the arms, so two writers. | `home()` refuses while the episode thread is alive. |
| 9 | LOW (runbook) | `RUNBOOK.md:19-21` | `~/dream-env/bin/python -m pytest -q` needs pytest, and the pip line did not install it. | Added `pytest` to the pip line. |

## Findings, not fixed (questions / speculative)

1. **`unplug_usb` rubric vs proprio contradict** (`config.yaml` unplug_usb). The rubric accepts a plug that "lies clear of the rack", but `expect_holding: true` vetoes it. A drop after a clean unplug is therefore a failure. The retry then starts with the plug already out, and after 3 attempts `on_fail: abort` ends the whole task. Should it be `expect_holding: null`, or should the rubric be tightened to "held"? The team should decide.
2. **Station trace path is hard-coded** (`backends.py:28`, `StationBackend` ignores `station_file`). If `station.yaml` sets `trace_path`, the harness finds no frames and every attempt is unknown, so it fails, silently. Check `trace_path` in `~/.dreamlabs/station.yaml` at bring-up, or read it from the file.
3. **A mid-run VLM call blocks the poll loop** for up to 30 s (`judge.py:80` timeout). During that time `max_time_s` is not enforced and the policy keeps running. This is fine at about 1 s per call, but a lower mid-run timeout may be better.
4. **Station 409s crash the harness** (with the arms holding, which is safe). This happens on `/stop` during a `homing` (a key refused as `disabled` or `over_cap` triggers an automatic home, `station.py:969-993`), and on `/home` with a dead arm driver. Should recovery treat a 409 as "skip home and retry", or abort?
5. **The dry-run backend's `max_time_s` includes `dreamcli` process startup** (cameras, client, IK warm-up up to 20 s, `run.py:150-160`). A 25 s dry attempt may get only a few seconds of policy, so do not read dry-run timeouts as policy failures.
6. **Total budget is checked only between attempts** (`harness.py:158`), so the last attempt and its recovery can overrun `total_time_s` by about `max_time_s` + a home move.
7. **Primitive joint convention.** The kit's MakerMods arms speak "motor degrees, zero at the arm's calibration pose" (`dream_hackathon/arms/makermods.py:9`). `tools/waypoints.py extract` assumes these equal `robots` `positions_rad`. If the calibration zeros differ, the `max_approach_rad` 1.0 check is the only guard. Before the first live replay, compare a recorded waypoint against `waypoints.py record` at the same pose.
8. **Kit path.** `tools/ft.py:36` and the RUNBOOK say `~/physical-ai-hacks`, but the vendor repo is `physical-ai-hack-2026`. Confirm on the station.
9. **Mid-run newest JPEG may be half-written** (`control.py:1798` writes non-atomically). The likely result is an unknown verdict (VLM error), not a crash. Low.
10. `speed_scale` lowers `action_hz` handed to the dream client (`client_loop.py:82`). `client_ik` sizes its per-row speed caps from it (`client_ik.py:128`), so the caps scale consistently, but this is untested on hardware.

## Checked and OK
- **Arm-motion gates.** `StationBackend`, `ClientLoopBackend` (even dry) and `primitives.play` each refuse without `HILLCLIMB_ALLOW_ARMS=1`. The client backend is dry unless `--client-live`, and Metal and Makers both have `open_arm_read_only`. `play()` holds both arms in `finally`.
- **Station API contract.** `POST /start {prompt, checkpoint}`, `/stop`, `/home` and `GET /status["state"]` match `station_api.py:144-184`. The state names match `station.py:78-84`. Port 8091 = `DEFAULT_API_PORT`, and `/status.run` = `run_number` = the rollout folder's `_run<N>`.
- **Trace/frame paths.** `<trace stem>_inputs/<cam>/<t:.3f>.jpg` (`control.py:136-145,1798`), the CSV columns used by `proprio_summary` (`control.py:116-133`), and the `ControlLoop` client calls (`get_action(commanded=)`, `refill_due(executed, remaining)`) that `PrefixPacer` wraps.
- **Harvest schema.** `rollout.json` `task` / `run` / `started_at_utc` / `status` / `frames` (`rollouts.py:230-271`) and the attempt keys `report.py` and `harvest.py` read are all still written. `dream.fine_tuning.training_data(model)` returns `prompts`, as `ft.py prompts` expects.

## Follow-ups resolved by the coordinator (04:30)
- Q1: `unplug_usb` is now `expect_holding: null`. A clean unplug that drops the plug clear of the rack counts as success.
- Q2: `StationBackend` reads `trace_path` from `station_file` (`backends.trace_from_station_file`), with a test.
- Q4: recovery `home` treats a station 409 as "skip home and retry from where the arms are"; other errors still raise, with a test.
- Q3 and Q5 are left as notes: mid-run VLM calls take about 1 s, and dry-run timeouts include process startup.
