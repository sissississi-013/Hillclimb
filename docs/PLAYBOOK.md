# Hillclimb PLAYBOOK — how we take 1st

_Living doc. Owner: research/claude-scout. Last update: 2026-11-07 ~04:30 PST. Sources: `docs/research/01–05`._
_If you change the plan, post `PLAN-CHANGE:` on issue #1._

## TL;DR (read this if nothing else)
1. **The clock is set by the fine-tune queue, not by us.** One Dream-1 fine-tune takes about 7–8 h of training, plus queue time, plus up to 1 h before the model first answers. It can't be cancelled and costs $200 for 1000 steps. **That gives us 2 shots:** FT#1 queued by **~13:30 Sat** (arms only open at 10:00). FT#2 must be queued **before the venue closes at 22:00 Sat**: resume from FT#1 if it's done, otherwise a fresh run on all data. It's ready Sunday morning, and submissions are due **15:00 Sun**.
2. **Split the task into subtasks; don't use one long prompt.** The prompt is fixed for each Dream episode. The harness runs 4 short episodes, one per subtask, each with its own fixed prompt: `unplug_usb → open_backing → close_backing → plug_usb`. Each subtask has its own success check, capped retries and a recovery move. Splitting one connector task this way took it from ≤10% to 100% (arXiv 2610.40828).
3. **Put data into the weak skills.** Dream-1 zero-shot: insert&fit 24%, open&close 16%. Most demos go to **plug_usb** and **open_backing**. Record one dataset folder per subtask, each with its own `--task` text, and train one multi-task fine-tune.
4. **Corrections beat more clean demos.** Run the policy, and when it's about to fail, record a demo that starts from that failure state and recovers. This is about 10× more data-efficient (RaC 2509.07953; Dyna went from 80% to 99.4%). The organizers explicitly name "fine-tuning, DAgger, System-2 harness".
5. **Clean, consistent data.** One written strategy for all operators. Trim idle time at the start and end. Discard sloppy demos with B. Never move the cameras. 10–30 s per episode.
6. **Free accuracy from the IK and execution settings:** `keep_out` boxes around the rack body, a `region` box per arm, `solve:"traj"`. Execute **short chunk prefixes during contact** (~0.4 s) and longer ones (~1–1.5 s) in free space.

## Timeline (official schedule in `context/schedule.md`; PST)
**Hard constraints:** rules at 09:00 · **arms + station booking from 10:00** · venue closes **22:00 Sat** · reopens 10:00 Sun · **submissions due 15:00 Sun** (then live scored attempts on stage).

| When | Who | What |
|---|---|---|
| **09:00** | everyone, **on time** | Kickoff: rules + scoring → `context/rules.md`. Ask the organizer questions (below). |
| 09:10–09:50 | split | Quick-starts (arms, Dream-1 API). Declare the **Hill Climb** track at 09:50. |
| **10:00 sharp** | 1 person | **Book stations the moment booking opens: as many early slots as allowed**, plus a Sat-evening slot (eval FT#1) and Sun 10:00–14:30 slots. |
| 10:00–10:40 | harness lead + operator | Bring-up (RUNBOOK), zero-shot baseline with `--judge confirm`, prompt search, freeze prompts. |
| 10:40–13:15 | 2 operators + curator | **Data sprint 1** (DEMO-PROTOCOL budget). The curator QA's live. |
| **≤13:30** | curator (+ Sissi's OK) | `rehearse_pipeline.sh --parent hc-ft1` → `ft.py launch` **FT#1** (1000 steps, constant LR, base_share 0.5) → ready **~21:00–22:00**. |
| 13:30–21:30 | harness team | Tune the harness on the **base** model: judge `scene` notes + judge_eval, retries, recovery, `pre:` primitives, horizons. Every run is logged (rollouts on). |
| 13:30–21:30 | operators | Recovery demos from logged failure states (1:1 with expert), precision-phase plug demos, harvest judged successes. |
| **≤21:30** | curator | **Queue FT#2 before the venue closes.** If FT#1 is done and passes a quick check: `--resume-from FT#1 --steps 500`. **Otherwise launch FT#2 from base on ALL data (1000 steps).** No waiting on FT#1. Ready Sun ~05:00–07:00. |
| Sun 10:00–14:30 | all | A/B base / FT#1 / FT#2 per subtask (`report.py`), pick per-subtask checkpoints, harness polish, many full runs. Freeze config by 14:30. |
| **Sun 15:00** | — | Submission due. Live scored attempts. |

## Per-subtask tactics (evidence → action)
| Subtask | Expected difficulty | Tactics |
|---|---|---|
| **unplug_usb** | easy-ish: removals ~56% vs installs 16% (REBOOT) | Grasp the plug body, not the cable. Pull straight along the port axis. Don't over-squeeze: the metal-arm-harness notes say the gripper overheats and drops the plug. Low demo budget (~20). Proprio check: gripper not fully closed = holding something. |
| **open_backing** | hard: Dream-1 open&close 16% zero-shot | Fixed geometry → strongest candidate for `pre:` (canonical grasp pose at the handle/edge) or `mode: replay` if allowed. Consider bimanual: one arm braces the rack, the other opens. Judge: "inside of rack visible". 40 demos, varying only the panel start angle. |
| **close_backing** | medium | Push from a fixed point until flush, then retract. Replay is likely enough. Judge needs a `scene` note for what "flush/latched" looks like. |
| **plug_usb** | **hardest, decides the leaderboard** | Keep the plug orientation consistent (USB-A is keyed). Use `pre:` hover ~2 cm in front of the port, then Dream-1 with a short execution horizon (client backend). Recovery = retract, re-align, re-approach; never push through a jam. Gripper-only retry: `recovery: none`, so we don't home and drop the cable. 50–60 demos + 1:1 recovery demos + ~10 precision-phase demos. |

## Fine-tune gotchas (verified in dreamlabs-client 1.1.3 `facade.py:312-398`)
- `resume_from` requires the parent model to have used the **constant** LR schedule, with the same lr, batch size and use_eef. **FT#1 must be constant.** The kit's `finetune.py` exposes none of `steps`, `base_share` or `resume_from`. Use **`tools/ft.py`** for those.
- `steps` must be a multiple of 100 in [500, 2000]. A resumed job is charged only for its new steps.
- `base_share`: 0.5 (default) or 0.0. A resumed job inherits its parent's value.
- The account has a limit on active jobs (`conflict`). Ask the organizers whether 2 can run in parallel. If so, launch FT#1a (base_share 0.5) and FT#1b (0.0) together, or one plug-only specialist.

## The harness (System 2)
```
for subtask in [unplug_usb, open_backing, close_backing, plug_usb]:
    for attempt in range(max_retries[subtask]):
        reset_episode(); run Dream with fixed prompt[subtask], horizon[subtask], timeout[subtask]
        verdict = judge(subtask, frames)          # VLM describe→score w/ frozen rubric + proprio check
        if verdict.success: break
        recover(subtask, verdict)                 # back off to pre-pose / re-grasp / re-run
    log(attempt frames, verdict)                  # → DAgger data mining
```
- **Judge:** the VLM (Fireworks) first describes the scene, then scores it against a **frozen rubric per subtask**. Rewording the prompt can flip the verdict (2609.05401). Use 2+ camera views (overhead + the active wrist). Add a cheap second signal: gripper width or stall shows whether the cable is held, and a joint-current spike shows whether the plug seated.
- **Policy escalation:** for each subtask, pick whichever model (base, FT#1, FT#2) or scripted primitive scores best on our own test runs. Selection is decided by real runs, not training loss (1X).
- **Scripted fallback (if the rules allow):** a small spiral or wiggle plus press for the last few mm of USB insertion. Use `metal-arm-harness`'s `tip`/`nudge`.
- Code: `hillclimb/` — `python -m hillclimb.harness --backend fake|dry-run|station [--only plug_usb]`. The station backend drives `dreamcli robot up`'s local API (127.0.0.1:8091) and needs `HILLCLIMB_ALLOW_ARMS=1`; a human runs it with a hand on the e-stop. Logs go to `logs/harness/<run>/attempts.jsonl`.
- `home` recovery opens the grippers, so plug_usb retries from where it is (`recovery: none`).
- Questions to add for 08:00: may scored runs use `dreamcli robot up` + our harness? Is `rollouts: enabled` allowed in the station file?

## Our edge (the "data engine")
**Nothing in Dream's stack detects success or ends an episode** (05 §0). Every team gets the same model. The winner is the team that gets the most out of each robot-minute. Our loop:
1. **Judged autonomous runs.** The harness runs the policy subtask by subtask. Every attempt gets a VLM + proprio verdict and is logged.
2. **Filtered self-rollouts → training data.** With `rollouts: {enabled: true}` in the station file, every policy run is saved as a LeRobot episode. `tools/harvest.py` keeps only the **judged-successful** runs, re-labels them with the frozen subtask prompt, and adds them to the FT#2 dataset. That is expert iteration: robot time spent evaluating also produces training data. _(06-harvest.md: rollouts are 20 fps on Metal and FaaS takes only 30/15/10/5, so harvest sessions need `control_hz: 30` or `--retime-fps 30`; the gripper unit and camera keys are bridged.)_
3. **Failure-state curriculum.** Logged failures tell operators which states to stage for recovery demos (same task string).
4. **Per-subtask control knobs** via our own `DreamClient` loop (`--backend client`): short execution horizon during insertion (phase-aware replanning: 51%→70%, 2606.00537), long in free space, IK `keep_out` around the rack. _(Only if the organizers allow a custom loop in scored runs.)_
5. **Policy escalation per subtask:** base vs FT#1 vs FT#2, chosen by our own logged success rates.

### Replay primitives (fixed geometry → less start-state variance)
The rack, port and panel never move, so the harness can put the arms in a **canonical pre-contact pose** with a deterministic joint replay before Dream takes over: e.g. "hover 2 cm in front of the port" before `plug_usb`, "gripper at the panel edge" before `open_backing`. Dream then always starts the contact-rich part from (nearly) the same state.
- **Defensible form = `pre:` + Dream.** A subtask's `pre: prims/<file>.yaml` plays before each attempt's Dream episode (after recovery); success still comes from the learned policy. This is the "System-2 harness" story.
- **`mode: replay`** makes the whole subtask the primitive, then judged (same retries/recovery). Plausible for open/close panel, which is fixed geometry. **ONLY if the organizers allow non-learned motions in scored runs** (question 8 below). Assume not until they say yes.
- Code: `tools/waypoints.py` (extract from a LeRobot episode, or `record` the current pose read-only), `hillclimb/primitives.py` (play), `ClientLoopBackend.play`. Needs `--backend client`: the station API has no joint move, since `dreamcli robot up` owns the arms.
- Safety: refuses without `HILLCLIMB_ALLOW_ARMS=1`; dry unless `--client-live`. Moves are our own conservative clamp: ≤0.5 rad/s per joint (also ≤ the station's `ramp_max_rad_s`), a straight joint blend to the first target, trajectories time-stretched under the cap, gripper ≤1/s. It refuses a first target >1 rad from the measured pose. The drivers still clamp to the soft joint limits. Holds on exit and after any fault.
- TODO(event): none of this has run on the station. A replay that starts from a different cable/panel state than the demo does not adapt to it. Re-record waypoints whenever the rig is touched.

_Dropped:_ world-model divergence as a failure detector. `get_action` returns no predicted frames (05 §0).

## Questions for organizers at 08:00 (must-ask)
1. The scoring rubric: partial credit per subtask? Time limit? Number of attempts, best-of or average? Tiebreaker?
2. Are our own control loop + VLM API + scripted moves allowed during **scored** runs? Can a human reset or touch anything between subtasks?
3. Fine-tune credits per team? Is there a priority queue for the event? Can they keep our model warm?
4. Are scored runs on the same station and prop we collect on? USB type (A/C) and orientation? What kind of panel (hinge/latch/slide)? What counts as "plugged"?
5. Is Dream-1-flash available? Which base checkpoint id should we use?
6. Can we scp datasets off the station? Does the station have internet access for our harness (Fireworks)?
7. Final leaderboard deadline and when the stage attempts happen.
8. May scored runs include **non-learned, scripted joint motions**: (a) a replay to a fixed pre-contact pose before the model runs (`pre:`), (b) a whole subtask such as open/close panel done by pure replay (`mode: replay`)?

## Roles (5-person team)
- **Operator A / B** — Quest teleop, one written strategy (`docs/DEMO-PROTOCOL.md`).
- **Curator** — dashboard playback, drops bad episodes, runs add_eef/finetune, owns the credits.
- **Harness lead** — `hillclimb/` state machine + recovery, on the base model all day.
- **Judge/eval lead** — VLM judge, eval logs, per-subtask success table, model selection.

## Avoid
- Training LoRA only (4/110 vs 130/200 for full action-expert fine-tunes in 2610.37170). Use the platform defaults.
- One long 4-step episode with a single prompt.
- Executing full 2.4 s chunks during contact.
- Moving cameras; letting different operators use different strategies; idle time in demos.
- Burning a fine-tune shot on a dataset we haven't checked by playback.
