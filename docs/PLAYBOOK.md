# Hillclimb PLAYBOOK — how we take 1st

_Living doc. Owner: research/claude-scout. Last update: 2026-11-07 ~04:30 PST. Sources: `docs/research/01–05`._
_If you change the plan, post `PLAN-CHANGE:` on issue #1._

## TL;DR (read this if nothing else)
1. **The clock is set by the fine-tune queue, not by us.** One Dream-1 fine-tune takes about 7–8 h of training, plus queue time, plus up to 1 h before the model first answers. It can't be cancelled and costs $200 for 1000 steps. **That gives us 2 shots:** FT#1 queued by **~13:00 Sat**, then FT#2 (`resume_from` FT#1, adding correction data) queued by **~23:00 Sat**, ready Sun morning.
2. **Split the task into subtasks; don't use one long prompt.** The prompt is fixed for each Dream episode. The harness runs 4 short episodes, one per subtask, each with its own fixed prompt: `unplug_usb → open_backing → close_backing → plug_usb`. Each subtask has its own success check, capped retries and a recovery move. Splitting one connector task this way took it from ≤10% to 100% (arXiv 2610.40828).
3. **Put data into the weak skills.** Dream-1 zero-shot: insert&fit 24%, open&close 16%. Most demos go to **plug_usb** and **open_backing**. Record one dataset folder per subtask, each with its own `--task` text, and train one multi-task fine-tune.
4. **Corrections beat more clean demos.** Run the policy, and when it's about to fail, record a demo that starts from that failure state and recovers. This is about 10× more data-efficient (RaC 2509.07953; Dyna went from 80% to 99.4%). The organizers explicitly name "fine-tuning, DAgger, System-2 harness".
5. **Clean, consistent data.** One written strategy for all operators. Trim idle time at the start and end. Discard sloppy demos with B. Never move the cameras. 10–30 s per episode.
6. **Free accuracy from the IK and execution settings:** `keep_out` boxes around the rack body, a `region` box per arm, `solve:"traj"`. Execute **short chunk prefixes during contact** (~0.4 s) and longer ones (~1–1.5 s) in free space.

## Timeline (Sat Nov 7 → Sun Nov 8, PST)
| When | Who | What |
|---|---|---|
| 08:00–09:00 | everyone | Get the rules (see questions below). SSH into the station, `--dry-run`, read `dreamcli` zero-shot on each subtask. **Measure the zero-shot baseline per subtask** (5 tries each). |
| 09:00–12:30 | 2 operators + 1 curator | **Data sprint 1**: ~40–60 demos for plug_usb, ~40 for open_backing, ~25 each for unplug/close, + ~5 full runs. The curator watches dashboard playback live and drops bad episodes. |
| 12:30–13:00 | curator | `add_eef.py --arm metal` per folder → parent folder → `finetune.py --check` → **queue FT#1** (confirm credits with organizers and Sissi first). Settings: 1000 steps, `lr_schedule cosine`, `base_share 0.5`. |
| 13:00–22:00 | harness team | Build and tune the harness on the **base** model: success judge, retries, recovery, IK safety boxes, execution horizon per subtask. Log every failure state. |
| 13:00–22:00 | operators | **Recovery-data sprint**: start demos from the logged failure states (cable dropped, plug misaligned, panel half-open). Add more plug_usb demos with varied cable start poses. |
| ~22:00 | all | FT#1 ready → A/B against base on each subtask (≥10 tries each). Pick per-subtask which model to use (policy escalation). |
| ~23:00 | curator | **Queue FT#2** = `resume_from` FT#1 + recovery + correction data. Consider `base_share 0.0`. |
| Sun AM | all | FT#2 ready → final per-subtask model selection, harness tuning, many full scored-style runs. Push the leaderboard. |

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
- Code: `hillclimb/` (harness.py, judge.py, config.yaml) — being written against the real `dreamlabs` client API; see `docs/research/05-dream-client-api.md`.

## Innovation bets (try once the basics work)
- **Use the world model as a failure detector.** Dream-1 jointly predicts future frames. If the API exposes the predicted frames, compare predicted vs. actual: a large divergence means early failure, so retry before the subtask times out. _(Check in 05 whether the client returns predicted video.)_
- **Failure-state curriculum:** the harness logs where each attempt failed. Operators record recovery demos starting from exactly those states, so FT#2 learns from our real failure distribution.
- **Per-subtask execution horizon and speed**, tuned by real runs (phase-aware replanning took one robot from 51% to 70%, 2606.00537).

## Questions for organizers at 08:00 (must-ask)
1. The scoring rubric: partial credit per subtask? Time limit? Number of attempts, best-of or average? Tiebreaker?
2. Are our own control loop + VLM API + scripted moves allowed during **scored** runs? Can a human reset or touch anything between subtasks?
3. Fine-tune credits per team? Is there a priority queue for the event? Can they keep our model warm?
4. Are scored runs on the same station and prop we collect on? USB type (A/C) and orientation? What kind of panel (hinge/latch/slide)? What counts as "plugged"?
5. Is Dream-1-flash available? Which base checkpoint id should we use?
6. Can we scp datasets off the station? Does the station have internet access for our harness (Fireworks)?
7. Final leaderboard deadline and when the stage attempts happen.

## Roles (5-person team)
- **Operator A / B** — Quest teleop, one written strategy (`docs/DEMO-PROTOCOL.md`, TODO).
- **Curator** — dashboard playback, drops bad episodes, runs add_eef/finetune, owns the credits.
- **Harness lead** — `hillclimb/` state machine + recovery, on the base model all day.
- **Judge/eval lead** — VLM judge, eval logs, per-subtask success table, model selection.

## Avoid
- Training LoRA only (4/110 vs 130/200 for full action-expert fine-tunes in 2610.37170). Use the platform defaults.
- One long 4-step episode with a single prompt.
- Executing full 2.4 s chunks during contact.
- Moving cameras; letting different operators use different strategies; idle time in demos.
- Burning a fine-tune shot on a dataset we haven't checked by playback.
