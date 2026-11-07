# Demo protocol — operator card (print / keep open)

Full rationale: `docs/research/04-company-blogs.md` § Data collection protocol. Kit docs: `dreamlabs-co/physical-ai-hack-2026/docs/data_collection.md`.

## Fixed task strings — NEVER change (same at record and inference)
| Subtask | Folder | `--task` |
|---|---|---|
| unplug_usb | `~/hackathon_data/hc-unplug` | `unplug the usb cable from the rack` |
| open_backing | `~/hackathon_data/hc-open` | `open the rack back panel` |
| close_backing | `~/hackathon_data/hc-close` | `close the rack back panel` |
| plug_usb | `~/hackathon_data/hc-plug` | `plug the new usb cable into the rack` |
| full run (5–10 eps) | `~/hackathon_data/hc-full` | `unplug the usb, open and close the rack back panel, and plug in the new usb` |

_TODO(event): if the rules' wording or the prompt search (`docs/PROMPTS.md`, 10:25) picks different strings, rewrite them once before demo sprint 1 (10:40) and freeze them. They must match `hillclimb/config.yaml` (subtasks) and `FULL_TASK` in `tools/ft.py` (full run)._

```bash
# laptop
ssh -L 8080:localhost:8080 <user>@<station>
# station
tmux new -A -s collect
cd ~/physical-ai-hacks        # or ~/physical-ai-hack-2026: wherever the kit is checked out
uv run dhk-collect --config configs/station.yaml --dataset-dir ~/hackathon_data/hc-plug --task "plug the new usb cable into the rack"
```
Quest: **grip** = drive that arm (works like a clutch: release it, re-center your hand, grip again) · **trigger** = close gripper · **A** = start/save · **B** = discard · **right stick click** = arms back to start (not while recording). Keep your head still and facing the arms.

## Before the first demo (off-arm, 15 min)
1. Write the **canonical strategy** for each subtask below in one line, and get all operators to agree. One operator does ~70% of the demos.
2. Check feasibility: can the gripper pinch the USB plug body next to the other cables? Can it reach the panel latch or edge?
3. Photograph the camera mounts and the rack position (we restore them after other teams' slots).
4. Zero-shot baseline (on-arm, hand on the e-stop): RUNBOOK § 10:10, through the harness with `--judge confirm`. Results go in `evals/zero-shot.md`. Weight the demo budget toward the worst subtask.

## Canonical strategies (fill in at bring-up, 10:00–10:40, before demo sprint 1)
- **unplug_usb:** _which arm grips the plug body, how it pulls (straight out along the port axis), where it drops the cable_
- **open_backing:** _which arm, grip point, motion (hinge arc / slide), how far open = "open"_
- **close_backing:** _which arm, push point, until fully latched_
- **plug_usb:** _grasp the new cable at a fixed point on the plug body (orientation marked), hover ~2 cm in front of the port, align, pause briefly, push straight in until seated_

## Rules while recording
- **Start moving within 0.5 s** of pressing A. Press A to save as soon as the subtask is done. Leave no idle time at either end.
- Move smoothly and decisively at a consistent speed. Don't dither or freelance.
- Keep each episode 10–30 s (minimum 1.6 s).
- **Vary the scene, not the method:** cable start pose, cable drape, panel start angle, small rack offsets if allowed. Spread the variations evenly (~10 demos per variation).
- **Natural recovery is fine:** if the plug misses and you calmly re-align and finish, keep it. If the operator was sloppy, press **B**.
- No hands or people in frame. Don't touch the cameras. Keep force and speed low near the rack.
- The curator watches the dashboard and keeps a tally per folder: good / recovery / dropped.

## Lessons from REBOOT (arXiv 2609.22591: real bimanual USB-A/USB-C insert and remove, 60 expert + 60 recovery demos per task)
- **Insertion is the bottleneck:** with 60 expert demos, policies averaged **16% on installs vs 56% on removals**. Removal is mostly a free-space pull from a known grasp. Spend demos on plug_usb, not unplug_usb.
- **Where installs fail:** at the port. **Align(place)** (connector axis off the port centre) and **Engage(place)** (jams partway in). The fix they demonstrate: **retract, re-align, re-approach**. Never keep pushing on a jam.
- **USB-A has one valid orientation (C1 symmetry):** keyed connectors fail more at engage. **Grasp the new plug the same way every time, logo/orientation marked.** Ask the organizers whether the new cable's start orientation is fixed.
- **Diversity recipe:** vary the object's start pose within a **3–6 cm radius** of a nominal pose and keep the goal (port) fixed. That is enough that the policy can't memorise one path, and small enough for ~60 demos to cover.
- **Recovery ratio:** **1:1 expert : recovery**, with recovery episodes allocated by phase **in proportion to the failures logged in autonomous rollouts** (logs → `tools/report.py`). Record from the start of the subtask, through the induced failure, to success.

## Budget (slot 1, before FT#1 at ≤13:30)
plug_usb **50–60** · open_backing **40** · unplug_usb **20** · close_backing **25** · full **5**
_(Re-weight from the zero-shot results.)_

**Reality check:** one station, Quest teleop, scene reset between episodes ≈ **1–1.5 min per kept episode**, so 10:40–13:15 yields **~100–150**. Record in priority order so that a cut at any moment leaves a usable dataset:
1. **Minimum viable FT#1** (~60 eps, ~75 min): plug 25 · open 20 · unplug 8 · close 8. Enough to queue early if the queue is racing (PLAYBOOK § Queue race).
2. **Then top up:** plug → 50–60, open → 40, close → 25, unplug → 20, full → 5.
If two stations are booked, run two operators in parallel on different subtasks, each with their own `--dataset-dir`.

## Slot 2+ (while FT#1 trains → data for FT#2)
- **Recovery demos from staged failure states**, 5–10 each, using the **same** task string as the subtask:
  - plug resting just off the port → back off 1 cm, realign, insert
  - plug inserted at an angle or not seated → pull back, re-insert
  - cable dropped on the rack shelf → re-grasp and continue
  - panel ajar or half-closed → re-grip and finish
  - plug still partly in the port after a weak pull → re-grip and pull straight
- **Precision-phase demos (~10):** start the episode already hovering 2–3 cm in front of the port, then insert.
- Pull the harness's logged failure states (`logs/harness/<run>/attempts.jsonl`, `tools/report.py logs/harness`) and recreate the most common ones first.

## Hand-off to fine-tune (curator)
Rehearsed offline end to end: `docs/research/09-pipeline-rehearsal.md`. **Never `mv` the -eef folders**: a later `add_eef --replace` then rebuilds beside the source, and the parent silently keeps the old copy (Dream's check still passes).
```bash
cd ~/Hillclimb
# free, local, nothing uploaded: add_eef every hc-* folder into a temp parent + Dream's checks. Run it any time.
tools/rehearse_pipeline.sh
# build the REAL parent (one <folder>-eef session per subtask, videos hard-linked). Re-run it after adding episodes:
# it rebuilds the parent from scratch, so it can never go stale.
tools/rehearse_pipeline.sh --parent ~/hackathon_data/hc-ft1      # ends with ft.py check: must print PASS
# PAID ($200 per 1000 steps), cannot be cancelled. Get Sissi's OK + the organizers' credits first.
# Use OUR launcher (keeps lr_schedule=constant so FT#2 can resume_from FT#1). It re-checks for stale sessions first.
~/dream-env/bin/python tools/ft.py launch ~/hackathon_data/hc-ft1 --name hc-ft1 --suffix hc-ft1 --steps 1000
```
FT#2 (queue ≤21:30): harvest `~/dream-env/bin/python tools/harvest.py logs/harness/2026* --out ~/hackathon_data/self --retime-fps 30`, then `tools/rehearse_pipeline.sh --parent ~/hackathon_data/hc-ft2 --self` (demos + harvested `self/hc-*-self`), then
- FT#1 landed and passes a quick check: `ft.py launch ~/hackathon_data/hc-ft2 --name hc-ft2 --suffix hc-ft2 --steps 500 --resume-from dream-1-ft:<uid8>-hc-ft1`.
- FT#1 not landed yet: same, but `--steps 1000` and no `--resume-from` (fresh from base on ALL data). Don't wait for FT#1.

Keep recording into the same `hc-<subtask>` folders; never edit `hc-ft1/` (it is the record of what FT#1 trained on).
