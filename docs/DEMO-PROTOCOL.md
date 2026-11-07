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

_TODO(event): if the rules' wording differs, rewrite these strings once at 08:30 and freeze them. They must match `hillclimb/config.yaml`._

```bash
# laptop
ssh -L 8080:localhost:8080 <user>@<station>
# station
tmux new -A -s collect
cd ~/physical-ai-hacks
uv run dhk-collect --config configs/station.yaml --dataset-dir ~/hackathon_data/hc-plug --task "plug the new usb cable into the rack"
```
Quest: **grip** = drive that arm (works like a clutch: release it, re-center your hand, grip again) · **trigger** = close gripper · **A** = start/save · **B** = discard · **right stick click** = arms back to start (not while recording). Keep your head still and facing the arms.

## Before the first demo (off-arm, 15 min)
1. Write the **canonical strategy** for each subtask below in one line, and get all operators to agree. One operator does ~70% of the demos.
2. Check feasibility: can the gripper pinch the USB plug body next to the other cables? Can it reach the panel latch or edge?
3. Photograph the camera mounts and the rack position (we restore them after other teams' slots).
4. Run zero-shot on each subtask, 3–5 tries each: `dreamcli robot run --task "<string>" --dry-run`, then live with a hand on the e-stop. Log results in `evals/zero-shot.md`. Weight the demo budget toward the worst subtask.

## Canonical strategies (fill in at 08:30 when we see the rig)
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

## Budget (slot 1, before FT#1 at ~12:30)
plug_usb **50** · open_backing **40** · unplug_usb **25** · close_backing **25** · full **5**
_(Re-weight from the zero-shot results.)_

## Slot 2+ (while FT#1 trains → data for FT#2)
- **Recovery demos from staged failure states**, 5–10 each, using the **same** task string as the subtask:
  - plug resting just off the port → back off 1 cm, realign, insert
  - plug inserted at an angle or not seated → pull back, re-insert
  - cable dropped on the rack shelf → re-grasp and continue
  - panel ajar or half-closed → re-grip and finish
  - plug still partly in the port after a weak pull → re-grip and pull straight
- **Precision-phase demos (~10):** start the episode already hovering 2–3 cm in front of the port, then insert.
- Pull the harness's logged failure states (`runs/`) and recreate the most common ones first.

## Hand-off to fine-tune (curator)
```bash
# per folder, rebuild with --replace after adding episodes
~/dream-env/bin/python tools/add_eef.py ~/hackathon_data/hc-plug --arm metal --replace
# collect all -eef folders under one parent → one multi-task dataset
mkdir -p ~/hackathon_data/hc-ft1 && mv ~/hackathon_data/hc-*-eef ~/hackathon_data/hc-ft1/   # TODO(event): verify the parent layout works
~/dream-env/bin/python tools/finetune.py ~/hackathon_data/hc-ft1 --arm metal --check          # free, local
# PAID ($200 per 1000 steps), cannot be cancelled. Get Sissi's OK + the organizers' credits first.
# Use OUR launcher (keeps lr_schedule=constant so FT#2 can resume_from FT#1):
~/dream-env/bin/python ~/Hillclimb/tools/ft.py launch ~/hackathon_data/hc-ft1 --name hc-ft1 --suffix hc-ft1 --steps 1000
```
