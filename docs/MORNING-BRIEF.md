# Morning brief (Sat Nov 7) — read in 2 minutes

## Be there at 09:00 sharp
The rules and scoring are announced at 09:00. Arms and station booking open at **10:00**. The venue closes at 22:00. Submissions are due **Sun 15:00**.

## Decisions only you (Sissi) can make
1. **Fine-tune spend.** Plan: FT#1 at 1000 steps (~$200) by 13:30, then FT#2 at 500 resumed steps (~$100) by 21:30. Use organizer credits if they exist. Nothing paid launches without your OK; `tools/ft.py launch` asks you to type QUEUE.
2. **Roles** (PLAYBOOK § Roles): harness lead, 2 Quest operators, a curator (also books stations and watches the FT queue), and an eval lead.
3. **Merge PR #2** (`research/claude-scout` → `main`) once you've skimmed it. Codex review: clean at a21a3bc (all earlier findings fixed with regression tests).

## The first 3 actions
1. **09:00:** fill in `context/rules.md` during kickoff and ask the organizer questions (PLAYBOOK § Questions for organizers). The most important: are a custom loop, replay and the VLM judge allowed in scored runs? How does training time scale with steps? Are there team credits?
2. **10:00:** the curator books the earliest station slots, plus Sat evening and Sun 10:00–14:30, and runs `tools/ft.py queue`.
3. **10:00:** the harness lead follows `docs/RUNBOOK.md` top to bottom: bring-up, zero-shot baseline, prompt search, freeze prompts, then demo sprint 1 from 10:40.

## What's ready (all offline-tested, 65 tests; nothing has run on real arms yet)
- **Harness:** 4 subtask episodes, a VLM judge (92% on REBOOT frames, 0 false successes), a proprio veto, retries and recovery. Judge modes are `vlm`, `confirm` and `human`.
- **Fine-tune tooling:** `ft.py` (steps, base_share, resume_from, constant LR, queue) and `rehearse_pipeline.sh` (free dry run of the data → FT pipeline).
- **Data engine:** recovery demo protocol, `harvest.py` (judged successes become training data) and `report.py` (success rates with confidence intervals).
- **Options gated on the rules:** our own client loop (per-subtask horizon, keep-out boxes) and replay primitives.

## Biggest unknowns (fill in at the rig)
Rack/port geometry and USB orientation · camera names · base checkpoint id · gripper thresholds · whether the station has internet for Fireworks · whether `rollouts` recording is allowed.
