# RUNBOOK: first 90 minutes at the station (Sat Nov 7)

Goal: by **09:30** we have a measured zero-shot baseline per subtask, frozen strategy and task strings, and demos flowing. By **~12:30** FT#1 is queued.
Safety: everything that moves an arm is run by a **human with a hand on the e-stop**. Agents only print commands.

## 08:00 — rules + logistics (whole team, 15 min)
- [ ] Ask the questions in `PLAYBOOK.md` § Questions. **Paste the answers into `context/rules.md` and push.** Every agent reads it.
- [ ] Get: station login, booking slots, team API key / credits, base checkpoint id, station internet (Fireworks reachable?).
- [ ] Photograph the rig: rack, port, panel, cable start, camera mounts → `context/rig/`.
- [ ] Freeze the 4 task strings. If you edit them, change `docs/DEMO-PROTOCOL.md` and `hillclimb/config.yaml` together. The test `test_prompts_match_demo_protocol` enforces this.

## 08:15 — station bring-up (harness lead)
```bash
# laptop
ssh -L 8080:localhost:8080 <user>@<station>
# station
tmux new -A -s hc
git clone https://github.com/sissississi-013/Hillclimb ~/Hillclimb   # private: use a PAT or scp a tarball:  (laptop) git archive HEAD | ssh <user>@<station> 'mkdir -p ~/Hillclimb && tar -x -C ~/Hillclimb'
cd ~/Hillclimb && ~/dream-env/bin/python -m pip install pyyaml numpy pyarrow pytest   # harness runs in the Dream env (needs `dream`, `robots`)
echo "FIREWORKS_API_KEY=..." > .env && chmod 600 .env                        # never commit. The harness loads it; other tools: set -a; . ./.env; set +a
~/dream-env/bin/python -m pytest -q                                           # all green, no hardware
bash tools/api_check.sh                                                       # our code vs the INSTALLED dream/robots API: expect 0 problems
~/dream-env/bin/python -c "import dream; print([m.id for m in dream.DreamClient().get_models()])"   # → base checkpoint id → config.yaml
```

## 08:25 — zero-shot baseline (operator on e-stop + eval lead)
```bash
source ~/dream-env/bin/activate
# 1) dry: model called, arms never move
python -m hillclimb.harness --backend dry-run --only unplug_usb
# 2) live via the station API (start `dreamcli robot up` in another tmux pane first)
HILLCLIMB_ALLOW_ARMS=1 python -m hillclimb.harness --backend station --only unplug_usb
```
- [ ] The harness must NOT print `!! FIREWORKS_API_KEY is not set` (then every attempt is judged a failure).
- [ ] `checkpoint: null` is NOT "the station file's model" once any Start named one: the station keeps the last checkpoint it was sent until restarted. The logged `checkpoint` is now the one `/status` reports after the Stop; pass `--checkpoint` explicitly for A/B runs.
- [ ] Use `--judge confirm` for the baseline: the VLM decides early stops, and at the end of each attempt the operator presses Enter to accept or y/n to override. Overrides land in `logs/harness/<run>/judge_disagreements.json`, which are free calibration labels. **If the station has no internet for Fireworks: `--judge human`.**
- [ ] 3–5 attempts per subtask. Paste each `logs/harness/<run>/attempts.jsonl` summary into `evals/zero-shot.md`, as a table of subtask × success/attempts + failure notes.
- [ ] **Check the judge against your own eyes** on every attempt. If the VLM disagrees with a human more than 1 in 5 times, fix the rubric before trusting it. Then freeze it.
- [ ] **Calibrate the judge on OUR rig (eval lead, ~15 min, ~40 Fireworks calls).** Write a one-line `rubric.scene` per subtask in `hillclimb/config.yaml` saying what is where (which block is the port, what seated looks like). Without it, the judge looked at the wrong part and gave confident false successes (07-judge-eval.md). Save ~20 pos + ~20 neg frames per subtask from the baseline runs into `evals/judge/<subtask>/{pos,neg}/<sample>/<cam>.jpg`, then run `python tools/judge_eval.py evals/judge`. Target: **0 false successes**. Missed successes only cost a retry.
- [ ] Re-weight the demo budget in `DEMO-PROTOCOL.md` toward the worst subtask.

## 08:35 — zero-shot prompt search (`docs/PROMPTS.md`, ~20 min), then freeze prompts

## 08:50 — demo sprint 1 starts (operators + curator)
Follow `docs/DEMO-PROTOCOL.md`. Order: **plug_usb → open_backing → unplug_usb → close_backing → full**. The curator keeps a tally and drops bad episodes from the dashboard.
- [ ] Set `rollouts: {enabled: true}` in `~/.dreamlabs/station.yaml` (if the organizers allow it) so every policy run from now on is logged for `tools/harvest.py`.

## ~12:30 — queue FT#1 (curator + Sissi's OK)
Rehearsed offline: `docs/research/09-pipeline-rehearsal.md`. Do a free dry run at ~12:00 (`tools/rehearse_pipeline.sh`, no args) so a surprise costs nothing.
```bash
cd ~/Hillclimb
# kit not at ~/physical-ai-hacks or ~/physical-ai-hack-2026? export HC_KIT=<kit checkout> first
tools/rehearse_pipeline.sh --parent ~/hackathon_data/hc-ft1     # add_eef x5 -> hc-ft1/hc-<t>-eef, then ft.py check. Must print PASS
~/dream-env/bin/python tools/ft.py launch ~/hackathon_data/hc-ft1 --name hc-ft1 --suffix hc-ft1 --steps 1000   # constant LR, PAID
```
- [ ] `ft.py check` lists every session with its episode count and task text: compare with the curator's tally. Any `ERROR:` line (raw 14-wide folder, STALE copy, a file beside the sessions) blocks the upload.
- [ ] Do NOT `mv ~/hackathon_data/hc-*-eef` into the parent (the old instruction). It validates, but after the next `add_eef --replace` the parent keeps the old copy (silently trains on old data), and the next `mv` fails with "Directory not empty".
Post the job id on issue #1. **When the model lands:** `tools/ft.py prompts <model>` and check that every `hillclimb/config.yaml` prompt appears there verbatim. A prompt that doesn't match is one the fine-tune never learned (`dream/trained_inputs.py`). While it trains: harness tuning on the base model, recovery demos, harvesting (PLAYBOOK § Our edge).

## Replay primitive how-to (only if the rules allow; PLAYBOOK § Replay primitives)
1. **Record a demo** of reaching the pre-contact pose (teleop, normal dhk-collect folder), or put the arms there and run `HILLCLIMB_ALLOW_ARMS=1 python tools/waypoints.py record --out prims/plug_hover.yaml` (reads only, never powers the arms).
2. **Extract a waypoint:** `python tools/waypoints.py extract ~/hackathon_data/hc-plug --episode 3 --frame 150 --out prims/plug_hover.yaml` (or `--frames 120:180` for a trajectory; use `--grip-scale 1` for station rollouts). Then set `pre: prims/plug_hover.yaml` on the subtask in `hillclimb/config.yaml`.
3. **Test dry:** `HILLCLIMB_ALLOW_ARMS=1 python -m hillclimb.harness --backend client --only plug_usb`. The arms are read and the plan is checked (`primitive.json` has the ticks and peak rad/s), and nothing is written.
4. **Test live with a human on the e-stop:** add `--client-live`, with `defaults.primitive.max_rad_s: 0.25` (and `speed_scale: 0.5` for a trajectory) on the first run. Watch the approach, then go back to the defaults.
5. If the arms stop short or the run refuses "first target is X rad", re-record. Never raise `max_approach_rad` or `max_rad_s` to make it pass.

## Harness backends cheat-sheet
| backend | moves arms? | use |
|---|---|---|
| `fake` | no | laptop, tests |
| `dry-run` | no | wraps `dreamcli robot run --dry-run` |
| `station` | **yes** (`HILLCLIMB_ALLOW_ARMS=1`) | drives `dreamcli robot up`'s local API; the default for scored runs |
| `client` | dry unless `--client-live` (+ env var) | our own DreamClient loop with per-subtask `execution_horizon` / IK boxes; only if the organizers allow a custom loop |
