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
cd ~/Hillclimb && ~/dream-env/bin/python -m pip install pyyaml numpy pyarrow   # harness runs in the Dream env (needs `dream`, `robots`)
echo "FIREWORKS_API_KEY=..." > .env && chmod 600 .env                        # never commit
~/dream-env/bin/python -m pytest -q                                           # all green, no hardware
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
- [ ] 3–5 attempts per subtask. Paste each `logs/harness/<run>/attempts.jsonl` summary into `evals/zero-shot.md`, as a table of subtask × success/attempts + failure notes.
- [ ] **Check the judge against your own eyes** on every attempt. If the VLM disagrees with a human more than 1 in 5 times, fix the rubric before trusting it. Then freeze it.
- [ ] Re-weight the demo budget in `DEMO-PROTOCOL.md` toward the worst subtask.

## 08:35 — zero-shot prompt search (`docs/PROMPTS.md`, ~20 min), then freeze prompts

## 08:50 — demo sprint 1 starts (operators + curator)
Follow `docs/DEMO-PROTOCOL.md`. Order: **plug_usb → open_backing → unplug_usb → close_backing → full**. The curator keeps a tally and drops bad episodes from the dashboard.
- [ ] Set `rollouts: {enabled: true}` in `~/.dreamlabs/station.yaml` (if the organizers allow it) so every policy run from now on is logged for `tools/harvest.py`.

## ~12:30 — queue FT#1 (curator + Sissi's OK)
```bash
cd ~/physical-ai-hacks
for t in plug open unplug close full; do ~/dream-env/bin/python tools/add_eef.py ~/hackathon_data/hc-$t --arm metal --replace; done
mkdir -p ~/hackathon_data/hc-ft1 && mv ~/hackathon_data/hc-*-eef ~/hackathon_data/hc-ft1/
~/dream-env/bin/python ~/Hillclimb/tools/ft.py check ~/hackathon_data/hc-ft1
~/dream-env/bin/python ~/Hillclimb/tools/ft.py launch ~/hackathon_data/hc-ft1 --name hc-ft1 --suffix hc-ft1 --steps 1000   # constant LR, paid
```
Post the job id on issue #1. **When the model lands:** `tools/ft.py prompts <model>` and check that every `hillclimb/config.yaml` prompt appears there verbatim. A prompt that doesn't match is one the fine-tune never learned (`dream/trained_inputs.py`). While it trains: harness tuning on the base model, recovery demos, harvesting (PLAYBOOK § Our edge).

## Harness backends cheat-sheet
| backend | moves arms? | use |
|---|---|---|
| `fake` | no | laptop, tests |
| `dry-run` | no | wraps `dreamcli robot run --dry-run` |
| `station` | **yes** (`HILLCLIMB_ALLOW_ARMS=1`) | drives `dreamcli robot up`'s local API; the default for scored runs |
| `client` | dry unless `--client-live` (+ env var) | our own DreamClient loop with per-subtask `execution_horizon` / IK boxes; only if the organizers allow a custom loop |
