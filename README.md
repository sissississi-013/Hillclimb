# Hillclimb

Team repo for the **MakerMods Physical AI Hackathon — Hill Climb track** (Nov 7–8 2026, Founders Inc SF).
Goal: **1st on the leaderboard.**

## Start here (humans and agents), in this order
1. `docs/PLAYBOOK.md` — **the plan**: TL;DR, timeline, fine-tune gotchas, harness, our edge, organizer questions
2. `docs/RUNBOOK.md` — exact commands from bring-up (10:00) to queueing FT#1 (≤13:30)
3. `docs/DEMO-PROTOCOL.md` — operator card: frozen task strings, Quest controls, demo budget, recovery demos
4. `docs/PROMPTS.md` — zero-shot prompt search at 10:25
5. `context/` — `schedule.md` (official times: these win), `luma-event.md` (full event page), `task-brief.md`; add `rules.md` as soon as the rules drop
6. `docs/research/` — 01 sponsors/Dream-1 · 02 arXiv · 03 GitHub · 04 company blogs · 05 Dream client API · 06 harvest · 07 judge eval · 08 code review · 09 FT pipeline rehearsal
7. Coordination: **issue #1**. Post `CLAIM:` / `DONE:` / `PLAN-CHANGE:` / `NEEDS-HUMAN:` there.

## Code
| Path | What |
|---|---|
| `hillclimb/harness.py` | System-2 harness: 4 subtask episodes, judge, retries, recovery, logs → `logs/harness/<run>/attempts.jsonl` |
| `hillclimb/judge.py` | Fireworks VLM judge (frozen rubric + `scene` notes) + gripper proprio veto |
| `hillclimb/backends.py` | `fake` · `dry-run` · `station` (drives `dreamcli robot up` API) |
| `hillclimb/client_loop.py` | `client` backend: our own DreamClient loop (per-subtask execution horizon, IK keep-out boxes) |
| `hillclimb/primitives.py` | replay primitives (`pre:` pose / `mode: replay`), velocity-clamped |
| `hillclimb/config.yaml` | prompts, rubrics, timeouts, retries, knobs |
| `tools/ft.py` | fine-tune launcher with steps / base_share / resume_from (kit's script hides them) |
| `tools/harvest.py` | judged-success rollouts → per-subtask LeRobot datasets (self-improvement data) |
| `tools/report.py` | success tables per subtask × checkpoint × prompt (Wilson CIs) |
| `tools/judge_eval.py` | measure judge accuracy / false-success rate on labelled frames |
| `tools/waypoints.py` | extract/record waypoints for replay primitives |
| `tools/rehearse_pipeline.sh` | add_eef every `hc-*` folder into one FT parent + `ft.py check` (free, never uploads) |
| `tools/api_check.sh` | verify our code against the station's installed Dream library |

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python pyyaml numpy pyarrow pytest
.venv/bin/python -m pytest -q          # no hardware needed
.venv/bin/python -m hillclimb.harness --backend fake   # laptop. On the station use ~/dream-env/bin/python (RUNBOOK)
```
`--backend fake` uses an offline fake judge by default (no network, no cost). Add `--judge vlm` to exercise the real Fireworks judge, which makes paid calls and needs real JPEGs.
**Safety:** anything that moves an arm needs `HILLCLIMB_ALLOW_ARMS=1` and is run by a human with a hand on the e-stop.

## Branches
- `main` — stable
- `research/claude-scout` — overnight scouting + harness (Claude, Sissi's machine)

Secrets live in a local `.env` (gitignored). Never commit keys.
