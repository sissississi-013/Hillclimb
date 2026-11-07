# Hill Climb rules — FILL IN at the 09:00 kickoff (template; replace every `?`)

_Captured by: ? at ?:?? · source: kickoff slides / organizer / Discord link: ?_

## Scoring
- Score formula: ? (per-subtask points? all-or-nothing? partial progress?)
- Points per subtask: unplug ? · open ? · close ? · plug ?
- Time limit per attempt: ? · does faster = more points / tiebreak? ?
- Attempts per team counted: ? · best-of / average / last? ?
- Who judges success (organizer eye / camera / sensor)? What counts as "plugged" (fully seated? LED?) ?
- Leaderboard deadline: ? · how scored attempts are scheduled / requested: ?
- Live stage attempts Sun 15:00: who qualifies (top N?) ?

## Allowed / not allowed in scored runs
- Our own control loop (DreamClient / `--backend client`)? ?
- `dreamcli robot up` + our harness (station API)? ?
- External APIs (Fireworks VLM judge) during scored runs? ?
- Non-learned / scripted / replayed motions (`pre:` / `mode: replay`)? ?
- Human touching anything between subtasks? Resets? Teleop takeover? ?
- `rollouts: enabled` / editing `~/.dreamlabs/station.yaml`? ?

## Setup facts
- Rack model / where the port is: ?
- USB type (A / C) and orientation: ? · where the new cable starts, and in what orientation: ?
- Panel: hinged / sliding / removable · latch? ?
- Is the scored station the same one we collect on (same cameras, same rack pose)? ?
- Station booking: slot length ? · slots per team ? · how to book ?

## Compute / fine-tuning
- Dream credits per team: ? · fine-tune price confirmed: ?
- Training time vs steps (500 vs 1000): ? · priority queue for the event? ? · concurrent jobs: ?
- Base checkpoint id: ? · Dream-1-flash available? ?
- Station internet (Fireworks reachable)? ?

## Anything else said that matters
- ?

_After filling this in: post `PLAN-CHANGE:` on issue #1 with whatever changes the PLAYBOOK, and update `hillclimb/config.yaml` (`total_time_s`, rubrics, `scene`)._
