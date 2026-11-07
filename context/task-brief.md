# Hillclimb task brief (from Sissi, 2026-11-07 03:20 PST)

## Known (pre-rules) task description
Data-center server-rack manipulation, long horizon:
1. Pull a USB cable out of a server rack
2. Open the rack backing (panel/door)
3. Close the backing
4. Connect a new USB cable into the rack

Exact setup / scoring rules TBD — organizers will publish "before the event". **Update this file the moment rules drop.**

## Stack we'll have
- MakerMods bimanual 6-axis "Metal Arm" stations (shared, booked time slots)
- Dream Labs **Dream-1** World Action Model — fine-tune via Dream API with our own demos
- We build the **harness**: task sequencing, success checking, failure recovery

## Our resources
- Fireworks credits (key in local `.env` → `FIREWORKS_API_KEY`) — VLM/LLM inference & fine-tuning
- River: 50k credits (key in `.env` → `RIVER_API_KEY`) — what River offers is unconfirmed
- Modal account with leftover credits (Sissi's) — GPU fallback
- Never commit `.env`.

## Goal
1st place on the Hillclimb leaderboard. Approach must be innovative, clean, simple, effective. Be the fastest team to implement at the event.

## Team
- Sissi (sissississi-013), Ian (ianalin123), more joining.
- Agent coordination: GitHub issue "Agent coordination board" in this repo.
