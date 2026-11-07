# 09: Fine-tune data pipeline rehearsal (offline, 2026-11-07 ~04:30)

Goal: no surprise at ≤13:30, when FT#1 is queued (a failure would waste a 7–8 h slot). The whole data path ran on a laptop with no hardware, no API key and no upload, using the **real** library (`~/hc-vendor/dream-env-test`: python 3.12, dreamlabs 1.0.37 + dreamlabs-client 1.1.3) and the **real** kit (`~/hc-vendor/physical-ai-hack-2026`).

## Fixtures (written by the real writers, not hand-made)
- **Demos:** the kit's `dream_hackathon.dataset.DatasetWriter`, called the way `collect.py` calls it on a Metal station: `layouts.to_vector("metal", ...)` (14-wide, gripper in mm), the cameras of `layouts.video_keys("makermods")`, 640×360, 30 fps, H.264. Three folders (`hc-plug`, `hc-open`, `hc-unplug`), each holding 3 episodes × 2 s with its DEMO-PROTOCOL task text. The joints oscillate around `[0, -0.9, 1.3, -0.4, 0, 0]` rad, inside the Metal soft limits (`robots/metal.py` JOINTS).
- **Rollouts:** `robots.rollouts.Recorder`, the class the station control loop uses, fed fake `RobotState`/`Action`/`Frame` ticks. Settings: cameras `scene`/`wrist_l`/`wrist_r`, 640×480, one run at 20 Hz (the Metal default) and one at 30 Hz, 4 s each. A matching `attempts.jsonl` was written in the harness format.
- Generator: `/private/tmp/claude-501/-Users-sissi/e6d47fc6-e5b2-4c91-b05c-5321ac36e77f/scratchpad/gen_fixtures.py` (scratch only, not in the repo).

## Results
| Step | Result |
|---|---|
| kit `tools/add_eef.py <hc-x> --arm metal` on each demo folder | PASS. It writes `<hc-x>-eef` (state 34 / action 32) and runs Dream's checks itself |
| validate a raw folder, or `~/hackathon_data` itself (raw + eef mixed) | FAIL, as it should: `actions have 14 dimensions; the embodiment's spans cover 32` |
| **parent of sessions** `hc-ft1/{hc-plug-eef,hc-open-eef,hc-unplug-eef}`, `validate_dataset(parent, makermods.json)` | **PASS**: 9 episodes, 540 frames, 30 fps. A folder of session folders is the documented multi-task layout (dream/datasets.py docstring) |
| RUNBOOK `mkdir hc-ft1 && mv hc-*-eef hc-ft1/` | PASS the first time |
| …then record 1 more demo, `add_eef hc-plug --replace` | **SILENT BUG.** No error: it writes a new `hc-plug-eef` beside the source. `hc-ft1/hc-plug-eef` keeps 3 episodes and **still validates**, so FT#1 would train on the old data |
| …then `mv hc-*-eef hc-ft1/` again | **FAIL**: `mv: ... Directory not empty` |
| parent of **symlinks** to the -eef folders | PASS. Validation follows links, and the upload zip (built locally with `_local_files` + `_write_zip`, nothing sent) holds the real files. Not used: `--out` (below) is simpler |
| **`add_eef <src> --arm metal --out <parent>/<src>-eef --replace`** | **PASS, the chosen layout.** Videos stay hard links (same inode). A re-run replaces in place. `rm -rf <parent>/x-eef` leaves the source intact. Output is byte-identical to the default `<src>-eef` |
| missing subtask folder (`hc-close` not recorded) | kit add_eef crashes with `FileNotFoundError ... meta/info.json`. The script skips it with a message |
| `harvest.py` on the 20 Hz rollouts, no flag | refused, as designed: `rollouts are 20 fps ... --retime-fps 30` |
| `harvest.py --retime-fps 30` (20 Hz) / no flag (30 Hz) / `--trim` (30 Hz) | PASS: `hc-plug-usb-self`, `hc-open-backing-self`, each 1 episode. Gripper 0.7 → 70 mm → 0.7 after add_eef (same unit as the demos) |
| validate a harvested folder raw, or the harvest `--out` root | FAIL: 14-wide; the root fails on `'harvest.jsonl' is not inside a session folder` |
| add_eef on each `hc-*-self` | PASS |
| **FT#2 parent: 3 demo -eef + 2 harvested -eef** | **PASS**: 11 episodes, 700 frames, 30 fps, no notes. **But** the rollout frames are 640×480 and the demos 640×360: Dream accepts the mix without a word (now a WARNING in `ft.py check`) |
| a `README.md` beside the sessions | skipped (stray). Any other file (`harvest.jsonl`, `.json`) is refused |
| add_eef speed | 2.8 s for 10 episodes × 450 frames; validation 0.3 s. Rebuilding the parent at queue time costs nothing |
| **pyarrow exit deadlock** | `validate_dataset` returns, then the interpreter hangs at exit in `arrow::ThreadPool::Shutdown` in **1 of 20 runs**. Seen with ft.py and with a bare 5-line validate script, so the kit's add_eef.py and finetune.py `--check` can hang the same way. Seen on macOS + pyarrow 26; the station is untested |

## Verified commands (station, from `~/Hillclimb`, `~/dream-env`)
```bash
# FT#1 (≤13:30). Free dry run any time before: tools/rehearse_pipeline.sh   (temp parent, then deleted)
tools/rehearse_pipeline.sh --parent ~/hackathon_data/hc-ft1
#  = for t in unplug open close plug full (those recorded):
#      add_eef.py ~/hackathon_data/hc-$t --arm metal --out ~/hackathon_data/hc-ft1/hc-$t-eef --replace
#    then ~/dream-env/bin/python tools/ft.py check ~/hackathon_data/hc-ft1   -> must end with PASS
~/dream-env/bin/python tools/ft.py launch ~/hackathon_data/hc-ft1 --name hc-ft1 --suffix hc-ft1 --steps 1000   # PAID

# FT#2: harvest, then the demos (all episodes so far) + self/hc-*-self in a NEW parent
~/dream-env/bin/python tools/harvest.py logs/harness/20261107-* --out ~/hackathon_data/self --retime-fps 30 --confirmed ok.txt
tools/rehearse_pipeline.sh --parent ~/hackathon_data/hc-ft2 --self
~/dream-env/bin/python tools/ft.py launch ~/hackathon_data/hc-ft2 --name hc-ft2 --suffix hc-ft2 --steps 500 \
    --resume-from dream-1-ft:<uid8>-hc-ft1                                                                  # PAID
```
Env: `HC_KIT=<kit checkout>` if the kit is not at `~/physical-ai-hacks` or `~/physical-ai-hack-2026`. `DREAM_PY` defaults to `~/dream-env/bin/python` and `DATA` to `~/hackathon_data`.

## What changed in our files
- `tools/rehearse_pipeline.sh` (new): source folders → `<parent>/<name>-eef` → `ft.py check`. It never uploads. With no `--parent`, it builds a hidden temp parent and deletes it. It rebuilds an existing parent from scratch, but only if the parent holds nothing except `*-eef` datasets. It refuses `~/hackathon_data` itself, `self/`, `-eef` folders passed as sources, and two sources with the same name.
- `tools/ft.py`:
  - `dream` is imported lazily, so the tool is testable.
  - The kit path comes from `HC_KIT` or is auto-detected (it was hard-coded to `~/physical-ai-hacks`, a TODO).
  - New `eef` subcommand: the kit's own `add_eef.convert()` plus Dream's checks, in-process.
  - `check`/`eef` hard-exit with `os._exit` (the pyarrow deadlock).
  - `check` inspects the parent before Dream's checks: it lists each session with episodes and task texts. ERROR on raw 14-wide sessions, on **STALE** sessions (the source in `~/hackathon_data` or `~/hackathon_data/self` has a different episode count), and on files beside the sessions. WARNING on task texts that are not frozen prompts and on mixed frame sizes.
  - `launch` refuses on those errors before any network call (verified offline). Both commands take `--allow-stale` for a deliberate snapshot.
- `tools/harvest.py`: warns when rollout frames are not 640×360. The docstring now uses `~/dream-env` and points to the FT#2 step. With `--replace`, an empty subtask now says to remove its stale `-eef`.
- `docs/RUNBOOK.md` (≤13:30), `docs/DEMO-PROTOCOL.md` (Hand-off), `docs/research/06-harvest.md`: the `mv` flow is replaced by the script. Harvest runs in `~/dream-env` (the station has no `~/Hillclimb/.venv`).
- `tests/test_ft_check.py`: tests for stale, raw, stray, unknown task and mixed frames, and for the prompt list (`FULL_TASK` must match DEMO-PROTOCOL).

## Gotchas
1. **Never `mv` the -eef folders** (see above). Build the parent with `--out`, and rebuild it whole after every new demo.
2. **Operators keep recording during assembly.** If `hc-plug` gains an episode between `rehearse_pipeline.sh --parent` and `launch`, launch refuses with STALE. Re-run the script, or pass `--allow-stale` if you want the snapshot.
3. **Do not reuse `hc-ft1/` for FT#2.** Build `hc-ft2/`. `hc-ft1/` is the record of what FT#1 saw.
4. The harvest `--out` root is not a parent, and raw `hc-*-self` folders are not sessions. Only their add_eef output is.
5. **Aspect ratio:** rollouts keep the raw camera size (640×480 here). Dream takes the mix silently. Set the station cameras to 16:9 before harvesting (06 TODO), or think twice before mixing.
6. The 20 Hz rollouts need `--retime-fps 30` (or `control_hz: 30`). Every session in one parent must share one fps (`SHARED_FIELDS`). Names and widths are rewritten by add_eef, so they always match.
7. Session folder names must match `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`. `hc-plug-usb-self-eef` is fine. A dataset `--name` is `[a-z0-9-]{1,64}`.
8. `cp -a` on macOS breaks the hard links (each copy holds its own videos). Assemble only with the script or with add_eef `--out`, on the same disk as `~/hackathon_data`.
9. If a kit tool hangs after printing `... Dream's dataset checks pass` / `ready to upload`, the work is done (the exit deadlock): Ctrl+C. Our script never hits this.

## Not covered offline
Real cameras and codecs on the station, real Metal joint frames vs rollout frames (06 TODO), training-side behaviour with mixed aspect ratios, and the upload/processing service.
