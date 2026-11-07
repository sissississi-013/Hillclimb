# 06: Self-rollout harvesting

_Read on 2026-11-07 from the source alone. Nothing vendor was imported or executed. Prefixes as in 05: **R/** = dreamlabs 1.0.37 `robots/`, **K/** = the kit, **C/** = dreamlabs-client 1.1.3 `dream/`, **S/** = its `shared/datasets/`._

## What it does
`tools/harvest.py` is filtered BC / expert iteration. Each judged-successful policy run becomes a training episode for FT#2.
1. The station file has `rollouts: {enabled: true}`. The control loop then saves every **live** run as a one-episode LeRobot v2.1 dataset in `~/.dreamlabs/rollouts/<local %Y-%m-%dT%H-%M-%S>_run<N>[_k]/`. A prompt change mid-run starts a new episode with a `_k` suffix. **Dry runs save nothing** (R/run.py:107-109).
2. The harness now logs these fields per attempt in `attempts.jsonl`: `prompt`, `checkpoint`, `started_at` / `ended_at` (UTC wall clock, ms) and `station_run` (`/status.run` from `status.json`, R/station.py:778).
3. harvest pairs each attempt with the rollout episode that matches all of these:
   - the same `task`;
   - `started_at_utc` inside `[started_at − 2 s, ended_at + 2 s]` (`started_at_utc` is truncated to whole seconds, R/rollouts.py:267-271);
   - the same `run` number when both sides know it (`rollout.json.run` is the same ordinal).

   Each episode is used at most once. The station's run counter restarts with `dreamcli robot up`, so the time window resolves that ambiguity.
4. It keeps the attempts with `success: true`. With `--confirmed ok.txt`, it keeps **only** the attempt ids listed there (`<run>/<NN-subtask-aK>`, one per line, `#` for comments). The human list is authoritative.
5. It writes `<out>/hc-<subtask>-self/`. Each one is a new LeRobot v2.1 dataset:
   - its only task is the frozen prompt from `hillclimb/config.yaml`;
   - episodes are renumbered 0..N, with a global `index`;
   - `meta/` is rebuilt.

   Every pairing and build decision goes to `<out>/harvest.jsonl`.

```bash
# station, after a harness session with rollouts on (and control_hz: 30, see below)
~/Hillclimb/.venv/bin/python ~/Hillclimb/tools/harvest.py logs/harness/20261107-* --out ~/hackathon_data/self
#   20 Hz station instead: add --retime-fps 30        cut idle head/tail: add --trim
#   curator-reviewed only: --confirmed ok.txt         rebuild: --replace
~/dream-env/bin/python tools/add_eef.py ~/hackathon_data/self/hc-plug-usb-self --arm metal   # then into the FT#2 parent
```
`--trim` cuts frames at the start and end where the command stays within 0.01 of the first or last command, with a 5-frame margin. The head is floored to a key frame (rollout VP9 uses `g=10`). The videos are remuxed by packet copy (PyAV), **never re-encoded**. This is valid because the writer emits no B-frames (R/rollouts.py:44-55, rollout_episode.py:213). The image stats are copied from the untrimmed episode.

## Compatibility: rollout writer vs dhk-collect vs add_eef vs FaaS validation
| Item | Rollouts (R/rollout_episode.py, rollouts.py) | dhk-collect (K/dataset.py, layouts.py, configs/metal.yaml) | Needed by | harvest.py |
|---|---|---|---|---|
| **fps** | `control_hz` = **20** on Metal by default (R/config.py:221-224) | 30 | S/content_checks.py:47 accepts only **30/15/10/5**, and validation.py:90 requires the **same fps in every session** of one upload | **BLOCKER.** It refuses 20 fps. Fix A: set `control_hz: 30` in the station file for harvest sessions (R warns that the zero-shot base swings near home at 30 Hz; a fine-tune may be fine). Fix B: `--retime-fps 30` relabels one tick as one 1/30 s frame and remuxes the video timestamps. Each tick commands one 30 Hz model row, so actions stay consistent, but the world was filmed at 2/3 speed. TODO(event): A/B test before trusting B. |
| Gripper unit | 0 closed .. 1 open (R/rollouts.py:58) | Metal jaw stroke in **mm** (×100, layouts.py GRIP_SCALE) | add_eef divides by 100 | **Bridged:** cols 6 and 13 ×`--grip-scale` (100). TODO(event): confirm that both use the same 0..1 mapping between gripper_closed/open_deg. |
| State/action width and names | 14 = `left_joint_0..5, left_gripper, right_…` | same order and names | add_eef: v2.1 and 14/14 | OK. Written as FixedSizeList float32 like the kit (rollouts write variable `list<float>`). |
| Action meaning | Joints commanded **after the IK, the ramp (ramp_s 2 s) and the speed clamp** (R/control.py:1062) | teleop joints commanded | add_eef FK → EEF action | OK in kind. The first ~2 s are the ramp into the first target, not policy output. `--trim` does not remove that motion. |
| Joint frame | motor radians in MakerMods' frame (05 §2) | follower angles (`joint_signs`) | add_eef FK | TODO(event): compare one rollout and one demo at the home pose. |
| Camera keys | `observation.images.<station camera name>` (serve roles, e.g. `scene`, `wrist_l`, `wrist_r`) | `observation.images.scene / left_wrist / right_wrist` (embodiments/makermods.json) | validation: each embodiment role needs its key | **Bridged:** alias map renames folders, features and stats. An unknown name is kept as is, and validation will name it. TODO(event): read the station file's camera names. |
| Video codec | **VP9** in MP4, CRF 32, g=10 | H.264 CRF 23, g=10 | C/datasets.py:294 accepts H.264/HEVC/AV1/VP9 | OK as is. |
| Frame size | **raw capture size** (station `width`/`height`, e.g. 640×480 or 1280×720), RGB | center-cropped 16:9 **640×360** | not a shared field in validation | **Not bridged** (it needs a re-encode). TODO(event): whether training resizes or crops per session; the model warns when the aspect differs from training (C/facade.py:597-601). Set the station cameras to 16:9 to be safe. |
| Extra columns | `rollout.t_monotonic_s`, `rollout.chunk`, `rollout.row` | none | unknown features are a risk for loaders | **Dropped** from the parquet, features and stats. |
| `index`, `episode_index` | 0..n-1, 0 per folder | global, renumbered | validation.py:440 requires 0,1,2… with no gaps | **Bridged.** |
| Task text | the prompt sent | `--task` | the same string at train and run time | **Rewritten** to the frozen config prompt. |
| Min length | — | 1.6 s | validation: `48·fps/30 + 1` frames | Shorter episodes are skipped. |
| Status | `complete` / `partial` / `empty` / `recording` | — | — | Only `complete` is used: a `partial` episode stopped recording early, maybe before the success. |

## Risks
- **Distribution:** the policy's own successes are narrower than human demos and carry its quirks: hesitations, re-grasps and 2/3-speed motion at 20 Hz. Use them alongside human demos and corrections, not instead of them. Cap them at roughly ≤50% of a subtask's episodes.
- **Judge false positives become training data.** Use `--confirmed` (curator playback) for anything going into a paid FT.
- **Clocks:** pairing assumes the harness and the station share a clock. Run the harness on the station box (05 §6 already requires that), or widen `--slack-s`.
- **Disk:** a run refuses to start with <2 GB free in the rollouts dir (R/rollouts.py:35). Hard links share the videos, so do not edit the source videos afterwards.
- **The harvest output is a separate session folder** per subtask. Put it beside the human `hc-plug-eef` etc. under the FT#2 parent. Every session in that parent must have the same fps (see the blocker above).
- **Untested on real rollouts:** the tests use synthetic folders, plus one real VP9/MP4 written with the rollout writer's options. Before an FT, run `add_eef.py` on one harvested folder, then `tools/ft.py check` (free).
