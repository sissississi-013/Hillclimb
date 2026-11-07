# 05: Dream client API, `dreamcli robot run` and the station, read from the source

Read on 2026-11-07 from the source alone. Nothing was imported or executed. Path prefixes used below:
- **C/** = `/Users/sissi/hc-vendor/dreamclient/src/dream/`: `dreamlabs-client` 1.1.3, the `dream` module. A byte-identical copy is in `hc-vendor/dreamlabs-client-1.1.3/`.
- **R/** = `/Users/sissi/hc-vendor/dreamlabs-pkg/src/robots/`: `dreamlabs` 1.0.37 (`dreamcli`, the station and the control loop).
- **K/** = `/Users/sissi/hc-vendor/physical-ai-hack-2026/`: the official kit.

## 0. TL;DR for the harness
- **There are three ways to drive the arms.**
  - (a) `dreamcli robot run --task ...` runs one episode until Ctrl-C (R/run.py:1-13, 82-106).
  - (b) **The station agent** (`dreamcli robot up`, R/station.py) exposes an **HTTP API on 127.0.0.1:8091**: `GET /status`, and `POST /start {"prompt","checkpoint"}`, `/stop`, `/home`, `/reset` and `/prompt` (R/station_api.py:9-15, 151-168; port R/station.py:97).
  - (c) Your own loop on `dream.DreamClient` (C/facade.py).
  - **Our harness uses (b).** Every Start builds a fresh client, so each subtask is a fresh episode (R/station.py:922-952).
- **Neither dreamcli nor the station has an episode end or success detection.** A run lasts until Stop, Ctrl-C, a fault or `steps` (R/control.py:962-998). The success check and the timeouts are entirely ours. That is the System-2 gap.
- **There are no predicted video frames.** `get_action()` returns only action chunks: `per_key` / `q_r_chunk` / `q_l_chunk`, plus "any other key the model adds is passed through" (C/client.py:331-348, C/facade.py:603-607). Neither the client nor the shared contracts parse or document a video output (`grep video|predict|future` in C/ and R/../shared/contracts/envelope.py finds nothing). **So predicted-vs-actual divergence is not available from the public API.** TODO(event): print `action.keys()` once in a custom loop, or `RoundTrip.action` in DebugView (C/debug_view.py:86-109), to see whether the serve passes anything extra.
- **Joint torques/currents are not exposed.** The Metal driver reads per-motor `torque_nm` (R/motor_arm.py:126-141), but `MotorArm.read()` drops it. Its Reading carries only positions, velocities, gripper and temperatures, with no `gripper_effort` (R/motor_arm.py:515-526). The cheap signals that remain are the **measured vs commanded gripper and joints**, which are written per tick to the trace CSV (§6).

## 1. Constructing the client and opening a policy
- `dream.DreamClient(api_key=None)` reads the key and URLs from `~/.dreamlabs/credentials` / `DREAM_API_KEY`. Nothing is sent on construction (C/facade.py:96-110). The public surface is listed in C/__init__.py:1-38 and 109-127.
- `client.policy(config)` picks the model and opens an empty frame window. Nothing is sent until the first `get_action()` (C/facade.py:434-541). Config keys (C/facade.py:511-540, C/__init__.py:151-201):
  - `checkpoint` (required): a `get_models()` id or a fine-tune's `fine_tuned_model`.
  - `obs_window` (9) and `obs_fps` (5): the frame history (C/_window.py:35-38).
  - `timeout_ms` (15000) and `strict` (True).
  - **Pacing:** `execution_horizon` (30 rows), `action_hz` (20), `sync_above_ms` (700) (C/__init__.py:158-163).
  - `kinematics`.
  - `ik`: `{"arm":"metal","head":"eef"}`, plus `keep_out`, `region`, `reach` and `solve` (C/__init__.py:170-189, C/client_ik.py:191-209, 536-545).
  - `robot`: `"yam"` or `"other_embodiment"` (C/__init__.py:190-197).
  - `host`/`port` are refused; they come from the credentials (C/facade.py:505-509).
- **Recording:** every policy streams every tick to Dream's platform in the background, best effort (C/facade.py:497-503).
- `wait_for_ik(timeout_s)` waits for the `traj` IK compile, which takes about 5-20 s (C/facade.py:701-715). dreamcli waits up to 60 s before it powers the arms (R/run.py:59, 147-151).

## 2. Observation format (Metal)
- `set_observation(frames, observation, timestamp=None)` (C/facade.py:543-582):
  - `frames` maps `video.<role>` to **one base64 JPEG string**. The client keeps the first tick, then one every 1/obs_fps s, holding the last `obs_window` frames per key.
  - `observation` holds the `state.<part>` arrays and `prompt`, and must never carry `embodiment`.
- **History:** 9 frames at 5 fps, about 1.8 s (C/_window.py:35-38; R/config.py:546-549 says the base checkpoint was trained on 30 fps data with video_stride 6).
- **Metal/Makers embodiment** (R/adapter.py:91-121):
  - State: `left_arm[0:6]`, `left_gripper[6:7]`, `right_arm[7:13]`, `right_gripper[13:14]`. With `ik`, the client adds `*_eef_xyz`, `*_eef_rot6` and `*_grip`; do not send these yourself (C/__init__.py:175-178).
  - Action: `right_eef_xyz`, `right_eef_rot6` and `right_grip`, then the same for the left arm, then `left_arm` and `right_arm`.
  - **Camera roles:** `scene`/`overhead`/`head` map to the scene camera, `wrist_l`/`left_wrist` to the left wrist and `wrist_r`/`right_wrist` to the right wrist. The kit names them `overhead`, `wrist_l` and `wrist_r` (K/configs/metal.yaml:49-52).
- **Units:** joints are motor radians in MakerMods' frame. The gripper runs from 0 (closed) to 1 (open), linear in motor angle between `gripper_closed_deg` and `gripper_open_deg` (R/metal.py:15-19, R/config.py:295-300). Metal joint soft limits are at R/metal.py:43-51.
- **Image size:** the station encodes frames at `policy_image_size` = **640x360** before JPEG, and "the serve resizes anyway" (R/config.py:538-540, 610-612). A custom loop could send smaller frames for less upload.

## 3. `get_action()`: return shape and timing
- `get_action(*, commanded=None)` sends the window plus the latest observation. The first call of an episode opens a session (C/facade.py:584-660).
- **Return value** (C/client.py:331-348):
  - A fine-tune answers `per_key` with `action.<part>` rows, **48 rows** for Dream-1.
  - With `ik` head `eef`, `q_r_chunk` and `q_l_chunk` come back as `[H, joints+1]`: the solved joints, then the grip. They are already blended into the previous chunk.
- **Errors:**
  - `DA-2000 pool_warming` (wait `retry_after_s`), `rate_limited`, `pool_full`.
  - `DreamTimeout` after 15 s.
  - `DreamSessionExpired`.
  - Details at C/facade.py:610-633.
- **Timing:** `client.last_timing` reports `rtt_ms`, `serve_infer_ms`, `gateway_ms`, `network_ms`, `up_mbps`, `request_bytes` and similar (C/client.py:549-574). `on_round_trip(fn)` delivers each RoundTrip, with the frames sent, the action and any error (C/facade.py:662-679, C/debug_view.py:86-109).
- **Seam smoothing (the client's own):**
  - A 16-row velocity blend between chunks (C/seam_join.py:28).
  - A Savitzky-Golay filter, window 11, order 3 (C/smoothing.py:44-45).
  - The loop adds **no** smoothing of its own (R/control.py:12-17).

## 4. How `dreamcli robot run` and the station execute chunks
- **Tick rate:** `control_hz` = 20 on Metal (R/config.py:223, 503-505). The checkpoint's rows are **30 Hz actions played at 20 Hz, so motion runs at 2/3 speed** (R/config.py:503-504). The client paces with `action_hz = control_hz` (R/run.py:468). A command thread interpolates writes to the arms at 250 Hz (R/config.py:534-535).
- **Refill:** the loop asks `client.refill_due(executed, remaining)` (R/control.py:563-565, 1222-1240). There are two modes:
  - **Async** (default): the request is prefetched so that its reply lands as `execution_horizon` rows have run. That is **30 rows = 1.5 s**, with rows 31-48 kept as runway (C/pacing.py:10-29, 123-144).
  - **Sync**: used when the median RTT exceeds `sync_above_ms`. The station sets this to **900 ms** instead of the client's 700 (R/run.py:64, 469). In sync the whole 48-row chunk runs and the arms hold during the request (C/pacing.py:24-29).
  - Rows already executed when a chunk lands are dropped, not replayed (R/control.py:499-505).
- **dreamcli/station never set `execution_horizon`,** so it is always 30 (R/run.py:421-515 has no such key). Changing it requires a custom loop.
- **Safety and shaping knobs** (station YAML → `loop_settings`, R/run.py:345-375; AppConfig at R/config.py:494-560):
  - `max_ee_speed_m_s`: end-effector speed clamp, **0.6 on the station** (R/config.py:519-521).
  - `max_gripper_step` = 0.05 per tick.
  - `ramp_s` = 2 and `ramp_max_rad_s` = 0.5: the ramp into the first target, and after every new episode.
  - `grip_force_cap` (YAM only; Metal reports no effort).
  - `action_timeout_s` = 3, `max_stall_ticks`, `first_chunk_timeout_s` = 120.
  - **A first chunk later than 60 s after the first tick stops the run before it moves** (R/control.py:84-89).
- **IK options** (per arm in the station file, R/config.py:313-330; passed to the client at R/run.py:483-506):
  - `keep_out`: boxes and capsules in the arm's base frame. Collision spheres stay 2 cm off them.
  - `reach`: a 2-D allowed area.
  - `region`: a 3-D box. Targets are clipped to it, and a row more than 1 cm outside it holds the arm.
  - `ik_solve` `tick` or `traj`. The effective default is **traj** (R/config.py:211, R/run.py:495).
- **Episodes:** a `/start` builds a new client and loop, which means a new session (R/station.py:922-952). `/prompt` mid-run calls `loop.set_task()`, which drops the buffer, resets the episode and re-ramps at the next safe point. That gives a cheaper subtask switch with no client rebuild and no home (R/station.py:718-731, R/control.py:925-940, 1250-1274).
- **Episode end:**
  - Stop: the arms **hold** (R/station.py:508-532).
  - Home: the station file's `home` pose with the grippers open, refused while a run exists (R/station.py:534-553, R/config.py:305-309).
  - `steps`: `run()` returns after `max_steps` and then homes (R/station.py:907-913, 915-920). HTTP `/start` does not pass `steps` (R/station_api.py:157-158).
  - `dreamcli robot run` homes at both start and end (R/run.py:96-102).
- **Dry run:** `dreamcli robot run --dry-run` calls the model for real but wraps every arm in `DryRun`, which drops writes, opens arms read-only and never homes (R/run.py:96-104, 285-348). **The station has no dry-run flag** (R/station.py:1339-1386).
- **Checkpoint:** pass `--checkpoint dream-1-ft:...` or `/start {"checkpoint": ...}` (R/station.py:436-499). A Metal body has no default model; it uses the station file's `checkpoint` or one given with the Start (R/bodies.py:142, R/run.py:128-133). TODO(event): read the base model id from station.yaml or `get_models()`; it is probably `dreamlabs/dream-1-beta` (R/bodies.py:62). The first call to a new model can sit in `pool_warming` for a long time. The loop waits up to 1800 s before the first chunk (R/control.py:84) (K/docs/running_your_model.md:22).
- **`/status` fields:** state, prompt, checkpoint, run, chunks, stalls, clamps, `max_ee_speed_requested_m_s`, arms alive, camera frame ages, session_id, last_error, timing, `run_elapsed_s` (R/station.py:757-804).

## 5. Hooks
- Our own loop: `on_round_trip`, `refill_due`, `reset_episode`, and `set_observation` from any thread (C/facade.py:88-94).
- Station: HTTP routes, plus the `/prompt` task change.
- **Live View:** `dream.LiveViewClient(robot_id)` has `stream(obs, status)`, `get_prompt()` and `register_callback(fn)`. The events are start, end, prompt, home and reset (C/live.py:1-25, 276-293, 451-536). The station uses it when its file has a `live:` section, via `dreamcli robot up`, then Start on the platform page (K/docs/running_your_model.md:28).
- `DebugView(client, port)` is a local page of each round trip (`--view` on run/up, R/cli.py:1253; R/run.py:152-161).
- **Local rollout recording** (DAgger data!): set `rollouts: {enabled: true}` in the station file to save **every run as a LeRobot v2.1 episode** in `~/.dreamlabs/rollouts` (R/config.py:197-199, R/rollouts.py:1-9). A prompt change starts a new episode (R/control.py:1261-1266). TODO(event): check whether these episodes go straight through `tools/add_eef.py` and into FaaS.

## 6. Cheap success signals we can actually read
- **Trace CSV**, `~/.dreamlabs/last-run.csv`, one row per commanded tick (R/config.py:132, 525-528):
  - Columns: `measured_l0..5`, `measured_r0..5`, `gripper_l/r`, `commanded_*`, `cmd_gripper_l/r`, `ee_speed_*`, `chunk_id`, `clamped` (R/control.py:116-133).
  - The file is flushed every 30 rows (R/control.py:92), so it can be tailed mid-run.
  - **"Holding something" = commanded grip near 0 but measured grip stuck above 0**: the Metal gripper always holds by torque (R/config.py:303-304). **Contact or jamming = high |commanded - measured| joint error.**
- **Model-input frames**, `~/.dreamlabs/last-run_inputs/<camera>/<t>.jpg` plus `inputs.jsonl`, are written **once a second** during the run with the state and prompt sent (R/control.py:96-99, 136-149, 1787-1803). These are free frames for the VLM judge, both mid-run and at the end.
- **Both are wiped at the next run's start** (R/control.py:1748-1762), so the harness copies them per attempt.
- The station must be local: the API binds to 127.0.0.1 (R/station.py:99). The harness must run on the station box or over an SSH tunnel; the files need the box. TODO(event).

## 7. Knobs that plausibly raise success rate
1. **Subtask prompts, one episode each.** Use the same text as the fine-tune's `--task` (K/docs/running_your_model.md:22). The model also warns at runtime when a prompt or camera aspect differs from training (C/facade.py:597-601).
2. **`/prompt` instead of stop/home/start** between subtasks that chain physically (unplug → open). This avoids a home move that opens the grippers and drops things (R/station.py:535).
3. **`max_ee_speed_m_s`** (0.6) and **`control_hz`** (20 means 2/3 speed). Slower is likely better for insertion; 30 Hz plays at the trained speed. TODO(event): A/B test on plug_usb.
4. **`region` and `keep_out`** around the rack: they prevent link collisions, which cause resets. **`ik_solve: traj`** gives smoother tool paths (R/config.py:313-330, 211).
5. **Execution horizon** (custom loop only). A shorter horizon (for example 16-20 rows) re-plans more often from fresh frames, which suits contact-rich insertion but costs more calls (rate limit about 1 call per 0.7 s; see 01 doc). A longer one gives smoother, more committed motions for open/close.
6. **Keep the RTT under 900 ms** so the station stays async and does not hold between chunks. Watch `/status.timing` and `last_timing`.
7. **`ramp_s` / `ramp_max_rad_s`:** shorter ramps waste less time per new episode but risk jerks.
8. **The `steps` cap** (Live View or the controller) gives a deterministic per-subtask cut. The harness's timer does the same job over HTTP.

## 8. Unverified (TODO(event))
- The station camera names (overhead/wrist_l/wrist_r vs scene).
- The base checkpoint id for Metal.
- The `trace_path` in the station file.
- Whether the harness runs on the station box.
- The thresholds for `holding()`: 0.15 cmd and 0.06 measured.
- Whether the serve returns extra keys in the action (for example video).
- Whether `/home` in recovery is acceptable under the scoring rules.
