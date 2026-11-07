# 03 — GitHub code scout: what to steal for Hillclimb

Scouted 2026-11-07 (star counts and push dates are from the GitHub API that day). Task: long-horizon bimanual server-rack
manipulation (unplug USB, open the rack backing, close it, plug in a new USB) on MakerMods arms, fine-tuning Dream-1
through the Dream API, with our own harness.

## TL;DR: the five finds that matter most

1. **`dreamlabs-co/dreamlabs`** (★10, pushed 2026-11-07) is the official Dream-1 Python client (`dreamcli`,
   `dream.DreamClient`, `dream.datasets`, `dream.fine_tuning`). It already has **configs for `metal` and `makers`
   bodies**. Its dataset format is **LeRobot v2.1** (not v3.0). Read this repo before anything else.
   https://github.com/dreamlabs-co/dreamlabs
2. **MakerMods publishes its whole stack**: `makermodslab` (web UI with record, curate, train, **DAgger**, eval,
   Modal inference), `metal-arm-harness` (a safety-enveloped CLI any agent can drive, returning a JPEG per camera
   after every op), `maker-arm-sdk`, `metal-python-ros`, `metal-data-collection`, and `lerobot-MakerMods`.
   https://github.com/makermods-robotics
3. **LeRobot v0.7.0** (released 2026-11-05) ships three ready-made **reward/progress models**: SARM (stage-aware
   progress), Robometer-4B (Qwen3-VL-4B with progress and success heads) and TOPReward (zero-shot Qwen3-VL logprob).
   It also ships the HIL-SERL **binary reward classifier**, `lerobot-annotate` (VLM subtask segmentation) and
   `lerobot-rollout --strategy.type=dagger`. That covers most of a success checker and progress estimator.
4. **REBOOT26 datasets on HF**: real bimanual (WidowX-AI) **USB-A / USB-C install, remove and recovery** demos in
   LeRobot v3 format, from the REBOOT failure-to-recovery benchmark (arXiv 2609.22591). These are the closest public
   data to our task, and they show what recovery demonstrations look like.
5. **`ehehee/open-gemini-robot-harness`** (★17) and **`robocurve/inspect-robots`** (★746) are the cleanest
   open "VLM in the loop" bimanual harnesses: observe, then program, then check, then execute, with rejection
   feedback and full run records.

---

## 1. LeRobot ecosystem

### Repos

| Repo | Stars / last push | What to steal |
|---|---|---|
| [huggingface/lerobot](https://github.com/huggingface/lerobot) | ★28.7k / 2026-11-07 (v0.7.0, 2026-11-05) | `lerobot-record`, `lerobot-rollout` (sync / RTC / remote inference, `--strategy.type=dagger`), `lerobot-edit-dataset`, `lerobot-annotate`, `lerobot-dataset-viz`. Policies in `src/lerobot/policies/`: act, diffusion, pi0, pi05, pi0_fast, smolvla, groot, xvla, molmoact2, rtc, plus about 20 newer WAM/VLA ports (cosmos3, fastwam, longwam, lawam…). Rewards in `src/lerobot/rewards/`: `classifier`, `sarm`, `robometer`, `topreward`. Robots include `bi_so_follower`, `bi_yam_follower`, `bi_openarm_follower`, `bi_rebot_b601_follower` |
| [makermods-robotics/lerobot-MakerMods](https://github.com/makermods-robotics/lerobot-MakerMods) | ★16 / 2026-09-29 | LeRobot fork behind the MakerMods UI (has `bi_so101_follower` and `xlerobot`) |
| [makermods-robotics/makermodslab](https://github.com/makermods-robotics/makermodslab) | ★21 / 2026-11-03 | One-tab loop: calibrate, teleop, record, curate (deselect bad episodes), train, deploy, eval, **DAgger** (hand control back to the leader mid-rollout), merge, fine-tune. Remote inference on Modal; training on HF Jobs or tailnet nodes. Files worth reading: `dagger_runner.py`, `eval_runner.py`, `train_weighted.py`, `merge.py`, `recording_preparation.py`, `can_recovery.py`, `metal_gripper_hold.py`. **This is probably what the event stations run.** |
| [makermods-robotics/metal-data-collection](https://github.com/makermods-robotics/metal-data-collection) | ★6 | ROS 2 leader/follower recording into HDF5 (leader joints are the action, follower joints are the state) |
| [AIDASLab/lerobot-so101-bimanual](https://github.com/AIDASLab/lerobot-so101-bimanual) | ★12 | Bimanual SO-101 config reference |
| [Vector-Wangel/XLeRobot](https://github.com/Vector-Wangel/XLeRobot) | ★5.7k | Dual-arm SO-101 platform; its docs have a **3-camera bimanual SmolVLA/ACT recipe** ([xlerobot docs](https://xlerobot.readthedocs.io/en/latest/software/getting_started/VLA_smol.html)) |
| [Tavish9/any4lerobot](https://github.com/Tavish9/any4lerobot) | ★1.2k | Dataset converters (HDF5 / RLDS / AgiBot to LeRobot), version up- and down-conversion |
| [dsta022/Loop-Engineering-for-VLA](https://github.com/dsta022/Loop-Engineering-for-VLA) | ★791 | Collect, merge, audit and visualize LeRobot datasets |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | ★14.6k / 2025-08 | pi0 / pi0.5 reference code (LeRobot has ports) |
| [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | ★8.3k / 2026-11-07 | GR00T N1.7; LeRobot has a `groot` policy |
| [MarkFzp/act-plus-plus](https://github.com/MarkFzp/act-plus-plus) | ★3.7k | Original ACT / Mobile ALOHA co-training |

### Data format gotcha (high priority)
- **The Dream API uploads LeRobot v2.1** (`<session>/meta/info.json`, `data/chunk-xxx/episode_xxxxxx.parquet`,
  `videos/chunk-xxx/<key>/episode_xxxxxx.mp4`). Current LeRobot (v0.7) **writes v3.0** (`file-xxx.parquet` with
  many episodes per file). LeRobot ships `convert_dataset_v21_to_v30.py`, but **not the reverse**. If we record with
  stock LeRobot we need a v3-to-v2.1 down-converter, or we record with whatever the station uses. Check
  `any4lerobot` for one and pre-write our own if it lacks one (see the module list at the end).
- Dream embodiment config: `{"state": {part: [start, end]}, "action": {...}, "video": {role: video_key}}`. The Metal
  camera roles are `scene`, `wrist_l` and `wrist_r`, which map to `observation.images.scene`, `.left_wrist` and
  `.right_wrist`, recorded at 1280x720. `dream.datasets.validate_dataset(path, config)` runs the checks locally with
  no API key: **run it on every recording session.**

### Dream fine-tune knobs (from `dream/fine_tuning.py`)
- `steps` takes 500–2000 in multiples of 100 (default 1000). Price and time scale with steps.
- `learning_rate` takes 1e-6 to 1e-4 (default 1e-5).
- `lr_schedule` is `constant` or `cosine`. The docstring says cosine is best for single-task specialization.
- `base_share` is 0.5 (mixes in Dream data, the default) or 0.0 (our data only).
- `resume_from=<dream-1-ft:...>` continues training, charged for the new steps only. **Use it for DAgger rounds.**
- `use_eef` trains end-effector pose; Metal and Makers fine-tunes answer with tool poses, and `dream.client_ik`
  solves the IK.
- Checkpoint ids look like `dream-1-ft:<uid8>-<suffix>`. The first `get_action()` returns `pool_warming`
  (retryable), so build warm-up into the harness.
- Dream's published real-robot zero-shot numbers: Maker Arm 86.2%, Metal Arm 70.5%
  ([dreamlabs.ai/research/dream-1](https://dreamlabs.ai/research/dream-1/)).
- Client API surface: `get_action(observation)`, `reset_episode()`, `execution_horizon`, `action_hz`,
  `refill_due(executed, remaining)`, `last_timing()`. There are also `smoothing.py`, `seam_join.py` (chunk
  stitching) and `slow_network.py`. The harness should call `reset_episode()` at every subtask boundary.

### Hackathon winners and data-collection lessons
- The LeRobot Worldwide Hackathon (June 2025) had 30+ winners
  ([HF org](https://huggingface.co/LeRobot-worldwide-hackathon)). The #1 US / #2 global team (shirt folding and
  medicine sorting on SO-100) used **SmolVLA trained for 40k steps** on 86 demos and **lost 36 episodes to camera
  issues** ([Lightning blog](https://lightning.ai/blog/lerobot-hackathon-winners)). Lesson: camera reliability
  decides the result.
- [ggando: SmolVLA on SO-101](https://ggando.com/blog/smolvla-so101/) (repo [ggand0/vla-so101](https://github.com/ggand0/vla-so101)):
  - **75 clean, consistent-strategy demos beat 81 mixed-strategy ones.**
  - **50 demos spread over 30 cm failed, while 75 over 10 cm worked.** Shrink the workspace and increase density.
  - An overhead camera helped (SmolVLA dual-camera 100% vs ACT or wrist-only 80%).
  - A USB-2 port silently throttled the RealSense; check with `lsusb -t`.
  - LeRobot's synchronous `record_loop` makes the follower lag, so move slowly.
- [sherryxychen: ACT on SO-101](https://huggingface.co/blog/sherryxychen/train-act-on-so-101) (repo
  [sherrychen1120/so101_bench](https://github.com/sherrychen1120/so101_bench)):
  - Write **udev rules** by physical USB path, because identical webcams swap indices.
  - Lock exposure and white balance, and tape the camera and arm positions.
  - **Teleoperate only from the camera feed**, never by looking at the arm.
  - Calibration files lost after a power cycle broke the evaluation, so **back up calibration**.
- [Seeed wiki](https://wiki.seeedstudio.com/lerobot_so100m_new/): no USB hubs for cameras; use the right-arrow key to
  end an episode early.
- [Glen Yu (Medium)](https://medium.com/@glen.yu/what-i-learned-training-an-act-policy-for-the-lerobot-so-arm101-856c0f4d1328):
  2 encoder threads per camera; standardize on 30 fps; going from 50 to 200 to 325 episodes kept improving results,
  with targeted "awkward placement" episodes added last.
- XLeRobot docs: aim for one-shot grasps (no micro-corrections), and start with 20 episodes before scaling.
- MakerMods' own HF datasets (`makermods/merged_200ep_100corr3x_...`,
  `makermods/300ep_..._with_corrections`) show the pattern they use internally: **base demos plus DAgger corrections
  upweighted 3x**.

---

## 2. Harness / orchestration (VLM planner, low-level policy, success checks, retry)

| Repo | Stars / last push | What to steal |
|---|---|---|
| [makermods-robotics/metal-arm-harness](https://github.com/makermods-robotics/metal-arm-harness) | ★15 / 2026-09-29 | **Built for our exact arm.** `serve` runs a session and `op tip x y z pitch` / `op nudge forward=0.02` / `op gripper N` return joints, tool pose and one JPEG per camera. Its safety envelope (measured table floor, slow zone, 35°/move cap, over-temperature veto, "recovery never descends") is a good scripted fallback for resets and recovery moves. `docs/OPERATING.md` lists the bench facts: gripper 0 is closed and 112 is open; **never raise the gripper lead cap above 8°**, because it overheats and latches off within seconds and drops the plug. |
| [ehehee/open-gemini-robot-harness](https://github.com/ehehee/open-gemini-robot-harness) | ★17 / 2026-10-21 | Open bimanual harness on the "Gemini 4 Argon for Robotics" design. Each turn draws a **2 cm world grid and pinch-point reticle on the images**, the VLM replies with a skill program (`right.grasp`, `left.goto`, `parallel(...)`), and every step is IK- and collision-checked. Rejections come back as one word (`IK_FAILURE` / `COLLISION`). It records full runs with HTML reports. Steal the observation overlays and the rejection-feedback loop. |
| [robocurve/inspect-robots](https://github.com/robocurve/inspect-robots) (MakerMods fork: [makermods-inspect-robots](https://github.com/makermods-robotics/makermods-inspect-robots)) | ★746 / 2026-11-04 | "Inspect AI for robotics": benchmark plus any policy (VLA or LLM agent) on any rig, with grader scores, a transcript and **Rerun `.rrd` logging**. Plugin architecture (`inspect-robots-yam`, `inspect-robots-agent`). Steal its grader / scorer abstraction and Rerun logging for our eval loop and demo video. |
| [robocurve/clapboardbench](https://github.com/robocurve/clapboardbench) | ★3 | Real bimanual traces of `claude-opus-5` vs `gemini-robotics-er-2-preview`, one motion per call. Good reference prompts and 224x224 frame budgets. |
| [google-deepmind/gemini-robotics-sdk](https://github.com/google-deepmind/gemini-robotics-sdk) | ★614 | Safari SDK; mostly Trusted-Tester gated. The ER models are usable through the Gemini API for **task-progress / success estimation**. The GR 1.5 tech report reports agent success-detection failure rates of only 4–6% ([PDF](https://storage.googleapis.com/deepmind-media/gemini-robotics/Gemini-Robotics-1-5-Tech-Report.pdf)). |
| [GitHub30/Awesome-Gemini-Robotics](https://github.com/GitHub30/Awesome-Gemini-Robotics) | ★22 | Prompt gallery for ER 1.5 (pointing, boxes, progress) |
| [Grigorij-Dudnik/RoboCrew](https://github.com/Grigorij-Dudnik/RoboCrew) | ★143 | LLM agent with **VLA policies as tools** (XLeRobot). A simple pattern: `tools=[run_vla("unplug usb"), ...]` |
| [rokbenko/quackd](https://github.com/rokbenko/quackd) | ★290 / 2026-11-07 | CLI that gives each robot an LLM "brain"; worth skimming for its tool schema |
| [ok-robot/ok-robot](https://github.com/ok-robot/ok-robot) | ★617 / 2024 | Modular navigate, pick, drop pipeline; dated, but its failure taxonomy is useful |
| LeRobot `lerobot-annotate` ([docs](https://github.com/huggingface/lerobot/blob/main/docs/source/annotation_pipeline.mdx)) | — | A **VLM subtask segmenter** using timestamped contact sheets (frames every 0.5 s tiled into a JPEG grid with the time burned in) and a describe-then-segment two-pass prompt with event-boundary rules ("object becomes held / released / lid changes state"). PR #3896 runs it on Qwen3.6-27B via vLLM. **Use the contact-sheet trick for our VLM success checks; it is much cheaper than one image per frame.** |

Hierarchical-policy references (papers, not code): Hi Robot (VLM plus pi0, no official code), Gemini Robotics 1.5
agent (orchestrator plus VLA, with success detection as a subtask), "Critic in the Loop" tri-system VLA
(arXiv 2603.05185, which triggers a VLM replan when it detects an execution anomaly), and the "Automating Real-World
Robots using VLMs" sub-task verification module ([PDF](https://robot-ma.github.io/MA_paper.pdf)).

**Design takeaway:** no public repo combines (a) a WAM/VLA executor per subtask, (b) a fast learned success
classifier, and (c) a VLM tie-breaker with retry and recovery policy. That combination is our harness. Take the
observation overlays from open-gemini-harness, the scorer and logging from inspect-robots, the safety and reset
primitives from metal-arm-harness, and the DAgger loop from makermodslab / `lerobot-rollout`.

---

## 3. Cable / USB insertion and door / latch opening

| Source | What to steal |
|---|---|
| **REBOOT26** HF datasets ([org](https://huggingface.co/REBOOT26)): `usbA-install`, `usbA_remove`, `USB-A_recovery_install`, `USB-A_recovery_remove`, `usb-c_install/remove`, `usbc_recovery_install`, plus RJ45, HAN10e, RCA, M12, NEMA plugs | Real **bimanual** WidowX-AI, LeRobot v3, 30 fps, 14-D joint action, about 60 episodes and 54k frames per task. Paper: *REBOOT: From Failure to Recovery* ([arXiv 2609.22591](https://arxiv.org/html/2609.22591v1)), with 2,160 trajectories over 18 connector tasks (1,080 expert plus 1,080 recovery), **recovery failures distributed to match real policy rollouts**, and per-phase natural-language success and failure descriptions. **Copy its phase definitions and success/failure wording for our checker prompts, and copy its recall-to-recovery recording protocol.** |
| [Xense/insert-usb-ethernet-multipin-0901](https://huggingface.co/datasets/Xense/insert-usb-ethernet-multipin-0901), `villekuosmanen/agilex_takeout_insert_usb`, `lyl472324464/insert_usb_port_hub`, `YOLO2431/plug-usb-v4.2`, `AivexRoboticsGroup/omy_f3m_left_insert-usb-*` | More USB-insertion LeRobot datasets on other embodiments; useful for looking at camera placement and episode length |
| `RoboCOIN/AI2_Alphabot_2_unplug_data_cable`, `RoboCOIN/R1_Lite_connect_the_router_cable` | Unplug / connect cable on bimanual humanoids |
| Door opening: `RoboCOIN/Airbot_MMK2_open_door_right` / `close_doors`, `IPEC-COMMUNITY/nyu_door_opening_surprising_effectiveness_lerobot` | Panel / door open and close references |
| [UD-Y/latchbench](https://github.com/UD-Y/latchbench) + [UD77/latchbench-so101-v1](https://huggingface.co/datasets/UD77/latchbench-so101-v1) | SO-101 RJ45 insertion in Isaac Lab-Arena with **staged success predicates** and clearance tiers. The expert **approaches the estimated port, detects contact from servo load, then runs a spiral search**, and **`observation.state` includes `Present_Load`**. Trick: add motor current / load to state so the policy (and our checker) can feel contact. |
| Visuo-Tactile Pretraining for Cable Plugging ([arXiv 2403.11898](https://arxiv.org/html/2403.11898v1)) | USB plug-in with ACT vs Diffusion: vision-only ACT 20%, DP 45%. DP's longer replan horizon "jumps around the port". Expect insertion to be the bottleneck subtask. |
| HIL-SERL ([rail-berkeley/hil-serl](https://github.com/rail-berkeley/hil-serl), ★1.6k; LeRobot `docs/hilserl.mdx`) | Reached near-100% on **USB grasp-insert** with a learned binary reward classifier plus human interventions. Too slow to run RL at the event, but the **reward classifier** part is reusable (see section 4). |
| [Ice-Citron/Project-Automaton__Quantarune__](https://github.com/Ice-Citron/Project-Automaton__Quantarune__) (Intrinsic AI for Industry challenge) | OpenCV contour port detection from the wrist camera, plus a ground-truth-guided insertion data logger |

**Practical tricks for our task**
- Treat insertion as its own subtask with its own fine-tune or prompt. End it on the checker's verdict, not on a timer.
- On a failed insert: back off 1–2 cm (scripted, via the metal-arm-harness `nudge`), re-run the insertion policy, and
  **record those recoveries as DAgger data** (the REBOOT recipe).
- Have the **non-inserting arm stabilize** the rack or cable. Bimanual helps here: one arm holds the panel or cable
  while the other acts.
- Grasp the plug body consistently (same orientation every demo). Consistency beats diversity at about 50–100 demos.
- Watch the gripper: the Metal gripper overheats and drops the plug if squeezed harder than the 8° lead cap (see
  OPERATING.md).

---

## 4. Success classifier / progress estimator

Ranked by **speed × reliability** under event constraints.

| Option | Code | Notes |
|---|---|---|
| **A. Frozen DINOv2/v3 or SigLIP features + logistic regression / small MLP**, one binary head per subtask ("usb_out", "panel_open", "panel_closed", "usb_in") | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) ★13.5k, [dinov3](https://github.com/facebookresearch/dinov3) ★11.7k, [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) ★14.2k | **Recommended primary.** Trains in seconds on CPU or MPS from frames we already record (label the last 10–15% of each subtask segment positive and the earlier frames negative, and add failed rollouts as hard negatives). Runs in ~10 ms, deterministic, and can use a wrist crop for "plug seated" checks. Probes often beat prompt engineering on fine discrimination. Use temporal smoothing (k-of-n frames). |
| **B. LeRobot reward `classifier`** (HIL-SERL) | `src/lerobot/rewards/classifier/` | Pretrained-ResNet binary success classifier with a LeRobot-native train config. Basically option A with plumbing included. |
| **C. Robometer-4B** (zero-shot progress and success heads on Qwen3-VL-4B) | [robometer/robometer](https://github.com/robometer/robometer) ★279; `lerobot/Robometer-4B`; LeRobot `rewards/robometer` | Zero-shot dense progress in [0, 1] from 8 frames plus a task string. Needs a GPU (Modal). Good as a **progress estimator and stall detector** (progress flat for N seconds means retry). |
| **D. TOPReward** (zero-shot, logprob of "True" from Qwen3-VL-8B) | [TOPReward/TOPReward](https://github.com/TOPReward/TOPReward) ★120; LeRobot `rewards/topreward` | No training. Its offline labeller writes SARM-compatible progress parquet. Useful for **curating demos** (dropping bad episodes) before the Dream upload. |
| **E. SARM** (stage-aware reward model, CLIP features) | LeRobot `rewards/sarm`, `docs/sarm.mdx`, arXiv 2509.25358 | Predicts stage k plus within-stage progress τ, which suits a 4-stage task. Also enables **RA-BC** (reward-weighted BC) for filtering and weighting demos. Needs subtask annotations (from `lerobot-annotate` or our state-machine timestamps). |
| **F. Fine-tuned VLM yes/no checker** | Fireworks SFT for VLMs ([docs](https://docs.fireworks.ai/fine-tuning/fine-tuning-vlm), [blog](https://fireworks.ai/blog/vlm-fine-tuning): Qwen2.5-VL family, OpenAI-style `image_url` base64 messages; served only on on-demand deployments). Open alternatives: [2U1/Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune) ★2.0k (Qwen2/2.5/3-VL LoRA), [modelscope/ms-swift](https://github.com/modelscope/ms-swift) ★16k, [unslothai/unsloth](https://github.com/unslothai/unsloth) ★79k, [HF TRL VLM cookbook](https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl), vLLM to serve | Best for the **explainable tie-breaker** ("USB not fully seated: 2 mm gap visible") and for the pitch. Slower (0.5–2 s) and needs a dedicated deployment on Fireworks, which has a spin-up cost. Train only once we have about 500 labelled frames. Zero-shot Gemini-Robotics-ER / Qwen3-VL with a contact-sheet prompt is the day-one baseline. |

**Recommended stack:** a probe (A/B) runs every frame and gates subtask transitions. If the probe's confidence is
between 0.3 and 0.7, or a subtask times out, call the VLM (zero-shot ER or Qwen3-VL first, the fine-tuned
Fireworks Qwen2.5-VL later) on a 2x3 contact sheet of the last ~3 s. Robometer or SARM progress serves as a stall
detector. All verdicts get logged, and those logs become new probe training data.

---

## Code we should have pre-written before 8am

Plain Python, one package (`hillclimb/`). Each module is runnable on its own and testable without the robot (mock
camera and mock policy).

1. **`dream_io.py`**: a thin wrapper over `dream.DreamClient`.
   - Covers warm-up / `pool_warming` retry, `reset_episode()` per subtask, the action-chunk loop driven by
     `refill_due()` and `action_hz`, latency logging via `last_timing()`, and checkpoint switching per subtask
     (`dream-1-ft:<uid8>-<suffix>`).
   - Also holds the fine-tune launcher: `upload_dataset`, then poll until `processed`, then `fine_tuning.create`
     (steps, cosine, `base_share`), then poll `get()` until `fine_tuned_model` is set. Plus a one-shot
     `resume_from` for DAgger rounds.
2. **`dataset_tools.py`**:
   - LeRobot **v3.0 to v2.1 down-converter**, unless the station records v2.1 natively.
   - Our Metal/Makers embodiment config dict (state/action spans, `scene`/`wrist_l`/`wrist_r` roles).
   - `validate_dataset()` pre-flight on every session.
   - Episode filter / merge, and per-episode subtask boundary export (from state-machine timestamps).
3. **`state_machine.py`**: an explicit FSM.
   - Stages: `HOME → UNPLUG_USB → OPEN_PANEL → CLOSE_PANEL → PLUG_USB → DONE`.
   - Each state has an entry action (prompt text, checkpoint id), a policy loop, a success predicate, a timeout,
     `max_retries`, and an on-fail transition (`RETRY_SAME`, `BACKOFF_THEN_RETRY`, `SCRIPTED_RECOVERY`, or `SKIP`
     if the scoring allows it).
   - Pure logic with no I/O, so it is unit-testable.
4. **`checkers/probe.py`**:
   - Extracts DINOv2-S/14 (or SigLIP) embeddings from scene and wrist frames, then trains per-subtask logistic
     heads.
   - `label_from_episodes()` auto-labels from subtask boundaries.
   - Includes k-of-n temporal smoothing and a calibration report (precision at fixed recall).
   - Saves and loads with `joblib`. CPU/MPS only.
5. **`checkers/vlm.py`**:
   - A contact-sheet builder (6 frames with timestamps burned in, as in `lerobot-annotate`).
   - Per-subtask yes/no prompts with REBOOT-style failure wording, returning JSON `{done, confidence, failure_mode}`.
   - Backends: Gemini Robotics-ER, Fireworks (Qwen-VL, base or fine-tuned), Modal vLLM.
   - Caching, a 3 s timeout, and majority-of-3 for close calls.
6. **`checkers/progress.py`** (optional, Modal GPU): Robometer-4B / TOPReward wrapper giving progress in [0, 1], used
   for stall detection and demo curation.
7. **`recovery.py`**: scripted primitives that use `metal-arm-harness`-style safe moves (or direct SDK calls).
   - `retreat(dz)`, `home_arm(side)`, `open_gripper`, `wiggle_insert` (small spiral or lateral search),
     `regrasp`.
   - Every primitive is bounded by an envelope check.
8. **`runner.py`**: the main loop (cameras → FSM → policy → checker → recovery).
   - Writes **one run log**: Rerun `.rrd` or JSONL with frames, actions, verdicts and latencies.
   - Has a **human-override hotkey** that drops into DAgger / teleop and tags the segment as a correction.
9. **`fireworks_vlm_ft.py`**: converts logged (frame contact sheet, verdict) pairs into Fireworks VLM SFT JSONL
   (base64 `image_url`), launches the job via `firectl`, and deploys on demand. Do a dry run tonight with about 20
   fake samples so the format is proven.
10. **`station_preflight.sh`**:
    - udev / camera-index check (identical webcams swap), `lsusb -t` USB3 check, and locked exposure and white
      balance.
    - CAN-up and motor scan (`maker-arm doctor` / `scan`).
    - Calibration-file backup, and a disk-space check for video.
11. **`eval.py`**: runs N trials end-to-end and reports per-subtask success, retries used and time, giving the
    leaderboard-style numbers that tell us which checkpoint to submit (inspect-robots-style scorer).
12. **`demo_protocol.md`** (one page, for the teleoperators):
    - Fixed plug grasp; one strategy per subtask; teleoperate from the camera view only.
    - About 10 cm workspace variance; pause about 1 s at each subtask boundary (so the boundaries can be labelled).
    - Record about 20% **recovery demos** (start from a failed state) following REBOOT.
