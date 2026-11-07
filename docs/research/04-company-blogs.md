# 04 — Lessons from robotics company blogs (2024–2026)

Scope: practical lessons for fine-tuning Dream-1 (a World Action Model) on our own MakerMods Metal Arm teleop demos and wrapping it in a harness for the 4-stage server-rack task: unplug USB, open the rack backing, close it, plug in a new USB. Compiled 2026-11-07. "Secondary" marks a source that is not the company's own.

---

## 0. Dream Labs — Dream-1 (the model we will fine-tune)  ← read first

Source: https://dreamlabs.ai/research/dream-1/ , secondary explainer https://www.eesel.ai/blog/dream-labs-dream-1 , founders' prior paper DreamZero https://dreamzero0.github.io/

- **I/O contract.** Three cameras: a head view plus a wrist view on each arm, along with arm state and the task text. Output is imagined future frames plus an **action chunk covering 1.6 s** (the secondary source says 48 steps, which works out to about 30 Hz). Each API call covers 1.6 s of motion, so 2,250 calls per hour. After each chunk the real frame replaces the imagined one. Actions are joint angles for embodiments the model knows, or EE pose plus gripper for new ones. In EE-pose mode, zero-shot quality depends partly on **your IK solver**.
- **Metal Arm (MakerMods) is a listed embodiment.** Zero-shot: 80.5% in sim and **70.5% on the real robot** (relative score vs the YAM reference). Sim language-following is 76% zero-shot and **87% after FaaS** (fine-tuning as a service).
- **Fine-tune data on the beta robots ranged from 5 minutes to 98 minutes.** Fine-tuning raised language-following on 10 of 12 partner robots.
- **Our sub-tasks are its weakest categories.** Zero-shot success by category: Pick & place 38–40%, **Insert & fit 11–24%**, **Open & close 11–16%**, Tool use 6%. Unplug/plug falls under insert & fit and the backing panel under open & close. **So our gains will come from fine-tune data, not from prompting.**
- DreamZero (the predecessor): 14B autoregressive video diffusion model, about 150 ms per chunk (7 Hz replanning). It adapted to a new robot from **30 minutes of play data (55 trajectories)**, and it learns from heterogeneous data without needing repetitive demos.
- Pricing: $3/hr (Dream-1) and $2/hr (Dream-1-flash). Inference is cheap, so we can afford many autonomous eval rollouts.

---

## 1. Physical Intelligence (π)

**π\*0.6 / RECAP — "A VLA that Learns from Experience"** https://www.pi.website/blog/pistar06
- There are three data types, and **all three are used**. (1) Demonstrations teach the basics. (2) **Corrections**: an expert takes over teleop while the current policy is running and makes a mistake. This supervises exactly the states the policy actually reaches, which addresses compounding error. (3) Autonomous rollouts, labelled by outcome.
- The value function predicts −(steps to completion), and the policy is conditioned on advantage (the change in value). Failed and suboptimal data still go into training, tagged with an advantage label instead of being thrown away. At run time the model is asked for high-advantage actions.
- Results: **more than 2× throughput and about 2× fewer failures** on the hardest tasks, and **over 90% success**. Systems ran for hours: espresso from 5:30am to 11:30pm, and 59 boxes assembled in a real factory.

**RL Token (RLT) — precise manipulation with online RL** https://www.pi.website/research/rlt
- Tasks: **Ethernet insertion, power-cord insertion**, M3 screwdriving, zip ties. These are the closest public analogue to USB plugging.
- **15 minutes of robot data and 2 hours of wall clock** (the rest was resets). The most precise phases sped up by up to 3×. Median episode length: 66 steps for RLT, 146 for teleop, 228 for the base model. RLT ended up faster than any teleop demo.
- Recipe tips: the actor *edits* the VLA's action chunk rather than replacing it, the update is regularized toward the VLA action, reference-action dropout is used, and human interventions can be folded straight into the update.
- Lesson for us: **the base VLA already gets insertion roughly right, and its weak spot is the precise final phase.** If we cannot run RL, collect *extra* demos of only the final ~3 cm approach and insertion.

**Real-Time Chunking (RTC)** https://www.pi.website/research/real_time_chunking
- Synchronous chunk execution pauses between chunks. Those pauses never appear in the training data, so they hurt both performance and speed.
- RTC treats the next chunk as inpainting: actions that will run during the inference delay are frozen, and the overlapping middle actions are soft-masked. Settings: 50-step chunk = 1 s at 50 Hz, total latency 108–139 ms (97 ms of that is model time). With RTC, performance **did not change with +200 ms of injected latency**, while synchronous execution degraded badly. No retraining needed. LeRobot ships an RTC option (`--inference.rtc.execution_horizon=10`).
- Lesson for us: Dream-1 sits behind an API, so network latency is guaranteed. **Pipeline the next request before the current 1.6 s chunk ends** and blend at the boundary. Never stop-and-wait.

**π0.5** https://www.pi.website/blog/pi05
- **High-level subtask text is predicted first, then a 50-step low-level chunk.** Training data includes observations labelled with the current subtask (for example "pick up the pillow") and "verbal instruction" demos in which a person coaches step by step.
- Generalization grows with the number of distinct environments, at roughly 100 environments matching in-domain training. For us: vary the setup slightly rather than repeating one pose.

**Knowledge Insulation** https://www.pi.website/research/knowledge_insulation
- When action-expert gradients flow into the VLM they damage it: learning slows, the model forgets, and **language following degrades**. Stopping that gradient trains 7.5× faster and keeps instruction following intact.
- For us: if Dream FaaS exposes options, prefer settings that preserve language conditioning, because the harness depends on switching sub-task prompts.

**Robot Olympics** https://www.pi.website/blog/olympics
- Fine-tuning π0.6 needed **under 9 hours per hard task** (about 8 h for turning a sock inside out). Average success was 52% with 72% task progress. A VLM without robot pretraining solved none of the tasks.
- Failures came from **embodiment limits** (gripper too wide for a sleeve). **Check early whether the Metal Arm gripper can physically pinch a USB plug and the panel latch.**

**π0 / openpi** https://www.pi.website/blog/pi0 , https://www.pi.website/blog/openpi — pretrain + fine-tune paradigm; fine-tuning on 1–20 h of high-quality task data.

---

## 2. Dyna Robotics — reliability via reward model

**Dyna-1** https://www.dyna.co/research/dyna-1 ; **Talk "Robot Demos Are Easy. Reliability Is Hard"** (Jason Ma), summarized at https://ai.engineer/talks/Sjfz1TqxzEs-robot-demos-are-easy-reliability-is-hard and https://finance.biggo.com/news/03d0d18a242a1a23
- Results: **99.4% success** over 24 h with zero interventions (850+ napkins at about 60% of human speed). Performance went from failing after 5 minutes in week 1 to 24 h runs in week 6, through repeated cycles.
- **The standard pretrain + post-train recipe plateaued at about 80%.** The failure mode was that after a mistake the robot went out of distribution and got stuck. More successful demos did not fix it, because the missing data was **what to do after a mistake**.
- **Reward model = video progress estimator (0→1).** A successful attempt rises roughly monotonically, and **a non-monotonic dip marks the failure moment.** The reward model runs in the background and points humans at the failures. Humans collect **targeted recovery demos from those exact states**, then fine-tune and repeat (human-in-the-loop active learning).
- Recovery behaviour generalized to new error types after several cycles, partly by interpolating recovery skills from other tasks in pretraining.
- Ma's compounding math: 80–90% per attempt makes long uninterrupted chains unlikely (0.9^10 ≈ 35%; the "<0.1%" figure on the slide does not hold up). **That is exactly our 4-stage problem: 0.8^4 = 41% vs 0.95^4 = 81%.**

**Open-world dexterity** https://www.dyna.co/research/open-world-dexterity
- Tens of hours of post-training data from a handful of office stations held 99%+ on unseen stages, lobbies, and the CoRL hall for 3 days, robust to lighting and audience interference.
- They define production quality separately from success (grade ≥4/5). **Find out exactly what the judges count as "success" for each stage.**

**Dyna-2** https://www.dyna.co/dyna-2
- WAM pretrained on more than 1M hours of human ego video. Bottle-cap untwisting was post-trained on **about 10 minutes** of robot demos, and benchmark tasks use ≤10 h each. Language-following is scored 1 / ½ / 0, with ½ for the right primitive attempted but not completed. That rubric is a useful one to copy for our own eval.

---

## 3. Generalist AI — GEN-0 / GEN-1

https://generalistai.com/blog/gen-0 , https://generalistai.com/blog/gen-1 , https://generalistai.com/blog/the-real-breakthrough-behind-our-gtc-demo
- GEN-1 reaches **99% average success where earlier models reach 64%**, with about 3× faster execution, from **about 1 hour of robot data per task**. Its video list includes **"Robot plugs in ethernet cable."**
- GEN-0: more pretraining improves post-training predictably. The best results (up to 99%) need large pretraining **plus ample task data**. With only 5.6 h of task data, gains were real but smaller.
- GTC demo: no data from the venue hall, and the robot ran nonstop all show hours with the same performance as in the office. Pretraining absorbs the change of scene, so task data should cover task variation rather than venue variation.
- Secondary analysis https://archiesengupta.com/blog/embodied-scaling-laws-robot-data-wars: the scaling laws are **coverage laws, not volume laws**. Returns saturate at about 50 demos per environment-object pair, and long-tail data helps closed-loop performance disproportionately.

---

## 4. Figure — Helix

**Helix logistics** https://www.figure.ai/news/helix-logistics
- **Quality over quantity: curated demos gave 40% better throughput using ⅓ less data.** About 8 h of well-curated data was enough.
- They **filtered out slower, missed, or failed demos**, but **kept demos with natural corrections when the trigger was environmental randomness rather than operator error.**
- **Operators were made to "uniformize" their strategy.** Mixed strategies across operators hurt the model.
- **"Sport mode":** resample the action chunk to 0.8× its length and execute it at the same control rate, giving a free 20% speedup and up to 50% before performance collapses. Use this if scoring rewards time.
- Stereo/multi-scale vision gave +60% throughput. Learned visual self-calibration covers robot-to-robot differences. Relevant because **stations are shared, so camera mounts may drift between bookings.**

**Scaling Helix** https://www.figure.ai/news/scaling-helix-logistics
- Going from 10 h to 60 h of data: package handling time fell from 6.84 s to 4.31 s and success rose from 88.2% to 94.4%. More data mostly buys **speed and the last few percent of reliability**.

---

## 5. Google DeepMind — Gemini Robotics 1.5 / ER

https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/ , https://developers.googleblog.com/building-the-next-generation-of-physical-agents-with-gemini-robotics-er-15/ , tech report https://arxiv.org/html/2510.03342v3
- It is a two-level agent. An embodied-reasoning model (ER 1.5) plans, calls tools, and **detects success**, while the VLA executes short segments. The VLA can "think" in order to split long tasks into short ones that execute reliably.
- **Success detection is hard in two specific ways.** (1) **Multi-view is needed**: no single camera may see the outcome. (2) **Latency**: slow detectors give stale answers. The report evaluates real-time and offline success detection as separate settings.
- Progress estimation (percent complete, frame unshuffling) supports policy evaluation, data filtering, and long-horizon orchestration.
- For our harness: run a VLM success check on **head + wrist cams at each stage boundary only**, not on every frame. Use a fast model (Fireworks) during the chunk loop and a slower, careful check at stage gates.

---

## 6. 1X — World Model as evaluator

https://www.1x.tech/discover/redwood-ai-world-model , paper https://www.1x.tech/1x-world-model.pdf
- The world model predicts future frames **and whether the task succeeded**. It is used for **checkpoint selection**: with a real 15% gap between two policies, a world model that is 70% accurate picks the better one 90% of the time.
- **Imitation-learning validation loss did not correlate with real-world success.** Never pick a checkpoint by loss; pick it by rollouts.
- The world model fails on unseen objects and on articulated parts until it has task data (it treated an air-fryer tray and its body as one piece). Dream-1's imagined frames will likely do the same with the rack door until we fine-tune.
- Idea for us: **Dream-1 returns imagined frames.** Comparing the imagined frames with the real frame after each chunk is a free anomaly signal: a large divergence means something unexpected happened. Worth testing as a recovery trigger.

---

## 7. Toyota Research Institute — LBM

https://toyotaresearchinstitute.github.io/lbm1/ , https://arxiv.org/html/2507.05331v1 , https://www.tri.global/news/tri-pretrained-large-behavior-models-accelerate-robot-learning
- Pretrained LBMs needed **a fraction of the task data** to match single-task baselines, and they were more robust to distribution shift (1,700 h of data, 1,800 real plus 47k sim rollouts).
- **Filter low-motion (idle) segments.** Unfiltered models "take excessively long to start any motion." **Trim the idle head and tail of every episode.**
- Their evaluation was statistically careful: blind A/B tests, many trials, and confidence intervals. A comparison of 10 trials against 10 trials cannot separate 70% from 90%.

---

## 8. NVIDIA — GR00T N1.5 / N1.6 / N1.7

https://research.nvidia.com/labs/gear/gr00t-n1_6/ , https://github.com/NVIDIA/Isaac-GR00T , secondary https://www.naddod.com/ai-insights/a-deep-dive-into-pre-training-and-post-training-for-nvidia-isaac-gr00t-n1-6
- Post-training needs about **20–40 real demos + 10K–30K steps** per task (secondary source). N1.7 adds 20k hours of EgoScale human video to pretraining.
- EgoScale (secondary summary above): with a strong pretrain, about 4 h of robot data in the alignment stage replaces corpora that used to take 88 h.

---

## 9. Hugging Face — LeRobot

https://huggingface.co/blog/lerobot-datasets , https://huggingface.co/docs/lerobot/en/smolvla , https://huggingface.co/blog/smolvla (async inference)
- **About 50 episodes per task as a starting point, structured as about 10 episodes per variation** (5 positions × 10). **25 episodes was not enough.**
- Recording hygiene:
  - Fixed, non-shaking cameras.
  - **The leader arm and human limbs must stay out of frame.** The only moving things should be the follower arm and the object.
  - Stable, neutral lighting.
  - Resolution ≥ 640×480 at about 30 fps.
  - Consistent camera names.
  - Concise task strings of 25–50 characters.
  - **Use the same task string at inference as in recording.**
- Replay recorded episodes on the robot before training to validate them (secondary source: https://medium.com/correll-lab/fine-tuning-smolvla-for-new-environments-code-included-af266c56d632).
- Async inference separates prediction from execution, which is the same lesson as RTC.

---

## 10. Skild AI

https://www.skild.ai/blogs/learning-by-watching , https://www.skild.ai/blogs/building-the-general-purpose-robotic-brain , secondary https://www.eesel.ai/blog/skild-ai-s1
- New skills come from video plus **less than 1 h of robot data**. "For every dollar we spend on collecting data, we spend three on quality control."
- S1: one in-context demo is worth about 380 teleop demos. It also showed an emergent recovery behaviour of retrying after a mistake. The stack is a two-level hierarchy (slow planner, fast controller).

## 11. Sunday Robotics, Mecka AI, Tesla Optimus

- Sunday ACT-1 https://www.sunday.ai/blog/no-robot-data: zero robot data. The Skill Capture Glove is co-designed to match the robot hand's geometry ("if you can do it with the glove, Memo can"). The lesson is to keep the **embodiment gap small**: operators should only do motions the follower can reproduce, and should avoid leader-arm tricks the follower cannot follow.
- Mecka https://www.mecka.ai/: data collection from human wearables and iPhones. Not relevant to tomorrow.
- Tesla Optimus: no technical engineering blog. Public material is video posts about learning from human video. No actionable detail found.

## 12. Data-center / cable-handling automation

- Meta (WIRED, 2026-08) https://www.wired.com/story/inside-metas-experiments-with-data-center-robots/
  - Dual-arm Watney robots are being piloted for cabling. Kinova and ABB arms are used for reseating parts and power cycling.
  - Pain points: earlier robots **crushed servers**, a grayscale camera could not read LED status, and racks are designed for human hands.
  - Lesson: **limit force and speed near the rack**, and do not rely on LED color unless the camera is color.
- Microsoft Research modular data-center robotics (transceiver/fiber insertion) https://www.datacenterdynamics.com/en/news/microsoft-research-details-early-stage-modular-robotics-for-data-centers/
- Visuo-tactile USB plugging (academic) https://arxiv.org/html/2403.11898v1
  - ACT reached 95% USB insertion with tactile input. Diffusion replanning every 8 steps "jumps around the port."
  - **Without tactile sensing, high-frequency closed-loop correction near the port matters.**
  - Unrecoverable state = dropped cable.

---

## Cross-company synthesis (what moves the success rate)

1. **Pretraining gets you to roughly 60–80%. The last 15–20 points come from recovery and correction data on the policy's own failures.** Sources: Dyna 80%→99.4%, RECAP 2× fewer failures, Figure keeping natural corrections. This is the main point of the whole survey.
2. **Quality and consistency beat volume.** One uniform strategy across operators; filter slow, failed, and idle segments; ⅓ less data gave 40% better results (Figure, TRI).
3. **Data budget:** frontier recipes adapt to a task with 10 minutes to about 1 hour of robot data per task (Dream-1 5–98 min, Dyna-2 10 min, GEN-1 about 1 h, RLT 15 min, DreamZero 30 min). Hard dexterous tasks need more (PI Olympics up to about 9 h). **For us: about 40–60 min of clean data per stage is a realistic target.**
4. **Decompose long-horizon tasks** into language-labelled sub-tasks, with a high level that picks the next step and verifies success (π0.5, Gemini ER, Skild). **This is our harness design.**
5. **Progress and reward models** locate failures for targeted data (Dyna) and select checkpoints (1X). Loss does not predict success (1X); only rollouts do.
6. **Latency:** never stop between chunks. Pipeline requests (RTC / async). Sport-mode resampling gives a free 20–50% speedup (Figure).

---

## Data collection protocol we should follow tomorrow

Assumptions: 5 people, one shared bimanual Metal Arm station booked in slots, and Dream-1 FaaS turnaround measured in hours. Adjust slot math when the rules drop. **Arm time is the scarce resource. Nothing that can be done off-arm happens on-arm.**

### Roles (fixed for the day)
- **P1 — Lead operator.** Does about 70% of demos so the strategy stays uniform (Figure).
- **P2 — Second operator / reset tech.** Resets the scene between episodes, and swaps with P1 every 30–40 min to avoid fatigue. P2 learns P1's exact strategy first.
- **P3 — Data QA.** Live-labels each episode (good / slow / fail / recovery), trims idle segments, writes sub-task boundary timestamps, and uploads to FaaS.
- **P4 — Harness engineer.** Owns the stage sequencer, success checks, and recovery logic, plus the pipelined Dream-1 client (RTC-style).
- **P5 — Eval + VLM checker.** Builds the stage-success classifier (Fireworks VLM on head + wrist frames) and the progress-dip detector, and runs eval rollouts and tallies.

### Before touching the arm (night before / first hour, off-arm)
1. **Write the canonical strategy as a one-page script per stage.** Cover which arm does what, the grasp point on the USB plug, how to pull (straight out along the port axis), which hand opens the backing and where it grips, and the approach pose before insertion (hover about 2 cm in front of the port, align, then push straight). **Everyone memorizes the same script.**
2. **Fix the four task strings now** and never change them (25–50 chars, LeRobot):
   - `unplug the usb cable from the rack`
   - `open the rack back panel`
   - `close the rack back panel`
   - `plug the new usb cable into the rack`
3. Pre-check embodiment feasibility (PI Olympics lesson). Can the gripper pinch the USB plug body and fit beside adjacent cables? Can it reach the panel latch? If not, report it now. Do not burn demos on an infeasible grasp.
4. Prepare the recording-hygiene checklist (LeRobot):
   - Cameras locked down, with mount positions photographed so we can restore them after other teams' slots.
   - Leader arm and bodies out of frame.
   - Constant lighting.
   - At least 640×480 at 30 fps.
   - Head + both wrist cams, because that is Dream-1's input contract.

### Arm slot 1 (target about 60–75 min): baseline + core demos
5. **First 10 min: zero-shot probe.** Run Dream-1 un-fine-tuned on each stage prompt, about 3 tries each. Record videos. This tells us which stages are worst (expect insert & fit and open & close to be weakest, per Dream-1's category table) and where to spend demos. P5 logs failure modes.
6. **Demo collection, per-stage segmented episodes** (one stage = one episode, about 10–40 s each):
   - Target per stage: **about 40 clean demos** (well above LeRobot's 25-is-too-few, around GR00T's 20–40). That works out to about 10 demos × 4 variations.
   - Variations to cycle, about 10 episodes each:
     - (a) USB port index or position on the rack, if the rules allow.
     - (b) Rack/panel start pose, a few cm or degrees of offset.
     - (c) Cable drape and the starting location of the new USB.
     - (d) Small lighting or background change.
   - Vary across positions rather than repeating one. Diversity beats volume (coverage laws, π0.5).
   - **Weight demos toward the bottleneck stages from step 5.** If plug-in is worst, give it 60 demos and give close-panel 25.
7. **Also record about 5 full end-to-end episodes** (all 4 stages back to back, with boundary timestamps marked by P3). They teach the transitions, and the stage starts in them are realistic, unlike the clean resets in the segmented episodes.
8. **Operator rules during collection:**
   - Move smoothly and decisively, without dithering, and at a consistent speed. Slow demos get filtered (Figure).
   - **Start moving within about 0.5 s of the episode start.** P3 trims idle head and tail anyway (TRI low-motion lesson).
   - For insertion, use the same approach every time: a straight-line approach, a brief align pause, then insert. Do **not** wiggle or search unless that is the canonical strategy.
   - **If something random goes wrong mid-demo** (cable snags, plug misses the port), **recover naturally and keep going.** Tag the episode "recovery". Keep it if the failure was caused by the environment; drop it if it was operator sloppiness (Figure rule).
   - Don't talk to or reach into the scene during an episode.

### Off-arm (while FaaS trains): QA + harness
9. P3 filters to good + recovery-tagged episodes, deletes failed and slow ones, and replays a sample on the robot (or in a viewer) to validate. Upload **v1 = all four stages in one multi-task fine-tune**, conditioned on the stage strings.
10. P4/P5 build the harness around the stage loop: `prompt(stage) → chunked execution → stage-success check`. On a failed check, retry the stage or run a recovery prompt. Success checks run at stage gates on head + wrist views (Gemini ER multi-view lesson), for example "is the USB plug fully out of the port", "is the panel open past X", "is the plug seated flush". Also log the per-chunk imagined-vs-real frame divergence for later use as an anomaly signal (1X idea).

### Arm slot 2: evaluate v1, then targeted recovery and correction data (Dyna / RECAP loop)
11. **Run v1 autonomously through the harness for about 10 full attempts.** P5 records per-stage pass/fail and the video. **Pick checkpoints by these rollouts, never by loss** (1X).
12. **For each failure, the operator takes over from the failure state** (DAgger/RECAP-style intervention) and demonstrates the recovery, then completes the stage.
    - Examples: the plug is misaligned at the port, so back off 1 cm, realign, and insert. The panel is half-closed, so re-grip and push. The USB is dropped, so re-grasp it.
    - **Better still, deliberately stage the common failure states** (plug resting just off-port, panel ajar, cable fallen) and record 5–10 recovery demos for each.
    - Label these with the *same* stage string, so the policy learns that recovery is part of the stage.
13. Collect about **10 extra demos of only the final precise phase** for insertion (from hover 2–3 cm in front of the port to seated). That is where base VLAs are weakest (PI RLT).
14. Upload v2 = v1 data + recovery/correction data + precision-phase data. Aim for a total of roughly 2–3 h of operator time across the day, with 40–60 min of clean data per stage.

### Arm slot 3: final eval + tuning harness knobs
15. Run ≥15–20 full attempts with v2. Compare v1 vs v2 per stage, and note that 10 trials cannot resolve small gaps (TRI). Tune: retry limits per stage, success-check thresholds, sport-mode speedup (try 1.0× / 1.2× / 1.4× only if scoring rewards time; Figure found ≤1.5× safe), and the request-pipelining lead time.
16. If time remains, run one more Dyna-style cycle: recovery demos for the top remaining failure state, then fine-tune v3.

### Hard rules
- One canonical strategy. No freelancing.
- Fixed task strings, identical between recording and inference.
- Keep natural recoveries and drop operator-error episodes.
- Trim idle segments.
- Restore camera mounts after every booking gap.
- Low force and speed near the rack (Meta: robots crushed servers).
- Every rollout gets logged (video + per-stage outcome). That log is our data engine and our proof for the leaderboard.
