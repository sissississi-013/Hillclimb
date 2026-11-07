# 02 — arXiv frontier scan: few-demo WAM/VLA fine-tuning, precision insertion, long-horizon harness, test-time tricks

Scope: server-rack task (unplug USB → open rack backing → close → plug new USB) on shared low-cost bimanual 6-axis arms, fine-tuning **Dream-1** (Dream Labs' World Action Model, sold via API) on our teleop demos, plus our own harness. Budget ~1.5 days, limited robot time. Compiled 2026-11-07.

Numbers are as the papers report them, usually on other hardware. Treat them as signs of direction, not as forecasts for our setup.

---

## 0. What we know about Dream-1 (it drives several choices below)

- Dream Labs was founded by former NVIDIA researchers, the authors of **DreamZero, "World Action Models are Zero-shot Policies" (arXiv:2602.15922)**. Public reporting says Dream-1 takes camera images + robot state + text, returns **1.6 s of actions per call**, predicts **end-effector target pose + gripper**, and costs about $3/h ($2/h "flash"). Sources: [eesel.ai explainer](https://www.eesel.ai/blog/dream-labs-dream-1), [BusinessWire/Morningstar funding PR](https://www.morningstar.com/news/business-wire/20261019230613/dream-labs-raises-90m-seed-to-build-the-brain-for-everyday-robots), [everydev](https://www.everydev.ai/tools/dream-1). Confirm everything against the official docs at the event.
- What the DreamZero paper says that matters for us (2602.15922):
  - Chunks are 1.6 s (H=48 at 30 Hz), closed loop at about 7 Hz, with an autoregressive video+action context of about 6.6 s.
  - **Idle actions are filtered out of the training data.** We should trim idle starts and ends from our demos too.
  - **Full fine-tuning beat LoRA** for them. If the API offers a choice, ask for full fine-tuning or action-expert fine-tuning.
  - **Diverse, non-repetitive data beat repeated demos for the same number of hours.**
  - **Adapting to a new embodiment took 30 min of play data.** Video-only data from humans or other robots (10–20 min) gave +42% relative on unseen tasks.
- What this implies: an EE-pose action space means our arm's IK/servo quality sets the floor on insertion precision. The 1.6 s chunks mean the harness has to decide how much of each chunk to execute (sections 4.2 and 4.3).

---

## 1. Few-demo fine-tuning of VLAs / WAMs

| Paper (arXiv) | One-line claim | Concrete number |
|---|---|---|
| **2602.15922** DreamZero (WAM) | A WAM built on a video-diffusion backbone generalizes to new motions better than VLAs | >2× generalization vs SOTA VLAs; 30 min play data adapts it to a new robot; full fine-tuning > LoRA |
| **2610.37170** ACT / DP / SmolVLA / π0.5 on SO-101 | On a low-cost arm, the pretrained VLA wins when data is scarce, and the **fine-tune recipe matters more than the amount of data** | π0.5: 16/40 at 12 demos, 28/40 at 25 demos, plateaus near 70%. LoRA 4/110 vs action-expert fine-tune 130/200. ACT from scratch reaches 37/40 at 100 demos. SmolVLA "never started moving" in 47% of episodes, and DP rarely started until the idle lead-in was trimmed |
| **2610.40828** RJ45 insertion ablation (281 policies, about 1,700 trials) | Representation choices dominate: **absolute rather than relative actions**, skill decomposition, second camera, recovery episodes | Absolute actions took π0.5 insert from 30→100%. Second view + 108 recovery episodes took π0.5 insert from 0→100%. Decomposed pipelines inserted in every trial, monolithic grasp+insert in ≤10% |
| **2509.07953** RaC (recovery and correction) | Human-in-the-loop takeovers that **rewind to an in-distribution state, then finish the subtask**, scale far better than more full demos | About 10× less data for the same success on bimanual long-horizon tasks; success scales with the number of retries at test time |
| **2511.14759** π*0.6 / RECAP | Advantage-conditioned offline RL on demos + autonomous rollouts + teleop corrections | More than doubles throughput and roughly halves the failure rate on the hardest tasks |
| **2604.23073** RL Token (PI) | Small actor-critic on a VLA readout token refines **only the critical phase** (Ethernet and charger insertion) | Up to 3× faster in the critical phase. Beats baselines after **5 min of critical-phase data (about 40 min wall clock)**. Emergent "press and wiggle" insertion. Needs human reward labels and interventions |
| **2410.21845** HIL-SERL | Human-in-the-loop RL reaches near-perfect success on precise insertion (USB, RAM, …) | Near 100% in 1–2.5 h of real training. RLT showed single-step RL fails on long sparse tasks; chunked RL works |
| **2608.15088** Max-Q selective imitation (ACT Q-chunk) | Chunk-level MC critic + imitate whichever is better, human or policy | **USB pick-and-insert: 99% in 30 min of HIL training from 20 demos** (HIL-SERL took about 5 h) |
| **2609.30023** Res-HIL | Zero-initialized residual RL on a frozen IL policy, supervised by interventions | Beat 5×-more-demo IL after **10 min** of online training (20 demos) |
| **2502.05450** ConRFT | Offline BC+Q, then online consistency-policy RL with interventions | 96.3% across 8 real tasks in 45–90 min online |
| **2604.16683 / 2503.03707 / 2506.19121** Rewind-IL, Demo-SCORE, CUPID | Curate demos by outcome | Demo-SCORE: +15–35% absolute by filtering out unreliable demo strategies |
| **2604.23000** RINSE, **2605.01544** PSD | Filtering demos by smoothness is cheap and works | SAL filter: +16% with 1/6 of the data. TED: +20% with half the data |
| **2606.22907** ISR resampling | Remove pauses and speed jitter from demos | About +25% vs uniform 3× downsampling |
| **2605.30695** Primitive subspaces | Segmenting episodes into primitives with **per-primitive language prompts** improves transfer | 3× sample efficiency (3 vs 10 demos reach 78% of the upper bound) |
| **2601.16163** Cosmos Policy | Video model → policy in one post-training stage; it also predicts future frames and values for planning | 98.5% LIBERO; best on real bimanual tasks; planning adds more |
| **2609.38057** EVO-WAM | WAM self-improves on its own VLM- and IDM-verified imagined rollouts | Cosmos3 real long-horizon: 20→76.7% |
| **2604.11351** WM-DAgger | World model synthesizes OOD recovery data for eye-in-hand few-shot | 93.3% with 5 demos (soft bag push) |
| **2506.16211** ControlVLA | Object-centric ControlNet-style adapter | 76.7% with 10–20 demos |

Takeaways:
1. **Spend demos where failures happen.** RaC-style recover-then-correct data is about 10× more data-efficient than more clean demos. Record "botched insert → back off → re-align → insert" episodes on purpose.
2. **Consistency and smoothness beat volume.** Use one operator per skill if possible, trim idle frames, and drop jerky or hesitant demos (cheap PSD/SAL filter).
3. **Use diverse starting poses, not 50 copies of the same trajectory** (DreamZero, π0.5 SO-101 data).
4. **The fine-tune recipe matters more than demo count.** Full or action-expert fine-tuning, not LoRA, if the Dream API lets us choose. Use absolute EE targets, which Dream-1 already appears to use.
5. **Real-robot RL is feasible in our window only for the insertion critical phase** (RLT / Max-Q / Res-HIL: 10–40 min). It needs an API that exposes features or allows a residual. We can do it locally with a small residual head over our own encoder if Dream-1 is black-box.

---

## 2. Precision insertion (USB / connector)

| Paper (arXiv) | Claim | Number |
|---|---|---|
| **2604.23073** RL Token | Base VLA "probes" near contact; RL learns a decisive insert + press-and-wiggle | Critical phase 3× faster; full-task success +40–60% on the hardest tasks |
| **2610.40828** RJ45 ablation | **Decompose into reach (classical/scripted) + learned insert**; second camera; recovery episodes; absolute actions | Decomposed pipeline 100% vs ≤10% monolithic |
| **2608.15088** Max-Q | USB pick-and-insert with ACT + chunk critic | 99% in 30 min HIL |
| **2609.28312** VGM-VS | Visual servoing to a **reference image taken at the goal pose**, using a pretrained geometry model | Sub-mm on USB-C cable insertion; 90–100% with a moving target; 30 Hz |
| **2609.15198** PredTac | Touch predicted from vision+state, with no sensor at deployment | Real USB insertion 70% vs visual ACT about 21% (3-task mean) |
| **2512.04960** Teleop Augmentation Primitives | Operator-triggered **axis locks / pose anchors / scripted routines** recorded into demos, then triggered by the policy | Wrist-cam peg insertion 0.37→0.53; policy-triggered routine 67% vs 38% |
| **2610.35626** 3DGS reach | Pre-contact reaching from wrist cam only + a **reach verifier** that stops before contact | π0.5: 80% vs 58.7% with 150 teleop demos |
| **2610.23849** Blind proprioceptive assembly | Joint-position history alone can find holes through contact | 73% real zero-shot peg insertion |
| **2509.17053** FILIC | Force estimated from joint torques + an impedance loop, with no F/T sensor | Beats vision-only on contact-rich tasks |
| **2607.14021** Industrial Dexterity Benchmark (datacenter cable board) | Multi-view RGB raises grasp+insert from 36→78%, about 100 demos per phase | Our domain exactly (datacenter cabling) |
| **2609.23800** ContactDP | Wrist RGB + hybrid position-force controller for tight connectors | Large gains over vision-only DP |
| **2505.11858** Model-based + residual RL insertion | Potential-field base + residual RL | <1 mm clearance, sim2real |

Insertion tricks that need no force sensor (ranked by how cheaply we can do them):
1. **Wrist camera on the inserting arm, framed so the port and plug tip are both visible at the moment of insertion.** Every top result relies on wrist views. Add a second static view of the rack face.
2. **Split the skill.** Do a coarse "reach pre-insert pose" (scripted, or a VLA subtask) first, then a short learned "align + insert" with the shortest replan interval. Prompt each phase separately.
3. **Collect deliberate miss-and-retry demos** for insertion: plug lands 2–3 mm off, back off 5 mm, re-center, push. This teaches the "probe/wiggle" recovery (RaC, RJ45 ablation, RLT emergent behavior).
4. **Scripted micro-search as a harness fallback.** When the progress/success detector says "not seated" after a push, run a small Archimedean spiral (r≤2–3 mm, 0.5 mm pitch) or a ±yaw wiggle in the EE frame, then push. This is the classical peg-in-hole fallback; the RLT finding that wiggling exploits compliance supports it.
5. **Use compliance we already have.** Lower the position-servo gains or stiffness on the final approach if the MakerMods arm exposes them, or command small overshoots ("press") the way RLT learned to. Detect seating from a joint-current spike or tracking-error plateau (FILIC-style force estimate) instead of an F/T sensor.
6. **Reference-image servoing for the last cm** (VGM-VS idea, simplified): store a wrist-cam image of the plug correctly pre-aligned with the port. In the final approach, keep only the action chunk or nudge that most reduces image distance (also see best-of-N in section 4).

---

## 3. Long-horizon decomposition, planners, success/progress detection, recovery

| Paper (arXiv) | Claim | Number |
|---|---|---|
| **2606.10267** What matters in Hi-VLA (GDM) | Best hierarchy uses a **thinking VLM, success-based termination (not a fixed timer), contact-aware observation descriptions, cross-episode memory**. In-episode memory and summaries are neutral or negative | Systematically beats flat VLA and a naive hierarchy on real ALOHA |
| **2504.16054** π0.5 / **2502.19417** Hi Robot | Same model predicts a high-level subtask in language, then actions conditioned on it | Basis of subtask-conditioned prompting |
| **2510.03342** Gemini Robotics 1.5 | ER model plans and runs success detection; VLA executes | Thinking before acting improves multi-step |
| **2603.05185** Critic in the Loop | Lightweight visual critic routes control: VLA for routine steps, VLM replan on stagnation or failure; **rules to break infinite retry loops** | SOTA on long-horizon benchmarks while cutting VLM calls |
| **2604.21924** LoHo-Manip | Manager re-predicts the **remaining plan** (done + remaining) every call, so failed steps persist and get retried automatically | Implicit closed-loop recovery with no hand-coded logic |
| **2605.13119** VLAs-as-Tools | VLM agent calls bounded VLA "tools" with progress feedback and event-triggered replanning | Real Franka 16.7→58.3% |
| **2603.02115** Robometer | Reward model trained with frame progress + trajectory preference (RBM-1M) | Generalizes better than prior RMs; open weights |
| **2509.15937** VLAC | Pairwise-observation process reward: progress delta + done flag; one-shot in-context transfer | Real RL 30→90% in 200 episodes |
| **2411.04549** GVL | Ask the VLM to **order shuffled frames** rather than read progress directly; zero-shot value | 300+ tasks zero-shot; used for success detection and filtering |
| **2611.03622** D&A (Describe & Assess) | Prompt-only: first describe the scene objectively, then score against an explicit success rubric. This cuts confirmation bias, where naive prompting "hallucinates progress" | Matches post-trained reward models; frontier VLMs otherwise collapse to binary |
| **2602.19313** TOPReward | Use the VLM's **token probability of "completed"** as a dense reward instead of generated numbers | Competitive with trained RMs, training-free |
| **2601.00675** RoboReward | No VLM excels across tasks; a trained 8B RM beats Gemini-ER 1.5 | Use negatives and near-misses when calibrating |
| **2609.05401** ROBORMBENCH | **Paraphrasing the instruction can flip a VLM judge's success/fail call** | Freeze one exact prompt per check |
| **2604.16683** Rewind-IL | Training-free failure detector from **inter-chunk disagreement** (TIDE) + rewind to a VLM-verified checkpoint state | Better long-horizon reliability without failure data |
| **2509.07953** RaC | Retry behavior learned from recovery data drives test-time scaling | Success rises linearly with the number of retries |
| **2603.11558** RoboClaw | **Forward + inverse paired skills** (e.g. open/close) form self-resetting loops for data collection | +25% long-horizon success, 53.7% less human time |

Takeaways for the harness:
- **Our task already decomposes into four subtasks**, and open/close are inverse pairs (RoboClaw). Each subtask gets its own prompt string for Dream-1 (Hi Robot / primitive-prompt result 2605.30695).
- **Use success-based termination for every subtask**, with a VLM judge on fixed rubrics (D&A style) and an exact frozen prompt (ROBORMBENCH). Feed it wrist + overview frames before and after. Where possible, also check proprioception (gripper closed with an object, joint-current spike at seating).
- **The retry policy matters more than the planner.** A fixed four-step plan with a state machine is enough; a VLM "planner" adds little for a known sequence. Use the VLM only as the judge and to pick a recovery branch, and cap retries per subtask (Critic-in-the-Loop).
- **Cross-episode memory helps but in-episode memory doesn't** (GDM study). After practice runs, write a short affordance note ("plug needs +2 mm left bias") into the judge or recovery prompt.

---

## 4. Test-time tricks (cheap success-rate gains)

| Paper (arXiv) | Claim | Number |
|---|---|---|
| **2506.07339** RTC (PI) | Async execution: freeze the committed actions, inpaint the rest of the next chunk; no retraining | Smooth and robust to latency; enables precise tasks (match lighting) |
| **2512.05964** Training-time RTC | Simulate the delay during training instead of inpainting at inference | Parity with RTC, no inference overhead |
| **2509.23224** A2C2 / **2605.08168** async comparison | Small per-step residual correction head on top of chunks | Most robust at high delay (+23 pts vs RTC on Kinetix) |
| **2610.28019** PACT-VLA | Generate K candidates, **pick the one most consistent with the latest (pre-commit) observation** | Real π0.5 60→82% over naive async |
| **2606.00537** PACE | **Replan at low-speed valleys** of the predicted chunk instead of a fixed horizon; training-free | Real robots 50.7→70.4% vs full-chunk execution |
| **2606.11408** DEHP | Short execution horizons in fine stages, long in free space | Large gains on high-precision tasks |
| **2506.17811** RoboMonkey | Sample N actions + VLM verifier picks one | +9% in-distribution, +25% OOD |
| **2602.07399** VGAS | Best-of-N chunk selection with a geometric critic for **few-shot near-miss** errors | Higher few-shot success and robustness |
| **2603.18091** TokenPB | VLA self-verifies sampled candidates | +17.1 real-world |
| **2610.21985** VeriVLA | **Only trigger** best-of-N when 2 samples disagree (critical states) | Pen insertion etc.: 50.8→65.5% (+326 ms per trigger) |
| **2610.14588** ADS | After fine-tuning, samples collapse (proposal contraction), so best-of-N gains shrink | Warning: check sample diversity before relying on best-of-N |
| **2605.30660** BOKBO | When all K candidates are bad, abstain | Add an "abstain → recovery" branch |
| **2609.27167** Median temporal ensembling | Coordinate-wise **median** of overlapping chunk predictions instead of the mean (ACT) | Never worse than the mean; robust to blank frames |
| **2608.02547** Why chunking works | Chunking ≈ implicit ensemble of delayed policies | Supports overlap/ensembling of chunks |
| **2410.13816** V-GPS | Value function re-ranks sampled actions from a generalist policy | Consistent gains across policies |

How these apply to Dream-1 (black-box, 1.6 s chunks, about 7 Hz):
- **Don't execute the full 1.6 s chunk during insertion.** Execute about 0.3–0.5 s, then re-query (overlap). Execute about 0.8–1.2 s in free-space transport. A simple version of PACE: cut at the first low-velocity point in the predicted EE trajectory.
- **Asynchronous querying**: request the next chunk while the current one executes, then blend over the overlap (RTC-lite: linear crossfade over the first k steps). Linear blending also avoids jerks at chunk boundaries, which hurt insertion.
- **Best-of-N only at the insertion phase** and only if the API returns diverse samples (seed or temperature). Score candidates by distance between predicted EE endpoint and port pose, or by the predicted future frame if Dream exposes video (Cosmos Policy / τ0-WM rank by predicted futures). Watch out for proposal contraction (ADS).

---

## 5. Recipe for our hackathon, ranked by expected gain ÷ implementation cost

| # | Technique | Why (evidence) | Cost | Gain |
|---|---|---|---|---|
| 1 | **Four-state harness with a separate Dream-1 prompt per subtask + success-based termination + capped retries** (unplug → open → close → plug) | Hi-VLA study 2606.10267; RJ45 decomposition 100% vs ≤10%; primitive prompts 3× efficiency 2605.30695 | Low (Python state machine) | Very high: it's how partial and long-horizon credit is earned |
| 2 | **Demo protocol**: about 20–40 demos per subtask, diverse start poses, one consistent operator style, idle start/end trimmed, PSD/SAL smoothness filter, label each episode with its subtask string | DreamZero idle filter + diversity; SO-101 "failure to start" from idle lead-in 2610.37170; RINSE 2604.23000 | Low | High |
| 3 | **Recovery/correction demos (RaC protocol)** after the first fine-tune: run the policy, take over when failure looks imminent, **rewind to a known state, then finish the subtask**, end the episode. Focus on USB insertion and the panel latch | RaC 2509.07953 (~10× data efficiency); RJ45 recovery episodes; WM-DAgger | Medium (needs a takeover button in teleop) | Very high |
| 4 | **VLM success judge** (Fireworks VLM): before/after wrist + overview frames, describe-then-rubric prompt (D&A), frozen wording, with a cheap proprio signal (gripper state, joint-current spike at seating) as a second vote. Calibrate on 10 labeled success/fail clips | D&A 2611.03622; GVL 2411.04549; ROBORMBENCH 2609.05401; RoboReward negatives 2601.00675 | Low-medium | High: it gates retries and prevents false "done" |
| 5 | **Chunk execution horizon by phase**: execute about 25–30% of each 1.6 s chunk in the insertion phase, 60–75% in transport; async prefetch + linear crossfade at boundaries | PACE 2606.00537; DEHP 2606.11408; RTC 2506.07339 | Low (client-side) | Medium-high on insertion |
| 6 | **Insertion micro-recovery primitive** (scripted): if the judge says "not seated", back off 5 mm, then a small spiral or ±yaw wiggle in the EE frame, then press; maximum 3 tries, then re-invoke the Dream-1 "align and insert" prompt | Classical peg-in-hole; RLT emergent wiggle 2604.23073; TAPs 2512.04960 | Low | Medium-high |
| 7 | **Camera placement**: wrist cam framing the plug tip + port, plus a static view of the rack face; keep camera poses identical between data collection and eval | RJ45 second view 0→100% (with recovery); IDB 36→78% | Low (but do it before collecting data) | High; irreversible if done wrong |
| 8 | **Train and eval on the same physical station/fixture** (shared arms vary); collect about 10 min of free-play on our station if switching | DreamZero 30 min embodiment adaptation | Low (logistics) | Medium |
| 9 | **Cross-episode "lessons" memory** injected into the judge/recovery prompts (e.g. offsets that worked) | 2606.10267 cross-episode memory helps, in-episode doesn't | Low | Low-medium |
| 10 | **Triggered best-of-N** at the insertion step: query Dream-1 2–4× in parallel only when 2 samples disagree; pick by predicted endpoint vs port-pose estimate, or by median aggregation | VeriVLA 2610.21985; RoboMonkey 2506.17811; median TE 2609.27167 | Medium (API cost/latency; needs sample diversity) | Medium |
| 11 | **Rewind-to-checkpoint on detected failure**: store EE poses at each subtask boundary; on failure, servo back to the last verified pose and restart that subtask | Rewind-IL 2604.16683; LoHo-Manip remaining-plan loop 2604.21924 | Low-medium | Medium |
| 12 | **Residual RL / HIL fine-tune on the insertion phase only** (Res-HIL / Max-Q / RLT style, 10–40 min) | 2609.30023, 2608.15088, 2604.23073 | High (needs API features or our own residual head + reward labeling); only if a day-2 robot slot allows it | High but risky |
| 13 | **Advantage/progress-weighted re-fine-tune** (RECAP-lite): label rollouts with judge progress, add a "good" or "bad" tag or drop low-progress chunks, re-submit to the Dream fine-tune | π*0.6 2511.14759; RedFlow 2607.27782; WARP-BC 2606.28320 | Medium (depends on API) | Medium |

**Day plan implied by this ranking:**
- **Day 1 AM:** fix the cameras (#7) and the teleop takeover button; collect clean subtask demos (#2); write the harness state machine and judge (#1, #4) while the first fine-tune runs.
- **Day 1 PM:** evaluate, then RaC recovery collection on the failing subtasks (#3), with insertion first; second fine-tune; tune execution horizon and crossfade (#5); add the spiral/wiggle fallback (#6).
- **Day 2:** rewind-on-failure (#11), triggered best-of-N (#10), and residual RL (#12) only if robot time is plentiful.

**Things to avoid:**
- LoRA-only fine-tunes (2610.37170: 4/110 success).
- Relative or delta action targets when absolute targets are available (2610.40828).
- Executing full long chunks during contact.
- Changing judge prompt wording mid-event.
- Trusting validation loss for checkpoint choice. Only real-robot trials predict success, and 10-episode evals are noisy (±2/10 run to run, 2610.37170).
