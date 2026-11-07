# 07 — Success-judge eval on real robot footage (REBOOT26 USB-A)

Run 2026-11-07. This measures `hillclimb/judge.py` (`VLMJudge`, called through its real code path by
`tools/judge_eval.py`) on 64 labelled frame pairs from a real bimanual USB rig. **It is not our rig**: see the caveats.

## TL;DR

- **The judge as configured did not work at all.** `qwen3p8-max` is a reasoning model. With `max_tokens: 400` the
  whole budget went to `reasoning_content`, the reply had no JSON, and **4/4 probe calls returned `unknown`**. The
  harness treats unknown as "not success", so it would never stop early and would retry every subtask to
  `max_attempts`.
- **Fixed** (now in `judge.py` / `config.yaml`): `reasoning_effort: none`, `response_format: json_object`
  (`json_mode: true`), `max_tokens: 800`, plus an optional per-subtask `rubric.scene` line that says what is where on
  the rig. The same model then scored **92% accuracy, 0 false positives out of 36 negatives (0 of 8 hard negatives),
  0 unknowns, latency p50 1.1 s / p90 1.4 s.**
- The other model tried (`glm-5p3-flash`) and the before/after 2-frame prompt were both worse. Keep `qwen3p8-max`
  and the single (after) frame.
- **At the event:** write `scene:` notes for our rack in `config.yaml`, collect about 20 pos and 20 neg frames per
  subtask from our own runs, and rerun `tools/judge_eval.py` (about 2 minutes, about 40 calls).

## Dataset

- **REBOOT26** on Hugging Face (arXiv 2609.22591): bimanual Trossen WidowX-AI, LeRobot v3, 30 fps, 640x480.
  - `usbA-install` ("Pick the USB connector and install on taskboard") and `USB-A_recovery_install`.
  - `usbA_remove` ("Remove the USB connector from the USB hub and place in the container") and
    `USB-A_recovery_remove`.
- Episodes 3, 11, 19, 27, 35, 43 and 51 of each of the 4 datasets. Cameras: `cam_high` (overhead, sent as `scene`) and
  `cam_low` (side view at board height, sent as `low`). The wrist cameras were left out because they show the room
  once the arm is home.
- **No videos were downloaded.** `tools/reboot26_frames.py` has ffmpeg seek inside the remote mp4 over HTTPS range
  requests. Only metadata parquets and JPEGs are on disk, 11 MB in total, in `/Users/sissi/hc-vendor/judge-data/`
  (`eval/` is the labelled set, `results/` holds the per-sample jsonl for each run, `sheets/` holds the spot-check
  contact sheets). Nothing from the dataset was executed.
- **Labels** (64 samples, 2 images each):

  | subtask | pos | neg |
  |---|---|---|
  | `plug_usb` | 14 install-episode last frames (plug seated on the hub) | 14 install-episode first frames (plug in the container) + **4 hard**: plug hovering just above the hub, metal tip showing (mid-episode, t=12–15 s) |
  | `unplug_usb` | 14 remove-episode last frames (hub bare, plug in the container) | 14 remove-episode first frames (plug in the hub) + **4 hard**: gripper pulling the cable, plug still in (t=3.5 s) |

- **Spot-check.** I viewed contact sheets of every sample's `low` frame, all four label groups, and the `scene` frames
  for the plug_usb negatives. All 56 weak labels matched what the image shows. I picked the 8 hard negatives by eye and
  checked them at full resolution.

## Results

Each run used all 64 samples with threshold 0.7 and 2 parallel workers. Latency is the wall time of one judge call
(2 images, about 775 prompt tokens) from this laptop.

Every variant ran on `qwen3p8-max` except (b), which ran on `glm-5p3-flash`.

| variant | model / settings | acc | FPR (FP/neg) | FNR | unknown | hard negs called success | p50 | p90 |
|---|---|---|---|---|---|---|---|---|
| a0 as configured (probe, n=4) | max_tokens 400, thinking on | 50% | 0/2 | 2/2 | **4/4** | — | 2.65 s | 2.68 s |
| (a) current model, thinking off | `reasoning_effort: none`, 800 tok | 53% | **5/36 (14%)** | 25/28 | 25 | 3/8 | 1.80 s | 3.83 s |
| (b) other model | `glm-5p3-flash`, default thinking, 800 tok | 47% | **8/36 (22%)** | 26/28 | 24 | 3/8 | 4.56 s | 5.28 s |
| **(a2) current model, fixed** | + `json_mode` + `rubric.scene` notes | **92%** | **0/36 (0%)** | 5/28 (18%) | 0 | **0/8** | **1.12 s** | **1.36 s** |
| (c) a2 + before/after frames | + `before=` start-of-attempt frames | 83% | 0/36 (0%) | 11/28 (39%) | 0 | 0/8 | 1.19 s | 1.42 s |

### Confusion matrices (rows = truth, cols = judge; unknown counts as "fail")

| | (a) qwen, thinking off | (b) glm-5p3-flash | **(a2) qwen fixed** | (c) a2 + before |
|---|---|---|---|---|
| truth success | 3 TP / 25 FN | 2 TP / 26 FN | **23 TP / 5 FN** | 17 TP / 11 FN |
| truth fail | 5 FP / 31 TN | 8 FP / 28 TN | **0 FP / 36 TN** | 0 FP / 36 TN |

Per subtask for (a2): `plug_usb` 13 TP / 1 FN / 0 FP / 18 TN (97%); `unplug_usb` 10 TP / 4 FN / 0 FP / 18 TN (88%).

Threshold sweep for (a2), from the saved scores with no new calls: every negative scored ≤ 0.10. TP stays at 23/28 for
any threshold from 0.5 to 0.7, drops to 22 at 0.8 and to 15 at 0.9. **Keep 0.7.**

### What went wrong in each variant

- **(a) and (b): truncation.** With thinking off, qwen "thinks aloud" in `content` ("Wait, but is the plug
  seated?..."). glm thinks by default. Both hit `finish_reason: length` before the JSON. `response_format:
  json_object` fixes this: the `description` field becomes the describe step, and the JSON always closes.
- **(a) and (b): wrong port.** Without scene notes, both models took the beige multi-pin (HAN) block for "the rack
  port" and judged plug state from it. That produced the confident false positives (score 0.8–0.95), including 3 of
  the 4 "gripper pulling, plug still in" hard negatives. **Generic rubric text is unsafe. The judge needs to be told
  which object is the port and what seated looks like.**
- **(a2): the remaining 5 errors are all false negatives.** In 4 unplug cases the model said a plug was still standing
  on the hub when the hub was bare. On this rig the bare USB hub is itself a tall black block, so it looks like a plug
  body. A false negative costs one extra attempt, not a skipped step.
- **(c): before/after hurt.** With both frames, the model kept describing the BEFORE state as the current one
  ("AFTER low view... still has a plug"), mostly on unplug, which went from 10 to 5 TP. FPs stayed at 0, latency rose
  by about 0.1 s, and the request carries twice as many images. **Not adopted in the harness.** `VLMJudge(...,
  before=...)` stays available, and `tools/judge_eval.py --before` re-tests it.

### Spend

278 Fireworks calls in total, all ≤ 3 images and ≤ 800 output tokens: 4 + 64 × 4 full runs + 18 probes and
debugging calls.

## Changes made

- `hillclimb/judge.py`:
  - `VLMJudge(reasoning_effort="none", max_tokens=800, json_mode=True)`. The request now carries
    `reasoning_effort` and `response_format`.
  - Optional `rubric.scene` line in the prompt.
  - Optional `before=` frames, labelled `[<cam> BEFORE]`.
- `hillclimb/harness.py`: `build_judge` reads `judge.reasoning_effort`, `max_tokens` and `json_mode`.
- `hillclimb/config.yaml`: those three keys, plus a TODO(event) to add `scene:` notes for our rack. Rubric success and
  failure text is unchanged.
- `tools/judge_eval.py`: folder of labelled images in, metrics out. The layout is
  `<root>/<subtask>/<pos|neg>/<sample>/<cam>.jpg` with optional `before/` frames. Options: `--model`,
  `--reasoning-effort` (`omit` drops the field), `--max-tokens`, `--no-json-mode`, `--before`, `--cams`, `--limit`,
  `--dry-run`. An optional `<root>/rubric.yaml` overrides rubric keys per subtask; the REBOOT26 scene notes live in
  `/Users/sissi/hc-vendor/judge-data/eval/rubric.yaml`.
- `tools/reboot26_frames.py`: rebuilds this eval set.
- `tests/test_harness.py`: 2 new tests. `pytest -q` passes (36 passed).

## Rerun at the event

```bash
set -a; . ./.env; set +a
# put frames from logs/<run>/<NN-subtask>-a<k>/inputs/<cam>/ into eval-rig/<subtask>/<pos|neg>/<sample>/<cam>.jpg
.venv/bin/python tools/judge_eval.py eval-rig --out eval-rig/results.jsonl   # 1 call per sample
```

Gate: **0 false positives on our negatives**, especially "plug at the port but not seated" and "panel ajar". If
`unplug_usb` misses a lot of successes, sharpen its `scene:` note before touching the threshold.

## Caveats

- **Not our rig.** The REBOOT26 rig has a WidowX-AI arm, a white taskboard and a USB-A hub, filmed from an overhead
  camera and a low side camera. Ours has a MakerMods Metal arm, a server rack and `scene`/wrist cameras at 1280x720.
  The plug and port will look different, and our wrist cameras may help or hurt. The numbers here only show that the
  judge pipeline works and that scene notes matter; they do not predict our accuracy.
- Only USB was tested. `open_backing` and `close_backing` (panel ajar vs flush) are untested and probably the hardest
  call for a single frame.
- The scene notes were written while I could see these images, so (a2) is somewhat optimistic. The notes for our rack
  should be written once, then measured on frames held out from that writing.
- Weak labels come from episode start and end frames; hard negatives come from 8 hand-picked frames. n=64 is small:
  0/36 FP has a 95% upper bound of about 8%.
- The latencies are from a laptop on home Wi-Fi with 2 concurrent requests, at Fireworks load on 2026-11-07. At about
  1.1 s p50, `check_every_s: 4` has plenty of headroom.
- `kimi-k3` (the config's fallback) was only smoke-tested: with thinking on it truncated, and with `none` it answered
  in 2.6 s. It needs its own eval before it is used.
