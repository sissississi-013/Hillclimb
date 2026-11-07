# Zero-shot prompt search (~10:25, before demos are recorded)

**Why:** fine-tuning starts from the base model's language grounding, and the fine-tune follows the exact task text its demos were recorded with (`dream/trained_inputs.py`). So: (1) find the phrasing the **base** model already acts on best, (2) record every demo with that phrasing, (3) freeze it. A cheap gain before spending any data.

**Style of Dream-1's own tasks** (from the launch post's examples, https://dreamlabs.ai/research/dream-1/): one short imperative sentence. It names the object by a visible attribute and names the place. Examples: "Pull the cable out of the box handle", "Close the top drawer", "Put the straw into the bottle neck", "Take the pen out of the drawer and place it in the pen holder", "Pick up the knife by its handle, never the blade, and place it in the tray".

## How to run (2–3 attempts per variant, 1 attempt per call)
```bash
HILLCLIMB_ALLOW_ARMS=1 python -m hillclimb.harness --backend station --only plug_usb --attempts 1 \
    --prompt "Plug the USB cable into the empty port on the rack"
python tools/report.py logs/harness --since 08:30      # the prompt is logged per attempt
```
Budget: ~20 min of arm time total. Spend it on the two weakest subtasks: **plug_usb** and **open_backing**. Pick the winner by progress seen on video, not only binary success. Zero-shot success will be low everywhere. Replace the `<colour>`/`<side>` placeholders below with what's on the rig.

## Variants
**unplug_usb**
1. `unplug the usb cable from the rack` (current frozen string)
2. `Pull the USB cable out of the port on the rack`
3. `Grab the USB plug and pull it straight out of the rack`

**open_backing**
1. `open the rack back panel` (current)
2. `Open the back panel of the server rack`
3. `Swing the rack's back door open` / `Slide the rack's back cover off`, depending on the panel type

**close_backing**
1. `close the rack back panel` (current)
2. `Close the back panel of the server rack`
3. `Push the rack's back door shut`

**plug_usb**
1. `plug the new usb cable into the rack` (current)
2. `Plug the USB cable into the empty port on the rack`
3. `Pick up the <colour> USB cable and insert it into the port on the rack`

## After choosing
Update **both** `docs/DEMO-PROTOCOL.md` (task-string table) and `hillclimb/config.yaml` (`prompt:`) to the winners. `pytest` fails if they differ. Then announce `PLAN-CHANGE: prompts frozen` on issue #1 **before** demo sprint 1 starts.
