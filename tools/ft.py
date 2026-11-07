"""Hillclimb fine-tune launcher: the kit's tools/finetune.py plus the knobs it hides (steps, base_share, resume_from).

Runs ON THE STATION in ~/dream-env (needs the `dream` client + the team API key in ~/.dreamlabs/credentials):

    # free, local: validate only
    ~/dream-env/bin/python tools/ft.py check ~/hackathon_data/hc-ft1
    # PAID, uncancellable. FT#1: new model. KEEP lr_schedule=constant so FT#2 can resume_from it.
    ~/dream-env/bin/python tools/ft.py launch ~/hackathon_data/hc-ft1 --name hc-ft1 --suffix hc-ft1 --steps 1000
    # FT#2: continue FT#1 on a new (bigger) dataset; charged for the new steps only
    ~/dream-env/bin/python tools/ft.py launch ~/hackathon_data/hc-ft2 --name hc-ft2 --suffix hc-ft2 --steps 500 \
        --resume-from dream-1-ft:<uid8>-hc-ft1
    # poll / list
    ~/dream-env/bin/python tools/ft.py status ftj_...
    ~/dream-env/bin/python tools/ft.py models
    ~/dream-env/bin/python tools/ft.py prompts dream-1-ft:<uid8>-hc-ft1   # verify harness prompts match training

Facts from dreamlabs-client 1.1.3 `DreamClient.finetune` docstring (dream/facade.py:312-398):
- steps: multiple of 100 in [500, 2000]; default 1000. Price + time scale with steps.
- resume_from: continues optimizer + LR schedule; charged for new steps only. REQUIRES the parent model was trained at
  the "constant" schedule, and learning_rate / batch_size / use_eef must match the parent. base_share is inherited.
- base_share: 0.5 (default, mix in Dream data) or 0.0 (ours only).
- lr_schedule: "constant" (default) or "cosine". Cosine model => cannot be resumed.
- Metal arm => use_eef=True and dataset must be the add_eef.py output (-eef folders).
"""

import argparse
import json
import pathlib
import sys
import time

import dream  # only available in ~/dream-env on the station

KIT = pathlib.Path("~/physical-ai-hacks").expanduser()  # TODO(event): confirm kit path on the station
EMBODIMENT_FILE = KIT / "src" / "dream_hackathon" / "embodiments" / "makermods.json"
EMBODIMENT_NAME = "hackathon-metal"


def _config() -> dict:
    return json.loads(EMBODIMENT_FILE.read_text())


def _embodiment(client: "dream.DreamClient", config: dict) -> str:
    try:
        return client.add_embodiment(EMBODIMENT_NAME, config).name
    except dream.DreamServerError as error:
        if error.code != "conflict":
            raise
    existing = next(e for e in client.get_embodiments() if e.name.endswith(f"/{EMBODIMENT_NAME}"))
    if existing.embodiment_config != config:
        sys.exit(f"{existing.name} registered with a different config")
    return existing.name


def cmd_check(args: argparse.Namespace) -> None:
    report = dream.datasets.validate_dataset(args.dataset.expanduser(), _config())
    print(f"ok={report.ok} episodes={report.episodes} frames={report.frames} fps={report.fps}")
    for note in report.notes:
        print("note:", note)
    if not report.ok:
        sys.exit(f"fix before upload: {report.error}")


def cmd_launch(args: argparse.Namespace) -> None:
    if args.resume_from and args.lr_schedule == "cosine":
        sys.exit("resume_from only works at the constant schedule")
    est = {500: "$100", 1000: "$200", 2000: "$400"}.get(args.steps, "see platform form")
    print(f"About to queue a PAID, UNCANCELLABLE job: steps={args.steps} (~{est}) resume_from={args.resume_from} "
          f"base_share={args.base_share} lr_schedule={args.lr_schedule}")
    if not args.yes and input("type QUEUE to continue: ").strip() != "QUEUE":
        sys.exit("aborted")
    client = dream.DreamClient()
    dataset_name = args.dataset_name
    if dataset_name is None:
        config = _config()
        path = str(args.dataset.expanduser())
        report = client.validate_dataset(path, config)
        if not report.ok:
            sys.exit(f"fix before upload: {report.error}")
        ds = client.upload_dataset(path, args.name, _embodiment(client, config))
        print(f"uploaded {ds.id} ({ds.name})", flush=True)
        while (ds := client.get_dataset_status(ds.id)).status not in ("processed", "failed"):
            print("dataset:", ds.status, flush=True)
            time.sleep(30)
        if ds.status == "failed":
            sys.exit(f"processing failed: {ds.error}")
        dataset_name = ds.name
    kwargs = dict(steps=args.steps, use_eef=True, resume_from=args.resume_from)
    if not args.resume_from:  # a resumed job inherits these from its parent
        kwargs.update(lr_schedule=args.lr_schedule, base_share=args.base_share, learning_rate=args.learning_rate)
    job = client.finetune(dataset_name, args.suffix, **kwargs)
    print(f"QUEUED {job['id']} position={job.get('queue_position')} hyper={job.get('hyperparameters')}")
    print(f"poll: ~/dream-env/bin/python tools/ft.py status {job['id']}")


def cmd_status(args: argparse.Namespace) -> None:
    client = dream.DreamClient()
    while True:
        job = client.get_finetuning_status(args.job_id)
        prog = job.get("progress") or {}
        print(f"{time.strftime('%H:%M')} {job['status']} q={job.get('queue_position')} {prog} -> {job['fine_tuned_model']}",
              flush=True)
        if job["fine_tuned_model"] or job["status"] in ("failed", "cancelled") or not args.watch:
            break
        time.sleep(60)


def cmd_prompts(args: argparse.Namespace) -> None:
    """The task texts a fine-tune was trained on (dream/fine_tuning.py:training_data). Harness prompts must match
    these verbatim, or the client warns that the model was never trained to follow them (dream/trained_inputs.py)."""
    info = dream.fine_tuning.training_data(args.model)
    print("cameras:", info.get("cameras"))
    for task in info.get("prompts", []):
        print("prompt:", repr(task))


def cmd_models(_: argparse.Namespace) -> None:
    for model in dream.DreamClient().get_models():
        print(model)


def main() -> None:
    p = argparse.ArgumentParser(description="Hillclimb Dream-1 fine-tune launcher")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("dataset", type=pathlib.Path)
    c.set_defaults(fn=cmd_check)
    lch = sub.add_parser("launch")
    lch.add_argument("dataset", type=pathlib.Path, nargs="?")
    lch.add_argument("--dataset-name", help="skip upload: an already-processed <user>/<name>")
    lch.add_argument("--name", help="new dataset name (upload)")
    lch.add_argument("--suffix", required=True)
    lch.add_argument("--steps", type=int, default=1000)
    lch.add_argument("--resume-from")
    lch.add_argument("--base-share", type=float, choices=(0.0, 0.5), default=0.5)
    lch.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    lch.add_argument("--learning-rate", type=float)
    lch.add_argument("--yes", action="store_true", help="skip the QUEUE prompt (only after a human OK)")
    lch.set_defaults(fn=cmd_launch)
    s = sub.add_parser("status")
    s.add_argument("job_id")
    s.add_argument("--watch", action="store_true")
    s.set_defaults(fn=cmd_status)
    sub.add_parser("models").set_defaults(fn=cmd_models)
    pr = sub.add_parser("prompts", help="task texts a fine-tune was trained on")
    pr.add_argument("model")
    pr.set_defaults(fn=cmd_prompts)
    args = p.parse_args()
    if args.cmd == "launch" and args.dataset is None and args.dataset_name is None:
        p.error("give a dataset folder or --dataset-name")
    if args.cmd == "launch" and args.dataset is not None and args.dataset_name is None and not args.name:
        p.error("--name is required to upload")
    args.fn(args)


if __name__ == "__main__":
    main()
