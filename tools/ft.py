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

Dataset layout (rehearsed offline, docs/research/09-pipeline-rehearsal.md): a parent folder holding one -eef session
folder per subtask, built with `add_eef.py <src> --arm metal --out <parent>/<src>-eef --replace`
(tools/rehearse_pipeline.sh does it). `check` inspects the parent before Dream's own checks: raw 14-wide sessions,
-eef sessions older than their source folder, unknown task texts, mixed frame sizes, files beside the sessions.
"""

import argparse
import json
import os
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
# The kit checkout on the station. HC_KIT overrides; else the first of these that exists.
KIT_CANDIDATES = ("~/physical-ai-hacks", "~/physical-ai-hack-2026", "~/hc-vendor/physical-ai-hack-2026")
EMBODIMENT_REL = pathlib.Path("src/dream_hackathon/embodiments/makermods.json")
EMBODIMENT_NAME = "hackathon-metal"
FULL_TASK = "unplug the usb, open and close the rack back panel, and plug in the new usb"  # DEMO-PROTOCOL hc-full
RAW_WIDTH, EEF_STATE, EEF_ACTION = 14, 34, 32  # dhk-collect vectors; makermods.json state / action spans
STRAY = (".txt", ".md")  # dream validation.is_stray(): notes at the parent root are skipped, anything else refused


def kit_dir() -> pathlib.Path:
    if os.environ.get("HC_KIT"):
        return pathlib.Path(os.environ["HC_KIT"]).expanduser()
    for cand in KIT_CANDIDATES:
        if (pathlib.Path(cand).expanduser() / EMBODIMENT_REL).is_file():
            return pathlib.Path(cand).expanduser()
    sys.exit(f"kit not found in {KIT_CANDIDATES}; set HC_KIT=<kit checkout>")


def _config() -> dict:
    return json.loads((kit_dir() / EMBODIMENT_REL).read_text())


def known_prompts() -> set[str]:
    try:
        import yaml
        cfg = yaml.safe_load((REPO / "hillclimb" / "config.yaml").read_text())
        return {s["prompt"] for s in cfg["subtasks"]} | {FULL_TASK}
    except Exception:  # noqa: BLE001 - no yaml / no config: skip the task-text check
        return set()


def _info(session: pathlib.Path) -> dict:
    return json.loads((session / "meta" / "info.json").read_text())


def _tasks(session: pathlib.Path) -> list[str]:
    path = session / "meta" / "tasks.jsonl"
    return [json.loads(x)["task"] for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def source_of(session: pathlib.Path, search: list[pathlib.Path]) -> pathlib.Path | None:
    """The collected folder an -eef session was built from: `<dir>/<name minus -eef>` in one of `search`."""
    name = session.resolve().name
    if not name.endswith("-eef"):
        return None
    for d in search:
        src = d / name.removesuffix("-eef")
        if (src / "meta" / "info.json").is_file():
            return src
    return None


def inspect_parent(parent: pathlib.Path, search: list[pathlib.Path] | None = None,
                   prompts: set[str] | None = None) -> tuple[list[str], list[str], list[str]]:
    """(info lines, warnings, errors) about a dataset folder, before Dream's checks. Needs no `dream`."""
    parent = parent.expanduser()
    if (parent / "meta" / "info.json").is_file():
        sessions = [parent]
        search = search or [parent.resolve().parent, parent.resolve().parent / "self"]
    else:
        sessions = sorted(p for p in parent.iterdir() if p.is_dir() and not p.name.startswith("."))
        search = search or [parent.resolve().parent, parent.resolve().parent / "self"]
        for f in sorted(p for p in parent.iterdir() if p.is_file() and not p.name.startswith(".")):
            if not (f.name.lower().startswith("readme") or f.name.lower().endswith(STRAY)):
                errors_hint = (f"{f.name} sits beside the session folders: Dream refuses it. Not a parent? (harvest's --out "
                               "root is not one: add_eef its hc-*-self folders into the FT parent)")
                return [], [], [errors_hint]
    info, warn, err = [], [], []
    sizes = {}
    for s in sessions:
        if not (s / "meta" / "info.json").is_file():
            err.append(f"{s.name}: not a dataset (no meta/info.json)")
            continue
        i = _info(s)
        w = (i["features"]["observation.state"]["shape"][0], i["features"]["action"]["shape"][0])
        cams = {k: tuple(v["shape"][:2]) for k, v in i["features"].items() if v.get("dtype") == "video"}
        tasks = _tasks(s)
        link = f" -> {s.resolve()}" if s.is_symlink() else ""
        info.append(f"{s.name}{link}: {i['total_episodes']} episodes, {i['total_frames']} frames, {i['fps']} fps, "
                    f"state/action {w[0]}/{w[1]}, tasks {tasks}")
        sizes[s.name] = sorted(set(cams.values()))
        if w == (RAW_WIDTH, RAW_WIDTH):
            err.append(f"{s.name}: raw dhk-collect/harvest folder (14-wide); put its add_eef output here instead")
        elif w != (EEF_STATE, EEF_ACTION):
            err.append(f"{s.name}: state/action {w[0]}/{w[1]}, expected {EEF_STATE}/{EEF_ACTION} (add_eef output)")
        src = source_of(s, search)
        if src is not None and _info(src)["total_episodes"] != i["total_episodes"]:
            err.append(f"{s.name}: STALE, {src} now has {_info(src)['total_episodes']} episodes, this copy "
                       f"{i['total_episodes']}: rerun add_eef.py {src} --arm metal --out {s} --replace")
        for t in tasks:
            if prompts and t not in prompts:
                warn.append(f"{s.name}: task {t!r} is not a frozen prompt (hillclimb/config.yaml / DEMO-PROTOCOL)")
    if len({tuple(v) for v in sizes.values()}) > 1:
        warn.append(f"frame sizes differ between sessions {sizes}: Dream accepts it, but the model sees two "
                    "aspect ratios (06-harvest.md: set the station cameras to 16:9 for rollouts)")
    return info, warn, err


def _embodiment(client, config: dict) -> str:
    import dream
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
    import dream.datasets

    lines, warns, errs = inspect_parent(args.dataset, prompts=known_prompts())
    for line in lines:
        print("session:", line)
    for line in warns:
        print("WARNING:", line)
    for line in errs:
        print("ERROR:", line)
    try:
        report = dream.datasets.validate_dataset(args.dataset.expanduser(), _config())
    except dream.DreamConfigError as error:
        sys.exit(f"fix before upload: {error}")
    print(f"ok={report.ok} episodes={report.episodes} frames={report.frames} fps={report.fps}")
    for note in report.notes:
        print("note:", note)
    if not report.ok:
        sys.exit(f"fix before upload: {report.error}")
    if errs := _stale_ok(errs, args.allow_stale):
        sys.exit(f"Dream's checks pass, but fix the {len(errs)} ERROR line(s) above before upload")


def cmd_eef(args: argparse.Namespace) -> None:
    """The kit's tools/add_eef.py, in-process: its own convert() (Metal FK, videos hard-linked), then Dream's checks.
    Same output as `add_eef.py SRC --arm metal --out OUT --replace`; exists so main() can hard-exit (see there)."""
    import importlib.util

    import dream.datasets

    spec = importlib.util.spec_from_file_location("add_eef", kit_dir() / "tools" / "add_eef.py")
    add_eef = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(add_eef)
    source = args.source.expanduser().resolve()
    target = (args.out or source.with_name(f"{source.name}-eef")).expanduser().resolve()
    add_eef.convert(source, target, "metal", replace=args.replace)
    report = dream.datasets.validate_dataset(target, _config())
    if not report.ok:
        sys.exit(f"{target} was written, but Dream's checks refuse it: {report.error}")
    print(f"wrote {target}: {report.episodes} episodes, {report.frames} frames; Dream's dataset checks pass")


def _stale_ok(errs: list[str], allow: bool) -> list[str]:
    return [e for e in errs if not (allow and "STALE" in e)]


def cmd_launch(args: argparse.Namespace) -> None:
    import dream

    if args.dataset is not None:
        _, _, errs = inspect_parent(args.dataset, prompts=known_prompts())
        if errs := _stale_ok(errs, args.allow_stale):
            sys.exit("refusing to upload: " + "; ".join(errs))
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
    import dream

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
    import dream.fine_tuning

    info = dream.fine_tuning.training_data(args.model)
    print("cameras:", info.get("cameras"))
    for task in info.get("prompts", []):
        print("prompt:", repr(task))


def cmd_models(_: argparse.Namespace) -> None:
    import dream

    for model in dream.DreamClient().get_models():
        print(model)


def main() -> None:
    p = argparse.ArgumentParser(description="Hillclimb Dream-1 fine-tune launcher")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("dataset", type=pathlib.Path)
    c.add_argument("--allow-stale", action="store_true", help="a frozen snapshot on purpose: don't fail on STALE")
    c.set_defaults(fn=cmd_check)
    e = sub.add_parser("eef", help="kit add_eef.py --arm metal, in-process (free, local)")
    e.add_argument("source", type=pathlib.Path)
    e.add_argument("--out", type=pathlib.Path)
    e.add_argument("--replace", action="store_true")
    e.set_defaults(fn=cmd_eef)
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
    lch.add_argument("--allow-stale", action="store_true")
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
    if args.cmd not in ("check", "eef"):
        args.fn(args)
        return
    # Local commands hard-exit: dream's validate_dataset leaves pyarrow's thread pool behind, and the interpreter's
    # exit then deadlocks in ThreadPool::Shutdown in ~1 of 20 runs (rehearsed on macOS, pyarrow 26; 09 gotchas).
    try:
        args.fn(args)
        code = 0
    except SystemExit as stop:
        code = stop.code if isinstance(stop.code, int) else 1
        if not isinstance(stop.code, int):
            print(stop.code, file=sys.stderr)
    except Exception:  # noqa: BLE001 - print it, then still hard-exit
        import traceback

        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
