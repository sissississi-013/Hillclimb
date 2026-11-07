"""System 2 harness: run the rack task as a chain of subtasks. Each subtask is its own Dream episode with its own
prompt.

Each attempt does these steps:
1. start a run with the subtask's prompt. A station Start builds a fresh client, so this is a fresh episode
   (station.py:922-952).
2. poll it. Every `check_every_s`, judge the newest model-input frame and stop early on success. Stop at
   `max_time_s` otherwise.
3. stop the run (the arms hold), copy the frames and trace into logs/<run>/<NN-subtask>-a<k>/, and judge the final
   frames plus proprio.
4. log one JSON line to attempts.jsonl with the verdict. That data feeds DAgger mining.
5. on failure, run the recovery action (`home` | `none`) and retry, up to `max_attempts` attempts.

Replay primitives (hillclimb/primitives.py): a subtask's `pre: <file>` is played before every attempt's Dream episode
(after any recovery), and `mode: replay` makes the attempt just that primitive, then judged. Only a backend with
`play()` can do this (`client`, `fake`); the station API has no joint move, since `dreamcli robot up` owns the arms.

python -m hillclimb.harness --config hillclimb/config.yaml                     # fake backend (default)
python -m hillclimb.harness --config hillclimb/config.yaml --backend dry-run   # dreamcli --dry-run, arms never move
HILLCLIMB_ALLOW_ARMS=1 python -m hillclimb.harness ... --backend station       # REAL ARMS
HILLCLIMB_ALLOW_ARMS=1 python -m hillclimb.harness ... --backend client        # our loop, dry: arms opened, never written
HILLCLIMB_ALLOW_ARMS=1 python -m hillclimb.harness ... --backend client --client-live   # our loop, REAL ARMS
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import pathlib
import time
from typing import Any, Callable

import yaml

from hillclimb import backends
from hillclimb.judge import CombinedJudge, Judge, ProprioJudge, Verdict, VLMJudge


@dataclasses.dataclass
class Attempt:
    subtask: str
    attempt: int
    success: bool
    score: float
    reason: str
    source: str
    elapsed_s: float
    end: str  # "early_success" | "timeout" | "run_ended" | "run_error" | "replay"
    dir: str
    # For tools/harvest.py: pairs this attempt with its rollout episode (rollout.json task/started_at_utc/run).
    prompt: str = ""
    checkpoint: str | None = None
    started_at: str = ""  # UTC wall clock just before the Start call, ISO 8601 with ms
    ended_at: str = ""  # UTC wall clock just after the Stop call
    station_run: int | None = None  # the station's run ordinal (/status `run`, station.py:778), when it answers one


class Harness:
    def __init__(self, cfg: dict[str, Any], backend: backends.Backend, judge: Judge, log_root: pathlib.Path,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 wall: Callable[[], float] = time.time) -> None:
        self.cfg, self.backend, self.judge = cfg, backend, judge
        self.clock, self.sleep, self.wall = clock, sleep, wall
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = log_root / stamp
        self.dir.mkdir(parents=True, exist_ok=True)
        self.attempts: list[Attempt] = []
        for sub in cfg["subtasks"]:  # refuse a bad primitive setup before anything moves
            mode = sub.get("mode", "dream")
            if mode not in ("dream", "replay") or (mode == "replay" and not sub.get("pre")):
                raise ValueError(f"{sub['name']}: mode is dream|replay, and replay needs `pre: <primitive file>`")
            if sub.get("pre") and not hasattr(backend, "play"):
                raise ValueError(f"{sub['name']}: `pre`/replay needs --backend client (station API: no joint move)")

    def _play(self, sub: dict[str, Any], dest: pathlib.Path | None = None) -> dict[str, Any]:
        opts = {**(self.cfg.get("defaults", {}).get("primitive") or {}), **(sub.get("primitive") or {})}
        return self.backend.play(sub["pre"], dest, opts)  # type: ignore[attr-defined]

    def _log(self, a: Attempt) -> None:
        self.attempts.append(a)
        with open(self.dir / "attempts.jsonl", "a") as f:
            f.write(json.dumps(dataclasses.asdict(a)) + "\n")
        print(f"[{a.subtask} #{a.attempt}] {'OK ' if a.success else 'FAIL'} {a.end} {a.elapsed_s:.1f}s "
              f"score={a.score:.2f} ({a.source}) {a.reason[:100]}")

    def run_attempt(self, index: int, sub: dict[str, Any], k: int) -> Attempt:
        d = self.cfg.get("defaults", {})
        max_time = sub.get("max_time_s", d.get("max_time_s", 30))
        every = sub.get("check_every_s", d.get("check_every_s", 0))
        attempt_dir = self.dir / f"{index:02d}-{sub['name']}-a{k}"
        checkpoint = sub.get("checkpoint", self.cfg.get("checkpoint"))
        if sub.get("mode") == "replay":
            return self._replay_attempt(sub, k, attempt_dir)
        if sub.get("pre"):
            self._play(sub)
        started_at = _iso(self.wall())
        self.backend.start(sub["prompt"], checkpoint)
        t0 = next_check = self.clock()
        end, early = "timeout", None
        while (elapsed := self.clock() - t0) < max_time:
            state = self.backend.state()
            if state in ("idle", "error") and elapsed > 1.0:
                end = "run_ended" if state == "idle" else "run_error"
                break
            if every and self.clock() >= next_check + every:
                next_check = self.clock()
                frames = self.backend.latest_frames()
                if frames:
                    v = self.judge(sub, frames, {})
                    if v.success:
                        end, early = "early_success", v
                        break
            self.sleep(self.cfg.get("poll_s", 0.5))
        self.backend.stop()
        ended_at = _iso(self.wall())
        proprio = self.backend.collect(attempt_dir)
        frames = backends.latest_frames_in(attempt_dir / "inputs")
        v = self.judge(sub, frames, proprio) if end != "run_error" else Verdict(False, reason="run error")
        if early is not None and v.success is None:
            v = early
        a = Attempt(sub["name"], k, bool(v.success), v.score, v.reason, v.source,
                    round(self.clock() - t0, 2), end, str(attempt_dir), sub["prompt"], checkpoint, started_at,
                    ended_at, _station_run(attempt_dir))
        return self._finish(a, attempt_dir)

    def _finish(self, a: Attempt, attempt_dir: pathlib.Path) -> Attempt:
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "verdict.json").write_text(json.dumps(dataclasses.asdict(a), indent=1))
        self._log(a)
        return a

    def _replay_attempt(self, sub: dict[str, Any], k: int, attempt_dir: pathlib.Path) -> Attempt:
        """The attempt is the primitive alone, then the same judge. No prompt logged: harvest must never pair it."""
        started_at, t0 = _iso(self.wall()), self.clock()
        attempt_dir.mkdir(parents=True, exist_ok=True)
        out = self._play(sub, attempt_dir)
        frames = backends.latest_frames_in(attempt_dir / "inputs")
        v = self.judge(sub, frames, out.get("proprio") or {})
        return self._finish(Attempt(sub["name"], k, bool(v.success), v.score, v.reason, v.source,
                                    round(self.clock() - t0, 2), "replay", str(attempt_dir), "", None,
                                    started_at, _iso(self.wall())), attempt_dir)

    def recover(self, sub: dict[str, Any]) -> None:
        # TODO(event): `home` opens the grippers (station.py:535), so a held cable drops. Use `none` (retry from
        # where the arms are) for plug_usb once a cable is in hand.
        action = sub.get("recovery", self.cfg.get("defaults", {}).get("recovery", "home"))
        if action == "home":
            self.backend.home()

    def run(self) -> bool:
        d = self.cfg.get("defaults", {})
        budget = self.cfg.get("total_time_s", float("inf"))
        t0 = self.clock()
        for i, sub in enumerate(self.cfg["subtasks"]):
            for k in range(1, sub.get("max_attempts", d.get("max_attempts", 3)) + 1):
                if self.clock() - t0 > budget:
                    print("total time budget spent; stopping")
                    return False
                if self.run_attempt(i, sub, k).success:
                    break
                self.recover(sub)
            else:
                print(f"{sub['name']}: out of attempts")
                if sub.get("on_fail", d.get("on_fail", "abort")) == "abort":
                    return False
        return all(any(a.success for a in self.attempts if a.subtask == s["name"]) for s in self.cfg["subtasks"])


def _iso(t: float) -> str:
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).isoformat(timespec="milliseconds")


def _station_run(attempt_dir: pathlib.Path) -> int | None:
    try:  # StationBackend.collect() saves /status after the Stop; its `run` names the rollout folder's _run<N>
        run = json.loads((attempt_dir / "status.json").read_text()).get("run")
    except (OSError, ValueError):
        return None
    return run if isinstance(run, int) else None


def build_judge(cfg: dict[str, Any]) -> Judge:
    j = cfg.get("judge", {})
    vlm = VLMJudge(j.get("model", VLMJudge().model), j.get("threshold", 0.7),
                   reasoning_effort=j.get("reasoning_effort", "none"), max_tokens=j.get("max_tokens", 800))
    return CombinedJudge(vlm, ProprioJudge())


def _client_backend(cfg: dict[str, Any], dry_run: bool) -> backends.Backend:
    from hillclimb.client_loop import ClientLoopBackend  # imports `robots` only when built

    return ClientLoopBackend(cfg, dry_run=dry_run)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(pathlib.Path(__file__).with_name("config.yaml")))
    p.add_argument("--backend", choices=["fake", "dry-run", "station", "client"], default="fake")
    p.add_argument("--client-live", action="store_true", help="client backend: write the motors (default: dry)")
    p.add_argument("--logs", default="logs/harness")
    p.add_argument("--only", help="run only this subtask")
    p.add_argument("--prompt", help="with --only: override the prompt (zero-shot prompt search, docs/PROMPTS.md)")
    p.add_argument("--checkpoint", help="override the checkpoint for every subtask (A/B base vs fine-tunes)")
    p.add_argument("--attempts", type=int, help="override max_attempts for every subtask (eval sweeps: use 1)")
    args = p.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    if args.only:
        cfg["subtasks"] = [s for s in cfg["subtasks"] if s["name"] == args.only]
        if not cfg["subtasks"]:
            p.error(f"no subtask named {args.only}")
    if args.prompt:
        if not args.only:
            p.error("--prompt needs --only")
        cfg["subtasks"][0]["prompt"] = args.prompt
    if args.checkpoint:
        cfg["checkpoint"] = args.checkpoint
        for sub in cfg["subtasks"]:
            sub.pop("checkpoint", None)
    if args.attempts:
        for sub in cfg["subtasks"]:
            sub["max_attempts"] = args.attempts
    backend = {"fake": backends.FakeBackend, "dry-run": backends.DreamcliDryRunBackend,
               "station": lambda: backends.StationBackend(cfg.get("station_url", "http://127.0.0.1:8091")),
               "client": lambda: _client_backend(cfg, dry_run=not args.client_live)}[args.backend]()
    try:
        ok = Harness(cfg, backend, build_judge(cfg), pathlib.Path(args.logs)).run()
    finally:
        if hasattr(backend, "close"):
            backend.close()
    print("TASK", "SUCCESS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
