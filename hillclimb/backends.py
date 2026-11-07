"""Ways to run one Dream episode. Every backend has the same five calls the harness uses.

- StationBackend: the station agent's HTTP API (`dreamcli robot up`, port 8091; robots/station_api.py:9-15).
  THIS MOVES THE ARMS. It refuses to build unless HILLCLIMB_ALLOW_ARMS=1 is set.
- DreamcliDryRunBackend: `dreamcli robot run --dry-run --task ...` in a subprocess. It calls the model for real and
  writes no target to an arm (robots/run.py:288-334). It needs the station agent to be down (cli.py:585).
- FakeBackend: no hardware, for laptop tests.

The artifacts come from the files the control loop writes for every run: the per-tick trace CSV and, beside it,
the JPEGs handed to the model once a second (robots/control.py:1748-1803, config.py:132). A new run wipes both, so
collect() copies them into the attempt folder before the next start.
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import shutil
import signal
import subprocess
import time
import urllib.request
from typing import Any, Protocol

# TODO(event): confirm the station file's trace_path. The default is ~/.dreamlabs/last-run.csv (config.py:132).
DEFAULT_TRACE = pathlib.Path("~/.dreamlabs/last-run.csv").expanduser()


def trace_from_station_file(station_file: str | None) -> pathlib.Path:
    """The station file's `trace_path` (robots/config.py:526, 774), else the default. A trace_path the harness doesn't
    know about means no frames and every attempt judged unknown (08-code-review.md Q2)."""
    if station_file:
        path = pathlib.Path(station_file).expanduser()
        if path.exists():
            import yaml

            raw = yaml.safe_load(path.read_text()) or {}
            if raw.get("trace_path"):
                return pathlib.Path(str(raw["trace_path"])).expanduser()
    return DEFAULT_TRACE
ACTIVE = {"starting", "running", "stopping", "homing", "resetting"}  # robots/station.py:85-91


class Backend(Protocol):
    def start(self, prompt: str, checkpoint: str | None) -> None: ...
    def state(self) -> str: ...  # "idle" | "starting" | "running" | "error" | ...
    def stop(self) -> None: ...  # the arms hold where they are (station.py:508)
    def home(self) -> None: ...  # home pose from the station file, grippers open (station.py:534)
    def collect(self, dest: pathlib.Path) -> dict[str, Any]: ...  # copy frames + trace, return proprio summary
    def latest_frames(self) -> dict[str, pathlib.Path]: ...  # newest model-input JPEG per camera, mid-run


def _trace_files(trace: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    return trace, trace.with_name(f"{trace.stem}_inputs")  # control.py:136-149


def latest_frames_in(inputs: pathlib.Path) -> dict[str, pathlib.Path]:
    out = {}
    for cam in sorted(p for p in inputs.glob("*") if p.is_dir()):
        jpgs = sorted(cam.glob("*.jpg"), key=lambda p: float(p.stem))
        if jpgs:
            out[cam.name] = jpgs[-1]
    return out


def proprio_summary(trace_csv: pathlib.Path, tail_rows: int = 20) -> dict[str, Any]:
    """Cheap signals from the last rows of the trace (columns: control.py:116-133). Grip is 0 closed .. 1 open."""
    if not trace_csv.exists():
        return {}
    with open(trace_csv, newline="") as f:
        rows = list(csv.DictReader(f))[-tail_rows:]
    if not rows:
        return {}
    out: dict[str, Any] = {"rows": len(rows)}
    for side in ("l", "r"):
        grip = [float(r[f"gripper_{side}"]) for r in rows]
        cmd = [float(r[f"cmd_gripper_{side}"]) for r in rows]
        joints = [k[len("measured_") + 1 :] for k in rows[-1] if k.startswith(f"measured_{side}")]
        err = max(abs(float(r[f"commanded_{side}{j}"]) - float(r[f"measured_{side}{j}"])) for r in rows for j in joints)
        out[side] = {"grip": grip[-1], "cmd_grip": cmd[-1], "max_track_err_rad": round(err, 4)}
    return out


def _copy_artifacts(trace: pathlib.Path, dest: pathlib.Path) -> dict[str, Any]:
    csv_path, inputs = _trace_files(trace)
    dest.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        shutil.copy2(csv_path, dest / "trace.csv")
    if inputs.exists():
        shutil.copytree(inputs, dest / "inputs", dirs_exist_ok=True)
    return proprio_summary(dest / "trace.csv")


STATUS_TIMEOUT_S = 2.0


class StationBackend:
    """The real station. POST /start {prompt, checkpoint}, /stop, /home; GET /status (station_api.py:151-168)."""

    def __init__(self, url: str = "http://127.0.0.1:8091", trace: pathlib.Path = DEFAULT_TRACE) -> None:
        if os.environ.get("HILLCLIMB_ALLOW_ARMS") != "1":
            raise RuntimeError("StationBackend moves real arms: set HILLCLIMB_ALLOW_ARMS=1 with a hand on the e-stop")
        self.url, self.trace = url.rstrip("/"), trace

    def _call(self, method: str, route: str, body: dict | None = None, timeout: float = 70) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + route, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:  # start/stop: the API waits up to 60 s (station_api.py:35)
            return json.load(r)

    def start(self, prompt: str, checkpoint: str | None) -> None:
        self._call("POST", "/start", {"prompt": prompt, **({"checkpoint": checkpoint} if checkpoint else {})})

    def state(self) -> str:
        # /status is a local read: bound it so a stuck request can't hold up the harness's deadline (PR #2 review).
        try:
            return self._call("GET", "/status", timeout=STATUS_TIMEOUT_S)["state"]
        except (TimeoutError, OSError):  # urllib wraps socket timeouts in URLError (an OSError)
            return "unknown"

    def stop(self) -> None:
        self._call("POST", "/stop", {})

    def home(self) -> None:
        self._call("POST", "/home", {})

    def collect(self, dest: pathlib.Path) -> dict[str, Any]:
        summary = _copy_artifacts(self.trace, dest)
        (dest / "status.json").write_text(json.dumps(self._call("GET", "/status"), indent=1))
        return summary

    def latest_frames(self) -> dict[str, pathlib.Path]:
        return latest_frames_in(_trace_files(self.trace)[1])


class DreamcliDryRunBackend:
    """Mirrors `dreamcli robot run --dry-run`. Ctrl-C (SIGINT) ends a run (run.py:9-12). Nothing ever moves."""

    def __init__(self, trace: pathlib.Path = DEFAULT_TRACE, dreamcli: str = "dreamcli") -> None:
        self.trace, self.dreamcli, self.proc = trace, dreamcli, None

    def start(self, prompt: str, checkpoint: str | None) -> None:
        argv = [self.dreamcli, "robot", "run", "--dry-run", "--task", prompt]
        self.proc = subprocess.Popen(argv + (["--checkpoint", checkpoint] if checkpoint else []))

    def state(self) -> str:
        if self.proc is None:
            return "idle"
        return "running" if self.proc.poll() is None else ("idle" if self.proc.returncode == 0 else "error")

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            self.proc.wait(timeout=60)

    def home(self) -> None:
        pass  # a dry run never homes (run.py:96-99)

    def collect(self, dest: pathlib.Path) -> dict[str, Any]:
        return _copy_artifacts(self.trace, dest)

    def latest_frames(self) -> dict[str, pathlib.Path]:
        return latest_frames_in(_trace_files(self.trace)[1])


class FakeBackend:
    """No hardware. Each run 'finishes' after run_s seconds of the injected clock. All calls are logged in .calls."""

    def __init__(self, clock=time.monotonic, run_s: float = 5.0, fail_start: set[str] | None = None) -> None:
        self.clock, self.run_s, self.fail_start = clock, run_s, fail_start or set()
        self.calls: list[tuple[str, ...]] = []
        self._started: float | None = None
        self._prompt = ""

    def start(self, prompt: str, checkpoint: str | None) -> None:
        self.calls.append(("start", prompt))
        self._prompt, self._started = prompt, self.clock()

    def state(self) -> str:
        if self._started is None:
            return "idle"
        if self._prompt in self.fail_start:
            return "error"
        return "running" if self.clock() - self._started < self.run_s else "idle"

    def stop(self) -> None:
        self.calls.append(("stop",))
        self._started = None

    def home(self) -> None:
        self.calls.append(("home",))

    def collect(self, dest: pathlib.Path) -> dict[str, Any]:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "inputs" / "overhead").mkdir(parents=True, exist_ok=True)
        (dest / "inputs" / "overhead" / "0.000.jpg").write_bytes(b"fake")
        return {"l": {"grip": 1.0, "cmd_grip": 1.0, "max_track_err_rad": 0.0}}

    def latest_frames(self) -> dict[str, pathlib.Path]:
        return {}

    def play(self, path: str, dest: pathlib.Path | None = None, opts: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append(("play", str(path)))
        if dest is not None:
            self.collect(dest)
        return {"ticks": 0, "dry_run": True, "proprio": {"l": {"grip": 1.0, "cmd_grip": 1.0, "max_track_err_rad": 0.0}}}
