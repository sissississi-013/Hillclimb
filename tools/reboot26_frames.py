"""Pull a labelled judge-eval set from the public REBOOT26 USB-A datasets (arXiv 2609.22591) without downloading videos.

ffmpeg seeks inside the remote mp4 over HTTPS range requests, so each frame costs a few hundred KB, not the 200 MB file.
Weak labels: the first frame of an install episode = plug NOT seated (neg for plug_usb), the last = seated (pos); the
first frame of a remove episode = plug still in (neg for unplug_usb), the last = plug out in the container (pos).
Output is the layout tools/judge_eval.py reads: <out>/<subtask>/<pos|neg>/<sample>/{<cam>.jpg, before/<cam>.jpg}.
`before/` holds the episode's first frame, for the before/after prompt variant.

    .venv/bin/python -I tools/reboot26_frames.py --meta /Users/sissi/hc-vendor/judge-data --out /Users/sissi/hc-vendor/judge-data/eval
`--meta` must hold <dataset>/meta/episodes/chunk-000/file-000.parquet (downloaded from the HF repo). Nothing from the
dataset is executed; parquet and mp4 are only parsed.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess

import pandas as pd

HF = "https://huggingface.co/datasets/REBOOT26/{ds}/resolve/main/videos/observation.images.{cam}/chunk-000/file-{f:03d}.mp4"
SETS = {  # dataset -> subtask it tests
    "usbA-install": "plug_usb", "USB-A_recovery_install": "plug_usb",
    "usbA_remove": "unplug_usb", "USB-A_recovery_remove": "unplug_usb",
}
CAMS = {"scene": "cam_high", "low": "cam_low"}   # our name -> REBOOT26 camera key


def grab(url: str, t: float, dest: pathlib.Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", url, "-frames:v", "1",
                    "-q:v", "3", "-y", str(dest)], check=True, timeout=120)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--episodes", default="3,11,19,27,35,43,51")
    ap.add_argument("--pad_s", type=float, default=0.4, help="stay this far inside the episode's start/end")
    a = ap.parse_args()
    eps = [int(e) for e in a.episodes.split(",")]
    for ds, sub in SETS.items():
        meta = pd.read_parquet(a.meta / ds / "meta/episodes/chunk-000/file-000.parquet").set_index("episode_index")
        for ep in eps:
            row = meta.loc[ep]
            times = {}
            for name, cam in CAMS.items():
                k = f"videos/observation.images.{cam}/"
                url = HF.format(ds=ds, cam=cam, f=int(row[k + "file_index"]))
                t0, t1 = float(row[k + "from_timestamp"]), float(row[k + "to_timestamp"])
                times[name] = (url, t0 + a.pad_s, t1 - a.pad_s)
            sid = f"{ds}-ep{ep:02d}"
            for name, (url, t_start, t_end) in times.items():
                start = a.out / sub / "neg" / f"{sid}-start"
                end = a.out / sub / "pos" / f"{sid}-end"
                grab(url, t_start, start / f"{name}.jpg")
                grab(url, t_end, end / f"{name}.jpg")
                (end / "before").mkdir(parents=True, exist_ok=True)
                (start / "before").mkdir(parents=True, exist_ok=True)
                (end / "before" / f"{name}.jpg").write_bytes((start / f"{name}.jpg").read_bytes())
                (start / "before" / f"{name}.jpg").write_bytes((start / f"{name}.jpg").read_bytes())
            print(sid, "ok", flush=True)


if __name__ == "__main__":
    main()
