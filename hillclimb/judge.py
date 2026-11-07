"""Success checks. A Judge returns a Verdict from the camera frames and the proprio summary of one attempt.

- VLMJudge: Fireworks' OpenAI-compatible chat API. Its frozen rubric makes the model describe the scene first and
  only then score it.
- ProprioJudge: cheap signals from the trace CSV (gripper holding something, tracking error). It can veto.
- CombinedJudge: VLM verdict with the proprio veto applied.
- HumanJudge: the operator types y/n after each attempt (no network; zero-shot baselines; VLM fallback).
- ConfirmJudge: shows the automatic verdict, the operator presses Enter to accept or y/n to override; every
  disagreement is logged as a calibration label for the VLM.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import os
import pathlib
import re
import urllib.request
from typing import Any, Protocol

FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
# Checked against /v1/models on 2026-11-07 (supports_image_input=True; a test call answered in about 1 s).
DEFAULT_VLM = "accounts/fireworks/models/qwen3p8-max"

RUBRIC = """You are a strict judge of a robot manipulation step in a server rack.
Step: {name}. Success means: {success}
Failure signs: {failure}{scene}
Images: {cams} (latest frames, after the attempt).
First DESCRIBE what you see about the rack, the USB cable(s), the rack backing and both grippers, in 2-4
factual sentences. Then SCORE it. Answer only JSON:
{{"description": "...", "score": <0.0-1.0 confidence the step succeeded>, "success": <true|false>}}"""
# Added to RUBRIC when the caller also passes the frames from the start of the attempt (VLMJudge(..., before=...)).
BEFORE_NOTE = """The [<cam> BEFORE] images are from the START of the attempt; the [<cam>] images are AFTER it. Judge the
AFTER state, and use BEFORE only to see what changed (the plug, cable and panel positions)."""


@dataclasses.dataclass
class Verdict:
    success: bool | None  # None = unknown (judge unavailable)
    score: float = 0.0
    reason: str = ""
    source: str = ""


class Judge(Protocol):
    def __call__(self, subtask: dict[str, Any], frames: dict[str, pathlib.Path], proprio: dict[str, Any]) -> Verdict: ...


def holding(side: dict[str, Any], closed_cmd: float = 0.15, min_open: float = 0.06) -> bool:
    """Close commanded but jaws stopped open, so something is between them. TODO(event): tune on the Metal gripper."""
    return side.get("cmd_grip", 1.0) < closed_cmd and side.get("grip", 0.0) > min_open


class ProprioJudge:
    """`subtask["proprio"]`: {expect_holding: true|false|null}. Returns success=None when it has no opinion."""

    def __call__(self, subtask, frames, proprio) -> Verdict:
        expect = (subtask.get("proprio") or {}).get("expect_holding")
        sides = [proprio[s] for s in ("l", "r") if s in proprio]
        if expect is None or not sides:
            return Verdict(None, source="proprio")
        held = any(holding(s) for s in sides)
        return Verdict(held == expect, 1.0 if held == expect else 0.0, f"holding={held} expected={expect}", "proprio")


class VLMJudge:
    def __init__(self, model: str = DEFAULT_VLM, threshold: float = 0.7, api_key: str | None = None,
                 post=None, reasoning_effort: str | None = "none", max_tokens: int = 800,
                 json_mode: bool = True) -> None:
        # reasoning_effort "none": qwen3p8-max and kimi-k3 are reasoning models. With thinking on, the 400-token
        # budget went entirely to reasoning_content and the reply had no JSON, so every verdict was "unknown"
        # (docs/research/07-judge-eval.md). None = leave the field out (for models that reject it).
        self.model, self.threshold = model, threshold
        self.reasoning_effort, self.max_tokens, self.json_mode = reasoning_effort, max_tokens, json_mode
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        self._post = post or self._http_post

    def _http_post(self, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(FIREWORKS_URL, data=json.dumps(body).encode(), headers={
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)

    def __call__(self, subtask, frames, proprio, before: dict[str, pathlib.Path] | None = None) -> Verdict:
        if not frames or (self._post == self._http_post and not self.api_key):
            return Verdict(None, reason="no frames or no FIREWORKS_API_KEY", source="vlm")
        rubric = subtask.get("rubric", {})
        scene = f"\nScene notes (what is where on this rig): {rubric['scene']}" if rubric.get("scene") else ""
        text = RUBRIC.format(name=subtask["name"], success=rubric.get("success", ""),
                             failure=rubric.get("failure", ""), scene=scene, cams=", ".join(frames))
        if before:
            text += "\n" + BEFORE_NOTE
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        shots = [(f"{cam} BEFORE", p) for cam, p in (before or {}).items()] + list(frames.items())
        try:  # reading a frame can fail too: the loop wipes the inputs folder when a new run opens its trace
            for cam, path in shots:
                b64 = base64.b64encode(pathlib.Path(path).read_bytes()).decode()
                content += [{"type": "text", "text": f"[{cam}]"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]
            body = {"model": self.model, "temperature": 0, "max_tokens": self.max_tokens,
                    "messages": [{"role": "user", "content": content}]}
            if self.json_mode:  # forces a JSON reply, so a model that "thinks aloud" cannot run out of tokens first
                body["response_format"] = {"type": "json_object"}
            if self.reasoning_effort is not None:
                body["reasoning_effort"] = self.reasoning_effort
            reply = self._post(body)["choices"][0]["message"]["content"]
            data = json.loads(re.search(r"\{.*\}", reply, re.S).group(0))
            score = float(data.get("score", 0.0))
            # `is True`: a reply of "success": "false" (a string) must not pass as bool("false") == True
            return Verdict(data.get("success") is True and score >= self.threshold, score,
                           data.get("description", ""), "vlm")
        except Exception as exc:  # a judge failure is "unknown", never a crash mid-run
            return Verdict(None, reason=f"vlm error: {exc}", source="vlm")


class CombinedJudge:
    def __init__(self, vlm: Judge, proprio: Judge) -> None:
        self.vlm, self.proprio = vlm, proprio

    def __call__(self, subtask, frames, proprio) -> Verdict:
        p = self.proprio(subtask, frames, proprio)
        if p.success is False:
            return p  # the cheap veto: no VLM call
        v = self.vlm(subtask, frames, proprio)
        if v.success is not None:
            return v
        # Proprio can only veto. "Not holding" is true of nearly every close_backing end, and "holding" of a gripper
        # clamped on a cable still in the port, so a VLM outage must stay unknown (= retry), never pass on proprio.
        return Verdict(None, p.score, f"{v.reason}; proprio: {p.reason or 'no opinion'}", v.source or "vlm")


class HumanJudge:
    """Operator verdict at the end of each attempt. `final_only`: the harness skips mid-run checks with it."""

    final_only = True

    def __init__(self, ask=input) -> None:
        self.ask = ask

    def __call__(self, subtask, frames, proprio) -> Verdict:
        while True:
            a = self.ask(f"[{subtask['name']}] success? y/n: ").strip().lower()
            if a in ("y", "n"):
                return Verdict(a == "y", 1.0 if a == "y" else 0.0, "operator", source="human")


class ConfirmJudge:
    """The automatic judge decides mid-run (early stop). At the end, the operator confirms or overrides it."""

    def __init__(self, auto: Judge, ask=input) -> None:
        self.auto, self.ask = auto, ask
        self.midrun_judge = auto  # the harness's mid-run early-stop checks use the automatic judge, no prompt
        self.disagreements: list[dict[str, Any]] = []

    def __call__(self, subtask, frames, proprio) -> Verdict:
        v = self.auto(subtask, frames, proprio)
        shown = {True: "SUCCESS", False: "FAIL", None: "UNKNOWN"}[v.success]
        a = self.ask(f"[{subtask['name']}] judge says {shown} ({v.score:.2f}: {v.reason[:80]}). Enter=accept, y/n: ")
        a = a.strip().lower()
        if a not in ("y", "n"):
            return v
        human = a == "y"
        if human != v.success:
            self.disagreements.append({"subtask": subtask["name"], "auto": v.success, "human": human,
                                       "frames": {k: str(p) for k, p in frames.items()}})
        return Verdict(human, 1.0 if human else 0.0, f"operator override (judge: {shown}; {v.reason[:120]})",
                       source="human")


class FakeJudge:
    """Offline judge for `--backend fake`: no network, no cost. Fails each subtask's first attempt, passes the rest."""

    def __init__(self) -> None:
        self.seen: dict[str, int] = {}

    def __call__(self, subtask, frames, proprio) -> Verdict:
        if not proprio and not frames:
            return Verdict(None, source="fake")
        n = self.seen[subtask["name"]] = self.seen.get(subtask["name"], 0) + (1 if proprio is not None else 0)
        return Verdict(n >= 2, 1.0 if n >= 2 else 0.0, "fake judge", source="fake")
