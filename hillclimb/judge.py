"""Success checks. A Judge returns a Verdict from the camera frames and the proprio summary of one attempt.

- VLMJudge: Fireworks' OpenAI-compatible chat API. Its frozen rubric makes the model describe the scene first and
  only then score it.
- ProprioJudge: cheap signals from the trace CSV (gripper holding something, tracking error). It can veto.
- CombinedJudge: VLM verdict with the proprio veto applied.
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
Failure signs: {failure}
Images: {cams} (latest frames, after the attempt).
First DESCRIBE what you see about the rack, the USB cable(s), the rack backing and both grippers, in 2-4
factual sentences. Then SCORE it. Answer only JSON:
{{"description": "...", "score": <0.0-1.0 confidence the step succeeded>, "success": <true|false>}}"""


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
                 post=None) -> None:
        self.model, self.threshold = model, threshold
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        self._post = post or self._http_post

    def _http_post(self, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(FIREWORKS_URL, data=json.dumps(body).encode(), headers={
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)

    def __call__(self, subtask, frames, proprio) -> Verdict:
        if not frames or (self._post == self._http_post and not self.api_key):
            return Verdict(None, reason="no frames or no FIREWORKS_API_KEY", source="vlm")
        rubric = subtask.get("rubric", {})
        text = RUBRIC.format(name=subtask["name"], success=rubric.get("success", ""),
                             failure=rubric.get("failure", ""), cams=", ".join(frames))
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for cam, path in frames.items():
            b64 = base64.b64encode(pathlib.Path(path).read_bytes()).decode()
            content += [{"type": "text", "text": f"[{cam}]"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]
        body = {"model": self.model, "temperature": 0, "max_tokens": 400,
                "messages": [{"role": "user", "content": content}]}
        try:
            reply = self._post(body)["choices"][0]["message"]["content"]
            data = json.loads(re.search(r"\{.*\}", reply, re.S).group(0))
            score = float(data.get("score", 0.0))
            return Verdict(bool(data.get("success")) and score >= self.threshold, score,
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
        return v if v.success is not None else p
