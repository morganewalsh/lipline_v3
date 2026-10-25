#!/usr/bin/env python3
"""
lipread.py: turn lip movements from your webcam into text.

L1  live        Teach it a small vocabulary with your own mouth, then read it back live.
                MediaPipe Face Landmarker tracks the lips; a DTW nearest-neighbour
                classifier matches each utterance against your recorded samples.

L2  record      Record a 25 fps webcam clip (no audio).
    setup-l2    One-time setup for L2: creates a separate Python 3.12 environment with uv
                (.venv-vsr), installs the model's packages, downloads the model code (vsr/)
                and tells you which model files are still missing.
    transcribe  Record clips (or use an existing one) and run the pretrained Auto-AVSR
                visual speech recognition model on them (open vocabulary, English).

Both show live LipLine captions in your browser (lip-read words get a purple underline)
and write the transcript to a PDF and a text file. Add --screen to read a speaker's video
tile from a Google Meet / Zoom window instead of your webcam.

Install (your main Python):   pip install opencv-python mediapipe numpy reportlab uv mss

Examples:
    python lipread.py live --words hello yes no water stop thank_you please help
    python lipread.py setup-l2
    python lipread.py transcribe --record 6 --auto
    python lipread.py transcribe --record 6 --auto --screen     (read someone in a Meet call)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
import urllib.request
import zipfile
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import numpy as np

# =============================================================================
# Configuration
# =============================================================================
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")
FACE_MODEL_PATH = Path("face_landmarker.task")

# Face Mesh landmark indices around the lips: outer contour, then inner contour.
OUTER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
INNER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
LIPS = OUTER_LIP + INNER_LIP
LEFT_EYE, RIGHT_EYE = 33, 263      # outer eye corners: used for scale and head roll
MOUTH_LEFT, MOUTH_RIGHT = 61, 291  # mouth corners: used as the origin

# MediaPipe blendshapes that describe the mouth (each is a score from 0 to 1).
MOUTH_BLENDSHAPES = [
    "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper", "mouthPressLeft", "mouthPressRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthSmileLeft", "mouthSmileRight", "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthUpperUpLeft", "mouthUpperUpRight",
]

RESAMPLE_N = 24      # every utterance is stretched or squeezed to this many frames
DTW_BAND = 6         # how far DTW may warp time, in resampled frames
COORD_WEIGHT = 4.0   # weight of lip geometry relative to blendshapes
CALIBRATION_S = 2.0  # seconds of "mouth still" used to set the motion thresholds
ARM_TIMEOUT_S = 5.0  # how long a training key waits for you to say the word
CAPTION_PORT = 8766  # LipLine caption page: http://127.0.0.1:8766
REGION_FILE = Path("screen_region.json")  # the part of the screen chosen with --screen

# L2: the pretrained model runs in its own Python 3.12 environment (it needs older
# PyTorch and MediaPipe versions than L1), created with uv.
VSR_ENV = Path(".venv-vsr")
VSR_PYTHON_VERSION = "3.12"
VSR_REPO = Path("vsr")
VSR_REPO_GIT = "https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages"
VSR_REPO_ZIP = ("https://codeload.github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages"
                "/zip/refs/heads/master")
VSR_CONFIG = "./configs/LRS3_V_WER19.1.ini"
TORCH_PACKAGES = ["torch==2.4.1", "torchvision==0.19.1", "torchaudio==2.4.1"]
VSR_PACKAGES = ["setuptools",            # brings back distutils, which Python 3.12 removed
                "numpy<2", "opencv-python<4.11", "mediapipe==0.10.14", "av==12.3.0",
                "scipy", "scikit-image", "six"]
# Model files the config expects, with where to download them (from the repo's README).
VSR_MODEL_FILES = {
    "benchmarks/LRS3/models/LRS3_V_WER19.1/model.pth":
        ("Visual-only model, ~900 MB", "http://bit.ly/40EAtyX"),
    "benchmarks/LRS3/models/LRS3_V_WER19.1/model.json":
        ("Visual-only model, ~900 MB", "http://bit.ly/40EAtyX"),
    "benchmarks/LRS3/language_models/lm_en_subword/model.pth":
        ("Language model, ~190 MB", "http://bit.ly/3FE4XsV"),
    "benchmarks/LRS3/language_models/lm_en_subword/model.json":
        ("Language model, ~190 MB", "http://bit.ly/3FE4XsV"),
}


# =============================================================================
# Step 1 · Face tracking
# =============================================================================
def ensure_face_model() -> None:
    if not FACE_MODEL_PATH.exists():
        print(f"Downloading {FACE_MODEL_PATH} (~4 MB)…")
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)


class FaceTracker:
    """Wraps MediaPipe Face Landmarker: frame in, (478×2 pixel points, blendshapes) out."""

    def __init__(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        ensure_face_model()
        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)  # must strictly increase
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)
        if not result.face_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        points = np.array([[p.x * w, p.y * h] for p in result.face_landmarks[0]], dtype=np.float32)
        blendshapes = ({c.category_name: c.score for c in result.face_blendshapes[0]}
                       if result.face_blendshapes else {})
        return points, blendshapes

    def close(self) -> None:
        self._landmarker.close()


# =============================================================================
# Step 2 · Features: one vector per frame describing the mouth
# =============================================================================
def lip_features(points: np.ndarray, blendshapes: dict) -> np.ndarray:
    """Lip points relative to the mouth centre, scaled by eye distance and rotated so the
    eyes are level (so moving or tilting your head doesn't change the features),
    followed by the mouth blendshape scores. Length: 40*2 + 18 = 98."""
    left, right = points[LEFT_EYE], points[RIGHT_EYE]
    eye_dist = max(float(np.linalg.norm(right - left)), 1e-6)
    angle = np.arctan2(right[1] - left[1], right[0] - left[0])
    c, s = np.cos(-angle), np.sin(-angle)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
    centre = (points[MOUTH_LEFT] + points[MOUTH_RIGHT]) / 2
    lips = ((points[LIPS] - centre) / eye_dist) @ rotation.T
    shapes = [blendshapes.get(name, 0.0) for name in MOUTH_BLENDSHAPES]
    return np.concatenate([COORD_WEIGHT * lips.ravel(), shapes]).astype(np.float32)


# =============================================================================
# Step 3 · Finding where words start and stop
# =============================================================================
class MotionMeter:
    """Smoothed speed of the mouth: how much the feature vector changes per second."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.value, self._prev, self._prev_t = 0.0, None, 0.0

    def update(self, features: np.ndarray, t: float) -> float:
        if self._prev is not None:
            dt = max(t - self._prev_t, 1e-3)
            speed = float(np.linalg.norm(features - self._prev)) / dt
            self.value = 0.6 * self.value + 0.4 * speed
        self._prev, self._prev_t = features, t
        return self.value


class Calibrator:
    """Watches your still mouth for a couple of seconds and picks the motion thresholds."""

    def __init__(self, seconds: float = CALIBRATION_S) -> None:
        self.seconds, self.start, self.values = seconds, None, []
        self.on = self.off = None

    @property
    def done(self) -> bool:
        return self.on is not None

    def update(self, activity: float, t: float) -> bool:
        if self.start is None:
            self.start = t
        self.values.append(activity)
        if t - self.start < self.seconds:
            return False
        xs = np.array(self.values[len(self.values) // 5:])   # skip the first moments
        mean, sd = float(xs.mean()), float(xs.std())
        self.on = max(3 * mean, mean + 4 * sd, 1e-3)          # "speaking" above this
        self.off = 0.6 * self.on                              # "quiet" below this
        return True


def trim(frames: list, threshold: float) -> np.ndarray:
    """Drop still frames at both ends of an utterance, keeping 2 frames of margin."""
    if not frames:
        return np.empty((0, 0), dtype=np.float32)
    a, b = 0, len(frames) - 1
    while a < b and frames[a][1] < threshold:
        a += 1
    while b > a and frames[b][1] < threshold:
        b -= 1
    a, b = max(0, a - 2), min(len(frames) - 1, b + 2)
    return np.stack([f for f, _ in frames[a:b + 1]])


class Segmenter:
    """Cuts the stream of frames into utterances: starts when the mouth moves,
    ends after a short pause (or after max_s)."""

    def __init__(self, on: float, off: float, pause_s: float = 0.45,
                 min_s: float = 0.25, max_s: float = 3.0, preroll: int = 5) -> None:
        self.on, self.off = on, off
        self.pause_s, self.min_s, self.max_s = pause_s, min_s, max_s
        self._preroll = deque(maxlen=preroll)
        self.reset()

    def reset(self) -> None:
        self.active, self._frames, self._start, self._quiet_since = False, [], 0.0, None

    def update(self, features: np.ndarray, activity: float, t: float):
        """Feed one frame. Returns a (frames × 98) array when an utterance ends, else None."""
        frame = (features, activity)
        self._preroll.append(frame)
        if not self.active:
            if activity > self.on:
                self.active, self._frames = True, list(self._preroll)
                self._start, self._quiet_since = t, None
            return None

        self._frames.append(frame)
        if activity < self.off:
            self._quiet_since = self._quiet_since or t
        else:
            self._quiet_since = None
        paused = self._quiet_since is not None and t - self._quiet_since > self.pause_s
        if not (paused or t - self._start > self.max_s):
            return None

        duration, frames = t - self._start, self._frames
        self.reset()
        return trim(frames, self.off) if duration >= self.min_s else None


# =============================================================================
# Step 4 · The word classifier (resample + dynamic time warping + nearest neighbour)
# =============================================================================
def resample(seq: np.ndarray, n: int = RESAMPLE_N) -> np.ndarray:
    """Linearly interpolate a (T × D) sequence to (n × D)."""
    if len(seq) == 1:
        return np.repeat(seq, n, axis=0)
    pos = np.linspace(0, len(seq) - 1, n)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(seq) - 1)
    w = (pos - lo)[:, None]
    return (seq[lo] * (1 - w) + seq[hi] * w).astype(np.float32)


def dtw_distance(a: np.ndarray, b: np.ndarray, band: int = DTW_BAND) -> float:
    """Cost of the best alignment between two sequences, allowing local speed changes."""
    n, m = len(a), len(b)
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(acc[n, m] / (n + m))


class WordModel:
    """Stores resampled samples per word and classifies new utterances."""

    def __init__(self) -> None:
        self.samples: list[tuple[str, np.ndarray]] = []

    def add(self, word: str, seq: np.ndarray) -> None:
        self.samples.append((word, resample(seq)))

    def remove_word(self, word: str) -> None:
        self.samples = [(w, s) for w, s in self.samples if w != word]

    def count(self, word: str) -> int:
        return sum(1 for w, _ in self.samples if w == word)

    def classify(self, seq: np.ndarray):
        """Returns (word, margin, runner_up) or None. margin is 0..1: how much closer the
        best word is than the second best. Small margin means "not sure"."""
        if not self.samples:
            return None
        query = resample(seq)
        best: dict[str, float] = {}
        for word, sample in self.samples:
            d = dtw_distance(query, sample)
            best[word] = min(d, best.get(word, np.inf))
        ranked = sorted(best.items(), key=lambda kv: kv[1])
        word, d1 = ranked[0]
        runner_up, d2 = ranked[1] if len(ranked) > 1 else (None, 2 * d1)
        return word, (d2 - d1) / (d2 or 1.0), runner_up

    def save(self, path: Path) -> None:
        data = [{"label": w, "seq": np.round(s, 4).tolist()} for w, s in self.samples]
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path, words: list[str]) -> "WordModel":
        model = cls()
        if path.exists():
            for item in json.loads(path.read_text()):
                if item["label"] in words:
                    model.samples.append((item["label"], np.array(item["seq"], dtype=np.float32)))
        return model


# =============================================================================
# Step 4b · Transcript PDF
# =============================================================================
def write_transcript_pdf(path: Path, title: str, entries: list[dict], note: str = "") -> None:
    """entries: [{"time": "14:02:11", "text": "...", "uncertain": [indices of words]}]."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError:
        print("PDF skipped: run  python -m pip install reportlab")
        return

    styles = getSampleStyleSheet()
    meta = ParagraphStyle("meta", parent=styles["Normal"], textColor=colors.grey, fontSize=9)
    stamp = ParagraphStyle("stamp", parent=styles["Normal"], textColor=colors.grey, fontSize=8, spaceBefore=8)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=13, leading=19)

    story = [Paragraph(escape(title), styles["Title"]),
             Paragraph(datetime.now().strftime("%B %d, %Y  %H:%M"), meta)]
    if note:
        story.append(Paragraph(escape(note), meta))
    story.append(Spacer(1, 0.25 * inch))
    if not entries:
        story.append(Paragraph("<i>No words were transcribed.</i>", body))
    for entry in entries:
        words = entry["text"].split()
        unsure = set(entry.get("uncertain", []))
        marked = [f'<font color="grey"><i>{escape(w)}?</i></font>' if i in unsure else escape(w)
                  for i, w in enumerate(words)]
        story.append(Paragraph(escape(entry.get("time", "")), stamp))
        story.append(Paragraph(" ".join(marked) or "<i>(nothing recognised)</i>", body))

    SimpleDocTemplate(str(path), pagesize=letter, title=title,
                      leftMargin=inch, rightMargin=inch, topMargin=inch, bottomMargin=inch).build(story)
    print(f"Saved {path}")


# =============================================================================
# Step 4c · LipLine caption page (browser) with purple-underlined lip-read words
# =============================================================================
class CaptionFeed:
    """Captions shared between the reading loop and the caption page (thread-safe)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.version, self.status, self.entries = 0, "Starting", []

    def _changed(self) -> None:
        self.version += 1

    def set_status(self, status: str) -> None:
        with self._lock:
            if status != self.status:
                self.status = status
                self._changed()

    def add_entry(self, label: str, text: str) -> None:
        with self._lock:
            words = [w.lower() for w in text.split()]       # the model writes in capitals
            words = [w.capitalize() if w in ("i", "i'm", "i'll", "i've", "i'd") else w for w in words]
            if words:
                words[0] = words[0][:1].upper() + words[0][1:]
            self.entries.append({"time": label, "words": [{"w": w, "sure": True} for w in words]})
            self._changed()

    def add_word(self, label: str, word: str, sure: bool) -> None:
        """L1: words collect in one running caption line per session."""
        with self._lock:
            if not self.entries or self.entries[-1]["time"] != label:
                self.entries.append({"time": label, "words": []})
            self.entries[-1]["words"].append({"w": word.replace("_", " "), "sure": sure})
            self._changed()

    def pop_word(self) -> None:
        with self._lock:
            if self.entries and self.entries[-1]["words"]:
                self.entries[-1]["words"].pop()
                self._changed()

    def clear(self) -> None:
        with self._lock:
            self.entries = []
            self._changed()

    def to_json(self) -> str:
        with self._lock:
            return json.dumps({"version": self.version, "status": self.status, "entries": self.entries})


CAPTION_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LipLine · live captions</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  :root {
    --navy: #000180; --digital: #006EE9; --mauve: #E7A0FE; --frost: #CFF1FB;
    --lime: #D0FFA3; --ink: #000180; --muted: #4b4f8f; --paper: #ffffff;
    --font: "Space Grotesk", "Segoe UI", system-ui, sans-serif;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: var(--font); color: var(--ink); background: var(--paper); }
  header { display: flex; align-items: center; gap: 14px; padding: 18px 28px;
           background: linear-gradient(90deg, #fff 0%, var(--frost) 100%); }
  .logo { position: relative; background: var(--frost); color: var(--digital); font-weight: 700;
          font-size: 30px; line-height: 1; padding: 4px 10px 6px; letter-spacing: -1px; }
  .logo::after { content: ""; position: absolute; left: 0; right: 0; top: 52%; height: 3px; background: var(--mauve); }
  .brand { font-weight: 700; font-size: 26px; }
  .tag { color: var(--muted); font-size: 15px; }
  .spacer { flex: 1; }
  .status { display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px; border-radius: 999px;
            background: var(--navy); color: #fff; font-size: 14px; font-weight: 500; }
  .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--lime); }
  .status.busy .dot { animation: pulse 1s infinite; }
  @keyframes pulse { 50% { opacity: .25; } }
  main { max-width: 1000px; margin: 0 auto; padding: 22px 28px 40px; }
  .captions { background: var(--navy); color: #fff; border-radius: 18px; padding: 26px 30px; min-height: 140px;
              display: flex; flex-direction: column; justify-content: center; }
  .captions .label { color: var(--lime); font-size: 13px; letter-spacing: .08em; text-transform: uppercase; margin-bottom: 10px; }
  .captions .line { font-size: 38px; font-weight: 500; line-height: 1.45; }
  .captions .waiting { color: #b8bbf0; font-size: 22px; }
  .lip { text-decoration-line: underline; text-decoration-color: var(--mauve); text-decoration-thickness: 4px;
         text-underline-offset: 7px; text-decoration-skip-ink: none; }
  .lip.unsure { text-decoration-style: dotted; opacity: .75; }
  .new { animation: rise .45s ease-out; }
  @keyframes rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
  .legend { display: flex; align-items: center; gap: 10px; margin: 14px 4px 26px; color: var(--muted); font-size: 15px; }
  .legend .swatch { width: 34px; height: 4px; background: var(--mauve); border-radius: 2px; }
  .bar { display: flex; gap: 10px; margin-bottom: 14px; align-items: center; }
  .bar h2 { margin: 0; font-size: 20px; flex: 1; }
  button { font-family: var(--font); font-size: 15px; font-weight: 500; border: 0; border-radius: 10px;
           padding: 9px 16px; cursor: pointer; background: var(--frost); color: var(--navy); }
  button.primary { background: var(--digital); color: #fff; }
  button:focus-visible { outline: 3px solid var(--mauve); outline-offset: 2px; }
  .transcript { border-top: 2px solid var(--frost); }
  .entry { display: grid; grid-template-columns: 130px 1fr; gap: 16px; padding: 14px 4px; border-bottom: 1px solid #e8eefc; }
  .entry time { color: var(--muted); font-size: 13px; padding-top: 5px; }
  .entry p { margin: 0; font-size: 21px; line-height: 1.6; }
  .empty { color: var(--muted); padding: 18px 4px; }
  footer { color: var(--muted); font-size: 13px; margin-top: 26px; }
  /* floating caption window */
  body.pip { background: var(--navy); }
  body.pip .captions { border-radius: 0; min-height: 100vh; padding: 16px 20px; }
  body.pip .captions .line { font-size: 30px; }
</style>
</head>
<body>
<header>
  <div class="logo" aria-hidden="true">Lip</div>
  <div><div class="brand">LipLine</div><div class="tag">see what you mean</div></div>
  <div class="spacer"></div>
  <div class="status" id="status" role="status" aria-live="polite"><span class="dot"></span><span id="statusText">Connecting…</span></div>
</header>
<main>
  <section class="captions" id="captions" aria-live="polite" aria-label="Live captions">
    <div class="label">Live captions</div>
    <div class="line" id="line"><span class="waiting">Waiting for the first words…</span></div>
  </section>
  <div class="legend"><span class="swatch"></span>A purple line means the words were read from the speaker's lips, not heard.</div>
  <div class="bar">
    <h2>Transcript</h2>
    <button class="primary" id="float">Float captions</button>
    <button id="download">Download transcript</button>
  </div>
  <section class="transcript" id="transcript"><div class="empty">No captions yet.</div></section>
  <footer>Runs on this computer. Video is processed locally and clips are deleted after reading. Lip reading makes mistakes, so treat captions as a guide.</footer>
</main>
<script>
(function () {
  "use strict";
  var lastVersion = -1, lastCount = 0, data = { entries: [] };
  var lineEl = document.getElementById("line");
  var listEl = document.getElementById("transcript");
  var statusEl = document.getElementById("status");
  var statusText = document.getElementById("statusText");

  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; });
  }
  function wordsHtml(words, animateFrom) {
    return words.map(function (w, i) {
      var cls = "lip" + (w.sure ? "" : " unsure") + (i >= animateFrom ? " new" : "");
      return '<span class="' + cls + '">' + esc(w.w) + "</span>";
    }).join(" ");
  }
  function render() {
    var entries = data.entries || [];
    statusText.textContent = data.status || "Ready";
    statusEl.classList.toggle("busy", /record|read|load/i.test(data.status || ""));
    if (!entries.length) {
      lineEl.innerHTML = '<span class="waiting">Waiting for the first words…</span>';
      listEl.innerHTML = '<div class="empty">No captions yet.</div>';
      lastCount = 0;
      return;
    }
    var last = entries[entries.length - 1];
    var total = entries.reduce(function (n, e) { return n + e.words.length; }, 0);
    var fresh = Math.max(0, last.words.length - Math.max(0, total - lastCount));
    lineEl.innerHTML = last.words.length ? wordsHtml(last.words, fresh) : '<span class="waiting">(nothing recognised)</span>';
    lastCount = total;
    listEl.innerHTML = entries.slice().reverse().map(function (e) {
      return '<div class="entry"><time>' + esc(e.time) + "</time><p>" +
        (e.words.length ? wordsHtml(e.words, 1e9) : '<span class="waiting">(nothing recognised)</span>') + "</p></div>";
    }).join("");
  }
  function poll() {
    fetch("/feed.json", { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(function (d) { if (d.version !== lastVersion) { lastVersion = d.version; data = d; render(); } })
      .catch(function () { statusText.textContent = "Engine stopped"; statusEl.classList.remove("busy"); })
      .finally(function () { setTimeout(poll, 600); });
  }
  poll();

  document.getElementById("download").addEventListener("click", function () {
    var text = (data.entries || []).map(function (e) {
      return "[" + e.time + "] " + e.words.map(function (w) { return w.sure ? w.w : w.w + "?"; }).join(" ");
    }).join("\n");
    var a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([text + "\n"], { type: "text/plain" }));
    a.download = "lipline-transcript.txt";
    document.body.appendChild(a); a.click(); a.remove();
  });

  // "Float captions": an always-on-top window over the meeting (Chrome/Edge Document Picture-in-Picture).
  var captions = document.getElementById("captions");
  var home = captions.parentNode, homeNext = captions.nextSibling;
  document.getElementById("float").addEventListener("click", function () {
    if (window.documentPictureInPicture) {
      documentPictureInPicture.requestWindow({ width: 640, height: 190 }).then(function (pip) {
        Array.prototype.forEach.call(document.querySelectorAll("style, link[rel=stylesheet]"), function (n) {
          pip.document.head.appendChild(n.cloneNode(true));
        });
        pip.document.body.className = "pip";
        pip.document.body.appendChild(captions);
        pip.addEventListener("pagehide", function () { home.insertBefore(captions, homeNext); });
      });
    } else {
      window.open("/?float=1", "lipline-captions", "width=640,height=220");
    }
  });
  if (/float=1/.test(location.search)) {
    document.body.className = "pip";
    document.querySelector("header").style.display = "none";
    Array.prototype.forEach.call(document.querySelectorAll("main > :not(#captions)"), function (n) { n.style.display = "none"; });
    document.querySelector("main").style.padding = "0";
  }
})();
</script>
</body>
</html>
"""


def start_caption_page(feed: CaptionFeed, port: int, open_browser: bool = True) -> None:
    """Serve the caption page on this computer only (127.0.0.1) and open it in the browser."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/feed.json":
                body, kind = feed.to_json().encode("utf-8"), "application/json"
            elif path in ("/", "/index.html"):
                body, kind = CAPTION_PAGE.encode("utf-8"), "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):          # keep the terminal quiet
            pass

    for p in range(port, port + 10):           # skip ports that are already taken
        try:
            server = ThreadingHTTPServer(("127.0.0.1", p), Handler)
            break
        except OSError:
            continue
    else:
        print("Caption page unavailable: no free port.")
        return
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{p}/"
    print(f"LipLine captions: {url}  (open in Chrome or Edge; 'Float captions' keeps them on top of Meet)")
    if open_browser:
        webbrowser.open(url)


# =============================================================================
# Step 4d · Frame sources: webcam, or a region of the screen (a Meet / Zoom video tile)
# =============================================================================
def pick_screen_region(monitor: int = 1) -> dict:
    """Show a screenshot and let you drag a box around the speaker's video tile."""
    import mss

    with mss.mss() as sct:
        mon = sct.monitors[monitor]
        shot = np.ascontiguousarray(np.array(sct.grab(mon))[:, :, :3])
    scale = min(1.0, 1280 / shot.shape[1])
    view = cv2.resize(shot, None, fx=scale, fy=scale) if scale < 1 else shot.copy()
    put(view, "Drag a box around the speaker's face/video, then press ENTER", (10, 30), 0.7, (90, 214, 255), 2)
    x, y, w, h = cv2.selectROI("LipLine - select the speaker", view, showCrosshair=False)
    cv2.destroyAllWindows()
    if w < 20 or h < 20:
        sys.exit("No region selected.")
    region = {"left": mon["left"] + int(x / scale), "top": mon["top"] + int(y / scale),
              "width": int(w / scale) // 2 * 2, "height": int(h / scale) // 2 * 2}
    REGION_FILE.write_text(json.dumps(region))
    print(f"Saved screen region {region} to {REGION_FILE} (use --pick-region to choose again)")
    return region


def screen_region(args) -> dict | None:
    """The region to read for --screen: the saved one, or ask for one."""
    if not getattr(args, "screen", False):
        return None
    if REGION_FILE.exists() and not args.pick_region:
        return json.loads(REGION_FILE.read_text())
    return pick_screen_region(args.monitor)


class FrameSource:
    """Gives BGR frames from the webcam (mirrored) or from a fixed screen region."""

    def __init__(self, camera: int, region: dict | None, mirror: bool = True) -> None:
        self.region, self.cap, self.sct, self.mirror = region, None, None, mirror
        if region:
            import mss
            self.sct = mss.mss()
            self._next = 0.0
        else:
            self.cap = open_camera(camera)

    def read(self):
        if self.cap is not None:
            ok, frame = self.cap.read()
            return ok, (cv2.flip(frame, 1) if ok and self.mirror else frame)  # mirror view, like a selfie
        delay = self._next - time.monotonic()                  # pace screen grabs at ~30 fps
        if delay > 0:
            time.sleep(delay)
        self._next = time.monotonic() + 1 / 30
        return True, np.ascontiguousarray(np.array(self.sct.grab(self.region))[:, :, :3])

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
        if self.sct is not None:
            self.sct.close()


# =============================================================================
# Step 5 · L1 live app: train and read in one window
# =============================================================================
def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {index}. Try --camera 1, or close other apps using it.")
    return cap


def put(img, text, xy, scale=0.55, color=(235, 235, 235), thick=1):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def transcript_text(items: list[tuple[str, bool]]) -> str:
    return " ".join(w if sure else f"[{w}?]" for w, sure in items)


def run_live(args) -> None:
    words = args.words[:9]
    samples_path, transcript_path = Path(args.samples), Path(args.transcript)
    model = WordModel() if args.start_over else WordModel.load(samples_path, words)
    region = screen_region(args)
    tracker, source = FaceTracker(), FrameSource(args.camera, region)
    feed = CaptionFeed()
    if args.captions:
        start_caption_page(feed, args.port, not args.no_browser)
    session = f"Live · {datetime.now():%H:%M}"
    meter, calib, seg = MotionMeter(), Calibrator(), None
    transcript: list[tuple[str, bool]] = []
    armed, armed_at, reading = None, 0.0, True
    message = "Calibrating: keep your mouth closed and still…"

    print("Keys: 1-9 record a sample of that word · SPACE pause/resume reading · "
          "u undo · x clear · c recalibrate · q quit")
    try:
        while True:
            ok, frame = source.read()
            if not ok:
                break
            t = time.monotonic()
            found = tracker.process(frame)

            if found is None:
                meter.reset()
                message = "No face found: face the camera in good light."
            else:
                points, blendshapes = found
                if message.startswith("No face"):            # face is back: clear the stale warning
                    message = "Ready." if model.samples else "Ready. Press 1-9 to teach it words."
                for x, y in points[LIPS]:
                    cv2.circle(frame, (int(x), int(y)), 1, (90, 210, 150), -1)
                features = lip_features(points, blendshapes)
                activity = meter.update(features, t)

                if not calib.done:
                    if calib.update(activity, t):
                        seg = Segmenter(calib.on, calib.off, pause_s=args.pause)
                        message = "Ready." if model.samples else "Ready. Press 1-9 to teach it words."
                else:
                    utterance = seg.update(features, activity, t)
                    if armed and not seg.active and t - armed_at > ARM_TIMEOUT_S:
                        message, armed = f"Didn't catch '{armed}'. Press its key and try again.", None
                    if utterance is not None and len(utterance) >= 4:
                        if armed:                             # training sample
                            model.add(armed, utterance)
                            model.save(samples_path)
                            message = f"Saved sample for '{armed}' ({model.count(armed)} total)"
                            armed = None
                        elif reading and model.samples:       # reading
                            word, margin, runner_up = model.classify(utterance)
                            transcript.append((word, margin >= args.min_margin))
                            transcript_path.write_text(transcript_text(transcript))
                            feed.add_word(session, word, margin >= args.min_margin)
                            message = f"{word}  (margin {margin:.2f}, runner-up {runner_up})"

            feed.set_status("Calibrating" if not calib.done else "Training" if armed
                            else "Reading lips (live)" if reading else "Paused")
            draw_hud(frame, words, model, meter, calib, seg, armed, reading, transcript, message)
            cv2.imshow("lipread - L1", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if ord("1") <= key <= ord(str(len(words))) and calib.done:
                armed, armed_at = words[key - ord("1")], t
                seg.reset()
                message = f"Say '{armed}' now…"
            elif key == ord(" "):
                reading = not reading
            elif key == ord("u") and transcript:
                transcript.pop()
                transcript_path.write_text(transcript_text(transcript))
                feed.pop_word()
            elif key == ord("x"):
                transcript.clear()
                transcript_path.write_text("")
                feed.clear()
            elif key == ord("c"):
                calib, seg, message = Calibrator(), None, "Calibrating: keep your mouth still…"
    finally:
        source.release()
        tracker.close()
        cv2.destroyAllWindows()
        feed.set_status("Stopped")
    print("\nTranscript:", transcript_text(transcript))
    if args.pdf:
        entry = {"time": f"Session ended {datetime.now():%H:%M:%S}",
                 "text": " ".join(w for w, _ in transcript),
                 "uncertain": [i for i, (_, sure) in enumerate(transcript) if not sure]}
        write_transcript_pdf(Path(args.pdf), "Lip-reading transcript (L1, trained words)", [entry],
                             "Grey words ending in ? were uncertain guesses.")


def draw_hud(frame, words, model, meter, calib, seg, armed, reading, transcript, message):
    h, w = frame.shape[:2]
    put(frame, message, (10, 24), 0.6, (90, 214, 255))
    # word list with sample counts
    for i, word in enumerate(words):
        color = (120, 80, 255) if word == armed else (235, 235, 235)
        put(frame, f"{i + 1} {word}: {model.count(word)}", (w - 190, 50 + 22 * i), 0.5, color)
    # motion meter with threshold marker
    x0, y0, bw = 10, 40, 200
    scale = (calib.on * 3) if calib.done else max(meter.value * 1.5, 1e-6)
    cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + 10), (60, 50, 45), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + int(bw * min(meter.value / scale, 1)), y0 + 10), (150, 210, 95), -1)
    if calib.done:
        tx = x0 + int(bw * min(calib.on / scale, 1))
        cv2.line(frame, (tx, y0 - 3), (tx, y0 + 13), (120, 80, 255), 2)
    state = "speaking" if seg and seg.active else ("reading" if reading else "paused")
    put(frame, state, (x0 + bw + 8, y0 + 10), 0.5)
    # transcript (last ~60 characters)
    cv2.rectangle(frame, (0, h - 40), (w, h), (30, 25, 20), -1)
    put(frame, transcript_text(transcript)[-60:], (10, h - 14), 0.7)


# =============================================================================
# Step 6 · L2: record a clip and run the pretrained model
# =============================================================================
def record_clip(out: str, seconds: float, camera: int, fps: int = 25,
                region: dict | None = None, feed: CaptionFeed | None = None) -> Path:
    """Record `seconds` of video (webcam, or a screen region) and save it at exactly `fps` fps.
    In screen mode there is no preview window, so nothing covers the video being read."""
    source = FrameSource(camera, region, mirror=False)       # save the video unmirrored
    status = feed.set_status if feed else (lambda _s: None)
    try:
        t0 = time.monotonic()
        while (left := 3 - (time.monotonic() - t0)) > 0:      # countdown
            ok, frame = source.read()
            if not ok:
                sys.exit("Video source stopped delivering frames.")
            status(f"Recording in {int(left) + 1}…")
            if region is None:
                view = cv2.flip(frame, 1)
                put(view, f"Recording in {int(left) + 1}…", (10, 30), 0.9, (90, 214, 255), 2)
                cv2.imshow("lipread - record", view)
                cv2.waitKey(1)
        print("Recording…")
        status("Recording")
        frames, times = [], []
        t0 = time.monotonic()
        while (t := time.monotonic() - t0) < seconds:
            ok, frame = source.read()
            if not ok:
                break
            frames.append(frame)
            times.append(t)
            if region is None:
                view = cv2.flip(frame, 1)
                put(view, f"REC  speak a full sentence ({seconds - t:.0f}s)", (10, 30), 0.8, (80, 80, 255), 2)
                cv2.imshow("lipread - record", view)
                cv2.waitKey(1)
    finally:
        source.release()
        cv2.destroyAllWindows()
    if not frames:
        sys.exit("No frames recorded.")

    # Webcams rarely deliver exactly 25 fps, so pick the nearest real frame for each 1/25 s.
    times = np.array(times)
    targets = np.arange(int(seconds * fps)) / fps
    picks = np.clip(np.searchsorted(times, targets), 0, len(frames) - 1)
    h, w = frames[0].shape[:2]
    out_path = Path(out)
    raw_path = out_path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in picks:
        writer.write(frames[i])
    writer.release()

    if shutil.which("ffmpeg"):                               # re-encode to widely readable H.264
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_path), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-r", str(fps), str(out_path)], check=True)
        raw_path.unlink()
    else:
        raw_path.replace(out_path)
    print(f"Saved {out_path} ({len(picks)} frames at {fps} fps)")
    return out_path


def vsr_python() -> Path:
    """Path to the Python inside .venv-vsr (Windows and macOS/Linux layouts)."""
    win, unix = VSR_ENV / "Scripts" / "python.exe", VSR_ENV / "bin" / "python"
    return win if win.exists() or os.name == "nt" else unix


def uv_command() -> list[str]:
    """`uv` if it's on PATH, otherwise `python -m uv` (installing uv first if needed)."""
    if shutil.which("uv"):
        return ["uv"]
    if subprocess.run([sys.executable, "-m", "uv", "--version"], capture_output=True).returncode != 0:
        print("Installing uv…")
        subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
    return [sys.executable, "-m", "uv"]


def run_step(title: str, cmd: list[str]) -> None:
    print(f"\n== {title}\n   " + " ".join(cmd))
    if subprocess.run(cmd).returncode != 0:
        sys.exit(f"Step failed: {title}. Read the error above.")


def download_repo(repo: Path) -> None:
    if (repo / "pipelines").is_dir():
        print(f"\n== Model code already in {repo}/")
        return
    if shutil.which("git"):
        run_step("Downloading the model code (git)", ["git", "clone", "--depth", "1", VSR_REPO_GIT, str(repo)])
        return
    print("\n== Downloading the model code (zip)")
    zip_path = Path("vsr_repo.zip")
    urllib.request.urlretrieve(VSR_REPO_ZIP, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        top = z.namelist()[0].split("/")[0]
        z.extractall(".")
    Path(top).rename(repo)
    zip_path.unlink()


def fix_ctc_type(repo: Path) -> None:
    """The model files may ask for the warpctc training library; inference only needs the
    built-in CTC, so switch it to avoid an import error."""
    conf = repo / "benchmarks/LRS3/models/LRS3_V_WER19.1/model.json"
    if conf.exists():
        text = conf.read_text(encoding="utf-8")
        if '"warpctc"' in text:
            conf.write_text(text.replace('"warpctc"', '"builtin"'), encoding="utf-8")
            print("Switched ctc_type to 'builtin' in model.json")


def missing_model_files(repo: Path) -> list[str]:
    return [rel for rel in VSR_MODEL_FILES if not (repo / rel).is_file()]


def report_model_files(repo: Path) -> bool:
    missing = missing_model_files(repo)
    if not missing:
        fix_ctc_type(repo)
        print("\n== Model files: all present ✓")
        return True
    print("\n== Model files still needed. Download these zips and extract them so the files land here:")
    shown = set()
    for rel in missing:
        name, url = VSR_MODEL_FILES[rel]
        if name not in shown:
            print(f"\n   {name}: {url}")
            shown.add(name)
        print(f"     -> {repo / rel}")
    print("\nThen run  python lipread.py setup-l2  again to check.")
    return False


def run_setup_l2(args) -> None:
    uv = uv_command()
    py = vsr_python()
    if py.exists():
        print(f"== Environment {VSR_ENV}/ already exists (delete the folder to rebuild it)")
    else:
        run_step(f"Creating {VSR_ENV}/ with Python {VSR_PYTHON_VERSION}",
                 uv + ["venv", str(VSR_ENV), "--python", VSR_PYTHON_VERSION])
    index = "https://download.pytorch.org/whl/" + ("cu121" if args.cuda else "cpu")
    run_step("Installing PyTorch (large download)",
             uv + ["pip", "install", "--python", str(py), *TORCH_PACKAGES, "--index-url", index])
    run_step("Installing the model's other packages", uv + ["pip", "install", "--python", str(py), *VSR_PACKAGES])
    run_step("Checking the environment", [str(py), "-c",
             "import distutils, torch, torchvision, mediapipe; "
             "assert mediapipe.solutions; print('   torch', torch.__version__, '· GPU:', torch.cuda.is_available())"])
    download_repo(Path(args.repo))
    if report_model_files(Path(args.repo)):
        print("\nL2 is ready:  python lipread.py transcribe --record 6 --loop")


# The model runs in the repo's own Python environment (it needs PyTorch and an older
# MediaPipe), so we start it as a separate process that loads the model once and then
# transcribes every clip path we send it on stdin.
VSR_WORKER = r"""
import sys, torch
from pipelines.pipeline import InferencePipeline
config, want_gpu = sys.argv[1], sys.argv[2] == "1"
device = torch.device("cuda:0" if want_gpu and torch.cuda.is_available() else "cpu")
print("INFO:loading model on " + str(device), flush=True)
pipeline = InferencePipeline(config, device=device, detector="mediapipe", face_track=True)
print("READY", flush=True)
for line in sys.stdin:
    clip = line.strip()
    if not clip:
        continue
    try:
        print("HYP:" + str(pipeline(clip)).strip(), flush=True)
    except Exception as e:
        print("ERR:" + repr(e).replace("\n", " "), flush=True)
"""


class VSRModel:
    def __init__(self, repo: str, config: str, python: str, cpu: bool) -> None:
        repo_path = Path(repo).resolve()
        if not (repo_path / "pipelines").is_dir():
            sys.exit(f"{repo_path} doesn't contain the model code. Run  python lipread.py setup-l2  first.")
        if not (repo_path / config).is_file():
            sys.exit(f"Config {config} not found inside {repo_path}.")
        if not Path(python).exists() and not shutil.which(python):
            sys.exit(f"{python} not found. Run  python lipread.py setup-l2  first.")
        if missing_model_files(repo_path) and config == VSR_CONFIG:
            report_model_files(repo_path)
            sys.exit(1)
        fix_ctc_type(repo_path)
        if Path(python).exists():
            python = str(Path(python).resolve())        # the worker runs inside the repo folder
        # PYTHONUTF8=1 makes the model's open() calls read files as UTF-8. Without it, Windows
        # uses its old cp1252 encoding and fails on the tokenizer's word list.
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        self.proc = subprocess.Popen([python, "-c", VSR_WORKER, config, "0" if cpu else "1"],
                                     cwd=repo_path, env=env, text=True, encoding="utf-8",
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        print("Loading the lip-reading model (this can take a minute)…")
        self._read_until("READY")

    def _read_until(self, prefix: str) -> str:
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            if line.startswith(prefix):
                return line[len(prefix):]
            if line.startswith("ERR:"):
                print("Model error:", line[4:])
                return ""
            if line.startswith("INFO:"):
                print(line[5:])
        sys.exit("The model process stopped. Read the error printed above.")

    def transcribe(self, clip: Path) -> str:
        print("Reading lips…")
        self.proc.stdin.write(str(Path(clip).resolve()) + "\n")
        self.proc.stdin.flush()
        return self._read_until("HYP:")

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)


def run_transcribe(args) -> None:
    region = screen_region(args) if args.record > 0 else None
    feed = CaptionFeed()
    if args.captions:
        start_caption_page(feed, args.port, not args.no_browser)
    feed.set_status("Loading model")
    model = VSRModel(args.repo, args.config, args.python, args.cpu)
    repeat = args.auto or args.loop
    entries, n = [], 0
    try:
        while True:
            n += 1
            if args.record > 0:
                clip = record_clip(args.clip, args.record, args.camera, region=region, feed=feed)
            else:
                clip = Path(args.clip)
            feed.set_status("Reading lips")
            text = model.transcribe(clip)
            if args.record > 0 and not args.keep_clips:
                Path(clip).unlink(missing_ok=True)            # privacy: no video is kept
            label = f"{datetime.now():%H:%M:%S}"
            print(f"\n>>> {text or '(nothing recognised)'}\n")
            feed.add_entry(label, text)
            feed.set_status("Ready")
            entries.append({"time": f"Clip {n} · {label}", "text": text})
            with open(args.transcript, "a", encoding="utf-8") as f:
                f.write(text + "\n")
            if args.pdf:
                write_transcript_pdf(Path(args.pdf), "LipLine transcript (Auto-AVSR)", entries,
                                     "Visual-only speech recognition. Expect some wrong words.")
            if not (repeat and args.record > 0):
                break
            if not args.auto and input("Press Enter to record another clip, "
                                       "or type q then Enter to finish: ").strip().lower() == "q":
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        feed.set_status("Stopped")
        model.close()
    if args.captions:
        try:
            input("Captions page is still open. Press Enter to close it… ")
        except EOFError:
            pass


# =============================================================================
# Command line
# =============================================================================
def add_capture_options(p) -> None:
    p.add_argument("--screen", action="store_true", help="read a video tile on screen (Meet/Zoom) instead of the webcam")
    p.add_argument("--pick-region", action="store_true", help="choose the screen region again")
    p.add_argument("--monitor", type=int, default=1, help="which monitor to capture with --screen")
    p.add_argument("--no-captions", dest="captions", action="store_false", help="don't start the caption page")
    p.add_argument("--no-browser", action="store_true", help="start the caption page without opening a browser")
    p.add_argument("--port", type=int, default=CAPTION_PORT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    live = sub.add_parser("live", help="L1: train your own words and read them live")
    live.add_argument("--words", nargs="+", required=True, help="up to 9 words (keys 1-9)")
    live.add_argument("--samples", default="lip_samples.json")
    live.add_argument("--transcript", default="lip_transcript.txt")
    live.add_argument("--start-over", action="store_true", help="ignore saved samples")
    live.add_argument("--min-margin", type=float, default=0.08, help="below this, a word is marked uncertain")
    live.add_argument("--pause", type=float, default=0.45, help="seconds of stillness that end a word")
    live.add_argument("--camera", type=int, default=0)
    live.add_argument("--pdf", default="lip_transcript.pdf", help="PDF written when you quit ('' to skip)")
    add_capture_options(live)

    rec = sub.add_parser("record", help="L2: record a 25 fps clip")
    rec.add_argument("--seconds", type=float, default=6)
    rec.add_argument("--out", default="clip.mp4")
    rec.add_argument("--camera", type=int, default=0)

    setup = sub.add_parser("setup-l2", help="L2: one-time setup of the model environment")
    setup.add_argument("--repo", default=str(VSR_REPO), help="where to put the model code")
    setup.add_argument("--cuda", action="store_true", help="install the NVIDIA GPU build of PyTorch")

    tr = sub.add_parser("transcribe", help="L2: run the pretrained model on a clip")
    tr.add_argument("--repo", default=str(VSR_REPO), help="folder with the model code (from setup-l2)")
    tr.add_argument("--clip", default="clip.mp4")
    tr.add_argument("--record", type=float, default=0, help="record this many seconds first")
    tr.add_argument("--config", default=VSR_CONFIG)
    tr.add_argument("--python", default=None, help="python of the model environment (default: .venv-vsr)")
    tr.add_argument("--cpu", action="store_true", help="force CPU even if a GPU is available")
    tr.add_argument("--loop", action="store_true", help="keep recording and transcribing clips")
    tr.add_argument("--transcript", default="vsr_transcript.txt")
    tr.add_argument("--pdf", default="vsr_transcript.pdf", help="PDF updated after every clip ('' to skip)")
    tr.add_argument("--camera", type=int, default=0)
    tr.add_argument("--auto", action="store_true", help="record and transcribe clips nonstop (Ctrl+C to stop)")
    tr.add_argument("--keep-clips", action="store_true", help="keep recorded clips (deleted by default)")
    add_capture_options(tr)

    args = parser.parse_args()
    if args.command == "live":
        run_live(args)
    elif args.command == "record":
        record_clip(args.out, args.seconds, args.camera)
    elif args.command == "setup-l2":
        run_setup_l2(args)
    elif args.command == "transcribe":
        args.python = args.python or str(vsr_python().resolve())
        run_transcribe(args)


if __name__ == "__main__":
    main()
