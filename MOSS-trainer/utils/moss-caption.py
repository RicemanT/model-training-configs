#!/usr/bin/env python3
"""
MOSS-Image captioning for the MageTrail dataset, through any of four OpenAI-compatible providers.

Writes captions in MOSS-Image's own format (OpenMOSS USERGUIDE.md, "Image Captioning"): one JSON object per
image with three caption lengths, the watermarks with their boxes, and the borders:

    <image>_moss.json   {"caption": {"long", "medium", "short"}, "watermark": [{"box_2d", "label"}],
                         "borders": [...], "source": {...}}
    <image>_nl.txt      caption.long, for the Studio's caption editor, counts and caption check

Captions never name the artist: the trigger (e.g. "@artist name") is added at training time, so its form can
still change. Watermark boxes are [y_min, x_min, y_max, x_max] normalized to 0-1000 of the full image, so they
do not depend on how the image was resized; the trainer converts them for each training crop.

MODE
  "layout"   images without a `_layout.json`: a MOSS layout (summary + character / object / text items with
             boxes, MOSS's prompt-enhancer vocabulary) in <image>_layout.json.
             - Text and watermark boxes come from a local detector (TEXT_ENGINE). Its regions are fused first
               where they overlap or repeat one another's text (the bigger box and the longer text win), then
               numbered for the model, which says what each region is (text, watermark, or a false detection),
               reads its text itself and captions it. Text or watermarks no region covers are boxed by the
               model (MODEL_TEXT_BOXES); where its box and a detector box are one block, the detector's box
               wins. Watermark boxes also replace the `watermark` field of <image>_moss.json (WATERMARKS_TO_MOSS).
             - Character details (accessories, worn weapons, distinctive outfit pieces, hair ornaments) are
               object items with "part_of" = the index of their character item in the layout, so training can
               keep or drop them (MOSS itself only sees plain objects).
             - Character and object boxes are drawn by the model; in comics and collages it boxes every
               appearance of a character panel by panel, even when the same character repeats.
             - Panels (MageTrail's extension of MOSS's vocabulary, label "panel"): the panel detector's boxes
               (DETECT_PANELS), numbered for the model, which writes one captioned panel item per real panel
               and boxes panels the detector missed itself.
             - Which images: every image whose tags show text or a watermark (TEXT_TAGS: speech bubble,
               english text, sound effects, watermark, artist name...), plus a random LAYOUT_SHARE of the rest.
             - No two items keep the same box: overlapping text items are fused, and near-identical boxes of the
               same label keep one.
             TEXT_ENGINE "anzhc" (GPU): Anzhc's illustration text segmenter (speech bubbles, titles, sound
             effects, signs, watermarks). Needs: pip install ultralytics huggingface_hub, plus a CUDA torch.
             "anzhc_vl" also reads each region with PaddleOCR-VL-1.6 (pip install "transformers>=5"): only shown
             in the Studio's Detections layer, never sent to the model. "rapidocr": PP-OCR lines on the CPU
             (pip install rapidocr onnxruntime). DETECT_FACES / DETECT_PANELS (off) add Anzhc's face and panel
             segmenters as hints. The models download once from Hugging Face.
  "shorten"  images with a `_layout.json` whose items lack a `short` caption (layouts made before short
             captions existed): a text-only call writes them from the full item captions.
  "detect"   only the local detectors (no API calls, no key): fills <image>_ocr.json for every image, to inspect
             in the Studio's "Detections" layer; "layout" reuses the files. Run it on a GPU box.
  "caption"  images without a `_moss.json`: the image plus its tags and character facts (the "original caption
             reference" of MOSS's prompt) -> the full JSON. Images that already have a MageTrail `_nl.txt` are
             left to "convert" unless CAPTION_OVER_OLD_NL = True.
  "convert"  images with a MageTrail `_nl.txt` but no `_moss.json`: a text-only call (no image, so cheap and
             rarely refused) turns the old caption into long/medium/short. Watermarks and borders are unknown
             for these (null) until the layout pass looks at the image.

PROVIDER  "a6api" | "linkapi" | "openrouter" | "opencode": endpoint, key variable, model and error handling.
Keys come from the environment only (export A6API_KEY=... etc.); never put one in this file.

Safety (as in the per-provider scripts): a stop on errors that every call would repeat (key, balance, token
quota, price protection, usage limit), backoff on 429 and 5xx, no retry of refused requests, a per-folder guard
that gives up on a folder whose first images all fail, a stop after many failures in a row, and MAX_SPEND.
"""
import asyncio
import base64
import io
import json
import os
import re
import threading
import time
from datetime import datetime, timezone

from openai import (APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, InternalServerError,
                    RateLimitError)
from PIL import Image
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

# ---- PROVIDER ----
PROVIDER = "openrouter"
PROVIDERS = {
    "a6api":      {"base_url": "https://api.a6api.com/v1", "key_env": "A6API_KEY", "model": "gemini-3.8-flash"},
    "linkapi":    {"base_url": "https://linkapi.ai/v1", "key_env": "LINKAPI_API_KEY", "model": "gemini-3.8-flash"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY",
                   "model": "meta/muse-spark-1.3-contributor"},
    "opencode":   {"base_url": "https://opencode.ai/inference/go/v1", "key_env": "OPENCODE_API_KEY",
                   "model": "muse-spark-1.3-contributor", "requests_per_minute": 30},
}
MODEL = None                   # None = the provider's default above
REASONING_EFFORT = "medium"    # MOSS recommends thinking; "low"/"medium"/"high" or None (dropped if refused)
USER_AGENT = "magetrail-captioner/1.0"

MODE = "shorten"               # "caption", "convert", "layout", "shorten" or "detect" (see the top of this file)
CAPTION_OVER_OLD_NL = False    # caption mode: also re-caption images that have a MageTrail _nl.txt

# Folders to caption, walked recursively. Library images live in <library>/images/<group>/<artist>/.
IMAGE_ROOTS = ["/home/jovyan/Artist-Collection-Dataset/images"]
ONLY_PATHS = ["gelbooru"]   # only images whose path contains one of these strings; [] = all
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
CHARACTER_FACTS_FILE = "/home/jovyan/Artist-Collection-Dataset/planner/exports/character-facts.json"
PROMPT_FILE = "moss-caption-prompt.md"
CONVERT_PROMPT_FILE = "moss-convert-prompt.md"
JSON_SUFFIX = "_moss"          # <image>_moss.json
LAYOUT_PROMPT_FILE = "moss-layout-prompt.md"
LAYOUT_SUFFIX = "_layout"      # <image>_layout.json
OCR_SUFFIX = "_ocr"            # <image>_ocr.json (cached OCR lines)
LAYOUT_SHARE = 1               # layout mode: this share of images (chosen by a stable hash of folder + name)...
LAYOUT_TEXT_ALWAYS = True      # ...plus every image whose tags show text or a watermark (TEXT_TAGS below)
MODEL_TEXT_BOXES = True        # text / watermarks / panels no detection covers: True = the model boxes them, False = left out
WATERMARKS_TO_MOSS = True      # confirmed watermark boxes replace the `watermark` field of <image>_moss.json
MERGE_OVERLAP = 0.6            # boxes this much inside one another are the same block and get fused
OCR_WORKERS = 4                # OCR engines running in parallel (CPU threads)
OCR_MIN_SCORE = 0.6            # OCR lines below this confidence are dropped
TEXT_ENGINE = "anzhc"          # text/watermark boxes: "anzhc" (GPU), "anzhc_vl" or "rapidocr" (CPU); see the top
DETECT_FACES = False           # GPU (or CPU, slowly): faces listed for the model as anchors of character boxes
DETECT_PANELS = True           # GPU: the panel detector's boxes become the layout's panel items
DETECTOR_DEVICE = "cuda:2"     # "cuda:2" = GPU 2 (as nvidia-smi numbers them), "auto" = the GPU with the most
                               # free memory, or "cpu"
MIN_FREE_GPU_GB = 4            # "auto" only picks a GPU with at least this much free memory
YOLO_REPO = "Anzhc/Anzhcs_YOLOs"            # model names below are files of this repo, or local .pt paths
TEXT_SEG_MODEL = "Anzhcs-text-seg-v9-y11m.pt"
FACE_MODEL = "Anzhc Face seg 640 v4 y11n.pt"
PANEL_MODEL = "Anzhc Manga Panels -seg.pt"
VL_MODEL = "PaddlePaddle/PaddleOCR-VL-1.6"   # or a local folder
TEXT_MIN_CONF = 0.35           # detections below these confidences are dropped
FACE_MIN_CONF = 0.45
PANEL_MIN_CONF = 0.5
VL_BATCH = 16                  # anzhc_vl: text regions read per PaddleOCR-VL call
VL_MAX_TOKENS = 256
MAX_LAYOUT_OBJECTS = 24       # object items per layout, character details included
MAX_DETAILS_PER_CHARACTER = 6  # distinctive looks + pose details (hands, legs) per character
NL_SUFFIX = "_nl"              # caption.long is also written to <image>_nl.txt

# Spend: OpenRouter reports each call's cost; the others are estimated from tokens if prices are set (per 1M
# tokens, in your account currency). MAX_SPEND stops the run at that amount (None = no limit).
PRICE_INPUT = None
PRICE_OUTPUT = None
MAX_SPEND = None

MAX_CONCURRENCY = 50
REQUESTS_PER_MINUTE = None     # None = the provider's default pace (only OpenCode Go has one)
LIMIT = None                   # stop after this many images (e.g. 20 for a first test); None = all
TEMPERATURE = 1.0              # MOSS's recommended request settings
TOP_P = 0.95
MAX_TOKENS = 16384

STRIP_TAGS = {"masterpiece", "best quality", "low quality", "very aesthetic", "aesthetic", "normal quality",
              "worst quality"}
MIN_LONG_WORDS = 60
RETRY_ON_BLOCK_ATTEMPTS = 3    # extra attempts after a refusal, a blank or an invalid JSON answer
FOLDER_FAIL_LIMIT = 0          # give up on a folder whose first N images all fail; 0 = never (mixed SFW/NSFW folders)
STOP_AFTER_FAILURES_IN_A_ROW = 0   # stop the run after N failures in a row; 0 = never
RATE_LIMIT_MAX_ATTEMPTS = 5
MAX_IMAGE_DIM = 2048
MAX_RAW_BYTES = 6_000_000

# ---------------------------------------------------------------------------------------------------------
_P = PROVIDERS[PROVIDER]
API_KEY = os.environ.get(_P["key_env"], "")
MODEL = MODEL or _P["model"]
PACE = REQUESTS_PER_MINUTE or _P.get("requests_per_minute")
# Detect mode calls no provider and keeps its own logs, so its failures never block caption or layout runs.
# Per provider AND model: a second model on the same provider (e.g. Muse after Gemini on A6API) retries what the
# first one failed instead of skipping it.
_LOG_NAME = PROVIDER + "_" + re.sub(r"[^A-Za-z0-9.-]+", "-", MODEL).strip("-")
OUTPUT_FILE = "moss_detect_progress.tsv" if MODE == "detect" else "moss_" + _LOG_NAME + "_progress.tsv"
FAILED_LOG = "moss_detect_failed_images.log" if MODE == "detect" else "moss_" + _LOG_NAME + "_failed_images.log"

client = AsyncOpenAI(base_url=_P["base_url"], api_key=API_KEY or "missing", max_retries=0,
                     default_headers={"User-Agent": USER_AGENT})
_options = {"reasoning_effort": REASONING_EFFORT}
_spent = {"cost": 0.0, "calls": 0, "input": 0, "output": 0}
_pace = {"next": 0.0, "lock": None}
_failed = set()
_folders = {}


def load_text(path):
    if not os.path.isfile(path):
        raise FileNotFoundError("Prompt file not found: " + path + " (keep it next to this script)")
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


# ---- request options, pacing and spend ----
def request_options():
    if not _options["reasoning_effort"]:
        return {}
    if PROVIDER == "openrouter":
        return {"extra_body": {"usage": {"include": True}, "reasoning": {"effort": _options["reasoning_effort"]}}}
    return {"reasoning_effort": _options["reasoning_effort"]}


async def pace():
    if not PACE:
        return
    if _pace["lock"] is None:
        _pace["lock"] = asyncio.Lock()
    async with _pace["lock"]:
        now = time.monotonic()
        wait = _pace["next"] - now
        _pace["next"] = max(now, _pace["next"]) + 60.0 / PACE
    if wait > 0:
        await asyncio.sleep(wait)


def record_cost(response):
    usage = getattr(response, "usage", None)
    _spent["calls"] += 1
    if usage is None:
        return
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    output = getattr(usage, "completion_tokens", 0) or 0
    _spent["input"] += prompt
    _spent["output"] += output
    reported = getattr(usage, "cost", None)
    if reported is None:
        reported = (getattr(usage, "model_extra", None) or {}).get("cost")
    if reported is not None:
        _spent["cost"] += float(reported)
    elif PRICE_INPUT is not None and PRICE_OUTPUT is not None:
        _spent["cost"] += (prompt * PRICE_INPUT + output * PRICE_OUTPUT) / 1e6


def over_budget():
    return MAX_SPEND is not None and _spent["cost"] >= MAX_SPEND


def spent_text():
    cost = ("~" + format(_spent["cost"], ".4f") + ", ") if (PRICE_INPUT is not None or PROVIDER == "openrouter") else ""
    return (cost + format(_spent["input"], ",") + " in / " + format(_spent["output"], ",") + " out tokens over "
            + str(_spent["calls"]) + " calls")


# ---- errors ----
class StopRun(RuntimeError):
    """Every further call would fail the same way: stop the run."""


class PromptBlocked(ValueError):
    """The provider refused the request itself: the same request gets the same answer."""


STOP_CODES = {   # new-api / one-api gateways (A6API, LinkAPI)
    "invalid_api_key": "The provider rejected the API key",
    "insufficient_user_quota": "Account balance is used up: top up",
    "token_quota_exhausted": "This token's allowance is used up: raise it in the provider's token settings",
    "model_not_allowed": "This token may not use the model: check its model restrictions",
    "access_denied": "The token's source-IP restriction blocks this machine",
    "group_access_denied": "The token's group restriction blocks this model",
    "marketplace_price_guard_triggered": "Price protection: the pinned route now costs more than your cap",
    "smart_route_all_candidates_over_cap": "Price protection: every route for this model is above your cap",
}
USAGE_LIMIT = re.compile(r"usage limit|limit (?:reached|exceeded)|quota|subscription|billing|insufficient|upgrade",
                         re.IGNORECASE)


def _detail(exc):
    return " ".join(str(getattr(exc, "message", "") or exc).split())[:240]


def _code(exc):
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        inner = body.get("error") if isinstance(body.get("error"), dict) else body
        return str(inner.get("code") or "")
    return ""


def _request_id(exc):
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    return headers.get("X-Oneapi-Request-Id") or headers.get("x-request-id") or ""


def status_error(exc):
    status, code, rid, detail = getattr(exc, "status_code", None), _code(exc), _request_id(exc), _detail(exc)
    tag = " [" + str(status) + (" " + code if code else "") + (", request " + rid if rid else "") + "]"
    if code in STOP_CODES:
        return StopRun(STOP_CODES[code] + tag + ": " + detail)
    if status in (401, 402) or (status in (403, 429) and USAGE_LIMIT.search(detail)):
        return StopRun("The provider refused the account, balance or usage limit" + tag + ": " + detail)
    if status == 404:
        return StopRun("Unknown model (404), check MODEL" + tag + ": " + detail)
    if status in (400, 403, 413, 422):
        return PromptBlocked("Request refused" + tag + ": " + detail)
    return ValueError("HTTP" + tag + ": " + detail)


def _transient(exc):
    if isinstance(exc, RateLimitError):
        return not USAGE_LIMIT.search(_detail(exc)) or _code(exc) in ("user_concurrency_limit", "model_request_rate_limit")
    return isinstance(exc, (InternalServerError, APIConnectionError, APITimeoutError))


@retry(wait=wait_random_exponential(min=2, max=60), stop=stop_after_attempt(RATE_LIMIT_MAX_ATTEMPTS),
       retry=retry_if_exception(_transient), reraise=True)
async def _create(**kwargs):
    await pace()
    return await client.chat.completions.create(**request_options(), **kwargs)


# ---- refusals (apostrophe-agnostic, as in the per-provider scripts) ----
def _norm(t):
    for a in ("’", "‘", "´", "`", "ʼ"):
        t = t.replace(a, "'")
    return t.replace("“", '"').replace("”", '"')


QUOTED = re.compile(r'"[^"]*"')
TASK_VERB = r"(?:create|describe|provide|generate|caption|depict|write|assist|comply|fulfill|produce|do\s+that|help)"
REFUSAL_I = re.compile(
    r"\bi(?:m|\s+am)?\s+unable\b"
    r"|\bi\s+(?:wont|will\s+not|cant|cannot|refuse\s+to|must\s+decline)\b[^.!?]{0,40}?" + TASK_VERB +
    r"|\bi\s*(?:m|am)?\s+not\s+going\s+to\b[^.!?]{0,40}?" + TASK_VERB +
    r"|\bi\s+(?:dont|do\s+not)\s+feel\s+comfortable\b|\bi\s+apologize\b", re.IGNORECASE)
PREFIXES = ("im sorry", "i cannot", "i cant", "i am unable", "as an ai", "sorry but i", "i must decline",
            "im not able", "i am not able", "i am programmed", "im not comfortable", "i apologize", "this request",
            "i wont", "i will not", "i refuse", "illegal", "i dont think i can", "im not going", "i dont feel")
META = ["against my safety", "against my programming", "against my guidelines", "against my rules", "violates my",
        "safety guidelines", "content policy", "sexual content involving minors", "as an ai"]
PROVIDER_BLOCK = re.compile(
    r"prompt could not be submitted|contains sensitive words|prohibited use policy|generative-ai/use-policy"
    r"|request was blocked|blocked due to|violates? (?:google|our|the) |content management policy", re.IGNORECASE)


def is_refusal(text):
    if not text:
        return None
    m = PROVIDER_BLOCK.search(text)
    if m:
        return "provider block: " + m.group(0)
    u = QUOTED.sub("", _norm(text.strip())).lower().replace("'", "")
    m = REFUSAL_I.search(u)
    if m:
        return "I+refusal verb: " + " ".join(m.group(0).split())
    for p in PREFIXES:
        if u.startswith(p):
            return "prefix: " + p
    for p in META:
        if p in u:
            return "meta: " + p
    return None


# ---- the JSON answer ----
BORDERS = {"left", "right", "up", "down"}


def parse_answer(text, need_layout_fields=True):
    """The model's JSON object, validated; raises ValueError (retried) with what is wrong."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        trig = is_refusal(raw)
        raise ValueError(("Refusal (" + trig + ")" if trig else "No JSON object in the answer") + ": "
                         + " ".join(raw.split())[:120])
    try:
        data = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON (" + str(exc) + ")") from exc
    caption = data.get("caption")
    if not isinstance(caption, dict):
        raise ValueError("JSON has no caption object")
    for key in ("long", "medium", "short"):
        value = caption.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("caption." + key + " is missing or empty")
        caption[key] = " ".join(value.split())
        trig = is_refusal(caption[key])
        if trig:
            raise ValueError("Refusal in caption." + key + " (" + trig + ")")
    words = len(caption["long"].split())
    if words < MIN_LONG_WORDS:
        raise ValueError("caption.long too short to be a caption (" + str(words) + " words)")
    out = {"caption": {k: caption[k] for k in ("long", "medium", "short")}}
    if need_layout_fields:
        marks = data.get("watermark", [])
        if not isinstance(marks, list):
            raise ValueError("watermark is not a list")
        clean = []
        for mark in marks:
            box = mark.get("box_2d") if isinstance(mark, dict) else None
            if (not isinstance(box, list) or len(box) != 4
                    or not all(isinstance(v, (int, float)) and -5 <= v <= 1005 for v in box)):
                raise ValueError("watermark box is not [y_min, x_min, y_max, x_max] in 0-1000: " + json.dumps(mark)[:80])
            y0, x0, y1, x1 = (min(1000, max(0, round(v))) for v in box)
            if y1 <= y0 or x1 <= x0:
                raise ValueError("watermark box has no area: " + json.dumps(box))
            clean.append({"box_2d": [y0, x0, y1, x1], "label": str(mark.get("label") or "").strip()})
        borders = data.get("borders", ["none"])
        if (not isinstance(borders, list) or not borders
                or (borders != ["none"] and not set(borders) <= BORDERS)):
            raise ValueError("borders must be [\"none\"] or a list of left/right/up/down: " + json.dumps(borders))
        out.update(watermark=clean, borders=sorted(set(borders)) if borders != ["none"] else ["none"])
    return out


# ---- character facts and tags (as in the per-provider scripts) ----
def load_character_facts(path):
    if not path or not os.path.isfile(path):
        print("Character facts: none (" + str(path) + " not found) -- tags only.")
        return {}
    with open(path, encoding="utf-8") as f:
        document = json.load(f)
    characters = document.get("characters", {})
    facts = dict(characters.get("e621", {}))
    facts.update(characters.get("danbooru", {}))
    print("Character facts: " + str(len(facts)) + " characters from " + path)
    return facts


def describe_character(tag, fact):
    line = "- " + tag + ": " + fact["name"]
    variant = [q for q in fact.get("qualifiers", []) if q not in fact.get("series", [])]
    if variant:
        line += " (" + ", ".join(variant) + ")"
    series = fact.get("series", [])
    if series:
        line += ", an original character" if series == ["original"] else ", from " + " / ".join(series)
    if fact.get("appearance"):
        line += ". Usually drawn with: " + ", ".join(fact["appearance"])
    return line + "."


def read_tags(image_path):
    path = os.path.splitext(image_path)[0] + ".txt"
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        tags = [t.strip() for t in f.read().split(",") if t.strip()]
    # The artist trigger (the folder's first tag, e.g. "drawn by x" or "@x") never reaches the captioner.
    if tags and (tags[0].lower().startswith("drawn by ") or tags[0].startswith("@")):
        tags = tags[1:]
    return [t for t in tags if t.lower() not in STRIP_TAGS]


def reference_text(tags):
    if not tags:
        return "There are no tags for this image; describe it from the image alone."
    text = "<tags>\n" + ", ".join(tags) + "\n</tags>\n"
    known = [(t, CHARACTER_FACTS[t.lower()]) for t in tags if t.lower() in CHARACTER_FACTS]
    if known:
        text += "<characters>\n" + "\n".join(describe_character(t, f) for t, f in known) + "\n</characters>\n"
    return text


# ---- images ----
def encode_image(path):
    need_shrink = os.path.getsize(path) > MAX_RAW_BYTES
    with Image.open(path) as im:
        need_shrink = need_shrink or max(im.size) > MAX_IMAGE_DIM
    if need_shrink:   # boxes are normalized 0-1000, so shrinking does not move them
        with Image.open(path) as im:
            if im.mode in ("RGBA", "LA", "P"):
                rgba = im.convert("RGBA")
                im = Image.new("RGB", rgba.size, (255, 255, 255))
                im.paste(rgba, mask=rgba.getchannel("A"))
            else:
                im = im.convert("RGB")
            scale = MAX_IMAGE_DIM / max(im.size)
            if scale < 1:
                im = im.resize((int(im.size[0] * scale), int(im.size[1] * scale)), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=90)
            return base64.b64encode(buf.getvalue()).decode(), "jpeg"
    with open(path, "rb") as f:
        raw = f.read()
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    return base64.b64encode(raw).decode(), ("jpeg" if ext in ("jpg", "jpeg") else ext)


def find_images(roots):
    found = []
    for root in roots:
        if not os.path.isdir(root):
            print("Folder not found, skipped: " + root)
            continue
        for directory, subdirs, files in os.walk(root):
            subdirs[:] = sorted(d for d in subdirs if not d.startswith(".") and d != "thumbnails")
            found += [os.path.join(directory, f) for f in sorted(files) if f.lower().endswith(IMAGE_EXTENSIONS)]
    if ONLY_PATHS:
        found = [p for p in found if any(part in p for part in ONLY_PATHS)]
    return found


def sidecar(image_path, suffix, ext=".txt"):
    return os.path.splitext(image_path)[0] + suffix + ext


def write_atomic(path, text):
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def log(path, line):
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.replace("\n", " ") + "\n")


# ---- layout: detections (text, watermarks, faces, panels) and the layout answer ----
LABELS = {"character", "object", "text", "watermark", "panel"}
CANVASES = {"single", "multi_panel", "collage"}
TEXT_ENGINES = ("anzhc", "anzhc_vl", "rapidocr")
_ocr_pool = {"queue": None}
_gpu = {"models": None, "loading": None, "lock": threading.Lock()}


def _ocr_engine():
    from rapidocr import RapidOCR   # pip install rapidocr onnxruntime
    return RapidOCR()


def _run_ocr(engine, path):
    with Image.open(path) as im:
        width, height = im.size
    result = engine(path)
    lines = []
    for box, text, score in zip(result.boxes if result.boxes is not None else [], result.txts or [], result.scores or []):
        text = " ".join(str(text).split())
        if not text or float(score) < OCR_MIN_SCORE:
            continue
        xs = [float(pt[0]) for pt in box]
        ys = [float(pt[1]) for pt in box]
        y0, x0 = max(0, round(min(ys) / height * 1000)), max(0, round(min(xs) / width * 1000))
        y1, x1 = min(1000, round(max(ys) / height * 1000)), min(1000, round(max(xs) / width * 1000))
        if y1 > y0 and x1 > x0:
            lines.append({"text": text, "box_2d": [y0, x0, y1, x1], "score": round(float(score), 3)})
    return {"size": [width, height], "lines": lines}


def _model_file(name):
    """A local .pt path, or a file of YOLO_REPO (downloaded once into the Hugging Face cache)."""
    if os.path.isfile(name):
        return name
    from huggingface_hub import hf_hub_download
    return hf_hub_download(YOLO_REPO, name)


def _free_gpus():
    """{GPU number as nvidia-smi shows it: free GiB}, read without touching the GPUs."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    free = {}
    for line in out.strip().splitlines():
        index, mib = (part.strip() for part in line.split(","))
        free[int(index)] = int(mib) / 1024
    return free


def detector_device():
    """The torch device for the detectors. GPU numbers follow nvidia-smi (PCI bus order), not CUDA's own."""
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")   # before CUDA starts: "cuda:2" = nvidia-smi's GPU 2
    choice = DETECTOR_DEVICE.strip().lower()
    if choice != "auto":
        return choice
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return "cuda"   # already narrowed down from outside
    free = _free_gpus()
    if not free:
        raise StopRun("DETECTOR_DEVICE = \"auto\" found no GPU (nvidia-smi); set \"cuda:N\" or \"cpu\"")
    best = max(free, key=free.get)
    if free[best] < MIN_FREE_GPU_GB:
        raise StopRun("No GPU has " + str(MIN_FREE_GPU_GB) + " GiB free (" + ", ".join(
            "GPU " + str(i) + ": " + format(gb, ".1f") + " GiB" for i, gb in sorted(free.items())) + ")")
    return "cuda:" + str(best)


class GpuModels:
    """Anzhc's illustration YOLOs and PaddleOCR-VL, loaded once and run under one lock."""

    def __init__(self):
        self.device = detector_device()
        print("Detectors on " + self.device)
        from ultralytics import YOLO   # pip install ultralytics "transformers>=5" huggingface_hub
        self.text = YOLO(_model_file(TEXT_SEG_MODEL)) if TEXT_ENGINE in ("anzhc", "anzhc_vl") else None
        self.faces = YOLO(_model_file(FACE_MODEL)) if DETECT_FACES else None
        self.panels = YOLO(_model_file(PANEL_MODEL)) if DETECT_PANELS else None
        self.vl = self.processor = None
        if TEXT_ENGINE == "anzhc_vl":
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
            dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
            self.vl = AutoModelForImageTextToText.from_pretrained(VL_MODEL, dtype=dtype).to(self.device).eval()
            self.processor = AutoProcessor.from_pretrained(VL_MODEL)
            self.processor.tokenizer.padding_side = "left"   # batched generation

    def boxes(self, model, image, conf):
        """(class name, confidence, [x0, y0, x1, y1] in pixels); overlapping boxes of any class are merged."""
        r = model.predict(image, conf=conf, agnostic_nms=True, device=self.device, verbose=False)[0]
        return [(r.names[int(k)], float(c), b)
                for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]

    def read(self, crops):
        """PaddleOCR-VL's "OCR:" reading of each crop."""
        import torch
        texts = []
        for start in range(0, len(crops), VL_BATCH):
            messages = [[{"role": "user", "content": [{"type": "image", "image": crop}, {"type": "text", "text": "OCR:"}]}]
                        for crop in crops[start:start + VL_BATCH]]
            inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                        return_dict=True, return_tensors="pt",
                                                        processor_kwargs={"padding": True})
            inputs = inputs.to(self.vl.device)
            with torch.inference_mode():
                out = self.vl.generate(**inputs, max_new_tokens=VL_MAX_TOKENS, do_sample=False)
            texts += self.processor.batch_decode(out[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return texts


def _norm_box(xyxy, width, height):
    x0, y0, x1, y1 = xyxy
    box = [max(0, round(y0 / height * 1000)), max(0, round(x0 / width * 1000)),
           min(1000, round(y1 / height * 1000)), min(1000, round(x1 / width * 1000))]
    return box if box[2] > box[0] and box[3] > box[1] else None


def _area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _share_inside(a, b):
    """The share of box a that lies inside box b."""
    inter = _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])
    return inter / max(_area(a), 1)


def _run_gpu(models, path):
    with Image.open(path) as im:
        image = im.convert("RGB")
    width, height = image.size
    out = {"size": [width, height]}
    if models.text is not None:
        regions = []
        for kind, conf, (x0, y0, x1, y1) in models.boxes(models.text, image, TEXT_MIN_CONF):
            box = _norm_box((x0, y0, x1, y1), width, height)
            if box:
                pad = max(4.0, 0.06 * max(x1 - x0, y1 - y0))   # a margin, so edge strokes are not cut
                crop = image.crop((max(0, int(x0 - pad)), max(0, int(y0 - pad)),
                                   min(width, int(x1 + pad) + 1), min(height, int(y1 + pad) + 1)))
                regions.append((kind.lower(), conf, box, crop))
        texts = models.read([r[3] for r in regions]) if regions and models.vl is not None else [""] * len(regions)
        out["lines"], out["watermarks"] = [], []
        for (kind, conf, box, _), text in zip(regions, texts):
            text = "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())[:500]
            if kind == "watermark":
                out["watermarks"].append({"text": text, "box_2d": box, "score": round(conf, 3)})
            elif text or models.vl is None:
                out["lines"].append({"text": text, "box_2d": box, "score": round(conf, 3), "kind": kind})
        out["lines"].sort(key=lambda line: (line["box_2d"][0], line["box_2d"][1]))
    if models.faces is not None:
        faces = [{"box_2d": box, "score": round(conf, 3)} for _, conf, xyxy in models.boxes(models.faces, image, FACE_MIN_CONF)
                 if (box := _norm_box(xyxy, width, height))]
        out["faces"] = sorted(faces, key=lambda f: (f["box_2d"][1], f["box_2d"][0]))
    if models.panels is not None:
        panels = []
        for _, conf, xyxy in sorted(models.boxes(models.panels, image, PANEL_MIN_CONF), key=lambda d: -d[1]):
            box = _norm_box(xyxy, width, height)
            # Not the whole page, and not nested in (or around) a surer panel.
            if box and _area(box) < 850_000 and not any(_share_inside(box, p["box_2d"]) > 0.8
                                                        or _share_inside(p["box_2d"], box) > 0.8 for p in panels):
                panels.append({"box_2d": box, "score": round(conf, 3)})
        out["panels"] = sorted(panels, key=lambda p: (p["box_2d"][0], p["box_2d"][1])) if len(panels) > 1 else []
    return out


def _gpu_run(path):
    with _gpu["lock"]:
        return _run_gpu(_gpu["models"], path)


def _detectors():
    return {"text": TEXT_ENGINE, "faces": DETECT_FACES, "panels": DETECT_PANELS}


def _cache_fits(data):
    have = data.get("detectors") or {"text": "rapidocr"}   # caches from before the detectors
    text_fits = have.get("text") == TEXT_ENGINE or (TEXT_ENGINE == "anzhc" and have.get("text") == "anzhc_vl")
    return (text_fits and (have.get("faces") or not DETECT_FACES)
            and (have.get("panels") or not DETECT_PANELS))


async def detections(path):
    """The image's text lines (and watermarks, faces, panels when those detectors are on), cached in
    <image>_ocr.json; boxes normalized 0-1000 yxyx. A cache made by other detectors is redone."""
    cache = sidecar(path, OCR_SUFFIX, ".json")
    if os.path.isfile(cache):
        with open(cache, encoding="utf-8") as f:
            data = json.load(f)
        if _cache_fits(data):
            return data
    result = {}
    if TEXT_ENGINE == "rapidocr":
        if _ocr_pool["queue"] is None:
            _ocr_pool["queue"] = asyncio.Queue()
            for _ in range(OCR_WORKERS):
                _ocr_pool["queue"].put_nowait(await asyncio.to_thread(_ocr_engine))
        engine = await _ocr_pool["queue"].get()
        try:
            result = await asyncio.to_thread(_run_ocr, engine, path)
        finally:
            _ocr_pool["queue"].put_nowait(engine)
    if TEXT_ENGINE != "rapidocr" or DETECT_FACES or DETECT_PANELS:
        try:
            if _gpu["models"] is None:
                if _gpu["loading"] is None:
                    _gpu["loading"] = asyncio.ensure_future(asyncio.to_thread(GpuModels))
                _gpu["models"] = await _gpu["loading"]
            result.update(await asyncio.to_thread(_gpu_run, path))
        except StopRun:
            raise
        except Exception as exc:
            # A full or failing GPU fails every image the same way: stop instead of marking images failed.
            text = str(exc)
            if ("CUDA" in text or "out of memory" in text.lower() or "not enough memory" in text.lower()
                    or "cuda" in type(exc).__name__.lower()):
                raise StopRun("GPU or memory error (" + text.split(". ")[0][:200] + "). Pick a free GPU with DETECTOR_DEVICE "
                              "(nvidia-smi shows which) or \"auto\".") from exc
            raise
    result["detectors"] = _detectors()
    write_atomic(cache, json.dumps(result, ensure_ascii=False) + "\n")
    return result


def moss_watermarks(path, marks):
    """The layout pass's watermarks (detector boxes, labels read by the model) replace the caption's own; with
    none confirmed, they only fill a caption whose watermarks are unknown (a converted one)."""
    if not WATERMARKS_TO_MOSS:
        return False
    target = sidecar(path, JSON_SUFFIX, ".json")
    try:
        with open(target, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict) or (not marks and data.get("watermark") is not None):
        return False
    data["watermark"] = [{"box_2d": m["box_2d"], "label": m["label"]} for m in marks]   # MOSS's own fields
    data["watermark_source"] = "layout pass: detector boxes (model boxes for what it missed), " + MODEL + " labels"
    write_atomic(target, json.dumps(data, ensure_ascii=False, indent=1) + "\n")
    return True


TEXT_TAGS = {"text", "speech bubble", "thought bubble", "sound effects", "sound effect", "onomatopoeia", "dialogue",
             "dialogue box", "translated", "subtitled", "comic", "4koma", "title", "character name",
             "copyright name", "sign", "logo", "watermark", "artist name", "signature", "web address", "url",
             "dated", "copyright notice", "username", "speech", "caption"}


def has_text_tags(tags):
    """Whether the tags show text or a watermark: TEXT_TAGS, "... text", "... username", "... logo", "... bubble"."""
    for tag in (t.lower().replace("_", " ").strip() for t in tags):
        if tag in TEXT_TAGS or tag.endswith((" text", " username", " logo", " bubble", " watermark")):
            return True
    return False


def layout_selected(path, tags):
    """Every image whose tags show text or a watermark (LAYOUT_TEXT_ALWAYS), plus a stable random share."""
    import hashlib
    if LAYOUT_TEXT_ALWAYS and has_text_tags(tags):
        return True
    key = os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path)
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF < LAYOUT_SHARE


def _union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def _plain(text):
    return re.sub(r"\W+", "", text or "").lower()


def _same_block(a, b, text_a="", text_b=""):
    """Two boxes of one block: mostly inside one another, or overlapping with one's text inside the other's."""
    inter = _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])
    if not inter:
        return False
    if inter / max(1, min(_area(a), _area(b))) >= MERGE_OVERLAP:
        return True
    ta, tb = _plain(text_a), _plain(text_b)
    return bool(ta and tb and (ta in tb or tb in ta))


def text_regions(det):
    """The detector's text and watermark regions, duplicates fused (the bigger box grows to cover both, the longer
    text and the surer kind are kept), numbered top to bottom for the model."""
    regions = [{"box_2d": list(line["box_2d"]), "kind": line.get("kind") or "text line", "text": line.get("text", ""),
                "score": line.get("score", 0)} for line in det.get("lines", [])]
    regions += [{"box_2d": list(m["box_2d"]), "kind": "watermark", "text": m.get("text", ""), "score": m.get("score", 0)}
                for m in det.get("watermarks", [])]
    changed = True
    while changed:
        changed = False
        regions.sort(key=lambda r: -_area(r["box_2d"]))
        fused = []
        for region in regions:
            keep = next((k for k in fused if _same_block(k["box_2d"], region["box_2d"], k["text"], region["text"])), None)
            if keep is None:
                fused.append(region)
                continue
            changed = True
            keep["box_2d"] = _union(keep["box_2d"], region["box_2d"])
            if len(_plain(region["text"])) > len(_plain(keep["text"])):
                keep["text"] = region["text"]
            if region["score"] > keep["score"]:
                keep["kind"], keep["score"] = region["kind"], region["score"]
        regions = fused
    regions.sort(key=lambda r: (r["box_2d"][0], r["box_2d"][1]))
    return [{"box_2d": r["box_2d"], "kind": r["kind"]} for r in regions]


def regions_reference(regions, det):
    """What the model is told: the numbered regions (boxes and kinds only, never the detector's reading)."""
    out = "<text_regions>\n" + ("\n".join(str(i) + ": box_2d " + json.dumps(r["box_2d"]) + " (" + r["kind"] + ")"
                                          for i, r in enumerate(regions)) or "(the detector found no text)") + "\n</text_regions>\n"
    if "faces" in det:
        out += "<faces>\n" + ("\n".join(str(i) + ": box_2d " + json.dumps(face["box_2d"]) for i, face in enumerate(det["faces"]))
                              or "(no faces found)") + "\n</faces>\n"
    if "panels" in det:
        out += "<panels>\n" + ("\n".join(str(i) + ": box_2d " + json.dumps(p["box_2d"]) for i, p in enumerate(det["panels"]))
                               or "(the detector found no panels)") + "\n</panels>\n"
    return out


def _iou(a, b):
    inter = _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])
    return inter / max(1, _area(a) + _area(b) - inter)


def _fuse_box(kept, other):
    """One block boxed twice: two boxes from the same source are joined; a detector box beats a model box."""
    if kept.get("box_from") == other.get("box_from"):
        kept["box_2d"] = _union(kept["box_2d"], other["box_2d"])
    elif other.get("box_from") == "detector":
        kept["box_2d"], kept["box_from"] = other["box_2d"], "detector"


def dedupe_items(items, watermarks):
    """No two items keep the same box: text items of one block are fused (one box, the longer text and its
    caption), watermarks likewise, and a character or object nearly identical to an earlier one of its label goes."""
    out = []
    for item in items:
        twin = None
        for kept in out:
            if kept["label"] != item["label"]:
                continue
            if item["label"] == "text":
                if _same_block(kept["box_2d"], item["box_2d"], kept["text"], item["text"]):
                    twin = kept
                    break
            elif _iou(kept["box_2d"], item["box_2d"]) >= 0.9:
                twin = kept
                break
        if twin is None:
            out.append(item)
        elif item["label"] == "text":
            _fuse_box(twin, item)
            if len(_plain(item["text"])) > len(_plain(twin["text"])):
                twin["text"], twin["caption"] = item["text"], item["caption"]
                if item.get("short"):
                    twin["short"] = item["short"]
    marks = []
    for mark in watermarks:
        twin = next((m for m in marks if _same_block(m["box_2d"], mark["box_2d"], m["label"], mark["label"])), None)
        if twin is None:
            marks.append(mark)
        else:
            _fuse_box(twin, mark)
            if len(mark["label"]) > len(twin["label"]):
                twin["label"] = mark["label"]
    return out, marks


def _box(value, where):
    if (not isinstance(value, list) or len(value) != 4
            or not all(isinstance(v, (int, float)) and -5 <= v <= 1005 for v in value)):
        raise ValueError(where + " box_2d is not [y_min, x_min, y_max, x_max] in 0-1000: " + json.dumps(value)[:80])
    y0, x0, y1, x1 = (min(1000, max(0, round(v))) for v in value)
    if y1 <= y0 or x1 <= x0:
        raise ValueError(where + " box_2d has no area: " + json.dumps(value))
    return [y0, x0, y1, x1]


def parse_layout(text, regions, faces=(), panels=()):
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        trig = is_refusal(raw)
        raise ValueError(("Refusal (" + trig + ")" if trig else "No JSON object in the answer") + ": " + " ".join(raw.split())[:120])
    try:
        data = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON (" + str(exc) + ")") from exc
    summary = data.get("summary")
    if not isinstance(summary, dict) or not isinstance(summary.get("caption"), str) or not summary["caption"].strip():
        raise ValueError("summary.caption is missing")
    trig = is_refusal(summary["caption"])
    if trig:
        raise ValueError("Refusal in summary.caption (" + trig + ")")
    canvas = summary.get("canvas", "single")
    if canvas not in CANVASES:
        raise ValueError("summary.canvas must be single, multi_panel or collage: " + str(canvas))
    meta = [str(m).strip() for m in summary.get("meta", []) if str(m).strip()] if isinstance(summary.get("meta", []), list) else []
    items, watermarks, objects, used, faces_used, panels_used = [], [], 0, set(), set(), set()
    details = {}
    for index, item in enumerate(data.get("items") or []):
        where = "items[" + str(index) + "]"
        if not isinstance(item, dict) or item.get("label") not in LABELS:
            raise ValueError(where + ".label must be character, object, text, watermark or panel")
        label = item["label"]
        text = "\n".join(" ".join(line.split()) for line in str(item.get("text") or "").splitlines() if line.strip())
        caption = " ".join(str(item.get("caption") or "").split())
        box = None
        if label in ("text", "watermark"):
            ids = item.get("regions")
            if ids:
                if (not isinstance(ids, list) or not all(type(i) is int and 0 <= i < len(regions) for i in ids)
                        or used.intersection(ids) or len(set(ids)) != len(ids)):
                    raise ValueError(where + ".regions must list unused region numbers: " + json.dumps(ids))
                used.update(ids)
                box = regions[ids[0]]["box_2d"]
                for i in ids[1:]:
                    box = _union(box, regions[i]["box_2d"])
            elif MODEL_TEXT_BOXES and item.get("box_2d") is not None:
                box = _box(item.get("box_2d"), where)   # missed by the detector: the model's box
            else:
                continue   # MODEL_TEXT_BOXES off: text no region covers is left out
            if label == "watermark":
                watermarks.append({"box_2d": box, "label": text or "watermark",
                                   "box_from": "detector" if ids else "model"})
                continue
            if not text:
                raise ValueError(where + ": a text item needs the text it reads")
        if label == "panel":
            number = item.get("panel")
            if type(number) is int and 0 <= number < len(panels) and number not in panels_used:
                panels_used.add(number)
                box, box_from = panels[number]["box_2d"], "detector"
            elif MODEL_TEXT_BOXES and item.get("box_2d") is not None:
                box, box_from = _box(item.get("box_2d"), where), "model"   # missed by the detector
            else:
                continue
        if not caption:
            raise ValueError(where + ".caption is empty")
        out = {"label": label, "caption": caption}
        if label == "panel":
            out["box_2d"], out["box_from"] = box, box_from
        elif label == "text":
            out["text"] = text
            out["box_2d"] = box
            out["box_from"] = "detector" if item.get("regions") else "model"
        else:
            if label == "object":
                objects += 1
                if objects > MAX_LAYOUT_OBJECTS:
                    continue
                owner = item.get("part_of")
                if type(owner) is int:
                    details[owner] = details.get(owner, 0) + 1
                    if details[owner] > MAX_DETAILS_PER_CHARACTER:
                        continue
                    out["_of"] = owner
            if label == "character" and type(item.get("id")) is int:
                out["_id"] = item["id"]
            out["box_2d"] = _box(item.get("box_2d"), where)
            face = item.get("face")
            if label == "character" and type(face) is int and 0 <= face < len(faces) and face not in faces_used:
                # The detected face is exact: the character's box must contain it.
                faces_used.add(face)
                out["box_2d"] = _union(out["box_2d"], faces[face]["box_2d"])
        short = clean_short(item.get("short"))
        if short:
            out["short"] = short
        tags = [str(t).strip() for t in item.get("tags", []) if str(t).strip()] if isinstance(item.get("tags"), list) else []
        if tags and label == "character":
            out["tags"] = tags
        items.append(out)
    items, watermarks = dedupe_items(items, watermarks)
    # Character ids (the model's own numbers) become positions in the final item list; a detail whose character
    # is missing stays a plain object.
    owners = {item.pop("_id"): index for index, item in enumerate(items) if "_id" in item}
    for item in items:
        owner = item.pop("_of", None)
        if owner in owners:
            item["part_of"] = owners[owner]
    if not items:
        raise ValueError("items is empty")
    return {"summary": {"meta": meta, "canvas": canvas, "caption": " ".join(summary["caption"].split())},
            "items": items, "watermarks": watermarks, "bbox_format": "yxyx", "bbox_coord_space": "normalized_1000"}


# ---- one image ----
ATTRIBUTION = re.compile(r"^\s*(?:drawn|art|illustrated)\s+by\s+[^.]{1,120}\.\s*", re.IGNORECASE)


async def ask(messages):
    """One answer text, with refusal/blank handling shared by both modes."""
    try:
        response = await _create(model=MODEL, messages=messages, temperature=TEMPERATURE, top_p=TOP_P,
                                 max_tokens=MAX_TOKENS)
    except APIStatusError as exc:
        raise status_error(exc) from exc
    record_cost(response)
    if not response.choices:
        raise ValueError("No choices returned (possible upstream block)")
    choice = response.choices[0]
    finish = getattr(choice, "finish_reason", None)
    if finish in ("content_filter", "stop_filter", "flagged"):
        raise ValueError("Blocked by content filter (finish_reason=" + str(finish) + ")")
    text = (choice.message.content or "").strip()
    if not text:
        raise ValueError("Blank answer (finish_reason=" + str(finish) + ")")
    block = PROVIDER_BLOCK.search(text)
    if block:
        raise PromptBlocked("Prompt rejected by the provider (" + block.group(0) + ")")
    return text


async def layout_one(path):
    tags = read_tags(path)
    if not layout_selected(path, tags):
        return None
    det = await detections(path)
    regions = text_regions(det)
    b64, mime = encode_image(path)
    messages = [{"role": "system", "content": LAYOUT_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": reference_text(tags) + regions_reference(regions, det)},
                    {"type": "image_url", "image_url": {"url": "data:image/" + mime + ";base64," + b64}}]}]
    attempts = 1 + max(0, RETRY_ON_BLOCK_ATTEMPTS)
    for n in range(1, attempts + 1):
        try:
            result = parse_layout(await ask(messages), regions, det.get("faces") or [], det.get("panels") or [])
            break
        except (PromptBlocked, StopRun):
            raise
        except ValueError as exc:
            if n == attempts:
                raise
            print("    [retry] " + os.path.basename(path) + ": " + str(exc))
            await asyncio.sleep(1)
    result["text_regions"] = regions
    result["source"] = {"provider": PROVIDER, "model": MODEL, "mode": MODE, "detector": TEXT_ENGINE,
                        "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_atomic(sidecar(path, LAYOUT_SUFFIX, ".json"), json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    moss_watermarks(path, result["watermarks"])
    return result


SHORTEN_PROMPT = """You write short prompts for the boxes of an image layout in a text-to-image dataset. The user message is a JSON list of layout items, each with its label and its full caption. Return exactly one JSON object {"short": [...]} with one string per item, in the same order.

Each string is how a person would quickly prompt that box: 2 to 8 words, the name (for a named character) or the kind of thing, plus the key action or look. Examples: "Reimu", "Marisa holding a spoon on a stool", "blonde hair doll", "white speech bubble", "purple clothes hanger", "panel: the two girls hugging".

Rules: keep the character's name when the caption names one; no position words (left, upper right, background), the box already gives the position; no full sentences and no final period; plain words, no euphemisms for nudity or sexual content when the caption states it; never name the artist. Return only the JSON object, without Markdown or explanations."""


def clean_short(value):
    """A short item caption: one line, at most 12 words, no final period; "" when unusable."""
    words = " ".join(str(value or "").split()).rstrip(".").split()
    return " ".join(words[:12]) if words else ""


def needs_short(layout):
    return any(isinstance(item, dict) and not item.get("short") for item in layout.get("items") or [])


async def shorten_one(path):
    """Short captions for an existing layout's items, from their full captions (text only, no image)."""
    target = sidecar(path, LAYOUT_SUFFIX, ".json")
    with open(target, encoding="utf-8") as f:
        layout = json.load(f)
    items = layout.get("items") or []
    listing = json.dumps([{"label": i.get("label"), "caption": i.get("caption", "")} for i in items], ensure_ascii=False)
    messages = [{"role": "system", "content": SHORTEN_PROMPT}, {"role": "user", "content": listing}]
    attempts = 1 + max(0, RETRY_ON_BLOCK_ATTEMPTS)
    for n in range(1, attempts + 1):
        try:
            raw = (await ask(messages)).strip()
            raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw)
            shorts = json.loads(raw[raw.find("{"):raw.rfind("}") + 1]).get("short")
            if not isinstance(shorts, list) or len(shorts) != len(items):
                raise ValueError("short must list one string per item (" + str(len(items)) + ")")
            shorts = [clean_short(s) for s in shorts]
            if not all(shorts):
                raise ValueError("an empty short caption")
            break
        except (PromptBlocked, StopRun):
            raise
        except (ValueError, AttributeError) as exc:
            if n == attempts:
                raise ValueError(str(exc)) from exc
            print("    [retry] " + os.path.basename(path) + ": " + str(exc))
            await asyncio.sleep(1)
    for item, short in zip(items, shorts):
        item["short"] = short
    layout.setdefault("source", {})["short_by"] = {"provider": PROVIDER, "model": MODEL}
    write_atomic(target, json.dumps(layout, ensure_ascii=False, indent=1) + "\n")
    return layout


async def detect_one(path):
    return await detections(path)


async def caption_one(path):
    if MODE == "detect":
        return await detect_one(path)
    if MODE == "layout":
        return await layout_one(path)
    if MODE == "shorten":
        return await shorten_one(path)
    if MODE == "convert":
        with open(sidecar(path, NL_SUFFIX), encoding="utf-8") as f:
            old = ATTRIBUTION.sub("", f.read().strip())
        messages = [{"role": "system", "content": CONVERT_PROMPT},
                    {"role": "user", "content": "Existing caption:\n" + old}]
    else:
        b64, mime = encode_image(path)
        messages = [{"role": "system", "content": CAPTION_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Here is the original caption reference:\n" + reference_text(read_tags(path))},
                        {"type": "image_url", "image_url": {"url": "data:image/" + mime + ";base64," + b64}}]}]
    attempts = 1 + max(0, RETRY_ON_BLOCK_ATTEMPTS)
    for n in range(1, attempts + 1):
        try:
            result = parse_answer(await ask(messages), need_layout_fields=(MODE == "caption"))
            break
        except (PromptBlocked, StopRun):
            raise
        except ValueError as exc:
            if n == attempts:
                raise
            print("    [retry] " + os.path.basename(path) + ": " + str(exc))
            await asyncio.sleep(1)
    if MODE == "convert":
        result.update(watermark=None, borders=None)   # unknown until the layout pass sees the image
        try:   # ...or the layout pass already ran: take its watermarks
            with open(sidecar(path, LAYOUT_SUFFIX, ".json"), encoding="utf-8") as f:
                marks = json.load(f).get("watermarks")
            if isinstance(marks, list):
                result["watermark"] = [{"box_2d": m["box_2d"], "label": m["label"]} for m in marks]
                result["watermark_source"] = "layout pass (detector boxes, model boxes for what it missed)"
        except (OSError, ValueError, KeyError, TypeError):
            pass
    result["source"] = {"provider": PROVIDER, "model": MODEL, "mode": MODE,
                        "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_atomic(sidecar(path, JSON_SUFFIX, ".json"), json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    if MODE == "caption":
        write_atomic(sidecar(path, NL_SUFFIX), result["caption"]["long"] + "\n")
    return result


def wanted(path):
    if path in _failed:
        return False
    if MODE == "layout":
        return not os.path.isfile(sidecar(path, LAYOUT_SUFFIX, ".json"))
    if MODE == "shorten":
        try:
            with open(sidecar(path, LAYOUT_SUFFIX, ".json"), encoding="utf-8") as f:
                return needs_short(json.load(f))
        except (OSError, ValueError):
            return False
    if MODE == "detect":
        try:
            with open(sidecar(path, OCR_SUFFIX, ".json"), encoding="utf-8") as f:
                return not _cache_fits(json.load(f))
        except (OSError, ValueError):
            return True
    if os.path.isfile(sidecar(path, JSON_SUFFIX, ".json")):
        return False
    has_old = os.path.isfile(sidecar(path, NL_SUFFIX))
    return has_old if MODE == "convert" else (CAPTION_OVER_OLD_NL or not has_old)


async def process(path, counts):
    if counts["stopped"]:
        return
    if over_budget():
        counts["stopped"] = 1
        print("STOPPING: spent " + spent_text() + ", the MAX_SPEND limit.")
        return
    folder = _folders.setdefault(os.path.dirname(path), {"ok": False, "fails": 0, "lock": asyncio.Lock()})
    if FOLDER_FAIL_LIMIT and not folder["ok"]:   # 0 = no folder guard: every image is tried
        async with folder["lock"]:
            if not folder["ok"]:
                if folder["fails"] >= FOLDER_FAIL_LIMIT:
                    counts["skipped_folder"] += 1
                    return
                folder["ok"] = await attempt(path, counts)
                folder["fails"] += not folder["ok"]
                if folder["fails"] == FOLDER_FAIL_LIMIT:
                    print("Giving up on " + os.path.dirname(path) + " after " + str(FOLDER_FAIL_LIMIT) + " failures")
                return
    await attempt(path, counts)


async def attempt(path, counts):
    try:
        result = await caption_one(path)
    except StopRun as exc:
        if not counts["stopped"]:
            counts["stopped"] = 1
            print("STOPPING: " + str(exc))
        return False
    except Exception as exc:
        _failed.add(path)
        counts["failed"] += 1
        counts["failed_in_a_row"] += 1
        counts["refused"] += isinstance(exc, PromptBlocked)
        log(FAILED_LOG, "[" + time.strftime("%Y-%m-%d %H:%M:%S") + "] " + path + " -- " + MODEL + ": " + str(exc))
        log(OUTPUT_FILE, path + "\t" + MODE + "\tERROR: " + str(exc))
        print(path + " -> ERROR: " + str(exc))
        if STOP_AFTER_FAILURES_IN_A_ROW and counts["failed_in_a_row"] >= STOP_AFTER_FAILURES_IN_A_ROW and not counts["stopped"]:
            counts["stopped"] = 1
            print("STOPPING: " + str(STOP_AFTER_FAILURES_IN_A_ROW) + " failures in a row; check " + FAILED_LOG)
        return False
    counts["failed_in_a_row"] = 0
    if result is None:   # layout mode: not in the share and no text
        counts["not_selected"] += 1
        return True
    counts["done"] += 1
    if MODE == "detect":
        counts["with_text"] += bool(result["lines"])
        counts["watermarks"] += bool(result.get("watermarks"))
        log(OUTPUT_FILE, path + "	" + MODE + "	" + str(len(result["lines"])) + " text, "
            + str(len(result.get("faces", []))) + " faces, " + str(len(result.get("panels", []))) + " panels")
    elif MODE in ("layout", "shorten"):
        counts["with_text"] += any(item["label"] == "text" for item in result["items"])
        log(OUTPUT_FILE, path + "\t" + MODE + "\t" + str(len(result["items"])) + " items: " + result["summary"]["caption"][:160])
    else:
        counts["watermarks"] += bool(result.get("watermark"))
        log(OUTPUT_FILE, path + "\t" + MODE + "\t" + result["caption"]["short"][:200])
    return True


async def preflight():
    if not API_KEY:
        print("No API key: export " + _P["key_env"] + "=... (never write it into this file).")
        return False
    if PROVIDER == "opencode" and "contributor" not in MODEL.lower():
        print("On OpenCode Go use the Contributor model (the discounted one); got " + MODEL)
        return False
    try:
        listed = [m.id async for m in client.models.list()]
        if listed and MODEL not in listed and MODEL not in {m.split("/")[-1] for m in listed}:
            near = [m for m in listed if MODEL.split("-")[0].split("/")[-1] in m.lower()]
            print("MODEL " + MODEL + " is not listed by " + PROVIDER + ". Similar: " + (", ".join(near[:15]) or "none"))
            return False
    except APIStatusError as exc:
        if exc.status_code in (401, 402, 403):
            print("Preflight failed: " + str(status_error(exc)))
            return False   # a missing /models endpoint (404) is not fatal: the test call below decides
    except Exception as exc:
        print("Could not reach " + _P["base_url"] + ": " + str(exc)[:200])
        return False
    for tries in range(2):
        try:
            await client.chat.completions.create(model=MODEL, max_tokens=32, messages=[{"role": "user", "content": "Reply with OK."}],
                                                 **request_options())
            print("Preflight ok: " + PROVIDER + " / " + MODEL
                  + (" / reasoning " + str(_options["reasoning_effort"]) if _options["reasoning_effort"] else ""))
            return True
        except APIStatusError as exc:
            if exc.status_code == 400 and _options["reasoning_effort"] and tries == 0:
                print("The provider refused the reasoning setting; continuing without it.")
                _options["reasoning_effort"] = None
                continue
            print("Preflight failed: " + str(status_error(exc)))
            return False
    return False


def load_failed():
    if os.path.isfile(FAILED_LOG):
        with open(FAILED_LOG, encoding="utf-8") as f:
            for line in f:
                if " -- " in line:
                    _failed.add(line.split(" -- ")[0].split("] ", 1)[-1].strip())


async def main():
    if MODE != "detect" and not await preflight():
        return
    load_failed()
    paths = [p for p in find_images(IMAGE_ROOTS) if wanted(p)]
    if LIMIT:
        paths = paths[:LIMIT]
    # Round-robin over folders: many folders are probed at once, none floods before one image worked.
    by_folder = {}
    for p in paths:
        by_folder.setdefault(os.path.dirname(p), []).append(p)
    groups = list(by_folder.values())
    paths = [g[i] for i in range(max((len(g) for g in groups), default=0)) for g in groups if i < len(g)]
    if MODE == "detect":
        print("detect: " + str(len(paths)) + " image(s) to do (no API calls)")
    else:
        print(MODE + ": " + str(len(paths)) + " image(s) to do with " + PROVIDER + " / " + MODEL + ", concurrency "
              + str(MAX_CONCURRENCY) + (", pace " + str(PACE) + "/min" if PACE else ""))
    counts = dict(done=0, failed=0, refused=0, watermarks=0, with_text=0, not_selected=0, skipped_folder=0,
                  failed_in_a_row=0, stopped=0)
    queue = asyncio.Queue()
    for p in paths:
        queue.put_nowait(p)
    started = time.time()

    async def worker():
        while not counts["stopped"]:
            try:
                p = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await process(p, counts)
            except Exception as exc:   # never let one image stop a worker
                print("Unhandled error on " + p + ": " + str(exc))
            handled = len(paths) - queue.qsize()
            if handled % 200 == 0:
                rate = counts["done"] / max(time.time() - started, 1) * 60
                print(str(handled) + "/" + str(len(paths)) + " handled, " + str(counts["done"]) + " done, "
                      + str(counts["failed"]) + " failed, " + str(round(rate)) + "/min, " + spent_text())

    await asyncio.gather(*(worker() for _ in range(MAX_CONCURRENCY)))
    print("Finished in " + str(round(time.time() - started)) + "s: " + json.dumps(counts) + " | " + spent_text())
    print("Progress: " + OUTPUT_FILE + " | failures: " + FAILED_LOG)


if TEXT_ENGINE not in TEXT_ENGINES:
    raise SystemExit("TEXT_ENGINE must be one of " + ", ".join(TEXT_ENGINES) + ", not " + repr(TEXT_ENGINE))
CHARACTER_FACTS = load_character_facts(CHARACTER_FACTS_FILE) if MODE in ("caption", "layout") else {}
CAPTION_PROMPT = load_text(PROMPT_FILE) if MODE == "caption" else ""
LAYOUT_PROMPT = load_text(LAYOUT_PROMPT_FILE) if MODE == "layout" else ""
CONVERT_PROMPT = load_text(CONVERT_PROMPT_FILE) if MODE == "convert" else ""

if __name__ == "__main__":
    asyncio.run(main())
