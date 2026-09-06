#!/usr/bin/env python3
# """
# LinkAPI.ai captioning script -- supports Gemini, Grok, and Claude models
# (Sonnet 5, Opus 4.8), all via the OpenAI-compatible /v1/chat/completions
# endpoint that LinkAPI exposes for every provider.
#
# CHANGES IN THIS VERSION vs. your last one:
#   1. Rate-limit (429) retry with exponential backoff, via tenacity, mirroring
#      the same fix already in your Vertex script. Previously, a 429 (or any
#      non-ValueError exception) was NOT caught by the block-retry loop in
#      caption_image_api -- it propagated straight to process_one's generic
#      except Exception handler, which permanently blacklists the image after
#      just one hit. A transient rate limit has nothing to do with the image
#      and shouldn't cost it a permanent failure.
#   2. Guard against response.choices being empty before indexing [0]. An
#      empty choices list would previously raise an unhandled IndexError --
#      same problem as #1: it skips the retry loop entirely and permanently
#      fails the image on what might just be an upstream safety block that
#      deserves a retry, exactly like the Vertex script's explicit
#      "no candidates returned" check.
#   3. FLAT is_refusal(): apostrophe-agnostic. All quote/apostrophe variants
#      (’ ‘ ´ ` ʼ) are straightened and then apostrophes are stripped from the
#      text, and every pattern is written apostrophe-free -- so "I can't",
#      "I can’t", and "I cant" all match identically. Strips quoted dialogue
#      first, then I+refusal-verb+task-verb window, prefixes, META, and the
#      gated short-text layer. Returns the trigger string (or None).
#   4. Image shrink/recompress to avoid 413 Request Entity Too Large errors
#      from the upstream nginx proxy. Files exceeding MAX_RAW_BYTES or
#      MAX_IMAGE_DIM are automatically resized and JPEG-compressed before
#      base64 encoding.
#
# MESSAGE STRUCTURE (the Grok punctuation fix):
#   - system message  = PROMPT_FILE content, byte-for-byte intact (goal, guidelines,
#     few-shot examples). Same text the vertex script uses, just moved to the
#     system role where Grok weights it heavily.
#   - user message    = per-image <tags> block (+ character-identity note) + the image.
#   - Grounded path   = Responses API with the same md text passed via `instructions=`
#     (the Responses API's system-role equivalent).
# Resume behavior: SKIP_IF_CAPTIONED=True skips any image that already has a
# caption sidecar for the active phase's suffix (no API call, no cost).
# Set it False to force a full re-caption (existing sidecars overwritten).
# """
import os
import re
import json
import base64
import time
import asyncio
import io
from datetime import datetime, timezone
from PIL import Image
from openai import AsyncOpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

# ---- LINKAPI CONFIG ----
API_KEY = ""
BASE_URL = "https://linkapi.ai/v1"
IMAGE_FOLDER = "/home/jovyan/booru_essence_processed/links/danbooru/safe"   # ONE bucket per run
OUTPUT_FILE = "captions_output.txt"
FAILED_LOG = "failed_images.log"

# Set True to ONLY process images that previously failed (retry mode)
RETRY_FAILED_ONLY = False

# RESUME TOGGLE (see docstring)
SKIP_IF_CAPTIONED = True

# How many extra attempts to make when a call comes back blocked/blank.
# 1 = try once more (2 attempts total) before giving up and logging to FAILED_LOG.
# NOTE: this is separate from rate-limit retries below -- a 429 no longer
# consumes one of these attempts, since it isn't evidence anything is wrong
# with the image itself.
RETRY_ON_BLOCK_ATTEMPTS = 2

# Rate-limit (429) retry, independent of the block-retry above. Same pattern
# as the Vertex script's _generate_content_with_retry.
RATE_LIMIT_MAX_ATTEMPTS = 5
RATE_LIMIT_WAIT_MIN_SECONDS = 2
RATE_LIMIT_WAIT_MAX_SECONDS = 60

MAX_CONCURRENCY = 40
PREPROCESS_ALPHA = False
ALPHA_VALUE = 0.9
PROMPT_FILE = "/home/jovyan/models/captioners/gemini-flash-2.5-fixed.md"

# ---- IMAGE SIZE CAPS (413 Request Entity Too Large prevention) ----
MAX_IMAGE_DIM = 2048        # long-side cap in pixels
MAX_RAW_BYTES = 6_000_000   # recompress if the file on disk is bigger than this (6MB)

# ---- SAMPLING (LinkAPI supported parameters; None = omit / provider default) ----
TEMPERATURE = 1.0          # you measured 1.0 behaving better than 0.1 for Grok prose
TOP_P = None               # 0-1 nucleus sampling; None = provider default
MAX_TOKENS = None          # hard cap per caption; None = provider default
FREQUENCY_PENALTY = None   # -2..2; None = provider default
PRESENCE_PENALTY = None    # -2..2; None = provider default
SEED = None                # deterministic sampling (best-effort); None = random

# ---- GROUNDING (Grok live web search, official Responses API method) ----
GROUNDING = "off"          # cheap group has no search -> keep "off"; use "responses" on official group
GROUNDING_LOG = "grounding_log.jsonl"   # best-effort audit trail of citations

# ---- PHASES ----
PHASES = [
    {"label": "Gemini",  "model": "gemini-3.7-flash", "suffix": "_nl", "enabled": False, "limit": None, "offset": 0},
    {"label": "Grok",    "model": "grok-4.6",         "suffix": "_nl",       "enabled": True,  "limit": None,   "offset": 0},
    {"label": "Sonnet5", "model": "claude-sonnet-5",  "suffix": "_nl_sonnet5",     "enabled": False, "limit": None, "offset": 0},
    {"label": "Opus5",   "model": "claude-opus-5",    "suffix": "_nl_opus5",       "enabled": False, "limit": None, "offset": 0},
]

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

# model_name(lowercased) -> set of filenames that previously failed for that model
_failed_by_model = {}
_grounding_warned = False


def load_prompt(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError("Prompt file not found: " + file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


# Static instructions + few-shot examples, sent verbatim as the system message
SYSTEM_PROMPT = load_prompt(PROMPT_FILE)


# ---- REFUSAL DETECTOR (apostrophe-agnostic; identical to the notebook scan cell) ----
def _norm(t):
    for a in ("’", "‘", "´", "`", "ʼ"):
        t = t.replace(a, "'")
    return t.replace("“", '"').replace("”", '"')

QUOTED = re.compile(r'"[^"]*"')   # strip quoted dialogue first (after _norm straightens curly quotes)

TASK_VERB = r"(?:create|describe|provide|generate|caption|depict|write|assist|comply|fulfill|produce|do\s+that|help)"

# All patterns written WITHOUT apostrophes; the text is flattened the same way,
# so ’ ‘ ´ ` or missing apostrophes can never slip through again.
REFUSAL_I = re.compile(
    r"\bi(?:m|\s+am)?\s+unable\b"
    r"|\bi\s+(?:wont|will\s+not|cant|cannot|refuse\s+to|must\s+decline)\b[^.!?]{0,40}?" + TASK_VERB +
    r"|\bi\s*(?:m|am)?\s+not\s+going\s+to\b[^.!?]{0,40}?" + TASK_VERB +
    r"|\bi\s+(?:dont|do\s+not)\s+feel\s+comfortable\b"
    r"|\bi\s+apologize\b",
    re.IGNORECASE)

PREFIXES = ("im sorry", "i cannot", "i cant", "i am unable", "as an ai",
            "sorry but i", "i must decline", "im not able", "i am not able",
            "i am programmed", "im not comfortable", "i apologize",
            "this request", "i wont", "i will not", "i refuse", "illegal",
            "i dont think i can", "im not going", "i dont feel")

META = ["against my safety", "against my programming", "against my guidelines",
        "against my rules", "violates my", "safety guidelines", "content policy",
        "sexual content involving minors", "as an ai"]


def is_refusal(text):
    """Returns the trigger string if the output is a refusal, else None.
    (Truthy string / falsy None -- `if is_refusal(x):` works as-is.)"""
    if not text: return None
    t = _norm(text.strip())
    unquoted = QUOTED.sub('', t)
    u = unquoted.lower().replace("'", "")

    m = REFUSAL_I.search(u)
    if m: return "I+refusal verb: " + " ".join(m.group(0).split())
    for p in PREFIXES:
        if u.startswith(p): return "prefix: " + p
    for p in META:
        if p in u: return "meta: " + p

    sentences = [s for s in re.split(r"[.!?]+", u) if len(s.strip()) > 1]
    if len(sentences) <= 2 and len(u) < 450:
        for w in [r"\bcsam\b", r"\bprepubescent\b", r"\bunderage\b",
                  r"\bminors\b", r"\bminor\b", r"\bunethical\b"]:
            if re.search(w, u): return "keyword: " + w
        policy_words = ["policy", "guidelines", "restrictions", "inappropriate",
                        "offensive", "harmful", "decline", "assist", "fulfill",
                        "comply", "safety", "violate", "cannot", "unable",
                        "sorry", "apologize"]
        hits = [w for w in policy_words if w in u]
        if len(hits) >= 2: return "density: " + ",".join(hits)
    return None


# ---- Rate-limit retry wrappers (429), independent of block-retry above.
@retry(
    wait=wait_random_exponential(min=RATE_LIMIT_WAIT_MIN_SECONDS, max=RATE_LIMIT_WAIT_MAX_SECONDS),
    stop=stop_after_attempt(RATE_LIMIT_MAX_ATTEMPTS),
    retry=retry_if_exception_type(RateLimitError),
)
async def _chat_completions_with_retry(**kwargs):
    return await client.chat.completions.create(**kwargs)


@retry(
    wait=wait_random_exponential(min=RATE_LIMIT_WAIT_MIN_SECONDS, max=RATE_LIMIT_WAIT_MAX_SECONDS),
    stop=stop_after_attempt(RATE_LIMIT_MAX_ATTEMPTS),
    retry=retry_if_exception_type(RateLimitError),
)
async def _responses_with_retry(**kwargs):
    return await client.responses.create(**kwargs)


def build_user_text(tags):
    """Per-image user text: the <tags> block plus the character-identity note.
    The image itself is attached after this text in the same user message."""
    if not tags:
        return ""
    return (
        "Grounded tags for this image, scraped from Danbooru or e621 (use them to inform "
        "the caption, but write natural prose, do not just list them back):\n"
        "<tags>\n" + tags + "\n</tags>\n"
        "Some of these tags are character-identity tags. Danbooru's naming convention "
        "for these is inconsistent: some are 'name (series)', some are 'name (alternate "
        "outfit)' with no series at all, some are 'name (outfit) (series)', and some are "
        "a bare name with no parentheses. Use your own knowledge of Danbooru and e621 tagging "
        "conventions to identify which tag(s), if any, name a specific character. If you "
        "recognize the character, treat that identification as ground truth and describe "
        "that exact character -- do not substitute a different, visually-similar character. "
        "If a tag appears to name a character or series you don't confidently recognize, "
        "use Google Search to verify before writing the caption rather than guessing from "
        "visual similarity alone."
    )


def preprocess_image_alpha(path, target_alpha=0.9):
    img = Image.open(path)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    a = img.getchannel("A")
    a = a.point(lambda p: int(p * target_alpha))
    img.putalpha(a)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def encode_image(path):
    if PREPROCESS_ALPHA:
        raw = preprocess_image_alpha(path, target_alpha=ALPHA_VALUE)
        return base64.b64encode(raw).decode("utf-8"), "png"

    need_shrink = os.path.getsize(path) > MAX_RAW_BYTES
    with Image.open(path) as im:
        if max(im.size) > MAX_IMAGE_DIM:
            need_shrink = True

    if need_shrink:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = MAX_IMAGE_DIM / max(w, h)
            if scale < 1:
                im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=90)
            raw = buf.getvalue()
        mime = "jpeg"
    else:
        with open(path, "rb") as f:
            raw = f.read()
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    return base64.b64encode(raw).decode("utf-8"), mime


def read_tags(image_path):
    tpath = os.path.splitext(image_path)[0] + ".txt"
    if os.path.isfile(tpath):
        with open(tpath, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None


def sidecar_path(image_path, suffix="_nl"):
    base, _ = os.path.splitext(image_path)
    return base + suffix + ".txt"


def log_failed(image_name, model, reason):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(FAILED_LOG, "a", encoding="utf-8") as f:
        f.write("[" + timestamp + "] " + image_name + " -- " + model + ": " + reason + "\n")


def verify_write(path, expected_content):
    with open(path, "r", encoding="utf-8") as f:
        actual = f.read().strip()
    if actual != expected_content.strip():
        print("    !! WARNING: file on disk does NOT match what was written.")
        return False
    return True


def build_sampling_kwargs(phase):
    """Standard sampling params; only non-None values are sent."""
    kw = {}
    for key, cfg in (("temperature", TEMPERATURE), ("top_p", TOP_P),
                     ("max_tokens", MAX_TOKENS), ("frequency_penalty", FREQUENCY_PENALTY),
                     ("presence_penalty", PRESENCE_PENALTY), ("seed", SEED)):
        val = phase.get(key, cfg)   # per-phase override wins over global config
        if val is not None:
            kw[key] = val
    return kw


def _grounding_mode(phase):
    """Normalize per-phase/global grounding setting to 'responses' or 'off'."""
    g = phase.get("grounding", GROUNDING)
    if g is True:
        return "responses"
    if g in ("responses", "off"):
        return g
    return "off"


def log_grounding(fname, model, sources):
    with open(GROUNDING_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "image": fname, "model": model, "sources": sources,
            "ts": datetime.now(timezone.utc).isoformat(),
        }) + "\n")


async def _grounded_call(user_text, b64, mime, model, sampling_kwargs):
    """Official xAI grounding: Responses API + web_search server tool.
    System text goes via `instructions`; tags+image via `input`.
    Returns (caption, citation_urls)."""
    rkw = dict(sampling_kwargs)
    if "max_tokens" in rkw:
        rkw["max_output_tokens"] = rkw.pop("max_tokens")   # Responses API rename

    content = []
    if user_text:
        content.append({"type": "input_text", "text": user_text})
    content.append({"type": "input_image", "image_url": "data:image/" + mime + ";base64," + b64})

    response = await _responses_with_retry(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=[{"role": "user", "content": content}],
        tools=[{"type": "web_search"}],
        **rkw,
    )

    caption = (getattr(response, "output_text", None) or "").strip()
    cites = list(getattr(response, "citations", None) or [])
    if not cites:   # some SDK shapes put citations on output items
        for item in (getattr(response, "output", None) or []):
            cites += list(getattr(item, "citations", None) or [])

    sources = []
    for c in cites:
        if isinstance(c, str):
            sources.append(c)
        elif isinstance(c, dict):
            if c.get("url"):
                sources.append(c["url"])
        else:
            u = getattr(c, "url", None)
            if u:
                sources.append(u)
    return caption, sources


async def caption_image_api(path, tags, semaphore, model, sampling_kwargs, grounding):
    global _grounding_warned
    fname = os.path.basename(path)
    if fname in _failed_by_model.get(model.lower(), set()):
        raise RuntimeError("skipped")

    user_text = build_user_text(tags)
    total_attempts = 1 + max(0, RETRY_ON_BLOCK_ATTEMPTS)
    last_error = None

    for attempt in range(1, total_attempts + 1):
        try:
            async with semaphore:
                b64, mime = encode_image(path)

                # Official xAI grounding path (Responses API + web_search tool), Grok only
                if grounding == "responses" and model.lower().startswith("grok"):
                    try:
                        caption, sources = await _grounded_call(user_text, b64, mime, model, sampling_kwargs)
                        if sources:
                            print("    [grounding] " + str(len(sources)) + " citation(s): " + str(sources[:3]))
                            log_grounding(fname, model, sources)
                        if caption:
                            trig = is_refusal(caption)
                            if trig:
                                raise ValueError("Blocked by text safety/refusal filter (LLM refusal: " + trig + ")")
                            return caption
                        raise ValueError("Blank or empty grounded response")
                    except ValueError:
                        raise
                    except Exception as ge:
                        if not _grounding_warned:
                            print("    [grounding] /v1/responses + web_search unavailable via this proxy ("
                                  + str(ge) + ") - falling back to no grounding.")
                            _grounding_warned = True
                        # fall through to the plain chat-completions path below

                # Plain path: system message = md verbatim; user = tags text + image
                user_content = []
                if user_text:
                    user_content.append({"type": "text", "text": user_text})
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": "data:image/" + mime + ";base64," + b64},
                })

                response = await _chat_completions_with_retry(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    **sampling_kwargs,
                )

                if not response.choices:
                    raise ValueError("No choices returned from API (possible upstream block)")

                choice = response.choices[0]
                finish_reason = getattr(choice, "finish_reason", None)

                # OpenAI-compatible APIs use "content_filter" or similar for safety blocks
                if finish_reason in ("content_filter", "stop_filter", "flagged"):
                    raise ValueError(f"Blocked by content filter (finish_reason={finish_reason})")

                caption = choice.message.content
                if caption is None or caption.strip() == "":
                    raise ValueError(f"Blank or empty response from API (finish_reason={finish_reason})")

                trig = is_refusal(caption)
                if trig:
                    raise ValueError("Blocked by text safety/refusal filter (LLM refusal: " + trig + ")")

                return caption.strip()

        except ValueError as e:
            # Blank, empty, or content filter -- the retryable case.
            last_error = e
            if attempt < total_attempts:
                print("    [retry] attempt " + str(attempt) + "/" + str(total_attempts) + " blocked/blank (" + str(e) + "), retrying " + fname + "...")
                await asyncio.sleep(1)  # brief pause before retry
                continue
            raise last_error


async def process_one(path, tags, semaphore, idx, total,
                      results, counts, model, suffix, label, phase):
    fname = os.path.basename(path)
    sidecar = sidecar_path(path, suffix)
    existed = os.path.isfile(sidecar)

    # --- RESUME TOGGLE: skip images that already have this phase's caption ---
    if existed and SKIP_IF_CAPTIONED:
        counts["skipped_existing"] = counts.get("skipped_existing", 0) + 1
        print("[" + label + " " + str(idx) + "/" + str(total) + "] " + fname + " -> SKIPPED (already captioned)")
        results.setdefault(label, {})[fname] = "SKIPPED_EXISTING"
        return False

    if tags:
        counts["tags_used"] += 1

    try:
        caption = await caption_image_api(
            path, tags, semaphore, model,
            build_sampling_kwargs(phase), _grounding_mode(phase),
        )
    except RuntimeError as e:
        if str(e) == "skipped":
            counts["skipped"] += 1
            print("[" + label + " " + str(idx) + "/" + str(total) + "] " + fname + " -> SKIPPED (previous failure)")
            results.setdefault(label, {})[fname] = "SKIPPED"
            return False
        raise
    except Exception as e:
        _failed_by_model.setdefault(model.lower(), set()).add(fname)
        log_failed(fname, model, str(e))
        counts["failed"] += 1
        print("[" + label + " " + str(idx) + "/" + str(total) + "] " + fname + " -> ERROR: " + str(e))
        results.setdefault(label, {})[fname] = "ERROR: " + str(e)
        return False

    print("[" + label + " " + str(idx) + "/" + str(total) + "] " + fname)
    print("    Caption: " + caption[:100] + "...")

    with open(sidecar, "w", encoding="utf-8") as sf:
        sf.write(caption + "\n")
        sf.flush()
        os.fsync(sf.fileno())

    ok = verify_write(sidecar, caption + "\n")
    print("    Wrote to: " + os.path.abspath(sidecar) + "  (verified=" + str(ok) + ")")

    if existed:
        key = "overwritten_" + label.lower()
        counts[key] = counts.get(key, 0) + 1
        print("    -> overwrote existing sidecar")

    results.setdefault(label, {})[fname] = caption
    counts["success"] += 1
    return True


async def run_phase(image_data, semaphore, counts, results, phase):
    model = phase["model"]
    suffix = phase["suffix"]
    label = phase["label"]
    limit = phase["limit"]
    offset = phase["offset"]

    subset = image_data[offset:]
    if limit is not None and limit > 0:
        subset = subset[:limit]

    if not subset:
        print("[" + label + "] No images to process.")
        return

    print("[" + label + "] " + model + " -- " + str(len(subset)) + " image(s) (offset=" + str(offset) + ", limit=" + str(limit) + ")")
    print("  sampling: " + str(build_sampling_kwargs(phase)))
    print("  grounding: " + _grounding_mode(phase) + (" (grok web_search via /v1/responses)" if model.lower().startswith("grok") else " (grok-only feature)"))
    print("  system prompt: " + str(len(SYSTEM_PROMPT)) + " chars (md verbatim)")
    print()

    tasks = []
    for i, item in enumerate(subset, 1):
        tasks.append(
            process_one(
                item["path"], item["tags"], semaphore,
                i, len(subset), results, counts,
                model, suffix, label, phase
            )
        )

    outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    for outcome in outcomes:
        if isinstance(outcome, Exception):
            print("[" + label + "] Unhandled exception in task: " + str(outcome))

    success = sum(1 for o in outcomes if o is True)
    failed = sum(1 for o in outcomes if o is False)
    counts[label.lower() + "_success"] = success
    counts[label.lower() + "_failed"] = failed
    print("[" + label + "] Done. Success: " + str(success) + ", Failed: " + str(failed) + "\n")


async def main():
    if not os.path.isdir(IMAGE_FOLDER):
        print("Folder not found: " + IMAGE_FOLDER)
        return

    images = [
        f for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]
    if not images:
        print("No images found.")
        return

    enabled_phases = [p for p in PHASES if p["enabled"]]

    print("Provider: linkapi.ai (" + BASE_URL + ")")
    print("Enabled models: " + ", ".join(p["model"] for p in enabled_phases))
    print("Concurrency: " + str(MAX_CONCURRENCY))
    print("Alpha preprocess: " + str(PREPROCESS_ALPHA) + " (" + (str(ALPHA_VALUE) if PREPROCESS_ALPHA else "N/A") + ")")
    print("Skip-if-captioned: " + str(SKIP_IF_CAPTIONED))
    print("Sampling: temp=" + str(TEMPERATURE) + " top_p=" + str(TOP_P) + " max_tokens=" + str(MAX_TOKENS)
          + " freq_pen=" + str(FREQUENCY_PENALTY) + " pres_pen=" + str(PRESENCE_PENALTY) + " seed=" + str(SEED))
    print("Image caps: MAX_DIM=" + str(MAX_IMAGE_DIM) + "px, MAX_BYTES=" + str(MAX_RAW_BYTES))
    print("Grounding: " + str(GROUNDING) + " (log=" + GROUNDING_LOG + ")")
    print("Rate-limit retry: up to " + str(RATE_LIMIT_MAX_ATTEMPTS) + " attempts, "
          + str(RATE_LIMIT_WAIT_MIN_SECONDS) + "-" + str(RATE_LIMIT_WAIT_MAX_SECONDS) + "s backoff")
    print("System prompt: " + str(len(SYSTEM_PROMPT)) + " chars from " + PROMPT_FILE)
    print("Tag sidecars are READ ONLY.\n")

    if os.path.isfile(FAILED_LOG):
        with open(FAILED_LOG, "r", encoding="utf-8") as f:
            for line in f:
                if " -- " in line:
                    fname = line.split(" -- ")[0].split("] ")[-1].strip()
                    model_name = line.split(" -- ")[1].split(":")[0].strip().lower()
                    _failed_by_model.setdefault(model_name, set()).add(fname)

        for model_name, fnames in _failed_by_model.items():
            print("Pre-skipping " + str(len(fnames)) + " for " + model_name)
        print()

    # --- STARTUP CLEANUP FOR EXISTING REFUSALS ---
    deleted_refusals = 0
    for fname in images:
        path = os.path.join(IMAGE_FOLDER, fname)
        for phase in enabled_phases:
            sc = sidecar_path(path, phase["suffix"])
            if os.path.isfile(sc):
                with open(sc, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if is_refusal(content):
                    os.remove(sc)
                    log_failed(fname, phase["model"], "Deleted existing refusal output")
                    _failed_by_model.setdefault(phase["model"].lower(), set()).add(fname)
                    deleted_refusals += 1
    if deleted_refusals > 0:
        print(f"Cleaned up {deleted_refusals} existing refusal sidecar(s) and logged them as failed.")
    # ---------------------------------------------

    image_data = []
    tags_used = 0
    for fname in images:
        path = os.path.join(IMAGE_FOLDER, fname)
        tags = read_tags(path)
        if tags:
            tags_used += 1
        image_data.append({"fname": fname, "path": path, "tags": tags})

    # RETRY MODE: only process previously failed images
    if RETRY_FAILED_ONLY:
        all_failed = set()
        for fnames in _failed_by_model.values():
            all_failed |= fnames
        if not all_failed:
            print("RETRY MODE: No failed images to retry.")
            return
        image_data = [item for item in image_data if item["fname"] in all_failed]
        print("RETRY MODE: Only processing " + str(len(image_data)) + " previously failed image(s)")
        # Clear skip sets so we actually retry them instead of skipping
        _failed_by_model.clear()
        print()

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    results = {}
    counts = {
        "tags_used": tags_used,
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "skipped_existing": 0,
    }

    start = time.time()
    for i, phase in enumerate(enabled_phases, 1):
        print("=" * 55)
        print("PHASE " + str(i) + ": " + phase["label"] + " (" + phase["model"] + ")")
        print("=" * 55)
        await run_phase(image_data, semaphore, counts, results, phase)

    elapsed = time.time() - start

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for phase in enabled_phases:
            label = phase["label"]
            f.write("=== " + label + " (" + phase["model"] + ") ===\n")
            phase_results = results.get(label, {})
            for fname in images:
                f.write(fname + "\t" + phase_results.get(fname, "NOT_PROCESSED") + "\n")
            f.write("\n")

    print("=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print("Total images: " + str(len(images)))
    for phase in enabled_phases:
        label = phase["label"]
        print("  " + phase["model"] + " success: " + str(counts.get(label.lower() + "_success", 0)))
        print("  " + phase["model"] + " failed:  " + str(counts.get(label.lower() + "_failed", 0)))
        overwritten = counts.get("overwritten_" + label.lower(), 0)
        print("  " + phase["model"] + " overwritten sidecars: " + str(overwritten) + " (" + phase["suffix"] + ")")
    print("  Skipped:         " + str(counts["skipped"]))
    print("  Skipped existing:" + str(counts["skipped_existing"]))
    print("  Tags used:       " + str(counts["tags_used"]))
    print("  Time:            " + str(round(elapsed, 1)) + "s")
    print("  Failed log:      " + FAILED_LOG)
    print("  Grounding log:   " + GROUNDING_LOG)
    print("  Output:          " + OUTPUT_FILE)


if __name__ == "__main__":
    asyncio.run(main())