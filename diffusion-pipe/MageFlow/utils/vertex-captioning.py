#!/usr/bin/env python3
# """
# Vertex AI (Gemini) captioning script with exponential backoff for 429 errors.
# Install: pip install google-genai tenacity --break-system-packages
# """
import os
import time
import json
import asyncio
import io
from PIL import Image
from google import genai
from google.genai import types
from google.api_core.exceptions import ResourceExhausted
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

# ---- VERTEX AI CONFIG ----
PROJECT_ID = ""
LOCATION = "global"
IMAGE_FOLDER = "/home/jovyan/booru_essence_processed/links/e621/explicit"
OUTPUT_FILE = "captions_output_vertex.txt"
FAILED_LOG = "failed_images_vertex.log"
SPEND_LEDGER = "vertex_spend_ledger.json"

RETRY_FAILED_ONLY = False
SKIP_IF_CAPTIONED = True
MAX_CONCURRENCY = 15
PREPROCESS_ALPHA = False
ALPHA_VALUE = 0.9
PROMPT_FILE = "/home/jovyan/models/captioners/gemini-flash-2.5-fixed.md"

# Pricing
PRICE_INPUT_PER_TOKEN = 0.75 / 1_000_000
PRICE_OUTPUT_PER_TOKEN = 3.75 / 1_000_000

ENABLE_GROUNDING = True
GROUNDING_PRICE_PER_QUERY = 14.0 / 1_000

MAX_BUDGET_USD = 270.00
RETRY_ON_BLOCK_ATTEMPTS = 1

SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
]

PHASES = [
    {"label": "Gemini37Flash", "model": "gemini-3.7-flash", "suffix": "_nl", "enabled": True, "limit": None, "offset": 0},
]

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
_failed_by_model = {}
_spend_lock = asyncio.Lock()
_spend_state = {"total_usd": 0.0}

def load_ledger():
    if os.path.isfile(SPEND_LEDGER):
        with open(SPEND_LEDGER, "r", encoding="utf-8") as f:
            data = json.load(f)
        _spend_state["total_usd"] = data.get("total_usd", 0.0)
        print("Loaded spend ledger: $" + format(_spend_state["total_usd"], ".4f") + " already spent this ledger.")

def save_ledger_sync():
    with open(SPEND_LEDGER, "w", encoding="utf-8") as f:
        json.dump(
            {"total_usd": _spend_state["total_usd"], "updated": time.strftime("%Y-%m-%d %H:%M:%S")},
            f, indent=2
        )

async def record_spend(input_tokens, output_tokens, grounding_queries=0):
    cost = (
        (input_tokens * PRICE_INPUT_PER_TOKEN)
        + (output_tokens * PRICE_OUTPUT_PER_TOKEN)
        + (grounding_queries * GROUNDING_PRICE_PER_QUERY)
    )
    async with _spend_lock:
        _spend_state["total_usd"] += cost
        save_ledger_sync()
    return cost

async def budget_remaining():
    async with _spend_lock:
        return MAX_BUDGET_USD - _spend_state["total_usd"]

def load_prompt(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError("Prompt file not found: " + file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

SYSTEM_PROMPT = load_prompt(PROMPT_FILE)

def is_refusal(text: str) -> bool:
    if not text: return False
    t = text.strip().lower()
    if t.startswith(("i'm sorry", "i cannot", "i can't", "i am unable", "as an ai", "sorry, but i", "i must decline")):
        if any(w in t[:200] for w in ["image", "caption", "describe", "generate", "picture", "request", "content", "fulfill"]):
            return True
        if len(t) < 150: 
            return True
    first_sentence = t.split('.')[0]
    keywords = ["content policy", "safety guidelines", "unable to generate", "unable to describe", 
                "cannot assist", "inappropriate", "explicit", "nsfw", "violates", "not allowed", 
                "refuse to", "harmful", "offensive", "cannot process"]
    if any(kw in first_sentence for kw in keywords):
        return True
    return False

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

def load_image_bytes(path):
    if PREPROCESS_ALPHA:
        raw = preprocess_image_alpha(path, target_alpha=ALPHA_VALUE)
        mime = "image/png"
    else:
        with open(path, "rb") as f:
            raw = f.read()
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        ext = "jpeg" if ext in ("jpg", "jpeg") else ext
        mime = "image/" + ext
    return raw, mime

def read_tags(image_path):
    tpath = os.path.splitext(image_path)[0] + ".txt"
    if os.path.isfile(tpath):
        with open(tpath, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None

def build_prompt(tags):
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

@retry(
    wait=wait_random_exponential(min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(ResourceExhausted)
)
async def _generate_content_with_retry(model, contents, config):
    return await client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=config
    )

async def _single_attempt(path, tags, semaphore, model):
    async with semaphore:
        raw, mime = load_image_bytes(path)
        prompt_text = build_prompt(tags)
        tools = [types.Tool(google_search=types.GoogleSearch())] if ENABLE_GROUNDING else None

        parts = []
        if prompt_text:
            parts.append(types.Part.from_text(text=prompt_text))
        parts.append(types.Part.from_bytes(data=raw, mime_type=mime))

        config = types.GenerateContentConfig(
            safety_settings=SAFETY_SETTINGS,
            tools=tools,
            system_instruction=SYSTEM_PROMPT,
        )

        response = await _generate_content_with_retry(model, parts, config)

        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0

        candidate = response.candidates[0] if response.candidates else None

        # --- DIAGNOSTIC: zero candidates = prompt-level block or silent drop.
        # Surface the real reason instead of the ambiguous finish_reason=None.
        if not response.candidates:
            pf = getattr(response, "prompt_feedback", None)
            block_reason = getattr(pf, "block_reason", None) if pf is not None else None
            raise ValueError("No candidates returned (block_reason=" + str(block_reason) + ", finish_reason=None)")

        grounding_queries = 0
        gmeta = getattr(candidate, "grounding_metadata", None) if candidate else None
        if gmeta is not None:
            queries = getattr(gmeta, "web_search_queries", None)
            grounding_queries = len(queries) if queries else 0
            if grounding_queries > 0:
                print("    [grounding] " + str(grounding_queries) + " search quer" + ("y" if grounding_queries == 1 else "ies") + " used: " + str(list(queries)))
                chunks = getattr(gmeta, "grounding_chunks", None) or []
                for chunk in chunks:
                    web = getattr(chunk, "web", None)
                    if web is not None:
                        title = getattr(web, "title", "?")
                        uri = getattr(web, "uri", "?")
                        print("    [grounding]   source: " + str(title) + " -- " + str(uri))

        await record_spend(input_tokens, output_tokens, grounding_queries)

        finish_reason = getattr(candidate, "finish_reason", None) if candidate else None
        blocked_reasons = ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII")
        if finish_reason is not None and str(finish_reason) in blocked_reasons:
            raise ValueError("Blocked by safety filter (finish_reason=" + str(finish_reason) + ")")

        try:
            caption = response.text
        except Exception as e:
            raise ValueError("Could not read response text (finish_reason=" + str(finish_reason) + "): " + str(e))

        if caption is None or caption.strip() == "":
            raise ValueError("Blank or empty response from API (finish_reason=" + str(finish_reason) + ")")
            
        if is_refusal(caption):
            raise ValueError("Blocked by text safety/refusal filter (LLM refusal)")
            
        return caption.strip()

async def caption_image_api(path, tags, semaphore, model):
    fname = os.path.basename(path)
    if fname in _failed_by_model.get(model.lower(), set()):
        raise RuntimeError("skipped")

    total_attempts = 1 + max(0, RETRY_ON_BLOCK_ATTEMPTS)
    last_error = None

    for attempt in range(1, total_attempts + 1):
        remaining = await budget_remaining()
        if remaining <= 0:
            raise RuntimeError("budget_exceeded")
        try:
            return await _single_attempt(path, tags, semaphore, model)
        except ValueError as e:
            last_error = e
            if attempt < total_attempts:
                print("    [retry] attempt " + str(attempt) + "/" + str(total_attempts) + " blocked (" + str(e) + "), retrying " + fname + "...")
                continue
            raise last_error

async def process_one(path, tags, semaphore, idx, total,
                      results, counts, model, suffix, label):
    fname = os.path.basename(path)
    sidecar = sidecar_path(path, suffix)
    existed = os.path.isfile(sidecar)

    if existed and SKIP_IF_CAPTIONED:
        counts["skipped_existing"] = counts.get("skipped_existing", 0) + 1
        print("[" + label + " " + str(idx) + "/" + str(total) + "] " + fname + " -> SKIPPED (already captioned)")
        results.setdefault(label, {})[fname] = "SKIPPED_EXISTING"
        return False

    if tags:
        counts["tags_used"] += 1

    try:
        caption = await caption_image_api(path, tags, semaphore, model)
    except RuntimeError as e:
        if str(e) == "skipped":
            counts["skipped"] += 1
            print("[" + label + " " + str(idx) + "/" + str(total) + "] " + fname + " -> SKIPPED (previous failure)")
            results.setdefault(label, {})[fname] = "SKIPPED"
            return False
        if str(e) == "budget_exceeded":
            counts["budget_stopped"] = counts.get("budget_stopped", 0) + 1
            print("[" + label + " " + str(idx) + "/" + str(total) + "] " + fname + " -> STOPPED (MAX_BUDGET_USD reached)")
            results.setdefault(label, {})[fname] = "BUDGET_STOPPED"
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
    print()

    tasks = []
    for i, item in enumerate(subset, 1):
        tasks.append(
            process_one(
                item["path"], item["tags"], semaphore,
                i, len(subset), results, counts,
                model, suffix, label
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
    load_ledger()

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

    print("Provider: Vertex AI (project=" + PROJECT_ID + ", location=" + LOCATION + ")")
    print("Enabled models: " + ", ".join(p["model"] for p in enabled_phases))
    print("Concurrency: " + str(MAX_CONCURRENCY))
    print("Alpha preprocess: " + str(PREPROCESS_ALPHA) + " (" + (str(ALPHA_VALUE) if PREPROCESS_ALPHA else "N/A") + ")")
    print("Skip-if-captioned: " + str(SKIP_IF_CAPTIONED))
    print("Budget cap: $" + format(MAX_BUDGET_USD, ".2f") + "  |  Already spent (ledger): $" + format(_spend_state["total_usd"], ".4f"))
    print("System prompt: " + str(len(SYSTEM_PROMPT)) + " chars from " + PROMPT_FILE + " (sent as system_instruction)")
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

    # Startup cleanup for refusals
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

    image_data = []
    tags_used = 0
    for fname in images:
        path = os.path.join(IMAGE_FOLDER, fname)
        tags = read_tags(path)
        if tags:
            tags_used += 1
        image_data.append({"fname": fname, "path": path, "tags": tags})

    if RETRY_FAILED_ONLY:
        all_failed = set()
        for fnames in _failed_by_model.values():
            all_failed |= fnames
        if not all_failed:
            print("RETRY MODE: No failed images to retry.")
            return
        image_data = [item for item in image_data if item["fname"] in all_failed]
        print("RETRY MODE: Only processing " + str(len(image_data)) + " previously failed image(s)")
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
        "budget_stopped": 0,
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
    print("  Budget-stopped:  " + str(counts["budget_stopped"]))
    print("  Tags used:       " + str(counts["tags_used"]))
    print("  Time:            " + str(round(elapsed, 1)) + "s")
    print("  Failed log:      " + FAILED_LOG)
    print("  Output:          " + OUTPUT_FILE)
    print("  Total spend (ledger, all-time): $" + format(_spend_state["total_usd"], ".4f"))
    print("  Budget remaining: $" + format(MAX_BUDGET_USD - _spend_state["total_usd"], ".4f"))

    if counts["budget_stopped"] > 0:
        print()
        print("  !! MAX_BUDGET_USD was reached before all images were processed.")
        print("     Raise MAX_BUDGET_USD and re-run with RETRY_FAILED_ONLY-style logic")
        print("     (budget-stopped images are recorded as BUDGET_STOPPED in " + OUTPUT_FILE + ",")
        print("     not in " + FAILED_LOG + ", since they were never actually attempted).")

if __name__ == "__main__":
    asyncio.run(main())