#!/usr/bin/env python3
# """
# RunPod vLLM (Qwen 27B) captioning script -- same structure as the LinkAPI script:
# PHASES with suffix/limit/offset, resume toggle, retry modes, failed log.
# Thinking is LEFT ON;  blocks are stripped from the stored caption.
# ONLY_MISSING_NL=True restricts the run to images that have no _nl.txt sidecar.
#
# GROUNDING (application-layer, vLLM has no native search) -- MAX QUALITY MODE:
#   1. Text-only planning call extracts candidate character-identity tags.
#   2. Candidates are validated against the booru tag-category API
#      (1=artist / 3=copyright are dropped, cached forever) so artist tags
#      never waste search time.
#   3. Each surviving character is searched via a 3-tier chain:
#        a) Danbooru/e621 wiki page for the exact tag
#        b) DuckDuckGo (3 attempts + backoff)
#        c) Wikipedia -- last resort
#      plus a persistent disk cache so each unique character is searched once ever.
#   4. Per-character <grounding> block injected with attribution instructions.
#   Never fatal: any grounding failure = caption proceeds ungrounded.
# """
import os
import re
import io
import json
import base64
import time
import random
import asyncio
from pathlib import Path
from PIL import Image
import requests
from openai import AsyncOpenAI, RateLimitError, APIConnectionError
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

# ---- VLLM CONFIG ----
BASE_URL = "https://bgm0cvjk5xw8qx-8000.proxy.runpod.net/v1"
API_KEY = ""
LINKS = Path("/home/jovyan/booru_essence_processed/links")
FOLDERS = ["danbooru/safe", "danbooru/explicit", "e621/safe", "e621/explicit"]
EXT = {".jpg", ".jpeg", ".png", ".webp"}
OUTPUT_FILE = "captions_output_qwen.txt"
FAILED_LOG = "failed_images_qwen.log"

ONLY_MISSING_NL = True
RETRY_FAILED_ONLY = False
SKIP_IF_CAPTIONED = True

RETRY_ON_BLOCK_ATTEMPTS = 1

RATE_LIMIT_MAX_ATTEMPTS = 5
RATE_LIMIT_WAIT_MIN_SECONDS = 2
RATE_LIMIT_WAIT_MAX_SECONDS = 60

MAX_CONCURRENCY = 16
MAX_IMAGE_DIM = 2048
TEMPERATURE = 0.1
MAX_TOKENS = 4096
PROMPT_FILE = "/home/jovyan/models/captioners/gemini-flash-2.5-fixed.md"

# ---- GROUNDING ----
ENABLE_GROUNDING = True
GROUNDING_MAX_CHARACTERS = 20
GROUNDING_MAX_RESULTS = 3
GROUNDING_LOG = "grounding_log_qwen.jsonl"
SEARCH_CONCURRENCY = 2
SEARCH_CACHE_FILE = "search_cache_qwen.json"        # query -> hits
TAG_CATEGORY_CACHE_FILE = "tag_category_cache.json" # "source|tag" -> category
UA = {"User-Agent": "riceman-repair/1.0"}

# ---- PHASES ----
PHASES = [
    {"label": "Qwen", "model": "Qwen/Qwen3.8-27B-FP8", "suffix": "_nl", "enabled": True, "limit": None, "offset": 0},
]

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

_failed_by_model = {}
_search_sem = asyncio.Semaphore(SEARCH_CONCURRENCY)
_cache_lock = asyncio.Lock()

_search_cache = {}
if os.path.isfile(SEARCH_CACHE_FILE):
    try:
        with open(SEARCH_CACHE_FILE, "r", encoding="utf-8") as f:
            _search_cache = json.load(f)
    except Exception:
        _search_cache = {}

_tagcat_cache = {}
if os.path.isfile(TAG_CATEGORY_CACHE_FILE):
    try:
        with open(TAG_CATEGORY_CACHE_FILE, "r", encoding="utf-8") as f:
            _tagcat_cache = json.load(f)
    except Exception:
        _tagcat_cache = {}


def _save_cache():
    tmp = SEARCH_CACHE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_search_cache, f, ensure_ascii=False)
    os.replace(tmp, SEARCH_CACHE_FILE)


def _save_tagcat():
    tmp = TAG_CATEGORY_CACHE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_tagcat_cache, f, ensure_ascii=False)
    os.replace(tmp, TAG_CATEGORY_CACHE_FILE)


def load_prompt(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError("Prompt file not found: " + file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


SYSTEM_PROMPT = load_prompt(PROMPT_FILE)

GROUNDING_SYSTEM = (
    "You are an assistant that extracts character-identity tags from booru tag lists. "
    "Character-identity tags name a specific fictional character or persona; per Danbooru/e621 "
    "conventions they may look like 'name (series)', 'name (alternate outfit)', "
    "'name (outfit) (series)', or a bare personal name. The FIRST tag in the list is usually "
    "the ARTIST tag, and artist tags (real-person or circle names) as well as series/copyright "
    "tags (e.g. 'touhou', 'one piece') and generic tags are NOT character-identity tags -- never "
    "extract them. Extract EVERY character-identity tag present, whether or not you recognize "
    "the character, exactly as written in the list. Reply with ONLY this JSON: "
    '{"characters": ["tag as written", ...]}'
)


# ---- FLAT REFUSAL DETECTOR ----
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
    """Returns the trigger string if the output is a refusal, else None."""
    if not text: return None
    t = _norm(text.strip())
    u = QUOTED.sub('', t).lower().replace("'", "")
    m = REFUSAL_I.search(u)
    if m: return "I+refusal verb: " + " ".join(m.group(0).split())
    for p in PREFIXES:
        if u.startswith(p): return "prefix: " + p
    for p in META:
        if p in u: return "meta: " + p
    sentences = [s for s in re.split(r"[.!?]+", u) if len(s.strip()) > 1]
    if len(sentences) <= 2 and len(u) < 450:
        policy_words = ["policy", "guidelines", "restrictions", "inappropriate",
                        "offensive", "harmful", "decline", "assist", "fulfill",
                        "comply", "safety", "violate", "cannot", "unable",
                        "sorry", "apologize"]
        hits = [w for w in policy_words if w in u]
        if len(hits) >= 2: return "density: " + ",".join(hits)
    return None


@retry(
    wait=wait_random_exponential(min=RATE_LIMIT_WAIT_MIN_SECONDS, max=RATE_LIMIT_WAIT_MAX_SECONDS),
    stop=stop_after_attempt(RATE_LIMIT_MAX_ATTEMPTS),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def _chat_completions_with_retry(**kwargs):
    return await client.chat.completions.create(**kwargs)


def build_user_text(tags, grounding=""):
    if not tags:
        return ""
    text = (
        "Grounded tags for this image, scraped from Danbooru or e621 (use them to inform "
        "the caption, but write natural prose, do not just list them back):\n"
        "<tags>\n" + tags + "\n</tags>\n"
        "Some of these tags are character-identity tags. Danbooru's naming convention "
        "for these is inconsistent: some are 'name (series)', some are 'name (alternate "
        "outfit)' with no series at all, some are 'name (outfit) (series)', and some are "
        "a bare name with no parentheses. Use your own knowledge of Danbooru and e621 tagging "
        "conventions to identify which tag(s), if any, name a specific character. If you "
        "recognize the character, treat that identification as ground truth and describe "
        "that exact character -- do not substitute a different, visually-similar character."
    )
    if grounding:
        text += "\n" + grounding
    return text


# ---- TAG CATEGORY VALIDATION (1=artist, 3=copyright, 4=character) ----
def _tag_category_once(tag, source):
    """Booru tag category for the exact tag; None if unknown. Never raises."""
    title = tag.replace(" ", "_")
    try:
        if source == "danbooru":
            r = requests.get("https://danbooru.donmai.us/tags.json",
                             params={"search[name]": title, "limit": 1}, headers=UA, timeout=15)
            data = r.json()
        else:
            r = requests.get("https://e621.net/tags.json",
                             params={"search[name]": title, "limit": 1}, headers=UA, timeout=15)
            data = r.json()
            if isinstance(data, dict):
                data = data.get("tags", [])
        if isinstance(data, list) and data:
            return data[0].get("category")
    except Exception as e:
        print("    [grounding] tag category error for '" + tag + "': " + str(e)[:80])
    return None


# ---- SEARCH BACKENDS (3-tier chain: booru wiki -> DDG -> Wikipedia) ----
def _ddgs_once(query, max_results):
    """Returns list of hits, [] for clean no-results, None on hard error."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        return list(DDGS().text(query, max_results=max_results) or [])
    except Exception as e:
        print("    [grounding] ddgs error for '" + query + "': " + str(e)[:100])
        return None


def _booru_wiki_search(tag, source):
    """The boorus' own wiki pages: community-curated notes for the exact tag."""
    title = tag.replace(" ", "_")
    sites = ["e621", "danbooru"] if source == "e621" else ["danbooru", "e621"]
    for site in sites:
        try:
            if site == "danbooru":
                url = "https://danbooru.donmai.us/wiki_pages.json"
                page = "https://danbooru.donmai.us/wiki_pages/"
            else:
                url = "https://e621.net/wiki_pages.json"
                page = "https://e621.net/wiki_pages/"
            r = requests.get(url, params={"search[title]": title, "limit": 1},
                             headers=UA, timeout=15)
            data = r.json()
            if isinstance(data, dict):
                data = data.get("wiki_pages", [])
            if data:
                w = data[0]
                body = (w.get("body") or "").strip()
                if body:
                    if len(body) > 1500:
                        body = body[:1500] + "..."
                    return [{"title": site + " wiki page for '" + tag + "'",
                             "body": body,
                             "href": page + str(w.get("id", ""))}]
        except Exception as e:
            print("    [grounding] " + site + " wiki error for '" + tag + "': " + str(e)[:80])
    return []


def _wiki_search(query, max_results):
    """Keyless Wikipedia search; last resort. Never raises."""
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "format": "json",
                    "srlimit": max_results, "srsearch": query},
            headers=UA, timeout=15,
        )
        hits = []
        for h in (r.json().get("query", {}) or {}).get("search", []):
            hits.append({
                "title": h.get("title", ""),
                "body": re.sub(r"<[^>]+>", "", h.get("snippet", "")),
                "href": "https://en.wikipedia.org/?curid=" + str(h.get("pageid", "")),
            })
        return hits
    except Exception as e:
        print("    [grounding] wiki error for '" + query + "': " + str(e)[:80])
        return []


def _robust_search(query, tag, source, max_results):
    """Booru wiki page -> DDG (3 attempts + backoff) -> Wikipedia. Never raises."""
    hits = _booru_wiki_search(tag, source)   # in-domain: exact tag's wiki page
    if hits:
        return hits
    for attempt in range(3):                 # fallback: general web (Fandom etc.)
        hits = _ddgs_once(query, max_results)
        if hits:
            return hits
        if hits is not None:
            break
        time.sleep(1.5 * (attempt + 1) + random.random())
    return _wiki_search(query, max_results)  # last resort


def _char_query(tag):
    """'hakurei reimu (touhou)' -> 'hakurei reimu touhou character appearance'."""
    flat = re.sub(r"[()]", " ", tag)
    flat = re.sub(r"\s+", " ", flat).strip()
    return flat + " character appearance"


def log_grounding(fname, model, queries, sources):
    with open(GROUNDING_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "image": fname, "model": model, "queries": queries, "sources": sources,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }) + "\n")


async def get_grounding(fname, tags, model, source, semaphore):
    """Extract character tags -> validate categories -> search -> <grounding>. Never raises."""
    try:
        async with semaphore:
            resp = await _chat_completions_with_retry(
                model=model,
                messages=[
                    {"role": "system", "content": GROUNDING_SYSTEM},
                    {"role": "user", "content": tags},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
        raw = (resp.choices[0].message.content or "") if resp.choices else ""
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
        chars = []
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                chars = json.loads(m.group(0)).get("characters", [])
            except Exception:
                chars = []
        if not isinstance(chars, list):
            chars = []
        chars = [c.strip() for c in chars if isinstance(c, str) and c.strip() and c.strip() in tags]

        # ---- validate against booru tag categories; drop artist(1)/copyright(3) ----
        async def categorize(c):
            key = source + "|" + c
            async with _cache_lock:
                if key in _tagcat_cache:
                    return c, _tagcat_cache[key]
            async with _search_sem:
                cat = await asyncio.to_thread(_tag_category_once, c, source)
            async with _cache_lock:
                _tagcat_cache[key] = cat
                _save_tagcat()
            return c, cat

        if chars:
            cat_results = await asyncio.gather(*[categorize(c) for c in chars])
            dropped = [c for c, cat in cat_results if cat in (1, 3)]
            chars = [c for c, cat in cat_results if cat not in (1, 3)]
        else:
            dropped = []

        chars = chars[:GROUNDING_MAX_CHARACTERS]
        if not chars:
            return ""

        async def search_one(tag):
            q = _char_query(tag)
            async with _cache_lock:
                cached = _search_cache.get(q)
            if cached is not None:
                return tag, cached, True
            async with _search_sem:
                hits = await asyncio.to_thread(_robust_search, q, tag, source, GROUNDING_MAX_RESULTS)
            async with _cache_lock:
                _search_cache[q] = hits
                _save_cache()
            return tag, hits, False

        results = await asyncio.gather(*[search_one(c) for c in chars])

        blocks, sources, n_hits, n_cached = [], [], 0, 0
        for tag, hits, cached in results:
            if cached:
                n_cached += 1
            lines = []
            for h in hits:
                lines.append("- " + str(h.get("title", "")) + ": " + str(h.get("body", "")))
                sources.append(str(h.get("href", "")))
            n_hits += len(lines)
            blocks.append("[" + tag + "]\n" + ("\n".join(lines) if lines else "- (no results)"))

        print("    [grounding] " + fname + ": " + str(len(chars)) + " character(s), "
              + str(len(dropped)) + " non-character dropped, " + str(n_cached) + " cached, "
              + str(n_hits) + " result(s)")
        log_grounding(fname, model, [_char_query(c) for c in chars], sources)
        return (
            "<grounding>\n"
            "Character reference notes gathered from web search, one block per character-identity "
            "tag. Use them to (1) confirm which character each tag names, (2) describe each "
            "character's appearance accurately (hair, outfit, accessories, colors), and (3) when "
            "multiple characters appear, attribute the correct name to each figure you describe by "
            "matching these appearance cues to what you see in the image. If a block has no useful "
            "results, fall back to your own knowledge.\n"
            + "\n".join(blocks) +
            "\n</grounding>"
        )
    except Exception as e:
        print("    [grounding] disabled for " + fname + " due to error: " + str(e)[:120])
        return ""


def encode_image(path):
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = MAX_IMAGE_DIM / max(w, h)
        if scale < 1:
            im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


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


async def caption_image_api(path, tags, semaphore, model):
    fname = os.path.basename(path)
    if fname in _failed_by_model.get(model.lower(), set()):
        raise RuntimeError("skipped")

    grounding = ""
    if ENABLE_GROUNDING and tags:
        source = Path(path).parent.parent.name   # "danbooru" or "e621"
        grounding = await get_grounding(fname, tags, model, source, semaphore)

    user_text = build_user_text(tags, grounding)
    total_attempts = 1 + max(0, RETRY_ON_BLOCK_ATTEMPTS)
    last_error = None

    for attempt in range(1, total_attempts + 1):
        try:
            async with semaphore:
                b64 = encode_image(path)

                user_content = []
                if user_text:
                    user_content.append({"type": "text", "text": user_text})
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64," + b64},
                })

                response = await _chat_completions_with_retry(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )

                if not response.choices:
                    raise ValueError("No choices returned from API (possible upstream block)")

                choice = response.choices[0]
                finish_reason = getattr(choice, "finish_reason", None)

                caption = choice.message.content or ""
                caption = re.sub(r"<think>.*?</think>", "", caption, flags=re.S).strip()

                if caption is None or caption.strip() == "":
                    raise ValueError(f"Blank or empty response from API (finish_reason={finish_reason})")

                trig = is_refusal(caption)
                if trig:
                    raise ValueError("Blocked by text safety/refusal filter (LLM refusal: " + trig + ")")

                return caption.strip()

        except ValueError as e:
            last_error = e
            if attempt < total_attempts:
                print("    [retry] attempt " + str(attempt) + "/" + str(total_attempts) + " blocked/blank (" + str(e) + "), retrying " + fname + "...")
                await asyncio.sleep(1)
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
    print("  suffix: " + suffix + " | temp=" + str(TEMPERATURE) + " max_tokens=" + str(MAX_TOKENS) + " (thinking ON)")
    print("  grounding: " + ("on (category-validated, booru wiki -> DDG -> wikipedia, cached)" if ENABLE_GROUNDING else "off"))
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
    enabled_phases = [p for p in PHASES if p["enabled"]]

    print("Provider: RunPod vLLM (" + BASE_URL + ")")
    print("Enabled models: " + ", ".join(p["model"] for p in enabled_phases))
    print("Only-missing-_nl: " + str(ONLY_MISSING_NL))
    print("Concurrency: " + str(MAX_CONCURRENCY) + " (search: " + str(SEARCH_CONCURRENCY) + ")")
    print("Skip-if-captioned: " + str(SKIP_IF_CAPTIONED))
    print("Search cache: " + SEARCH_CACHE_FILE + " (" + str(len(_search_cache)) + " cached queries)")
    print("Tag category cache: " + TAG_CATEGORY_CACHE_FILE + " (" + str(len(_tagcat_cache)) + " cached)")
    print("Grounding: " + str(ENABLE_GROUNDING) + " (log=" + GROUNDING_LOG + ")")
    print("System prompt: " + str(len(SYSTEM_PROMPT)) + " chars from " + PROMPT_FILE)
    print("Tag sidecars are READ ONLY.\n")

    image_data = []
    tags_used = 0
    for folder in FOLDERS:
        root = LINKS / folder
        if not root.exists():
            print(folder + ": folder not found, skipped")
            continue
        for fname in sorted(os.listdir(root)):
            p = root / fname
            if p.suffix.lower() not in EXT:
                continue
            if ONLY_MISSING_NL and (root / (p.stem + "_nl.txt")).exists():
                continue
            path = str(p)
            tags = read_tags(path)
            if tags:
                tags_used += 1
            image_data.append({"fname": fname, "path": path, "tags": tags})

    print("Target set: " + str(len(image_data)) + " image(s) without _nl sidecar\n")

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
    for item in image_data:
        path = item["path"]
        for phase in enabled_phases:
            sc = sidecar_path(path, phase["suffix"])
            if os.path.isfile(sc):
                with open(sc, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if is_refusal(content):
                    os.remove(sc)
                    log_failed(item["fname"], phase["model"], "Deleted existing refusal output")
                    _failed_by_model.setdefault(phase["model"].lower(), set()).add(item["fname"])
                    deleted_refusals += 1
    if deleted_refusals > 0:
        print(f"Cleaned up {deleted_refusals} existing refusal sidecar(s) and logged them as failed.")
    # ---------------------------------------------

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
            for item in image_data:
                f.write(item["fname"] + "\t" + phase_results.get(item["fname"], "NOT_PROCESSED") + "\n")
            f.write("\n")

    print("=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print("Total targets: " + str(len(image_data)))
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