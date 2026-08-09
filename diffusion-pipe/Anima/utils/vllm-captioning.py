#!/usr/bin/python3
# =============================================================================
# INSTALLATION (run once in your environment):
#   pip install vllm openai
#   pip install xxhash openai --break-system-packages
# VLLM SERVER STARTUP (quick start up with no speedup trick, used when desperate or the speedup trick waste too much precious compute to compile)
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# vllm serve /teamspace/studios/this_studio/Qwen3.6-27B-FP8 \
#   --port 8080 \
#   --served-model-name local-model \
#   --max-model-len 8192 \
#   --gpu-memory-utilization 0.95 \
#   --max-num-seqs 64 \
#   --enable-chunked-prefill \
#   --enable-prefix-caching \
#   --prefix-caching-hash-algo xxhash \
#   --reasoning-parser qwen3 \
#   --generation-config vllm \
#   --enforce-eager \
#   --gdn-prefill-backend triton \
#   --disable-uvicorn-access-log
# (with speedup tricks)
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# vllm serve /teamspace/studios/this_studio/Qwen3.6-27B-FP8 \
#   --port 8080 \
#   --served-model-name local-model \
#   --max-model-len 8192 \
#   --gpu-memory-utilization 0.95 \
#   --max-num-seqs 64 \
#   --enable-chunked-prefill \
#   --enable-prefix-caching \
#   --prefix-caching-hash-algo xxhash \
#   --block-size 16 \
#   --mamba-cache-dtype float16 \
#   --mamba-ssm-cache-dtype float16 \
#   --reasoning-parser qwen3 \
#   --speculative-config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 3}' \
#   --generation-config vllm \
#   --performance-mode throughput \
#   --renderer-num-workers 8 \
#   --mm-processor-cache-gb 0 \
#   --disable-uvicorn-access-log
# Only move on once you see it output INFO:     Uvicorn running on http://0.0.0.0:8080
# Then run this script in a separate terminal python /teamspace/studios/this_studio/vllm-label.py
# =============================================================================

import time
import asyncio
import base64
import mimetypes
from pathlib import Path
from openai import AsyncOpenAI

# =============================================================================
# CONFIG — edit these
# =============================================================================
input_dir   = Path("/teamspace/studios/this_studio/Anime-Background-Finetuning-V1.1")
prompt_file = Path("/teamspace/studios/this_studio/caption_groundBGFinetuning.md")
extensions  = [".png", ".jpg", ".jpeg", ".webp"]

MODEL_NAME  = "local-model"   # must match --served-model-name in vllm serve
CONCURRENCY = 64              # concurrent requests sent to vLLM at once
                              # H100 can handle high concurrency — tune up if GPU util is low
# =============================================================================

if not prompt_file.exists():
    raise FileNotFoundError(f"Missing required prompt file: {prompt_file}")

sys_prompt = prompt_file.read_text(encoding="utf-8").strip()

client = AsyncOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8080/v1",
    timeout=3600,
)


# --- Helpers (unchanged from original) ---------------------------------------

def encode_image_to_base64(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"
    with open(image_path, "rb") as f:
        return f"data:{mime_type};base64,{base64.b64encode(f.read()).decode('utf-8')}"


def trim_incomplete_sentence(text: str) -> str:
    terminators = ('.', '!', '?')
    last_idx = max(text.rfind(t) for t in terminators)
    if last_idx != -1:
        return text[:last_idx + 1].strip()
    return text.strip()


def clean_tag_string(tags: str, keep_tags_n: int = 100) -> str:
    tags = tags.replace('\\', '')
    return ", ".join(tags.split(', ')[:keep_tags_n])


# --- Core async worker -------------------------------------------------------

async def process_image(
    img_file: Path,
    semaphore: asyncio.Semaphore,
    index: int,
    total: int,
) -> None:
    async with semaphore:
        start = time.perf_counter()

        output_file = img_file.parent / f"{img_file.stem}_nl.txt"
        tags_file   = img_file.with_suffix(".txt")

        tags_content = ""
        if tags_file.exists():
            tags_content = clean_tag_string(tags_file.read_text(encoding="utf-8").strip())

        base64_image_url = encode_image_to_base64(img_file)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": sys_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": base64_image_url}},
                    {"type": "text", "text": (
                        f"Image Tags: {tags_content}\n\n"
                        "Instruction: Use the above Image Tags as a guideline to provide "
                        "natural language captions for the image. Avoid complicated "
                        "formatting like unusual punctuation marks, lists, or markdown."
                    )},
                ],
            },
        ]

        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=1024,
                temperature=1.0,
                top_p=0.95,
                presence_penalty=0.0,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )

            output = response.choices[0].message.content.strip()

            if not output:
                print(f"⚠️  [{index}/{total}] Empty response — {img_file.name}")
                return

            output = trim_incomplete_sentence(output)
            output_file.write_text(output, encoding="utf-8")

            elapsed = time.perf_counter() - start
            print(f"✅  [{index}/{total}] {img_file.name}  ({elapsed:.2f}s)")

        except Exception as e:
            print(f"❌  [{index}/{total}] {img_file.name} — {e}")


# --- Entry point -------------------------------------------------------------

async def main() -> None:
    # Collect all unprocessed images upfront (same skip logic as original)
    pending: list[Path] = []
    for img_file in input_dir.rglob("*"):
        if any(part.startswith('.') for part in img_file.parts):
            continue
        if img_file.suffix.lower() not in extensions:
            continue
        if not (img_file.parent / f"{img_file.stem}_nl.txt").exists():
            pending.append(img_file)

    total = len(pending)
    if total == 0:
        print("✅  All images already processed. Nothing to do.")
        return

    print(f"📋  {total} images to process | concurrency = {CONCURRENCY}\n")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [
        process_image(img_file, semaphore, i + 1, total)
        for i, img_file in enumerate(pending)
    ]
    await asyncio.gather(*tasks)

    print(f"\n🎉  Done. {total} images processed.")


if __name__ == "__main__":
    asyncio.run(main())