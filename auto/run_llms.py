#!/usr/bin/env python3
# run_llms.py
"""
Extract {"cause": "...", "effect": "..."} from each Context paragraph
via any of eight LLMs. The CSV you give must contain:

    id , Context            (required)
    Context_masked          (optional; takes priority)
    Cause , Effect          (optional; used only for scoring)

--------------------------------------------------------------------
Model key    Vendor            Concrete model / endpoint
--------------------------------------------------------------------
openai       OpenAI            GPT‑4o (chat)
claude       Anthropic         Claude 3.7 Sonnet
grok         xAI               Grok‑3 (chat)
gemini       Google DeepMind   Gemini 2.5 Pro (live)
deepseek     DeepSeek          DeepSeek R1 (local)
qwen         Alibaba           Qwen QwQ‑32B (HF API)
llama3       Meta              Llama 3.1 405B (local)
mistral      Mistral AI        Mistral Small 3 (local)
--------------------------------------------------------------------
"""

import os, json, time, pathlib, argparse, httpx
from typing import Callable, Dict
import pandas as pd
import backoff

# ─────────────────────────────────────────────────────────────────────────────
# 0. Prompt template
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = (
    "You are a concise information‑extraction assistant.\n"
    "Extract exactly one cause phrase and one effect phrase from the paragraph "
    "below and return VALID JSON with keys \"cause\" and \"effect\"—no prose.\n\n"
    "<paragraph>\n{context}\n</paragraph>"
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Retry logic
# ─────────────────────────────────────────────────────────────────────────────
def _retry():
    return backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=5,
        jitter=backoff.full,
        factor=2,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 2. Model wrappers
# ─────────────────────────────────────────────────────────────────────────────

# OpenAI GPT-4o
# https://platform.openai.com/docs/guides/text?api-mode=responses&prompt-example=code
@_retry()
def call_openai(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=256,
    )
    return resp.choices[0].message.content.strip()

# Anthropic Claude 3.7
# https://docs.anthropic.com/en/api/getting-started
@_retry()
def call_claude(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=256,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()

# xAI Grok-3
# https://docs.x.ai/docs/guides/chat#chat
@_retry()
def call_grok(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("XAI_API_KEY"),
        base_url="https://api.x.ai/v1"
    )
    resp = client.chat.completions.create(
        model="grok-3-latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=256,
    )
    return resp.choices[0].message.content.strip()

# Google Gemini 2.5 Pro (Live API)
# https://ai.google.dev/gemini-api/docs/live
@_retry()
def call_gemini(prompt: str) -> str:
    import asyncio
    import nest_asyncio
    from google import genai
    from google.genai.types import Part

    nest_asyncio.apply()
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    async def run():
        async with client.aio.live.connect(
            model="gemini-2.5-pro",
            config={"response_modalities": ["TEXT"]}
        ) as session:
            await session.send_client_content(
                turns={"role": "user", "parts": [Part(text=prompt)]},
                turn_complete=True
            )

            collected = ""
            async for response in session.receive():
                if response.text:
                    collected += response.text
                if getattr(response.server_content, "generation_complete", False):
                    break
            return collected.strip()

    return asyncio.run(run())

# DeepSeek R1 (local vLLM)
@_retry()
def call_deepseek(prompt: str) -> str:
    r = httpx.post(
        "http://localhost:8100/v1/chat/completions",
        json={
            "model": "deepseek-r1",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 256,
        },
        timeout=120,
    )
    return r.json()["choices"][0]["message"]["content"].strip()

# Alibaba Qwen QwQ‑32B (HF endpoint)
@_retry()
def call_qwen(prompt: str) -> str:
    url = os.getenv("QWEN_ENDPOINT")
    hdr = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"}
    payload = {"inputs": prompt, "parameters": {"temperature": 0, "max_new_tokens": 256}}
    r = httpx.post(url, json=payload, headers=hdr, timeout=120)
    r.raise_for_status()
    return r.json()[0]["generated_text"].strip()

# Meta Llama3.1 (local)
@_retry()
def call_llama3(prompt: str) -> str:
    r = httpx.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "llama3-405b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 256,
        },
        timeout=120,
    )
    return r.json()["choices"][0]["message"]["content"].strip()

# Mistral Small 3 (local)
@_retry()
def call_mistral(prompt: str) -> str:
    r = httpx.post(
        "http://localhost:8200/v1/chat/completions",
        json={
            "model": "mistral-7b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 256,
        },
        timeout=120,
    )
    return r.json()["choices"][0]["message"]["content"].strip()

# Registry
MODELS: Dict[str, Callable[[str], str]] = {
    "openai":  call_openai,
    "claude":  call_claude,
    "grok":    call_grok,
    "gemini":  call_gemini,
    "deepseek":call_deepseek,
    "qwen":    call_qwen,
    "llama3":  call_llama3,
    "mistral": call_mistral,
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. Utilities
# ─────────────────────────────────────────────────────────────────────────────
def safe_json(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {"cause": "", "effect": ""}

# ─────────────────────────────────────────────────────────────────────────────
# 4. Main driver
# ─────────────────────────────────────────────────────────────────────────────
def run_models(csv_path: str, keys):
    df = pd.read_csv(csv_path)
    if "id" not in df.columns:
        df["id"] = [f"row{i}" for i in range(len(df))]

    runs_dir = pathlib.Path("runs"); runs_dir.mkdir(exist_ok=True)

    for key in keys:
        fn = MODELS[key]
        out_fp = runs_dir / f"{key}__{pathlib.Path(csv_path).stem}.jsonl"
        print(f"▶ {key:8s} → {out_fp}")
        with out_fp.open("w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                ctx_column = "Context_masked" if "Context_masked" in row else "Context"
                prompt = PROMPT_TEMPLATE.format(context=row[ctx_column])

                try:
                    reply = fn(prompt)
                except Exception as e:
                    print(f"   ! {key} error (id={row['id']}):", e)
                    reply = "{}"

                record = {"id": row["id"], "pred": safe_json(reply)}
                if {"Cause", "Effect"}.issubset(row.index):
                    record["ground"] = {"cause": row["Cause"], "effect": row["Effect"]}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# 5. CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="call-ready CSV (id, Context[,_masked])")
    ap.add_argument(
        "--model",
        default="all",
        choices=["all"] + list(MODELS.keys()),
        help="run only one model (default: all)",
    )
    args = ap.parse_args()
    selected = list(MODELS.keys()) if args.model == "all" else [args.model]
    run_models(args.csv, selected)