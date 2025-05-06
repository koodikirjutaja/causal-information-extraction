
#!/usr/bin/env python3
# judge_llm_extractions.py
"""
Ask one (or all) LLMs to grade an extracted cause/effect pair against
ground truth. Each judge assigns a NUMERIC score per span:

    1.0  – wording matches exactly (ignoring case / punctuation)
    0.5  – clear synonym / paraphrase, meaning identical
    0.0  – wrong, missing, or too vague to decide

The judge must return JSON:
    {"cause_score": 0|0.5|1, "effect_score": 0|0.5|1}

------------------------------------------------------------------
CLI
------------------------------------------------------------------
python judge_llm_extractions.py \
        --pred  runs/openai__easy_masked_call.jsonl \
        --truth data/processed/v1/easy_masked.csv   \
        [--judge openai|claude|…|all]

Writes one JSONL per judge:
   runs/score_<judge>__<source>__<dataset>.jsonl

Each line:
   {"id": "...", "cause_score": 0/0.5/1, "effect_score": 0/0.5/1}
"""

import os, json, pathlib, argparse, httpx, backoff, pandas as pd
from typing import Dict, Callable

import openai
import anthropic

# ─────────────────────────────────────────────────────────────────────────────
# Prompt sent to each model during evaluation
# ─────────────────────────────────────────────────────────────────────────────
PROMPT = """id: {id}
predicted_by: {source_model}

original_cause : {orig_cause}
original_effect: {orig_effect}

responded_cause : {pred_cause}
responded_effect: {pred_effect}

SCORING RULE (choose numeric):
    1   – wording matches exactly (case/punct ignored)
    0.5 – synonym / paraphrase with identical meaning
    0   – wrong, missing, or too vague

Return VALID JSON only:
{{"cause_score": 0|0.5|1, "effect_score": 0|0.5|1}}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Retry logic
# ─────────────────────────────────────────────────────────────────────────────
def _retry():
    return backoff.on_exception(backoff.expo, Exception, max_tries=5, jitter=backoff.full)

# ─────────────────────────────────────────────────────────────────────────────
# LLM API Wrappers
# ─────────────────────────────────────────────────────────────────────────────

# GPT-4o (OpenAI)
@_retry()
def judge_openai(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=128,
    )
    return resp.choices[0].message.content.strip()

# Claude 3.7 Sonnet (Anthropic)
@_retry()
def judge_claude(prompt: str) -> str:
    cli = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = cli.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=128,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()

# Grok-3 (xAI) – same interface as OpenAI
@_retry()
def judge_grok(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("XAI_API_KEY"),
        base_url="https://api.x.ai/v1"
    )
    resp = client.chat.completions.create(
        model="grok-3-latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=128,
    )
    return resp.choices[0].message.content.strip()

# Gemini 2.5 Pro (Google Live API)
@_retry()
def judge_gemini(prompt: str) -> str:
    import asyncio, nest_asyncio
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

            reply = ""
            async for response in session.receive():
                if response.text:
                    reply += response.text
                if getattr(response.server_content, "generation_complete", False):
                    break
            return reply.strip()

    return asyncio.run(run())

# DeepSeek R1 (local)
@_retry()
def judge_deepseek(prompt: str) -> str:
    r = httpx.post("http://localhost:8100/v1/chat/completions",
        json={"model": "deepseek-r1", "messages": [{"role": "user", "content": prompt}],
              "temperature": 0, "max_tokens": 128}, timeout=120)
    return r.json()["choices"][0]["message"]["content"].strip()

# Qwen QwQ-32B (HF inference)
@_retry()
def judge_qwen(prompt: str) -> str:
    url = os.getenv("QWEN_ENDPOINT")
    headers = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"}
    payload = {"inputs": prompt, "parameters": {"temperature": 0, "max_new_tokens": 128}}
    r = httpx.post(url, json=payload, headers=headers, timeout=120)
    r.raise_for_status()
    return r.json()[0]["generated_text"].strip()

# LLaMA 3.1 (local)
@_retry()
def judge_llama3(prompt: str) -> str:
    r = httpx.post("http://localhost:8000/v1/chat/completions",
        json={"model": "llama3-405b-instruct", "messages": [{"role": "user", "content": prompt}],
              "temperature": 0, "max_tokens": 128}, timeout=120)
    return r.json()["choices"][0]["message"]["content"].strip()

# Mistral Small 3 (local or hosted)
@_retry()
def judge_mistral(prompt: str) -> str:
    r = httpx.post("http://localhost:8200/v1/chat/completions",
        json={"model": "mistral-7b-instruct", "messages": [{"role": "user", "content": prompt}],
              "temperature": 0, "max_tokens": 128}, timeout=120)
    return r.json()["choices"][0]["message"]["content"].strip()

# ─────────────────────────────────────────────────────────────────────────────
# Judge Registry
# ─────────────────────────────────────────────────────────────────────────────
JUDGES: Dict[str, Callable[[str], str]] = {
    "openai": judge_openai,
    "claude": judge_claude,
    "grok": judge_grok,
    "gemini": judge_gemini,
    "deepseek": judge_deepseek,
    "qwen": judge_qwen,
    "llama3": judge_llama3,
    "mistral": judge_mistral,
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def safe_json(txt: str):
    try:
        return json.loads(txt)
    except:
        return {"cause_score": 0, "effect_score": 0}

# ─────────────────────────────────────────────────────────────────────────────
# Judging logic
# ─────────────────────────────────────────────────────────────────────────────
def score(pred_jsonl: str, truth_csv: str, judge_keys):
    pred = [json.loads(line) for line in open(pred_jsonl, encoding="utf-8")]
    pred_df = pd.DataFrame(pred)
    src_model = pathlib.Path(pred_jsonl).stem.split("__")[0]

    truth = pd.read_csv(truth_csv)[["id", "Cause", "Effect"]].astype(str)
    truth.columns = ["id", "orig_cause", "orig_effect"]
    df = truth.merge(pred_df, on="id", how="left").fillna("")

    for jk in judge_keys:
        judge_fn = JUDGES[jk]
        out = pathlib.Path("runs") / f"score_{jk}__{src_model}__{pathlib.Path(truth_csv).stem}.jsonl"
        out.parent.mkdir(exist_ok=True)
        print(f"▶ {jk:8s} judging {src_model}  → {out.name}")

        with out.open("w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                prompt = PROMPT.format(
                    id=row["id"], source_model=src_model,
                    orig_cause=row["orig_cause"], orig_effect=row["orig_effect"],
                    pred_cause=row["pred"].get("cause", "") if isinstance(row["pred"], dict) else row["pred_cause"],
                    pred_effect=row["pred"].get("effect", "") if isinstance(row["pred"], dict) else row["pred_effect"],
                )
                try:
                    reply = judge_fn(prompt)
                except Exception as e:
                    print("  !", jk, "error id=", row["id"], ":", e)
                    reply = "{}"

                result = safe_json(reply)
                f.write(json.dumps({
                    "id": row["id"],
                    "cause_score": result.get("cause_score", 0),
                    "effect_score": result.get("effect_score", 0),
                }, ensure_ascii=False) + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="JSONL predictions from run_llms.py")
    ap.add_argument("--truth", required=True, help="CSV containing Cause & Effect ground truth")
    ap.add_argument("--judge", default="all", choices=["all"] + list(JUDGES.keys()), help="Judge LLM(s) to use")
    args = ap.parse_args()

    selected = list(JUDGES.keys()) if args.judge == "all" else [args.judge]
    score(args.pred, args.truth, selected)
