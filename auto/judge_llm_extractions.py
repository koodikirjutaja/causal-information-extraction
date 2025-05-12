#!/usr/bin/env python3
# judge_llm_extractions.py
"""
Judge the quality of cause-effect extractions by comparing predicted pairs 
against ground truth using multiple LLM judges. Each judge assigns a NUMERIC score:

    1.0  – Exact match or semantically correct
    0.5  – Partial match
    0.0  – No match or incorrect

The judge returns JSON:
    {"cause_score": 0|0.5|1, "effect_score": 0|0.5|1}

------------------------------------------------------------------
CLI
------------------------------------------------------------------
python auto/judge_llm_extractions.py \
        --input runs/*__dataset.jsonl \
        [--judge openai|claude|deepseek|llama4|mistral|qwen3|gemini|grok|all]

Outputs:
   - JSONL file: runs/score_<judge>__<source_model>__results.jsonl
   - CSV file: runs/score_<judge>__<source_model>__results.csv

"""

import os
import json
import pathlib
import argparse
import time
import requests
import pandas as pd
from typing import Dict, Callable, List, Optional

# Try to import backoff for retry logic
try:
    import backoff
    BACKOFF_AVAILABLE = True
except ImportError:
    BACKOFF_AVAILABLE = False
    print("WARNING: backoff package not installed. Retry logic disabled.")
    print("Install with: pip install backoff")

# Try to import necessary LLM APIs
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("WARNING: OpenAI SDK not installed. GPT and Grok models will not be available.")
    print("Install with: pip install openai")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("WARNING: Anthropic SDK not installed. Claude model will not be available.")
    print("Install with: pip install anthropic")

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False
    print("WARNING: Vertex AI SDK not installed. Gemini model will not be available.")
    print("Install with: pip install google-cloud-aiplatform")

# ─────────────────────────────────────────────────────────────────────────────
# Improved prompt for more consistent JSON outputs
# ─────────────────────────────────────────────────────────────────────────────
PROMPT = """Task: Evaluate how well the predicted cause-effect pair matches the ground truth pair.

id: {id}
predicted_by: {source_model}

GROUND TRUTH PAIR:
Cause: {orig_cause}
Effect: {orig_effect}

PREDICTED PAIR:
Cause: {pred_cause}
Effect: {pred_effect}

EVALUATION INSTRUCTIONS:
1. Compare the predicted cause with the ground truth cause
2. Compare the predicted effect with the ground truth effect
3. Assign a score to each:
   - 1.0: Exact match or semantically identical meaning
   - 0.5: Partial match (core concept is similar but missing important details)
   - 0.0: No match or incorrect

You MUST respond with ONLY a valid JSON object in this exact format:
{{"cause_score": 0|0.5|1, "effect_score": 0|0.5|1}}

Your response MUST ONLY contain this JSON. No other text or explanation.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Model Definitions - Match the models in run_llms.py
# ─────────────────────────────────────────────────────────────────────────────
# Together.ai models
TOGETHER_MODELS = {
    "deepseek": "deepseek-ai/DeepSeek-R1",
    "llama4":   "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "mistral":  "mistralai/Mistral-Small-24B-Instruct-2501",
}

# OpenRouter models
OPENROUTER_MODELS = {
    "qwen3": "qwen/qwen3-30b-a3b"  # Paid version
}

# ─────────────────────────────────────────────────────────────────────────────
# Retry decorator with proper backoff options
# ─────────────────────────────────────────────────────────────────────────────
def _retry_decorator(func):
    """Simple retry decorator if backoff module is not available"""
    if BACKOFF_AVAILABLE:
        return backoff.on_exception(
            backoff.expo, 
            Exception, 
            max_tries=5, 
            jitter=backoff.full_jitter  # Correctly using full_jitter
        )(func)
    else:
        # Simple retry logic if backoff not available
        def wrapper(*args, **kwargs):
            max_tries = 5
            for attempt in range(max_tries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_tries - 1:
                        raise
                    print(f"Error: {e}. Retrying ({attempt+1}/{max_tries})...")
                    # Simple exponential backoff
                    time.sleep(2 ** attempt)
            return func(*args, **kwargs)  # Final attempt
        return wrapper

# ─────────────────────────────────────────────────────────────────────────────
# LLM API Wrappers - All standardized for consistency
# ─────────────────────────────────────────────────────────────────────────────

# GPT (OpenAI)
@_retry_decorator
def judge_openai(prompt: str) -> str:
    if not OPENAI_AVAILABLE:
        return '{"cause_score": 0, "effect_score": 0}'
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="o3",
        messages=[{"role": "user", "content": prompt}],
        temperature=1
        # No max_tokens limit
    )
    return resp.choices[0].message.content.strip()

# Claude 3.7 Sonnet (Anthropic)
@_retry_decorator
def judge_claude(prompt: str) -> str:
    if not ANTHROPIC_AVAILABLE:
        return '{"cause_score": 0, "effect_score": 0}'
    
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=256,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    
    if hasattr(resp, 'content') and resp.content:
        if isinstance(resp.content, list):
            return next((item.text for item in resp.content if hasattr(item, 'text')), "{}")
        else:
            return resp.content[0].text if isinstance(resp.content, list) else resp.content
    return "{}"

# Gemini 2.5 Pro (Google Vertex AI)
@_retry_decorator
def judge_gemini(prompt: str) -> str:
    if not VERTEX_AVAILABLE:
        return '{"cause_score": 0, "effect_score": 0}'
    
    # Get project ID and location from environment variables
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
    
    if not project_id:
        print("ERROR: GOOGLE_CLOUD_PROJECT environment variable not set")
        return "{}"
    
    try:
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        
        # Load model and generate content
        model = GenerativeModel("gemini-2.5-pro-preview-05-06")
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0,
                # No max_tokens limit
            }
        )
        
        # Extract response text
        if hasattr(response, 'text'):
            return response.text.strip()
        return "{}"
        
    except Exception as e:
        print(f"Error calling Vertex AI Gemini: {str(e)}")
        import traceback
        traceback.print_exc()
        return "{}"

# Grok-3 (xAI) - same interface as OpenAI
@_retry_decorator
def judge_grok(prompt: str) -> str:
    if not OPENAI_AVAILABLE:
        return '{"cause_score": 0, "effect_score": 0}'
    
    try:
        client = OpenAI(
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
        resp = client.chat.completions.create(
            model="grok-3-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            # Removed max_tokens limit
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling Grok API: {str(e)}")
        import traceback
        traceback.print_exc()
        return "{}"

# Together.ai models (DeepSeek, Llama4, Mistral)
@_retry_decorator
def judge_together(prompt: str, model_key: str) -> str:
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("ERROR: No Together API key provided")
        return '{"cause_score": 0, "effect_score": 0}'
    
    model_id = TOGETHER_MODELS[model_key]
    
    # Removed system prompt for consistency with other models
    
    # Create payload
    payload = {
        "model": model_id,
        "messages": [
            # No system message, just user message
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        # Removed max_tokens limit
        "response_format": {"type": "text"}
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Simple rate limiting
    try:
        time.sleep(1.5)  # Wait 1.5 seconds between calls to avoid rate limits
        
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            print(f"Together API call succeeded: {response.status_code}")
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            return content
        else:
            print(f"Together API call failed: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return '{"cause_score": 0, "effect_score": 0}'
            
    except Exception as e:
        print(f"Error in Together API call: {str(e)}")
        import traceback
        traceback.print_exc()
        return '{"cause_score": 0, "effect_score": 0}'

# DeepSeek (Together.ai)
def judge_deepseek(prompt: str) -> str:
    return judge_together(prompt, "deepseek")

# Llama4 (Together.ai)
def judge_llama4(prompt: str) -> str:
    return judge_together(prompt, "llama4")

# Mistral (Together.ai)
def judge_mistral(prompt: str) -> str:
    return judge_together(prompt, "mistral")

# Qwen3 (OpenRouter)
@_retry_decorator
def judge_qwen3(prompt: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: No OpenRouter API key provided")
        return '{"cause_score": 0, "effect_score": 0}'
    
    model_id = OPENROUTER_MODELS["qwen3"]
    
    try:
        # Rate limiting
        time.sleep(2)
        
        if OPENAI_AVAILABLE:
            # Using OpenAI client for consistent handling
            print("Using OpenAI client with OpenRouter")
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            
            response = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://github.com",
                    "X-Title": "Causal Extraction Evaluation",
                },
                model=model_id,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                # No max_tokens limit
            )
            
            if hasattr(response, 'choices') and len(response.choices) > 0:
                content = response.choices[0].message.content.strip()
                return content
            return '{"cause_score": 0, "effect_score": 0}'
            
        else:
            # Fallback to direct requests if OpenAI client not available
            print("Using direct API request for OpenRouter")
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com",  # Required by OpenRouter
                "X-Title": "Causal Extraction Evaluation",  # Optional for rankings
            }
            
            # Create request payload - match other models by NOT using system message
            payload = {
                "model": model_id,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0,
                # No max_tokens limit
            }
            
            # Make the API call
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,  # Use json instead of data=json.dumps() for consistency
                timeout=60
            )
            
            # Check if the request was successful
            if response.status_code == 200:
                print(f"OpenRouter API call succeeded: {response.status_code}")
                result = response.json()
                
                # Extract content from the response
                if "choices" in result and len(result["choices"]) > 0:
                    if "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
                        content = result["choices"][0]["message"]["content"].strip()
                        return content
                    else:
                        print(f"Unexpected response structure: {result}")
                else:
                    print(f"No choices in response: {result}")
                
                return '{"cause_score": 0, "effect_score": 0}'
            else:
                print(f"OpenRouter API call failed: {response.status_code}")
                print(f"Error response: {response.text}")
                return '{"cause_score": 0, "effect_score": 0}'
                
    except Exception as e:
        print(f"Error in OpenRouter API call: {str(e)}")
        import traceback
        traceback.print_exc()
        return '{"cause_score": 0, "effect_score": 0}'

# ─────────────────────────────────────────────────────────────────────────────
# Judge Registry
# ─────────────────────────────────────────────────────────────────────────────
JUDGES: Dict[str, Callable[[str], str]] = {
    "openai": judge_openai,         # OpenAI o3
    "claude": judge_claude,         # Anthropic Claude 3.7 Sonnet
    "gemini": judge_gemini,         # Google Gemini 2.5 Pro
    "grok": judge_grok,             # X.AI Grok-3
    "deepseek": judge_deepseek,     # Together.ai DeepSeek
    "llama4": judge_llama4,         # Together.ai Llama-4
    "mistral": judge_mistral,       # Together.ai Mistral
    "qwen3": judge_qwen3,           # OpenRouter Qwen3
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def safe_json(txt: str):
    """Safely parse JSON, with multiple fallback methods if parsing fails"""
    if not txt or txt.strip() == "":
        return {"cause_score": 0, "effect_score": 0}
        
    # First try direct JSON parsing
    try:
        json_obj = json.loads(txt)
        # Check if it has the expected fields
        if "cause_score" in json_obj and "effect_score" in json_obj:
            # Normalize values to 0, 0.5, or 1
            cause_score = json_obj["cause_score"]
            effect_score = json_obj["effect_score"]
            
            if isinstance(cause_score, (int, float)):
                cause_score = round(cause_score * 2) / 2  # Round to nearest 0.5
                cause_score = min(max(cause_score, 0), 1)  # Clamp to [0, 1]
            else:
                cause_score = 0
                
            if isinstance(effect_score, (int, float)):
                effect_score = round(effect_score * 2) / 2  # Round to nearest 0.5
                effect_score = min(max(effect_score, 0), 1)  # Clamp to [0, 1]
            else:
                effect_score = 0
                
            return {"cause_score": cause_score, "effect_score": effect_score}
        else:
            print(f"JSON missing required fields: {json_obj}")
    except json.JSONDecodeError:
        # Continue with regex fallbacks
        pass
    except Exception as e:
        print(f"Unexpected error parsing JSON: {e}")
    
    print(f"JSON parsing failed, trying regex on: {txt[:200]}...")
    
    # Try extracting using regex patterns
    import re
    
    # Look for JSON-like patterns with different score formats
    # Pattern 1: Standard JSON format with numbers
    pattern1 = r'"cause_score"\s*:\s*([0-9.]+).*?"effect_score"\s*:\s*([0-9.]+)'
    # Pattern 2: JSON with string numbers
    pattern2 = r'"cause_score"\s*:\s*"([0-9.]+)".*?"effect_score"\s*:\s*"([0-9.]+)"'
    # Pattern 3: Looking for just numbers after cause_score/effect_score labels
    pattern3 = r'cause_score\s*:?\s*([0-9.]+).*?effect_score\s*:?\s*([0-9.]+)'
    # Pattern 4: Looking for 0, 0.5, 1 or other values
    pattern4 = r'cause_score\s*:?\s*(0|0\.5|1|one|zero|half).*?effect_score\s*:?\s*(0|0\.5|1|one|zero|half)'
    
    for pattern in [pattern1, pattern2, pattern3, pattern4]:
        match = re.search(pattern, txt, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                # Extract values
                cause_raw = match.group(1).lower()
                effect_raw = match.group(2).lower()
                
                # Convert text values to numbers if needed
                if cause_raw == "one":
                    cause_score = 1.0
                elif cause_raw == "half":
                    cause_score = 0.5
                elif cause_raw == "zero":
                    cause_score = 0.0
                else:
                    cause_score = float(cause_raw)
                    
                if effect_raw == "one":
                    effect_score = 1.0
                elif effect_raw == "half":
                    effect_score = 0.5
                elif effect_raw == "zero":
                    effect_score = 0.0
                else:
                    effect_score = float(effect_raw)
                
                # Normalize to 0, 0.5, 1
                cause_score = round(cause_score * 2) / 2
                cause_score = min(max(cause_score, 0), 1)
                effect_score = round(effect_score * 2) / 2
                effect_score = min(max(effect_score, 0), 1)
                
                print(f"Extracted scores via regex: cause={cause_score}, effect={effect_score}")
                return {"cause_score": cause_score, "effect_score": effect_score}
            except Exception as e:
                print(f"Error converting regex match to scores: {e}")
    
    # Last resort: Look for any numbers or score indicators in the text
    # This is a loose fallback that might misinterpret text
    score_indicators = {
        "0": 0.0, "zero": 0.0, "no match": 0.0, "incorrect": 0.0, "wrong": 0.0,
        "0.5": 0.5, "half": 0.5, "partial": 0.5, "partially": 0.5, "similar": 0.5,
        "1": 1.0, "one": 1.0, "exact": 1.0, "identical": 1.0, "correct": 1.0, "right": 1.0
    }
    
    cause_indicators = re.findall(r'cause.{0,20}(0|0\.5|1|no match|incorrect|wrong|half|partial|partially|similar|exact|identical|correct|right)', 
                                txt, re.IGNORECASE)
    effect_indicators = re.findall(r'effect.{0,20}(0|0\.5|1|no match|incorrect|wrong|half|partial|partially|similar|exact|identical|correct|right)', 
                                txt, re.IGNORECASE)
    
    # Default scores
    cause_score = 0.0
    effect_score = 0.0
    
    # Extract score from indicators if found
    if cause_indicators:
        indicator = cause_indicators[0].lower()
        for key, value in score_indicators.items():
            if key in indicator:
                cause_score = value
                break
    
    if effect_indicators:
        indicator = effect_indicators[0].lower()
        for key, value in score_indicators.items():
            if key in indicator:
                effect_score = value
                break
    
    print(f"Using fallback extraction method: cause={cause_score}, effect={effect_score}")
    return {"cause_score": cause_score, "effect_score": effect_score}

def read_jsonl(file_path: str) -> List[Dict]:
    """Read JSONL file into a list of dictionaries"""
    items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSONL line: {e}")
    return items

def save_jsonl(data: List[Dict], file_path: str):
    """Save a list of dictionaries to a JSONL file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def save_csv(data: List[Dict], file_path: str):
    """Save a list of dictionaries to a CSV file"""
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)

def get_source_model(input_file: str) -> str:
    """Extract source model name from input file path"""
    try:
        basename = pathlib.Path(input_file).stem
        if "__" in basename:
            return basename.split("__")[0]
        return basename
    except:
        return "unknown_model"

def check_api_key_available(judge_key: str) -> bool:
    """Check if the API key for a judge is available"""
    if judge_key in ["openai"]:
        return bool(os.getenv("OPENAI_API_KEY"))
    elif judge_key in ["claude"]:
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    elif judge_key in ["gemini"]:
        return bool(os.getenv("GOOGLE_CLOUD_PROJECT"))
    elif judge_key in ["grok"]:
        return bool(os.getenv("XAI_API_KEY"))
    elif judge_key in ["deepseek", "llama4", "mistral"]:
        return bool(os.getenv("TOGETHER_API_KEY"))
    elif judge_key in ["qwen3"]:
        return bool(os.getenv("OPENROUTER_API_KEY"))
    return False

# ─────────────────────────────────────────────────────────────────────────────
# Judging logic
# ─────────────────────────────────────────────────────────────────────────────
def judge_extractions(input_jsonl: str, judge_keys: List[str]):
    """
    Have one or more LLMs judge the quality of predicted cause-effect pairs
    
    Args:
        input_jsonl: Path to JSONL file with predictions and ground truth
        judge_keys: List of judge names to use (e.g., ['openai', 'claude'])
    """
    # Ensure output directory exists
    output_dir = pathlib.Path("runs")
    output_dir.mkdir(exist_ok=True)
    
    # Extract source model name from input file
    source_model = get_source_model(input_jsonl)
    
    # Read predictions
    print(f"Reading predictions from {input_jsonl}")
    predictions = read_jsonl(input_jsonl)
    print(f"Found {len(predictions)} predictions")
    
    # Process with each judge
    for judge_key in judge_keys:
        # Check if API key is available
        if not check_api_key_available(judge_key):
            print(f"\n⚠ Skipping {judge_key}: API key not available")
            continue
            
        judge_fn = JUDGES[judge_key]
        
        # Setup output paths
        output_base = f"score_{judge_key}__{source_model}__results"
        output_jsonl = output_dir / f"{output_base}.jsonl"
        output_csv = output_dir / f"{output_base}.csv"
        
        print(f"\n▶ Using {judge_key} to judge {source_model} extractions")
        print(f"  Output will be saved to {output_jsonl} and {output_csv}")
        
        results = []
        
        # Process each prediction
        for i, pred in enumerate(predictions):
            row_id = pred.get("id", f"item_{i}")
            
            # Get predicted and ground truth values
            pred_cause = pred.get("pred", {}).get("cause", "")
            pred_effect = pred.get("pred", {}).get("effect", "")
            
            # Get ground truth if available
            orig_cause = pred.get("ground", {}).get("cause", "")
            orig_effect = pred.get("ground", {}).get("effect", "")
            
            # Skip if missing ground truth
            if not orig_cause or not orig_effect:
                print(f"  ! Skipping {row_id}: Missing ground truth")
                continue
            
            # Format prompt
            prompt = PROMPT.format(
                id=row_id,
                source_model=source_model,
                orig_cause=orig_cause,
                orig_effect=orig_effect,
                pred_cause=pred_cause,
                pred_effect=pred_effect
            )
            
            # Call judge
            print(f"  Judging {row_id} ({i+1}/{len(predictions)})")
            try:
                response = judge_fn(prompt)
                scores = safe_json(response)
                
                # Ensure scores are 0, 0.5, or 1
                cause_score = scores.get("cause_score", 0)
                effect_score = scores.get("effect_score", 0)
                
                # Normalize scores to valid values
                if cause_score not in [0, 0.5, 1]:
                    cause_score = round(cause_score * 2) / 2
                    cause_score = min(max(cause_score, 0), 1)
                
                if effect_score not in [0, 0.5, 1]:
                    effect_score = round(effect_score * 2) / 2
                    effect_score = min(max(effect_score, 0), 1)
                
                # Create result record
                result = {
                    "id": row_id,
                    "cause_score": cause_score,
                    "effect_score": effect_score,
                    "avg_score": (cause_score + effect_score) / 2,
                    "pred_cause": pred_cause,
                    "pred_effect": pred_effect,
                    "orig_cause": orig_cause,
                    "orig_effect": orig_effect,
                    "judge_response": response
                }
                
                # Add to results
                results.append(result)
                print(f"    Scores: cause={result['cause_score']}, effect={result['effect_score']}")
                
                # Save intermediate results every 10 items
                if (i + 1) % 10 == 0 or i == len(predictions) - 1:
                    save_jsonl(results, output_jsonl)
                    save_csv(results, output_csv)
                    print(f"  ✓ Saved progress ({i+1}/{len(predictions)})")
                
            except Exception as e:
                print(f"  ! Error judging {row_id}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save final results
        if results:
            save_jsonl(results, output_jsonl)
            save_csv(results, output_csv)
            
            # Calculate and show summary statistics
            scores_df = pd.DataFrame(results)
            avg_cause = scores_df["cause_score"].mean()
            avg_effect = scores_df["effect_score"].mean()
            avg_total = scores_df["avg_score"].mean()
            
            print(f"\n▶ {judge_key} judgment summary for {source_model}:")
            print(f"  Average cause score: {avg_cause:.3f}")
            print(f"  Average effect score: {avg_effect:.3f}")
            print(f"  Average total score: {avg_total:.3f}")
            print(f"  Results saved to {output_jsonl} and {output_csv}")
        else:
            print(f"\n! No results generated for {judge_key}")

# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Judge cause-effect extractions using LLMs")
    parser.add_argument("--input", required=True, help="JSONL file with predictions and ground truth")
    parser.add_argument("--judge", default="all", 
                       choices=["all"] + list(JUDGES.keys()), 
                       help="Judge LLM(s) to use")
    args = parser.parse_args()
    
    # Determine which judges to use
    judge_keys = list(JUDGES.keys()) if args.judge == "all" else [args.judge]
    
    # Check for available judges
    valid_judges = []
    for jk in judge_keys:
        if check_api_key_available(jk):
            valid_judges.append(jk)
        else:
            print(f"WARNING: {jk} judge unavailable (missing API key), skipping")
    
    if not valid_judges:
        print("ERROR: No valid judges available")
        exit(1)
    
    # Run the judging process
    judge_extractions(args.input, valid_judges)