"""
Extract cause-effect relationships from text using multiple models:
- Together.ai models (DeepSeek, Llama4, Mistral)
- OpenRouter models (Qwen3)
- Google Gemini 2.5 Pro Preview (via Vertex AI)
- Anthropic Claude 3.7 Sonnet
- OpenAI o3 with Batch API support
- Grok models from X.AI with reasoning support

python auto/run_llms.py datasets/[dataset-hard|dataset-easy]/*.csv [--model claude|deepseek|gemini|gpt4o|grok3|llama4|mistral|qwen3|all]
"""

import os
import json
import argparse
import pathlib
import re
import time
import sys
import asyncio
from typing import List, Dict, Any, Optional
import pandas as pd
import requests
from openai import OpenAI, AsyncOpenAI

# Try to import the Vertex AI SDK for Gemini 2.5 Pro Preview
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False
    print("WARNING: Vertex AI SDK not installed. Gemini model will not be available.")
    print("Install with: pip install google-cloud-aiplatform")

# Try to import the Anthropic SDK for Claude 3.7 Sonnet
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("WARNING: Anthropic SDK not installed. Claude model will not be available.")
    print("Install with: pip install anthropic")

# Define models by provider
TOGETHER_MODELS = {
    "deepseek": "deepseek-ai/DeepSeek-R1",
    "llama4":   "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "mistral":  "mistralai/Mistral-Small-24B-Instruct-2501",
}

OPENROUTER_MODELS = {
    "qwen3": "qwen/qwen3-30b-a3b"
}

GEMINI_MODELS = {
    "gemini": "gemini-2.5-pro-preview-05-06"
}

ANTHROPIC_MODELS = {
    "claude": "claude-3-7-sonnet-20250219"
}

OPENAI_MODELS = {
    "o3": "o3",
}

GROK_MODELS = {
    "grok3": "grok-3-latest",
}

# Create prompt using concatenation (no string formatting)
def create_prompt(context):
    """Build prompt by concatenation to avoid string formatting issues"""
    return ("You are a causal-extraction specialist. Identify the cause and effect in the text.\n\n"
            "Output ONLY a JSON object with these fields:\n"
            "- \"cause\": the exact cause phrase from the text\n"
            "- \"effect\": the exact effect phrase from the text\n\n"
            "Your response must ONLY be the JSON. No other text or explanation.\n"
            "If no clear cause-effect, return {\"cause\": \"\", \"effect\": \"\"}.\n\n"
            "<paragraph>\n" + context + "\n</paragraph>")

def call_together(context, model_key, api_key=None):
    """Call Together.ai API with the specified model"""
    if not api_key:
        api_key = os.getenv("TOGETHER_API_KEY")
    
    if not api_key:
        print("ERROR: No Together API key provided")
        return "{}"
    
    model_id = TOGETHER_MODELS[model_key]
    prompt = create_prompt(context)
    
    print(f"Using Together model: {model_id}")
    
    system_prompt = "Return ONLY a JSON object with cause and effect fields."
    
    # Create payload
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 256,
        "response_format": {"type": "text"}
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Simple rate limiting
    try:
        time.sleep(1.5)
        
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            print(f"Direct API call succeeded: {response.status_code}")
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            print(f"API response: {content[:100]}...")
            return content
        else:
            print(f"Direct API call failed: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return "{}"
            
    except Exception as e:
        print(f"Error in direct API call: {str(e)}")
        return "{}"

def call_openrouter_openai(context, model_key, api_key=None):
    """Call OpenRouter API using the OpenAI client"""
    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("ERROR: No OpenRouter API key provided")
        return "{}"
    
    model_id = OPENROUTER_MODELS[model_key]
    prompt = create_prompt(context)
    
    print(f"Using OpenRouter model via OpenAI client: {model_id}")
    print(f"Prompt length: {len(prompt)} characters")
    
    try:
        print("Initializing OpenAI client with OpenRouter base URL...")
        # Rate limiting
        time.sleep(2)
        
        # Create OpenAI client with OpenRouter base URL
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        print("Sending request...")
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://github.com",
                "X-Title": "Causal Extraction",
            },
            extra_body={},
            model=model_id,
            messages=[
                {
                    "role": "system", 
                    "content": "Return ONLY a JSON object with cause and effect fields."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )
        
        print("Response received.")
        
        # Extract the content from the response
        if hasattr(completion, 'choices') and len(completion.choices) > 0:
            content = completion.choices[0].message.content
            print(f"OpenRouter response: {content[:100]}...")
            return content
        else:
            print("No choices in response.")
            return "{}"
        
    except Exception as e:
        print(f"Error calling OpenRouter: {str(e)}")
        import traceback
        traceback.print_exc()  # Print full stack trace
        return "{}"

def call_openrouter_direct(context, model_key, api_key=None):
    """Call OpenRouter API using direct requests approach - backup method"""
    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("ERROR: No OpenRouter API key provided")
        return "{}"
    
    model_id = OPENROUTER_MODELS[model_key]
    prompt = create_prompt(context)
    
    print(f"Using OpenRouter model with direct requests: {model_id}")
    print(f"Prompt length: {len(prompt)} characters")
    
    # Create headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com",  # Required by OpenRouter
        "X-Title": "Causal Extraction",        # Optional for rankings
    }
    
    # Create request payload
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "system", 
                "content": "Return ONLY a JSON object with cause and effect fields."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
    }
    
    try:
        print("Sending request to OpenRouter API...")
        # Rate limiting
        time.sleep(2)
        
        # Make the API call
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
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
                    print(f"OpenRouter response: {content[:100]}...")
                    return content
                else:
                    print(f"Unexpected response structure: {result}")
            else:
                print(f"No choices in response: {result}")
            
            return "{}"
        else:
            print(f"OpenRouter API call failed: {response.status_code}")
            print(f"Error response: {response.text}")
            return "{}"
            
    except Exception as e:
        print(f"Error in OpenRouter API call: {str(e)}")
        import traceback
        traceback.print_exc()  # Print full error details
        return "{}"

def call_vertex_gemini(context, model_key, api_key=None):
    """Call Vertex AI for Gemini 2.5 Pro Preview model"""
    if not VERTEX_AVAILABLE:
        print("ERROR: Vertex AI SDK not available. Cannot use Gemini model.")
        return "{}"
    
    model_id = GEMINI_MODELS[model_key]
    prompt = create_prompt(context)
    
    print(f"Using Vertex AI Gemini model: {model_id}")
    print(f"Prompt length: {len(prompt)} characters")
    
    try:
        # Rate limiting
        time.sleep(1.5)
        
        # Get project ID and location from environment variables
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
        
        if not project_id:
            print("ERROR: GOOGLE_CLOUD_PROJECT environment variable not set")
            return "{}"
        
        print(f"Using Google Cloud project: {project_id}")
        print(f"Using location: {location}")
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        
        print("Sending request to Vertex AI Gemini...")
        
        # Load model and generate content
        model = GenerativeModel(model_id)
        response = model.generate_content(prompt)
        
        # Extract response text
        if hasattr(response, 'text'):
            content = response.text.strip()
            print(f"Gemini response: {content[:100]}...")
            return content
        else:
            print("Unexpected response format from Gemini.")
            return "{}"
        
    except Exception as e:
        print(f"Error calling Vertex AI Gemini: {str(e)}")
        import traceback
        traceback.print_exc()
        return "{}"

def call_anthropic_claude(context, model_key, api_key=None):
    """Call Anthropic API for Claude 3.7 Sonnet model"""
    if not ANTHROPIC_AVAILABLE:
        print("ERROR: Anthropic SDK not available. Cannot use Claude model.")
        return "{}"
    
    model_id = ANTHROPIC_MODELS[model_key]
    prompt = create_prompt(context)
    
    print(f"Using Anthropic Claude model: {model_id}")
    print(f"Prompt length: {len(prompt)} characters")
    
    if not api_key:
        api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("ERROR: No Anthropic API key provided")
        return "{}"
    
    try:
        # Rate limiting
        time.sleep(1.5)
        
        # Initialize Anthropic client
        client = anthropic.Anthropic(api_key=api_key)
        
        print("Sending request to Anthropic Claude...")
        
        # Create message
        response = client.messages.create(
            model=model_id,
            max_tokens=256,
            temperature=0,
            system="Return ONLY a JSON object with cause and effect fields.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract response text
        if hasattr(response, 'content') and response.content:
            # Get the text content (handles potential list of content blocks)
            if isinstance(response.content, list):
                content = next((item.text for item in response.content if hasattr(item, 'text')), "")
            else:
                content = response.content[0].text if isinstance(response.content, list) else response.content
            
            print(f"Claude response: {content[:100]}...")
            return content
        else:
            print("Unexpected response format from Claude.")
            return "{}"
        
    except Exception as e:
        print(f"Error calling Anthropic Claude: {str(e)}")
        import traceback
        traceback.print_exc()
        return "{}"

def call_openai_simple(context, model_key, api_key=None):
    """Call OpenAI API with a simpler approach matching the o3_call.py example"""
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("ERROR: No OpenAI API key provided")
        return "{}"
    
    model_id = OPENAI_MODELS[model_key]
    
    print(f"Using OpenAI model: {model_id}")
    
    try:
        # Rate limiting
        time.sleep(0.5)
        
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        print("Sending request to OpenAI...")
        
        # Create completion with simplified prompt structure
        response = client.chat.completions.create(
            model=model_id,
            temperature=1,
            messages=[
                {"role": "system", 
                 "content": "You are a causal-extraction specialist. Return ONLY a JSON object."},
                {"role": "user", 
                 "content": f"Extract exactly one cause and one effect from the text. "
                            f"Respond ONLY as JSON {{\"cause\": \"...\", \"effect\": \"...\"}}. "
                            f"If no clear cause-effect, return {{\"cause\": \"\", \"effect\": \"\"}}.\n\n"
                            f"<paragraph>\n{context}\n</paragraph>"}
            ]
        )
        
        # Extract response text
        if hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content.strip()
            print(f"OpenAI response: {content[:100]}...")
            return content
        else:
            print("Unexpected response format from OpenAI.")
            return "{}"
        
    except Exception as e:
        print(f"Error calling OpenAI: {str(e)}")
        import traceback
        traceback.print_exc()
        return "{}"

def call_grok(context, model_key, api_key=None, use_reasoning=False):
    """Call Grok API with the specified model using the OpenAI client"""
    # Use API key from parameter or environment
    if not api_key:
        api_key = os.getenv("XAI_API_KEY")
    
    if not api_key:
        print("ERROR: No XAI API key provided")
        return "{}"
    
    model_id = GROK_MODELS[model_key]
    
    print(f"Using Grok model: {model_id}")
    
    # Determine if reasoning should be used
    supports_reasoning = model_key in ["grok3-mini-beta", "grok3-mini-fast-beta"]
    if use_reasoning and not supports_reasoning:
        print(f"WARNING: Model {model_id} does not support reasoning. Proceeding without reasoning.")
        use_reasoning = False
    
    try:
        # Rate limiting
        time.sleep(0.5)
        
        # Initialize OpenAI client with X.AI base URL
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
        
        print("Sending request to Grok API...")
        
        # Prepare request parameters
        request_params = {
            "model": model_id,
            "temperature": 0,
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a causal-extraction specialist. Return ONLY a JSON object."
                },
                {
                    "role": "user", 
                    "content": f"Extract exactly one cause and one effect from the text. "
                               f"Respond ONLY as JSON {{\"cause\": \"...\", \"effect\": \"...\"}}. "
                               f"If no clear cause-effect, return {{\"cause\": \"\", \"effect\": \"\"}}.\n\n"
                               f"<paragraph>\n{context}\n</paragraph>"
                }
            ]
        }
        
        # Add reasoning parameter if applicable
        if use_reasoning and supports_reasoning:
            request_params["reasoning_effort"] = "high"
        
        # Make the API call
        response = client.chat.completions.create(**request_params)
        
        # Extract response text
        if hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content.strip()
            
            # If reasoning was used, also log the reasoning content
            if use_reasoning and supports_reasoning and hasattr(response.choices[0].message, 'reasoning_content'):
                reasoning = response.choices[0].message.reasoning_content
                print(f"Reasoning: {reasoning[:100]}..." if len(reasoning) > 100 else f"Reasoning: {reasoning}")
            
            print(f"Grok response: {content[:100]}...")
            return content
        else:
            print("Unexpected response format from Grok.")
            return "{}"
        
    except Exception as e:
        print(f"Error calling Grok API: {str(e)}")
        import traceback
        traceback.print_exc()
        return "{}"

async def process_openai_batch_simple(batch_contexts, model_key, api_key=None):
    """Process a batch of contexts using OpenAI's API asynchronously with simplified prompt"""
    # Use API key from parameter or environment
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("ERROR: No OpenAI API key provided for batch processing")
        return ["{}" for _ in batch_contexts]
    
    model_id = OPENAI_MODELS[model_key]
    
    print(f"Using OpenAI model for batch processing: {model_id}")
    print(f"Batch size: {len(batch_contexts)}")
    
    try:
        # Initialize Async OpenAI client
        client = AsyncOpenAI(api_key=api_key)
        
        # Create tasks for each context
        tasks = []
        for context in batch_contexts:
            tasks.append(
                client.chat.completions.create(
                    model=model_id,
                    temperature=0,
                    max_tokens=256,
                    messages=[
                        {"role": "system", 
                         "content": "You are a causal-extraction specialist. Return ONLY a JSON object."},
                        {"role": "user", 
                         "content": f"Extract exactly one cause and one effect from the text. "
                                    f"Respond ONLY as JSON {{\"cause\": \"...\", \"effect\": \"...\"}}. "
                                    f"If no clear cause-effect, return {{\"cause\": \"\", \"effect\": \"\"}}.\n\n"
                                    f"<paragraph>\n{context}\n</paragraph>"}
                    ]
                )
            )
        
        # Wait for all tasks to complete
        print(f"Processing batch of {len(tasks)} requests...")
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                print(f"Error in batch item {i}: {str(response)}")
                results.append("{}")
            else:
                if hasattr(response, 'choices') and response.choices:
                    content = response.choices[0].message.content.strip()
                    results.append(content)
                else:
                    print(f"Unexpected response format from OpenAI for batch item {i}.")
                    results.append("{}")
        
        return results
        
    except Exception as e:
        print(f"Error in batch processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return ["{}" for _ in batch_contexts]

async def process_grok_batch(batch_contexts, model_key, api_key=None, use_reasoning=False):
    """Process a batch of contexts using Grok API asynchronously"""
    # Use API key from parameter or environment
    if not api_key:
        api_key = os.getenv("XAI_API_KEY")
    
    if not api_key:
        print("ERROR: No XAI API key provided for batch processing")
        return ["{}" for _ in batch_contexts]
    
    model_id = GROK_MODELS[model_key]
    
    # Determine if reasoning should be used
    supports_reasoning = model_key in ["grok3-mini-beta", "grok3-mini-fast-beta"]
    if use_reasoning and not supports_reasoning:
        print(f"WARNING: Model {model_id} does not support reasoning. Proceeding without reasoning.")
        use_reasoning = False
    
    print(f"Using Grok model for batch processing: {model_id}")
    print(f"Batch size: {len(batch_contexts)}")
    
    try:
        # Initialize AsyncOpenAI client with X.AI base URL
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
        
        # Create tasks for each context
        tasks = []
        for context in batch_contexts:
            # Build request parameters
            request_params = {
                "model": model_id,
                "temperature": 0,
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a causal-extraction specialist. Return ONLY a JSON object."
                    },
                    {
                        "role": "user", 
                        "content": f"Extract exactly one cause and one effect from the text. "
                                  f"Respond ONLY as JSON {{\"cause\": \"...\", \"effect\": \"...\"}}. "
                                  f"If no clear cause-effect, return {{\"cause\": \"\", \"effect\": \"\"}}.\n\n"
                                  f"<paragraph>\n{context}\n</paragraph>"
                    }
                ]
            }
            
            # Add reasoning parameter if applicable
            if use_reasoning and supports_reasoning:
                request_params["reasoning_effort"] = "high"
            
            # Create task
            tasks.append(
                client.chat.completions.create(**request_params)
            )
        
        # Wait for all tasks to complete
        print(f"Processing batch of {len(tasks)} requests...")
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                print(f"Error in batch item {i}: {str(response)}")
                results.append("{}")
            else:
                if hasattr(response, 'choices') and response.choices:
                    content = response.choices[0].message.content.strip()
                    
                    # If reasoning was used, also log the reasoning content
                    if use_reasoning and supports_reasoning and hasattr(response.choices[0].message, 'reasoning_content'):
                        reasoning = response.choices[0].message.reasoning_content
                        print(f"Reasoning for item {i}: {reasoning[:50]}...")
                    
                    results.append(content)
                else:
                    print(f"Unexpected response format from Grok for batch item {i}.")
                    results.append("{}")
        
        return results
        
    except Exception as e:
        print(f"Error in batch processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return ["{}" for _ in batch_contexts]

def call_openai_batch_simple(contexts, model_key, row_ids, api_key=None):
    """Process multiple contexts using batch API with simplified prompt"""
    # Use API key from parameter or environment
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("ERROR: No OpenAI API key provided for batch processing")
        return {row_id: "{}" for row_id in row_ids}
    
    # Execute the async function
    loop = asyncio.get_event_loop()
    batch_results = loop.run_until_complete(
        process_openai_batch_simple(contexts, model_key, api_key)
    )
    
    # Map results to row IDs
    return {row_id: result for row_id, result in zip(row_ids, batch_results)}

def call_grok_batch(contexts, model_key, row_ids, api_key=None, use_reasoning=False):
    """Process multiple contexts using batch API with Grok"""
    # Use API key from parameter or environment
    if not api_key:
        api_key = os.getenv("XAI_API_KEY")
    
    if not api_key:
        print("ERROR: No XAI API key provided for batch processing")
        return {row_id: "{}" for row_id in row_ids}
    
    # Execute the async function
    loop = asyncio.get_event_loop()
    batch_results = loop.run_until_complete(
        process_grok_batch(contexts, model_key, api_key, use_reasoning)
    )
    
    # Map results to row IDs
    return {row_id: result for row_id, result in zip(row_ids, batch_results)}

def extract_json(text):
    """Extract JSON from text with better handling of malformed/incomplete JSON strings"""
    if not text or text.strip() == "":
        return {"cause": "", "effect": ""}
    
    # Normalize double quotes (some models use ""cause"" format)
    text = text.replace('""', '"')
    
    # Try direct JSON parsing first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "cause" in obj and "effect" in obj:
            return {"cause": str(obj.get("cause", "")), "effect": str(obj.get("effect", ""))}
    except:
        pass
    
    # Check for thinking tags
    if "<think>" in text:
        print("Found thinking tags, extracting content...")
        # Extract content between think tags or all content after think tag
        if "</think>" in text:
            thinking = text.split("<think>")[1].split("</think>")[0]
        else:
            thinking = text.split("<think>")[1]
        
        # Replace text with thinking content for further processing
        text = thinking
    
    # Try to find cause/effect with more flexible patterns
    cause = ""
    effect = ""
    
    # Pattern for extracting cause - handles missing closing quote
    cause_pattern = r'"cause"\s*:\s*"(.*?)(?:"|,\s*"effect"|$)'
    cause_match = re.search(cause_pattern, text, re.DOTALL)
    if cause_match:
        cause = cause_match.group(1).strip()
        print(f"Found cause: {cause[:50]}..." if len(cause) > 50 else f"Found cause: {cause}")
    
    # Pattern for extracting effect - handles missing closing quote
    effect_pattern = r'"effect"\s*:\s*"(.*?)(?:"|,\s*"|\}|$)'
    effect_match = re.search(effect_pattern, text, re.DOTALL)
    if effect_match:
        effect = effect_match.group(1).strip()
        print(f"Found effect: {effect[:50]}..." if len(effect) > 50 else f"Found effect: {effect}")
    
    # If both empty, try parsing any JSON-like structure in the text
    if not cause and not effect:
        json_pattern = r'\{.*?\}'
        json_matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in json_matches:
            # Try the more flexible patterns on each JSON-like structure
            cause_match = re.search(cause_pattern, match, re.DOTALL)
            if cause_match:
                cause = cause_match.group(1).strip()
            
            effect_match = re.search(effect_pattern, match, re.DOTALL)
            if effect_match:
                effect = effect_match.group(1).strip()
            
            if cause or effect:
                break
    
    # Return results
    return {"cause": cause, "effect": effect}

# Helper function to check if API key is available
def api_key_available(model_key):
    """Check if the API key for a model is available"""
    if model_key in TOGETHER_MODELS:
        return bool(os.getenv("TOGETHER_API_KEY"))
    elif model_key in OPENROUTER_MODELS:
        return bool(os.getenv("OPENROUTER_API_KEY"))
    elif model_key in ANTHROPIC_MODELS:
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    elif model_key in OPENAI_MODELS:
        return bool(os.getenv("OPENAI_API_KEY"))
    elif model_key in GROK_MODELS:
        return bool(os.getenv("XAI_API_KEY"))
    return False

# Dictionary mapping model keys to their respective API call functions
MODEL_FUNCTIONS = {
    "deepseek": lambda ctx, key: call_together(ctx, "deepseek", key),
    "llama4":   lambda ctx, key: call_together(ctx, "llama4", key),
    "mistral":  lambda ctx, key: call_together(ctx, "mistral", key),
    "qwen3":    lambda ctx, key: call_openrouter_openai(ctx, "qwen3", key),
    "gemini":   lambda ctx, key: call_vertex_gemini(ctx, "gemini", key),
    "claude":   lambda ctx, key: call_anthropic_claude(ctx, "claude", key),
    "o3":     lambda ctx, key: call_openai_simple(ctx, "o3", key),
    "grok3":        lambda ctx, key: call_grok(ctx, "grok3", key),
}

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Extract cause-effect relationships using multiple AI models")
    parser.add_argument("csv", help="Path to input CSV file (must have 'Context' column)")
    
    # Create the initial list of model choices
    model_choices = list(TOGETHER_MODELS.keys()) + list(OPENROUTER_MODELS.keys()) + ["all"]
    
    # Add claude to model choices
    model_choices.append("claude")
    
    # Add openai models
    model_choices.extend(list(OPENAI_MODELS.keys()))
    
    # Add Grok models
    model_choices.extend(list(GROK_MODELS.keys()))
    
    # Add gemini to model choices if Vertex AI SDK is available
    if VERTEX_AVAILABLE:
        model_choices.append("gemini")
    
    parser.add_argument("--model", 
                       choices=model_choices,
                       default="all",
                       help="Model to use (default: all)")
    
    parser.add_argument("--together-key", help="Together API key (overrides environment variable)")
    parser.add_argument("--openrouter-key", help="OpenRouter API key (overrides environment variable)")
    parser.add_argument("--anthropic-key", help="Anthropic API key (overrides environment variable)")
    parser.add_argument("--openai-key", help="OpenAI API key (overrides environment variable)")
    parser.add_argument("--xai-key", help="X.AI API key for Grok (overrides environment variable)")
    parser.add_argument("--output", "-o", help="Output directory (default: ./runs)")
    parser.add_argument("--fallback", action="store_true", 
                        help="Use direct requests fallback for OpenRouter if OpenAI client fails")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Batch size for APIs that support batching (OpenAI, Grok) (0 = disabled)")
    parser.add_argument("--grok-reasoning", action="store_true", 
                        help="Enable reasoning for Grok mini models (applies to grok3-mini-beta and grok3-mini-fast-beta)")
    
    args = parser.parse_args()
    
    # Set API keys from command line if provided
    if args.together_key:
        os.environ["TOGETHER_API_KEY"] = args.together_key
    
    if args.openrouter_key:
        os.environ["OPENROUTER_API_KEY"] = args.openrouter_key
        
    if args.anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = args.anthropic_key
        
    if args.openai_key:
        os.environ["OPENAI_API_KEY"] = args.openai_key
        
    if args.xai_key:
        os.environ["XAI_API_KEY"] = args.xai_key
    
    # Check Vertex AI requirements for Gemini
    if (args.model == "gemini" or args.model == "all") and VERTEX_AVAILABLE:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            print("ERROR: GOOGLE_CLOUD_PROJECT environment variable not set")
            print("This is required for Vertex AI Gemini models")
            print("Set with: export GOOGLE_CLOUD_PROJECT=your-project-id")
            if args.model == "gemini":
                sys.exit(1)
            elif args.model == "all":
                print("Skipping Gemini model and continuing with others")
                # Remove gemini from available models
                if "gemini" in MODEL_FUNCTIONS:
                    del MODEL_FUNCTIONS["gemini"]
    
    # Check Anthropic requirements for Claude
    if (args.model == "claude" or args.model == "all") and not ANTHROPIC_AVAILABLE:
        print("ERROR: Anthropic SDK not available. Cannot use Claude model.")
        print("Install with: pip install anthropic")
        if args.model == "claude":
            sys.exit(1)
        elif args.model == "all":
            print("Skipping Claude model and continuing with others")
            # Remove claude from available models
            if "claude" in MODEL_FUNCTIONS:
                del MODEL_FUNCTIONS["claude"]
    
    # Check Grok requirements
    if any(model.startswith("grok") for model in ([args.model] if args.model != "all" else model_choices)):
        if not os.getenv("XAI_API_KEY"):
            print("ERROR: XAI_API_KEY environment variable not set")
            print("This is required for Grok models")
            print("Set with: export XAI_API_KEY=your-api-key")
            if args.model.startswith("grok"):
                sys.exit(1)
            elif args.model == "all":
                print("Skipping Grok models and continuing with others")
                # Remove Grok models from available models
                for model_key in list(MODEL_FUNCTIONS.keys()):
                    if model_key.startswith("grok"):
                        del MODEL_FUNCTIONS[model_key]
    
    # Determine which models to run
    available_models = list(TOGETHER_MODELS.keys()) + list(OPENROUTER_MODELS.keys()) + list(OPENAI_MODELS.keys())
    
    # Add Grok models if API key is available
    if os.getenv("XAI_API_KEY"):
        available_models.extend(list(GROK_MODELS.keys()))
    
    # Add gemini if available
    if VERTEX_AVAILABLE and "gemini" in MODEL_FUNCTIONS:
        available_models.append("gemini")
    
    # Add claude if available
    if ANTHROPIC_AVAILABLE and "claude" in MODEL_FUNCTIONS:
        available_models.append("claude")
    
    models = available_models if args.model == "all" else [args.model]
    
    # Set up output directory
    output_dir = pathlib.Path(args.output or "./runs").absolute()
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Test if we can write to the output directory
    try:
        test_file = output_dir / "test_write.txt"
        test_file.write_text("Test")
        test_file.unlink()
        print("✓ Output directory is writable")
    except Exception as e:
        print(f"ERROR: Cannot write to output directory: {e}")
        sys.exit(1)
    
    # Check API keys
    if any(model in TOGETHER_MODELS for model in models) and not os.getenv("TOGETHER_API_KEY"):
        print("WARNING: TOGETHER_API_KEY environment variable not set or empty")
    
    if any(model in OPENROUTER_MODELS for model in models) and not os.getenv("OPENROUTER_API_KEY"):
        print("WARNING: OPENROUTER_API_KEY environment variable not set or empty")
        
    if any(model in ANTHROPIC_MODELS for model in models) and not os.getenv("ANTHROPIC_API_KEY"):
        print("WARNING: ANTHROPIC_API_KEY environment variable not set or empty")
        
    if any(model in OPENAI_MODELS for model in models) and not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY environment variable not set or empty")
        
    if any(model in GROK_MODELS for model in models) and not os.getenv("XAI_API_KEY"):
        print("WARNING: XAI_API_KEY environment variable not set or empty")
    
    # Read input CSV
    try:
        df = pd.read_csv(args.csv)
        if "Context" not in df.columns:
            print(f"ERROR: Input CSV must have a 'Context' column. Found: {df.columns.tolist()}")
            sys.exit(1)
        print(f"✓ Input CSV has {len(df)} rows")
    except Exception as e:
        print(f"ERROR: Failed to read CSV file: {e}")
        sys.exit(1)
    
    # Add ID column if not present
    if "id" not in df.columns:
        df["id"] = [f"row{i}" for i in range(len(df))]
    
    # Process each model
    for model in models:
        print(f"\n======== Processing model: {model} ========")
        
        # Check if model is supported
        if model not in MODEL_FUNCTIONS:
            print(f"ERROR: Model '{model}' is not supported")
            continue
        
        output_jsonl = output_dir / f"{model}__dataset.jsonl"
        output_csv = output_dir / f"{model}__dataset.csv"
        
        print(f"Output files:\n - JSONL: {output_jsonl}\n - CSV: {output_csv}")
        
        results = []
        
        # Check if we should use batch processing
        use_batch = args.batch_size > 0 and (
            (model in OPENAI_MODELS) or 
            (model in GROK_MODELS)
        )
        
        if use_batch:
            print(f"Using batch processing with batch size {args.batch_size}")
            # Process in batches
            all_row_ids = df["id"].tolist()
            all_contexts = df.get("Context_masked", df["Context"]).tolist()
            
            # Process in batches
            for i in range(0, len(df), args.batch_size):
                batch_row_ids = all_row_ids[i:i+args.batch_size]
                batch_contexts = all_contexts[i:i+args.batch_size]
                
                print(f"\nProcessing batch {i//args.batch_size + 1}/{(len(df)-1)//args.batch_size + 1}")
                print(f"Batch contains rows {batch_row_ids}")
                
                # Strip out any non-string values and replace with empty strings
                batch_contexts = [
                    str(ctx) if not pd.isna(ctx) else "" 
                    for ctx in batch_contexts
                ]
                
                # Use the appropriate API key and batch function
                api_key = None
                batch_results = {}
                
                if model in OPENAI_MODELS:
                    api_key = os.getenv("OPENAI_API_KEY")
                    batch_results = call_openai_batch_simple(batch_contexts, model, batch_row_ids, api_key)
                elif model in GROK_MODELS: # This will not work
                    api_key = os.getenv("XAI_API_KEY")
                    use_reasoning = args.grok_reasoning and model in ["grok3-mini-beta", "grok3-mini-fast-beta"]
                    batch_results = call_grok_batch(batch_contexts, model, batch_row_ids, api_key, use_reasoning)
                
                # Process batch results
                for row_id, raw_response in batch_results.items():
                    print(f"\nProcessed row {row_id}")
                    
                    # Extract JSON
                    extracted = extract_json(raw_response)
                    
                    # Show extracted data
                    cause_preview = extracted.get("cause", "")[:50] + "..." if len(extracted.get("cause", "")) > 50 else extracted.get("cause", "")
                    effect_preview = extracted.get("effect", "")[:50] + "..." if len(extracted.get("effect", "")) > 50 else extracted.get("effect", "")
                    print(f"Extracted: cause='{cause_preview}', effect='{effect_preview}'")
                    
                    # Find the actual row from DataFrame to get ground truth if available
                    row = df[df["id"] == row_id].iloc[0] if any(df["id"] == row_id) else None
                    
                    # Add to results
                    results.append({
                        "id": row_id,
                        "cause": extracted.get("cause", ""),
                        "effect": extracted.get("effect", ""),
                        "raw_response": str(raw_response).replace("\n", "\\n").replace('"', '""')  # Escape for CSV
                    })
                    
                    # Create record for JSONL
                    record = {
                        "id": row_id,
                        "pred": extracted,
                        "raw_response": raw_response
                    }
                    
                    # Add ground truth if available
                    if row is not None and {"Cause", "Effect"}.issubset(row.index):
                        record["ground"] = {
                            "cause": row["Cause"],
                            "effect": row["Effect"]
                        }
                    
                    # Write to JSONL file (append mode)
                    try:
                        with open(output_jsonl, "a", encoding="utf-8") as f:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            f.flush()
                    except Exception as e:
                        print(f"ERROR writing to JSONL file: {e}")
                
                # Write intermediate CSV every batch
                try:
                    pd.DataFrame(results).to_csv(output_csv, index=False, quoting=1)
                    print(f"Saved progress ({len(results)} rows processed)")
                except Exception as e:
                    print(f"ERROR saving progress: {e}")
        else:
            # Process row by row (no batching)
            for _, row in df.iterrows():
                try:
                    row_id = row["id"]
                    
                    # Get context and handle non-string values
                    context = row.get("Context_masked", row["Context"])
                    
                    # Check for NaN or non-string values
                    if pd.isna(context):
                        print(f"\nProcessing row {row_id}")
                        print(f"WARNING: Row {row_id} has missing context (NaN)")
                        context = ""
                    elif not isinstance(context, str):
                        print(f"\nProcessing row {row_id}")
                        print(f"WARNING: Row {row_id} has non-string context: {type(context)}")
                        context = str(context)
                    else:
                        print(f"\nProcessing row {row_id}")
                        print(f"Context preview: {context[:100]}...")
                    
                    # Skip empty contexts
                    if not context:
                        print("Context is empty, skipping API call")
                        extracted = {"cause": "", "effect": ""}
                        raw_response = "{}"
                    else:
                        # Use the appropriate API key based on the model
                        api_key = None
                        if model in TOGETHER_MODELS:
                            api_key = os.getenv("TOGETHER_API_KEY")
                        elif model in OPENROUTER_MODELS:
                            api_key = os.getenv("OPENROUTER_API_KEY")
                        elif model in ANTHROPIC_MODELS:
                            api_key = os.getenv("ANTHROPIC_API_KEY")
                        elif model in OPENAI_MODELS:
                            api_key = os.getenv("OPENAI_API_KEY")
                        elif model in GROK_MODELS:
                            api_key = os.getenv("XAI_API_KEY")
                        
                        # Make the API call using the appropriate function
                        raw_response = MODEL_FUNCTIONS[model](context, api_key)
                        
                        # Try fallback method for OpenRouter if enabled and primary method failed
                        if args.fallback and model == "qwen3" and (not raw_response or raw_response == "{}"):
                            print("Primary OpenRouter method failed, trying fallback...")
                            raw_response = call_openrouter_direct(context, model, api_key)
                        
                        # Extract JSON with improved handling
                        extracted = extract_json(raw_response)
                    
                    # Show extracted data
                    cause_preview = extracted.get("cause", "")[:50] + "..." if len(extracted.get("cause", "")) > 50 else extracted.get("cause", "")
                    effect_preview = extracted.get("effect", "")[:50] + "..." if len(extracted.get("effect", "")) > 50 else extracted.get("effect", "")
                    print(f"Extracted: cause='{cause_preview}', effect='{effect_preview}'")
                    
                    # Add to results
                    results.append({
                        "id": row_id,
                        "cause": extracted.get("cause", ""),
                        "effect": extracted.get("effect", ""),
                        "raw_response": str(raw_response).replace("\n", "\\n").replace('"', '""')  # Escape for CSV
                    })
                    
                    # Create record for JSONL
                    record = {
                        "id": row_id,
                        "pred": extracted,
                        "raw_response": raw_response
                    }
                    
                    # Add ground truth if available
                    if {"Cause", "Effect"}.issubset(row.index):
                        record["ground"] = {
                            "cause": row["Cause"],
                            "effect": row["Effect"]
                        }
                    
                    # Write to JSONL file (append mode)
                    try:
                        with open(output_jsonl, "a", encoding="utf-8") as f:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            f.flush()
                    except Exception as e:
                        print(f"ERROR writing to JSONL file: {e}")
                    
                    # Write intermediate CSV every 10 rows
                    if len(results) % 10 == 0:
                        try:
                            pd.DataFrame(results).to_csv(output_csv, index=False, quoting=1)
                            print(f"Saved progress ({len(results)} rows processed)")
                        except Exception as e:
                            print(f"ERROR saving progress: {e}")
                            
                except Exception as e:
                    print(f"ERROR processing row: {e}")
                    print(f"Continuing with next row...")
                    continue
        
        # Write final CSV
        try:
            pd.DataFrame(results).to_csv(output_csv, index=False, quoting=1)
            print(f"\nCompleted processing for {model}.")
            print(f"Processed {len(results)} rows.")
            
            # Verify files exist and have content
            if output_jsonl.exists():
                print(f"JSONL file size: {output_jsonl.stat().st_size} bytes")
            
            if output_csv.exists():
                print(f"CSV file size: {output_csv.stat().st_size} bytes")
                
        except Exception as e:
            print(f"ERROR writing final CSV: {e}")

# Test function to verify the Anthropic Claude configuration
def test_anthropic_claude():
    """Test the Anthropic Claude API"""
    if not ANTHROPIC_AVAILABLE:
        print("ERROR: Anthropic SDK not available. Cannot use Claude model.")
        print("Install with: pip install anthropic")
        return
    
    # Check for Anthropic API Key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        print("Set with: export ANTHROPIC_API_KEY=your-api-key")
        return
    
    print("Using Anthropic API key from environment")
    
    try:
        # Initialize Anthropic client
        client = anthropic.Anthropic(api_key=api_key)
        
        # The model name
        model_id = ANTHROPIC_MODELS["claude"]
        print(f"Testing model: {model_id}")
        
        # Test prompt
        prompt = create_prompt("The expansion of financial literacy education has led to increased household savings rates.")
        
        # Create message
        print("Creating message and sending request...")
        response = client.messages.create(
            model=model_id,
            max_tokens=256,
            temperature=0,
            system="Return ONLY a JSON object with cause and effect fields.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        print("\nRaw response:")
        if hasattr(response, 'content') and response.content:
            # Handle potential list of content blocks
            if isinstance(response.content, list):
                content = next((item.text for item in response.content if hasattr(item, 'text')), "")
            else:
                content = response.content[0].text if isinstance(response.content, list) else response.content
            print(content)
        else:
            print("Unexpected response format")
        
        extracted = extract_json(response.content[0].text if isinstance(response.content, list) else response.content)
        print("\nExtracted content:")
        print(extracted)
        
    except Exception as e:
        print(f"Error testing Anthropic Claude: {e}")
        import traceback
        traceback.print_exc()
        print("\nPossible issues:")
        print("1. Authentication failed - check your API key")
        print("2. Model name incorrect or not available")
        print("3. Rate limiting or insufficient credits")
        print("Check Anthropic dashboard for more details")


# Test function to verify the Grok API
def test_grok():
    """Test the Grok API"""
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("ERROR: XAI_API_KEY environment variable not set")
        print("Set with: export XAI_API_KEY=your-api-key")
        return
    
    print("Using XAI API key from environment")
    
    # Test context
    context = "The expansion of financial literacy education has led to increased household savings rates."
    
    # Test with regular model
    try:
        print("\nTesting Grok API with standard model...")
        response = call_grok(context, "grok3", api_key)
        
        print("\nRaw response:")
        print(response)
        
        extracted = extract_json(response)
        print("\nExtracted content:")
        print(extracted)
        
    except Exception as e:
        print(f"Error testing Grok API (standard): {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test with mini model using reasoning
    try:
        print("\nTesting Grok API with mini model and reasoning...")
        response = call_grok(context, "grok3-mini-beta", api_key, use_reasoning=True)
        
        print("\nRaw response:")
        print(response)
        
        extracted = extract_json(response)
        print("\nExtracted content:")
        print(extracted)
        
    except Exception as e:
        print(f"Error testing Grok API (mini with reasoning): {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()