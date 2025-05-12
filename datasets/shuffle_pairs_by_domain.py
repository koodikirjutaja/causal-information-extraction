#!/usr/bin/env python3
# shuffle_pairs_by_domain.py
"""
Create adversarial negatives by swapping cause–effect pairs *within the
same domain* and rewriting their Context paragraphs.

Eligibility
-----------
• A row is eligible only if its Context already mentions *both* the
  Cause and the Effect once we apply a light normalisation (see _norm).
• Eligible rows are shuffled; --ratio controls how many of them actually
  get swapped (rounded down to the nearest even number so swaps happen
  in pairs).

Label semantics (flipped)
-------------------------
Label = 1   → row WAS swapped / changed  (negative instance)
Label = 0   → row kept as‑is             (positive / original)

ratio (float, 0‒1)
    0.0 → keep everything (all Label = 0)
    1.0 → turn every eligible row into a negative (Label = 1)

Input  : CSV with columns Domain, Cause, Effect, Context
Output : same columns + Label

Usage
-----
python shuffle_pairs_by_domain.py  in.csv  out.csv  --ratio 0.5
"""
import argparse
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# normalisation helpers
# ---------------------------------------------------------------------------
def _norm(text: str) -> str:
    # First, normalize unicode
    text = unicodedata.normalize("NFKD", text).lower()
    
    # Manual replacements for problematic characters
    replacements = [
        ("'", "'"), ("'", "'"),
        (""", '"'), (""", '"'),
        ("–", "-"), ("—", "-"), ("‑", "-")
    ]
    
    for old, new in replacements:
        text = text.replace(old, new)
    
    # Strip punctuation except hyphen
    text = re.sub(r"[^\w\s-]", " ", text)
    # Replace hyphens with spaces
    text = text.replace("-", " ")
    # Normalize whitespace
    return re.sub(r"\s+", " ", text).strip()


def contains_ci(text: str, phrase: str) -> bool:
    """Check if phrase is in text, case-insensitive and normalized"""
    return _norm(phrase) in _norm(text)


def find_phrase_boundaries(text: str, phrase: str) -> tuple:
    """
    Find the start and end positions of a phrase in text,
    respecting word boundaries and handling case-insensitive matching
    """
    # Normalize both texts for matching
    norm_text = _norm(text)
    norm_phrase = _norm(phrase)
    
    # Find all occurrences of the normalized phrase
    matches = [(m.start(), m.end()) for m in re.finditer(r'\b' + re.escape(norm_phrase) + r'\b', norm_text)]
    
    # If no exact word boundary match, try without boundary checks
    if not matches:
        matches = [(m.start(), m.end()) for m in re.finditer(re.escape(norm_phrase), norm_text)]
    
    # If still no matches, return None
    if not matches:
        return None
    
    # Take the first match
    start, end = matches[0]
    
    # Now map these positions back to the original text
    # This is complex because normalization can change string lengths
    
    # Build a mapping from normalized positions to original positions
    pos_map = []
    norm_pos = 0
    for orig_pos, char in enumerate(text):
        if norm_pos < len(norm_text) and _norm(char) == norm_text[norm_pos]:
            pos_map.append(orig_pos)
            norm_pos += 1
    
    # If our mapping is incomplete, fall back to simple approach
    if norm_pos < len(norm_text) or start >= len(pos_map) or end > len(pos_map):
        # Simple search in original text
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        match = pattern.search(text)
        if match:
            return match.span()
        return None
    
    # Map normalized positions to original positions
    orig_start = pos_map[start]
    orig_end = pos_map[min(end-1, len(pos_map)-1)] + 1
    
    return (orig_start, orig_end)


def replace_first_ci(text: str, old: str, new: str) -> str:
    """
    Replace first occurrence of old with new, case-insensitive and respecting word boundaries
    This version is improved to handle whitespace correctly
    """
    # Edge cases
    if not old or not text:
        return text
    
    # Find the boundaries of the phrase
    boundaries = find_phrase_boundaries(text, old)
    if not boundaries:
        # If not found with improved method, fall back to simpler approach
        pattern = re.compile(re.escape(old), re.IGNORECASE)
        return pattern.sub(new, text, count=1)
    
    start, end = boundaries
    
    # Execute the replacement
    return text[:start] + new + text[end:]


# ---------------------------------------------------------------------------
# core swapper
# ---------------------------------------------------------------------------
def swap_within_domain(block: pd.DataFrame, ratio: float, rng: np.random.Generator):
    df = block.copy()
    df["Label"] = 0  # default: unchanged / positive

    # Check if Context_masked exists
    has_masked = "Context_masked" in df.columns
    if has_masked:
        print("Context_masked column found - will update both context columns")
    
    elig_idx = [
        i
        for i, row in df.iterrows()
        if contains_ci(row["Context"], row["Cause"])
        and contains_ci(row["Context"], row["Effect"])
    ]
    if len(elig_idx) < 2:
        return df

    rng.shuffle(elig_idx)
    n_swap = int(len(elig_idx) * ratio) // 2 * 2  # even
    elig_idx = elig_idx[:n_swap]

    for a, b in zip(elig_idx[0::2], elig_idx[1::2]):
        cause_a, effect_a = df.at[a, "Cause"], df.at[a, "Effect"]
        cause_b, effect_b = df.at[b, "Cause"], df.at[b, "Effect"]

        # row a
        df.at[a, "Cause"], df.at[a, "Effect"] = cause_b, effect_b
        
        # Update regular Context
        ctx_a = replace_first_ci(df.at[a, "Context"], cause_a, cause_b)
        ctx_a = replace_first_ci(ctx_a, effect_a, effect_b)
        df.at[a, "Context"] = ctx_a
        
        # Also update Context_masked if it exists
        if has_masked and pd.notna(df.at[a, "Context_masked"]):
            ctx_masked_a = replace_first_ci(df.at[a, "Context_masked"], cause_a, cause_b)
            ctx_masked_a = replace_first_ci(ctx_masked_a, effect_a, effect_b)
            df.at[a, "Context_masked"] = ctx_masked_a
        
        df.at[a, "Label"] = 1

        # row b
        df.at[b, "Cause"], df.at[b, "Effect"] = cause_a, effect_a
        
        # Update regular Context
        ctx_b = replace_first_ci(df.at[b, "Context"], cause_b, cause_a)
        ctx_b = replace_first_ci(ctx_b, effect_b, effect_a)
        df.at[b, "Context"] = ctx_b
        
        # Also update Context_masked if it exists
        if has_masked and pd.notna(df.at[b, "Context_masked"]):
            ctx_masked_b = replace_first_ci(df.at[b, "Context_masked"], cause_b, cause_a)
            ctx_masked_b = replace_first_ci(ctx_masked_b, effect_b, effect_a)
            df.at[b, "Context_masked"] = ctx_masked_b
        
        df.at[b, "Label"] = 1

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(infile: Path, outfile: Path, ratio: float, seed: int, debug: bool):
    print(f"Reading input file: {infile}")
    df_in = pd.read_csv(infile)
    required = {"Domain", "Cause", "Effect", "Context"}
    if not required.issubset(df_in.columns):
        sys.exit(f"X CSV must contain columns: {required}")

    # Check for Context_masked column
    if "Context_masked" in df_in.columns:
        print(f"Found Context_masked column - will update both context columns during swapping")
    else:
        print(f"No Context_masked column found - will only update regular Context")

    # Run a debug test of the replacement function if requested
    if debug:
        test_text = "The experts studied how rising carbon taxes affect heavy industry operations."
        test_old = "rising carbon taxes"
        test_new = "Implementation of carbon taxes"
        result = replace_first_ci(test_text, test_old, test_new)
        print(f"\nDebug replacement test:")
        print(f"Original: {test_text}")
        print(f"Replace: '{test_old}' → '{test_new}'")
        print(f"Result: {result}")
        
        # More complex test
        test_text2 = "Though Escalation of trade tensions led to changes."
        test_old2 = "Escalation of trade tensions"
        test_new2 = "Rising import tariffs"
        result2 = replace_first_ci(test_text2, test_old2, test_new2)
        print(f"\nComplex replacement test:")
        print(f"Original: {test_text2}")
        print(f"Replace: '{test_old2}' → '{test_new2}'")
        print(f"Result: {result2}")

    print(f"\nProcessing data with shuffle ratio: {ratio}")
    rng = np.random.default_rng(seed)
    out_blocks = [
        swap_within_domain(block, ratio, rng)
        for _, block in df_in.groupby("Domain", sort=False)
    ]
    
    df_out = pd.concat(out_blocks, ignore_index=True)
    
    # Add Shuffled column to indicate this is from the shuffled dataset
    df_out["Shuffled"] = 1
    
    # Add Masked column if needed for consistency
    if "Context_masked" in df_out.columns and "Masked" not in df_out.columns:
        df_out["Masked"] = 1
    
    df_out.to_csv(outfile, index=False)

    n_changed = (df_out["Label"] == 1).sum()
    print(f"\n✔ wrote '{outfile}'")
    print(f"   swapped (Label = 1): {n_changed}")
    print(f"   kept    (Label = 0): {len(df_out) - n_changed}")
    
    # Verify consistency and show some examples
    if "Context_masked" in df_out.columns:
        consistent = 0
        examples = []
        for i, row in df_out.iterrows():
            if row["Label"] == 1:  # Only check swapped rows
                if (contains_ci(row["Context_masked"], row["Cause"]) and 
                    contains_ci(row["Context_masked"], row["Effect"])):
                    consistent += 1
                    if len(examples) < 3:  # Store a few examples for display
                        examples.append((row["id"] if "id" in row else i, row["Cause"], row["Effect"], row["Context_masked"]))
        
        print(f"   Context_masked verified: {consistent}/{n_changed} swapped rows contain new cause-effect")
        
        # Show example replacements
        if examples:
            print("\nExample replacements:")
            for idx, cause, effect, context in examples:
                print(f"   Row {idx}:")
                print(f"   - Cause: {cause}")
                print(f"   - Effect: {effect}")
                print(f"   - Context (preview): {context[:100]}..." if len(context) > 100 else f"   - Context: {context}")
                print("")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("infile", type=Path, help="input CSV")
    ap.add_argument("outfile", type=Path, help="output CSV")
    ap.add_argument("--ratio", type=float, default=0.5, help="fraction of eligible rows to swap")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--debug", action="store_true", help="run debug tests of the replacement function")
    args = ap.parse_args()
    main(args.infile, args.outfile, args.ratio, args.seed, args.debug)