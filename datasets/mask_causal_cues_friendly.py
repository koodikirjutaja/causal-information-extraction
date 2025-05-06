#!/usr/bin/env python3
# mask_causal_cues_friendly.py
"""
Mask or neutralise causal cue words in each Context paragraph.

Adds two new columns:
    Context_masked  – rewritten paragraph with cues replaced
    Masked          – 1 if any cue was replaced in the row, else 0
"""

import re
import pandas as pd
from collections import Counter

# ------------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------------
INFILE  = "dataset-easy.csv"
OUTFILE = "dataset-easy_masked.csv"

# ------------------------------------------------------------------
# 2. Cue → neutral‑word mapping (FULL list)
# ------------------------------------------------------------------
CUE_REPLACE = {
    # single‑word conjunctions & adverbs
    r"\bbecause\b":                "while",
    r"\bsince\b":                  "while",
    r"\bas\b":                     "while",
    r"\bso\b":                     "and",
    r"\bthus\b":                   "then",
    r"\bhence\b":                  "then",
    r"\btherefore\b":              "overall",
    r"\bthereby\b":                "in doing so",
    r"\bconsequently\b":           "subsequently",
    r"\baccordingly\b":            "subsequently",
    r"\bergo\b":                   "then",
    r"\bwhereby\b":                "in which",

    # prepositional phrases
    r"\bdue to\b":                 "amid",
    r"\bowing to\b":               "amid",
    r"\bon account of\b":          "amid",
    r"\bas a result of\b":         "amid",
    r"\bas a consequence of\b":    "amid",
    r"\bin light of\b":            "amid",
    r"\bin view of\b":             "amid",
    r"\bthanks to\b":              "amid",

    # verb triggers
    r"\bcauses?\b":                "is associated with",
    r"\bcaused by\b":              "associated with",
    r"\bleads? to\b":              "is accompanied by",
    r"\bleading to\b":             "accompanied by",
    r"\bresults? in\b":            "is observed with",
    r"\bresulting in\b":           "observed with",
    r"\bbrings about\b":           "accompanies",
    r"\bgives rise to\b":          "coincides with",
    r"\bsets off\b":               "coincides with",
    r"\bset off\b":                "coincides with",
    r"\btriggers?\b":              "coincides with",
    r"\bsparks?\b":                "coincides with",
    r"\bprovokes?\b":              "coincides with",
    r"\bprompts?\b":               "coincides with",
    r"\bdrives?\b":                "coincides with",
    r"\bfosters?\b":               "coincides with",
    r"\bgenerates?\b":             "coincides with",
    r"\bproduces?\b":              "coincides with",
    r"\byields?\b":                "coincides with",
    r"\bengenders?\b":             "coincides with",

    # noun signals
    r"\bcause of\b":               "factor in",
    r"\breason for\b":             "factor in",
    r"\bresult of\b":              "pattern in",
    r"\bconsequence of\b":         "pattern in",
    r"\boutcome of\b":             "pattern in",
    r"\beffect of\b":              "pattern in",
    r"\bimpact of\b":              "pattern in",
    r"\binfluence of\b":           "pattern in",

    # purpose / intent
    r"\bso that\b":                "so",
    r"\bin order to\b":            "to",
    r"\bwith the aim of\b":        "to",
    r"\bwith the intention of\b":  "to",

    # temporal chains (often imply cause)
    r"\bafter\b":                  "when",
    r"\bbefore\b":                 "when",
    r"\bonce\b":                   "when",
    r"\bwhen\b":                   "as",
    r"\bfollowed by\b":            "with",
    r"\bsubsequent to\b":          "following",
    r"\bprior to\b":               "earlier than",

    # soft hedges
    r"\bthereby making\b":         "and making",
    r"\bleading ultimately to\b":  "and ultimately",
    r"\bpaving the way for\b":     "and allowing",
    r"\bwhich in turn\b":          "which"
}

# compile each regex once
COMPILED = {re.compile(pat, flags=re.I): repl for pat, repl in CUE_REPLACE.items()}

# ------------------------------------------------------------------
# 3. Helper function
# ------------------------------------------------------------------
def replace_cues(text: str) -> tuple[str, int]:
    """Return (masked_text, num_replacements) for one paragraph."""
    total = 0
    for regex, repl in COMPILED.items():
        text, n = regex.subn(repl, text)
        total += n
    return text, total

# ------------------------------------------------------------------
# 4. Load CSV and process
# ------------------------------------------------------------------
df = pd.read_csv(INFILE)
if "Context" not in df.columns:
    raise ValueError("Input CSV must contain a 'Context' column.")

row_mask_flags = []
masked_paragraphs = []
cue_counter = Counter()

for ctx in df["Context"]:
    new_ctx, n_rep = replace_cues(str(ctx))
    masked_paragraphs.append(new_ctx)
    row_mask_flags.append(1 if n_rep else 0)
    cue_counter["TOTAL"] += n_rep

df["Context_masked"] = masked_paragraphs
df["Masked"] = row_mask_flags  # 1 = row had ≥1 replacement

# ------------------------------------------------------------------
# 5. Save + report
# ------------------------------------------------------------------
df.to_csv(OUTFILE, index=False)

print(f"✔  Saved masked dataset to '{OUTFILE}'.")
print(f"➜  Rows masked        : {sum(row_mask_flags)} / {len(df)}")
print(f"➜  Total replacements : {cue_counter['TOTAL']}")
