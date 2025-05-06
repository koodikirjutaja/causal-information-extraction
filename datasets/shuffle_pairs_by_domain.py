#!/usr/bin/env python3
# swap_pairs_rewrite_context.py
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
    0.0 → keep everything (all Label = 0)
    1.0 → turn every eligible row into a negative (Label = 1)

Input  : CSV with columns Domain, Cause, Effect, Context
Output : same columns + Label

Usage
-----
python swap_pairs_rewrite_context.py  in.csv  out.csv  --ratio 0.5
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
_PUNCT_TABLE = str.maketrans(
    {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "‑": "-"}
)


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).translate(_PUNCT_TABLE).lower()
    text = re.sub(r"[^\w\s-]", " ", text)  # strip punctuation except hyphen
    text = text.replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def contains_ci(text: str, phrase: str) -> bool:
    return _norm(phrase) in _norm(text)


def replace_first_ci(text: str, old: str, new: str) -> str:
    pat = re.compile(re.escape(old), flags=re.I)
    return pat.sub(new, text, count=1)


# ---------------------------------------------------------------------------
# core swapper
# ---------------------------------------------------------------------------
def swap_within_domain(block: pd.DataFrame, ratio: float, rng: np.random.Generator):
    df = block.copy()
    df["Label"] = 0  # default: unchanged / positive

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
        ctx_a = replace_first_ci(df.at[a, "Context"], cause_a, cause_b)
        ctx_a = replace_first_ci(ctx_a, effect_a, effect_b)
        df.at[a, "Context"] = ctx_a
        df.at[a, "Label"]   = 1

        # row b
        df.at[b, "Cause"], df.at[b, "Effect"] = cause_a, effect_a
        ctx_b = replace_first_ci(df.at[b, "Context"], cause_b, cause_a)
        ctx_b = replace_first_ci(ctx_b, effect_b, effect_a)
        df.at[b, "Context"] = ctx_b
        df.at[b, "Label"]   = 1

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(infile: Path, outfile: Path, ratio: float, seed: int):
    df_in = pd.read_csv(infile)
    required = {"Domain", "Cause", "Effect", "Context"}
    if not required.issubset(df_in.columns):
        sys.exit(f"X CSV must contain columns: {required}")

    rng = np.random.default_rng(seed)
    out_blocks = [
        swap_within_domain(block, ratio, rng)
        for _, block in df_in.groupby("Domain", sort=False)
    ]
    df_out = pd.concat(out_blocks, ignore_index=True)
    df_out.to_csv(outfile, index=False)


    df_out = pd.read_csv(outfile)
    n_changed = (df_out["Label"] == 1).sum()
    print(f"✔ wrote '{outfile}'")
    print(f"   swapped (Label = 1): {n_changed}")
    print(f"   kept    (Label = 0): {len(df_out) - n_changed}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("infile", type=Path, help="input CSV")
    ap.add_argument("outfile", type=Path, help="output CSV")
    ap.add_argument("--ratio", type=float, default=0.5, help="fraction of eligible rows to swap")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    args = ap.parse_args()
    main(args.infile, args.outfile, args.ratio, args.seed)
