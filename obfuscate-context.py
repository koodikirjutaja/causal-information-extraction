#ideas:
# mask domain hints (Replace “steel import tariffs” with “Policy-A")

from __future__ import annotations

import sys, random, re, textwrap, pandas as pd
from pathlib import Path

# --------------------------------------------------------------------------------------
# 1. resources
EXPLICIT_CUES = [
    r"\btherefore\b", r"\bthus\b", r"\buzz?[\s-]*resulted in\b", r"\bultimately\b",
    r"\bleading to\b", r"\bconsequently\b", r"\btrigger(?:ed)?(?: off)?\b",
    r"\bprecipitat(?:ed|ing)\b", r"\bin turn\b", r"\bas a result\b"
]

SOFTER_CONNECTIVES = [
    "after a sequence of events", "some time later", "in the aftermath",
    "in what many observers noted", "within a broader chain of factors"
]

DISTRACTORS = [
    "fluctuating market conditions", "parallel shifts in demographic trends",
    "changes in reporting methodology", "unrelated policy reforms under debate",
    "regional weather anomalies", "simultaneous infrastructure upgrades",
    "ongoing litigation in the sector", "a separate but contemporaneous study"
]

REFS = ["this development", "that occurrence", "the situation", "the phenomenon", "the episode"]

# --------------------------------------------------------------------------------------
# 2. helper
def obfuscate_context(text: str, cause: str, effect: str, seed: int | None = None) -> str:
    """Return a new sentence with cues diffused, entity boundaries blurred,
    and a distractor inserted – but *leave the original cause/effect phrases in place once*.
    """
    rnd = random.Random(seed)

    # 2-a swap explicit connective for a softer one
    def swap_connective(_match):
        return rnd.choice(SOFTER_CONNECTIVES)

    for cue in EXPLICIT_CUES:
        text = re.sub(cue, swap_connective, text, flags=re.IGNORECASE)

    # 2-b blur the *second* appearance of each entity (if present)
    def blur_second(pattern: str, replacements: list[str]) -> None:
        nonlocal text
        matches = list(re.finditer(re.escape(pattern), text, flags=re.IGNORECASE))
        if len(matches) >= 2:
            start, end = matches[1].span()
            text = text[:start] + rnd.choice(replacements) + text[end:]

    blur_second(cause,  [cause, rnd.choice(REFS)])
    blur_second(effect, [effect, rnd.choice(REFS)])

    # 2-c inject a distractor clause right after the first occurrence of the cause
    def inject(m: re.Match) -> str:
        return f"{m.group(0)}, together with {rnd.choice(DISTRACTORS)},"

    text = re.sub(re.escape(cause), inject, text, count=1, flags=re.IGNORECASE)

    return textwrap.fill(text, width=100)

# --------------------------------------------------------------------------------------
# 3. I/O plumbing
def main(inp: str, out: str) -> None:
    df = pd.read_csv(inp)

    # Column detection – handle either naming scheme
    possible_cause_cols  = ["Left phrase", "Cause"]
    possible_effect_cols = ["Right phrase", "Effect"]
    possible_text_cols   = ["Generated context", "Context", "Sentence"]

    cause_col  = next(c for c in possible_cause_cols  if c in df.columns)
    effect_col = next(c for c in possible_effect_cols if c in df.columns)
    text_col   = next(c for c in possible_text_cols   if c in df.columns)

    df["Obfuscated context"] = [
        obfuscate_context(row[text_col], row[cause_col], row[effect_col], seed=i)
        for i, row in df.iterrows()
    ]

    # keep original cols + new one
    cols = [cause_col, effect_col, text_col, "Obfuscated context"]
    df[cols].to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")

# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    src  = sys.argv[1] if len(sys.argv) > 1 else "all_pairs_context_varied.csv"
    dest = sys.argv[2] if len(sys.argv) > 2 else "all_pairs_context_obfuscated.csv"

    if not Path(src).is_file():
        sys.exit(f"Source file {src!r} not found.")
    main(src, dest)