#!/usr/bin/env python3
# make_call_ready.py
"""
Strip columns so LLM prompts can’t expose the answer.

Usage
-----
python make_call_ready.py  data/processed/v1/*.csv
"""

import sys, pathlib, pandas as pd, glob

def make_call_ready(fp: str):
    df = pd.read_csv(fp)
    ctx_col = "Context_masked" if "Context_masked" in df.columns else "Context"
    keep = ["id"] if "id" in df.columns else []
    keep.append(ctx_col)

    out_path = pathlib.Path(fp).with_suffix("")  # drop .csv
    out_path = out_path.with_name(out_path.name + "_call.csv")

    df[keep].rename(columns={ctx_col: "Context"}).to_csv(out_path, index=False)
    print("✔", out_path)

# -----------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("give at least one CSV path (wildcards ok)"); sys.exit(1)
    for pattern in sys.argv[1:]:
        for path in glob.glob(pattern):
            make_call_ready(path)
