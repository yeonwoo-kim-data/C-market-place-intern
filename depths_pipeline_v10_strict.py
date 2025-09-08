#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depths Pipeline (v10 strict slim)
---------------------------------
Adds Depth 1 (브랜드명) and Depth 2 (규격/스펙) to an input CSV using the final rules we converged on.
- Brand (Depth 1): 
    1) Copy from `brand_name`
    3) If empty, and `pdt_name_clean` contains "®", fill with the token before ® (whole prefix if no whitespace).
   (Step 2 (candidate mapping) is intentionally excluded for reliability.)

- Spec (Depth 2):
    Start from `pdt_spec`, then merge (with "/" delimiter, de-duplicated case-insensitively) the following extracted items:
    A) Units with decimals & integers (m/cm/mm, g/ml/l/리터) with correctness guard for decimals (avoid 1.5m -> 5m)
    B) Composed dimensions:
        - unit x unit (e.g., "150mm x 150mm" -> "150mmx150mm")
        - m x mm, generic "num x num unit" forms
    C) Paper sizes, including "glued" forms with Korean (e.g., "컬러레이저용지a4")
    D) Quantity combos (e.g., "250매x10권", supports x or ×)
    E) Product codes:
        - Hyphen/underscore mixed codes, requiring at least one letter and one digit, not starting with "no." (e.g., "clt-p407c")
        - Compact alnum codes (letters+digits, length>=3), not starting with "no." (e.g., "c6578da")
    F) Parentheses from original `pdt_name` as-is: "(...)" snippets

Output: slim CSV with columns:
    pdt_name, pdt_name_clean, pdt_cas, 브랜드명, 규격/스펙

Usage:
    python depths_pipeline_v10_strict.py --input df_new.csv --output out.csv

Notes:
- The script is idempotent: running multiple times keeps specs de-duplicated.
- It removes stale unit tokens before adding re-extracted units to fix decimal issues (e.g., 1.5m).
"""

import argparse
import re
import sys
import pandas as pd
import numpy as np

def merge_specs(existing, parts):
    """Merge parts into existing spec using '/' delimiter, de-duplicated case-insensitively."""
    base = []
    if not (pd.isna(existing) or str(existing).strip() == ""):
        base = [p.strip() for p in str(existing).split("/") if p.strip()]
    base_norm = [p.lower() for p in base]
    for p in parts:
        if not p:
            continue
        pl = str(p).strip()
        if not pl:
            continue
        if pl.lower() not in base_norm:
            base.append(pl)
            base_norm.append(pl.lower())
    if not base:
        return np.nan
    return "/".join(base)

def extract_brand_from_rmark(text):
    """Extract brand from '®' rule: take the token immediately before ® (or whole prefix if no whitespace)."""
    if pd.isna(text):
        return np.nan
    s = str(text)
    if "®" not in s:
        return np.nan
    prefix = s.split("®")[0].strip()
    if not prefix:
        return np.nan
    token = prefix.split()[-1] if " " in prefix else prefix
    if len(token) >= 2 and re.search(r"[가-힣A-Za-z]", token):
        return token
    return np.nan

# --- Regex blocks (compiled once) ---

# A) Units (with decimal support) & correctness guard for integers (avoid capturing the trailing part of decimals)
M_DEC = re.compile(r"\b\d+\.\d+\s?m\b", re.IGNORECASE)
M_INT = re.compile(r"(?<!\.)\b\d+\s?m\b", re.IGNORECASE)
CM_DEC = re.compile(r"\b\d+\.\d+\s?cm\b", re.IGNORECASE)
CM_INT = re.compile(r"(?<!\.)\b\d+\s?cm\b", re.IGNORECASE)
MM_DEC = re.compile(r"\b\d+\.\d+\s?mm\b", re.IGNORECASE)
MM_INT = re.compile(r"(?<!\.)\b\d+\s?mm\b", re.IGNORECASE)

G = re.compile(r"\b\d+(?:\.\d+)?\s?g\b", re.IGNORECASE)
ML = re.compile(r"\b\d+(?:\.\d+)?\s?ml\b", re.IGNORECASE)
LIT = re.compile(r"\b\d+(?:\.\d+)?\s?l\b|\b\d+(?:\.\d+)?\s?리터\b", re.IGNORECASE)

# B) Composed dimensions
COMPOSED_UNIT_UNIT = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:mm|cm|m)\s?x\s?\d+(?:\.\d+)?\s?(?:mm|cm|m)\b", re.IGNORECASE
)
COMPOSED_M_MM = re.compile(
    r"\b\d+(?:\.\d+)?\s?m\s?x\s?\d+(?:\.\d+)?\s?mm\b", re.IGNORECASE
)
COMPOSED_GENERIC = re.compile(
    r"\b\d+(?:\.\d+)?\s?x\s?\d+(?:\.\d+)?\s?(?:mm|cm|m)\b", re.IGNORECASE
)

# C) Paper sizes (ASCII boundary so glued Korean+`a4` is captured)
PAPER = re.compile(r"(?<![A-Za-z0-9])(a[0-5]|b[0-5]|letter|legal)(?![A-Za-z0-9])", re.IGNORECASE)

# D) Quantity combos (x or ×), remove spaces in final token
QTY = re.compile(
    r"\b\d+\s?(?:매|권|묶음|롤|팩|박스)\s?[x×]\s?\d+\s?(?:매|권|묶음|롤|팩|박스)?\b",
    re.IGNORECASE
)

# E) Product codes
#   - hyphen/underscore codes with at least one letter and one digit (exclude "no.")
HYCODE = re.compile(
    r"(?<!no\.)\b(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+\b",
    re.IGNORECASE
)
#   - compact alnum codes (letters+digits, len>=3), exclude "no."
ALNUM_CODE = re.compile(
    r"(?<!no\.)\b(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{3,}\b",
    re.IGNORECASE
)

# F) Parentheses from original pdt_name (single-level)
PAREN = re.compile(r"\([^()]*\)")

UNIT_TOKEN_TO_DROP = re.compile(r"^\d+(?:\.\d+)?(?:mm|cm|m|g|ml|l|리터)$", re.IGNORECASE)

def extract_units_and_composed(text):
    """Return (units_list, composed_list) normalized (lower, no spaces), with '×' normalized to 'x'."""
    if pd.isna(text):
        return [], []
    s = str(text).replace("×", "x")
    composed = []
    for rx in (COMPOSED_UNIT_UNIT, COMPOSED_M_MM, COMPOSED_GENERIC):
        for m in rx.finditer(s):
            composed.append(m.group(0))
    # normalize composed
    composed_norm = []
    seen = set()
    for c in composed:
        c2 = re.sub(r"\s+", "", c.lower())
        if c2 not in seen:
            seen.add(c2)
            composed_norm.append(c2)

    # units
    units = set()
    for rx in (M_DEC, CM_DEC, MM_DEC, M_INT, CM_INT, MM_INT, G, ML, LIT):
        for m in rx.finditer(s):
            units.add(m.group(0).lower().replace(" ", ""))

    return list(dict.fromkeys(units)), composed_norm

def extract_paper_sizes(text):
    if pd.isna(text):
        return []
    return [m.group(1).lower() for m in PAPER.finditer(str(text))]

def extract_qty_combos(text):
    if pd.isna(text):
        return []
    caps = [m.group(0) for m in QTY.finditer(str(text))]
    out = []
    seen = set()
    for c in caps:
        cc = c.replace("×", "x")
        cc = re.sub(r"\s+", "", cc).lower()
        if cc not in seen:
            seen.add(cc)
            out.append(cc)
    return out

def extract_hy_codes(text):
    if pd.isna(text):
        return []
    s = str(text)
    caps = [m.group(0) for m in HYCODE.finditer(s)]
    out, seen = [], set()
    for c in caps:
        cl = c.lower()
        if cl not in seen:
            seen.add(cl)
            out.append(c)
    return out

def extract_alnum_codes(text):
    if pd.isna(text):
        return []
    s = str(text)
    caps = [m.group(0) for m in ALNUM_CODE.finditer(s)]
    out, seen = [], set()
    for c in caps:
        cl = c.lower()
        if cl not in seen:
            seen.add(cl)
            out.append(c)
    return out

def extract_parentheses(text):
    if pd.isna(text):
        return []
    caps = [m.group(0) for m in PAREN.finditer(str(text))]
    out, seen = [], set()
    for c in caps:
        cl = c.lower()
        if cl not in seen:
            seen.add(cl)
            out.append(c)
    return out

def build_spec(row):
    """Rebuild and merge spec following v10 strict slim logic."""
    base_spec = row.get("pdt_spec", np.nan)
    # start with base spec tokens
    if pd.isna(base_spec) or str(base_spec).strip() == "":
        base_tokens = []
    else:
        base_tokens = [p for p in str(base_spec).split("/") if p.strip()]

    # remove old unit tokens (to avoid stale wrong values like 5m from 1.5m)
    base_tokens = [t for t in base_tokens if not UNIT_TOKEN_TO_DROP.match(t.strip())]

    # Extract from pdt_name_clean and pdt_name
    pnc = row.get("pdt_name_clean", "")
    pn = row.get("pdt_name", "")

    units_c, composed_c = extract_units_and_composed(pnc)
    units_n, composed_n = extract_units_and_composed(pn)

    paper = extract_paper_sizes(pnc) + extract_paper_sizes(pn)
    qty = extract_qty_combos(pnc) + extract_qty_combos(pn)
    hy = extract_hy_codes(pnc) + extract_hy_codes(pn)
    alnum = extract_alnum_codes(pnc) + extract_alnum_codes(pn)
    paren = extract_parentheses(pn)  # parentheses only from original name

    # Merge order: composed -> units -> paper -> qty -> hy-codes -> alnum-codes -> parentheses
    merged = merge_specs("/".join(base_tokens) if base_tokens else np.nan, composed_c + composed_n)
    merged = merge_specs(merged, units_c + units_n)
    merged = merge_specs(merged, paper)
    merged = merge_specs(merged, qty)
    merged = merge_specs(merged, hy)
    merged = merge_specs(merged, alnum)
    merged = merge_specs(merged, paren)

    return merged

def run(input_csv, output_csv, output_full_csv=None):
    df = pd.read_csv(input_csv)
    # Drop Unnamed columns if any
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    # Depth 1: 브랜드명
    df["브랜드명"] = df.get("brand_name")
    mask_step3 = df["브랜드명"].isna() & df.get("pdt_name_clean", pd.Series([""]*len(df))).astype(str).str.contains("®", regex=False)
    df.loc[mask_step3, "브랜드명"] = df.loc[mask_step3, "pdt_name_clean"].apply(extract_brand_from_rmark)

    # Depth 2: 규격/스펙
    df["규격/스펙"] = df.apply(build_spec, axis=1)

    # Slim columns
    slim_cols = ["pdt_name", "pdt_name_clean", "pdt_cas", "브랜드명", "규격/스펙"]
    missing = [c for c in slim_cols if c not in df.columns]
    for c in missing:
        df[c] = np.nan
    slim = df[slim_cols].copy()
    slim.to_csv(output_csv, index=False)

    if output_full_csv:
        df.to_csv(output_full_csv, index=False)

    return slim

def main():
    ap = argparse.ArgumentParser(description="Depths Pipeline v10 (strict slim)")
    ap.add_argument("--input", required=True, help="Path to input CSV (e.g., df_new.csv)")
    ap.add_argument("--output", required=True, help="Path to slim output CSV")
    ap.add_argument("--output-full", required=False, help="(Optional) Path to full output CSV with all columns")
    args = ap.parse_args()

    run(args.input, args.output, args.output_full)

if __name__ == "__main__":
    main()
