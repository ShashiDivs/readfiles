"""
R2R Data Inspector — look inside any Excel before running an engine.

Shows per sheet:
  - dimensions, detected header row, column names
  - data type of each column (numeric / date / text / mixed)
  - sample values per column
  - red flags: total/subtotal rows, SAP-style dates (01.04.2026),
    trailing-minus negatives (1,000.00-), text-formatted numbers,
    empty columns, merged-looking gaps

Usage:
  python inspect_data.py data/prepaid_register.xlsx
  python inspect_data.py data/                      (inspects every .xlsx inside)
"""
import re
import sys
from pathlib import Path

import pandas as pd

SAP_DATE_RE       = re.compile(r"^\d{2}\.\d{2}\.\d{4}$")          # 01.04.2026
TRAILING_MINUS_RE = re.compile(r"^[\d,]+\.?\d*-$")                 # 1,000.00-
NUMBER_AS_TEXT_RE = re.compile(r"^-?[\d,]+\.?\d*$")                # "1,234.56"
TOTAL_WORDS       = ("total", "subtotal", "grand total", "sum", "totals")


def classify_value(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "empty"
    if isinstance(val, (int, float)):
        return "numeric"
    if isinstance(val, pd.Timestamp) or hasattr(val, "year"):
        return "date"
    s = str(val).strip()
    if SAP_DATE_RE.match(s):
        return "sap_date_text"
    if TRAILING_MINUS_RE.match(s):
        return "trailing_minus_text"
    if NUMBER_AS_TEXT_RE.match(s):
        return "number_as_text"
    return "text"


def find_header_row(df: pd.DataFrame) -> int:
    """First row where at least 2 cells are strings — same rule the engines use."""
    for i, row in df.iterrows():
        non_null = row.dropna()
        if sum(1 for v in non_null if isinstance(v, str)) >= 2:
            return i
    return 0


def inspect_sheet(file_name: str, sheet: str, raw: pd.DataFrame):
    print(f"\n{'='*78}")
    print(f"FILE: {file_name}   SHEET: {sheet}")
    print(f"{'='*78}")

    raw = raw.dropna(how="all")
    if raw.empty:
        print("  (sheet is empty)")
        return

    header_row = find_header_row(raw)
    print(f"  Raw rows (non-empty)   : {len(raw)}")
    print(f"  Detected header row    : {header_row + 1} (Excel row number)")
    if header_row > 0:
        print(f"  ⚠  {header_row} row(s) above the header — title/filter rows, engines skip these")

    df = raw.copy()
    df.columns = df.iloc[header_row]
    df = df.iloc[header_row + 1:].reset_index(drop=True)
    df = df.dropna(how="all")
    df.columns = [str(c).strip() if c is not None else f"col_{i}" for i, c in enumerate(df.columns)]

    print(f"  Data rows after header : {len(df)}")
    print(f"  Columns ({len(df.columns)})")

    # ── Per-column analysis ────────────────────────────────────────────────────
    print(f"\n  {'Column':<32} {'Type(s)':<28} Sample values")
    print(f"  {'─'*32} {'─'*28} {'─'*30}")

    flags = []
    for col in df.columns:
        values = df[col].dropna()
        types  = values.map(classify_value).value_counts().to_dict() if len(values) else {}
        type_desc = ", ".join(f"{t}({n})" for t, n in list(types.items())[:3]) or "all empty"

        samples = [str(v)[:18] for v in values.head(3).tolist()]
        sample_desc = " | ".join(samples) if samples else "—"

        print(f"  {str(col)[:32]:<32} {type_desc[:28]:<28} {sample_desc[:38]}")

        if "sap_date_text" in types:
            flags.append(f"Column '{col}': SAP-style text dates (01.04.2026) — need conversion")
        if "trailing_minus_text" in types:
            flags.append(f"Column '{col}': trailing-minus negatives (1,000.00-) — need conversion")
        if "number_as_text" in types and "numeric" in types:
            flags.append(f"Column '{col}': MIXED numeric and text numbers — inconsistent formatting")
        elif "number_as_text" in types:
            flags.append(f"Column '{col}': numbers stored as text — need conversion")
        if not types:
            flags.append(f"Column '{col}': completely empty")

    # ── Total/subtotal row detection ───────────────────────────────────────────
    total_rows = []
    for idx, row in df.iterrows():
        row_text = " ".join(str(v).lower() for v in row.dropna().tolist())
        if any(w in row_text for w in TOTAL_WORDS):
            total_rows.append(idx + header_row + 2)  # back to Excel row numbers
    if total_rows:
        flags.append(f"Possible total/subtotal rows at Excel rows {total_rows[:8]} — must exclude from calculations")

    # ── Duplicate column names ─────────────────────────────────────────────────
    dupe_cols = pd.Series(df.columns)[pd.Series(df.columns).duplicated()].unique()
    if len(dupe_cols):
        flags.append(f"Duplicate column names: {list(dupe_cols)} — only the first is reachable")

    # ── Red flags summary ──────────────────────────────────────────────────────
    if flags:
        print(f"\n  RED FLAGS ({len(flags)}):")
        for f in flags:
            print(f"    ⚠  {f}")
    else:
        print(f"\n  ✓ No red flags — data looks clean for the engine")


def inspect_file(path: Path):
    try:
        xl = pd.ExcelFile(path)
    except Exception as e:
        print(f"\n  Cannot open {path.name}: {e}")
        return

    print(f"\n{'#'*78}")
    print(f"# {path.name} — {len(xl.sheet_names)} sheet(s): {xl.sheet_names}")
    print(f"{'#'*78}")

    for sheet in xl.sheet_names:
        raw = xl.parse(sheet, header=None)
        inspect_sheet(path.name, sheet, raw)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    target = Path(sys.argv[1])
    if target.is_dir():
        files = sorted(target.glob("*.xlsx")) + sorted(target.glob("*.xls"))
        if not files:
            print(f"  No Excel files found in {target}")
            sys.exit(1)
        for f in files:
            if not f.name.startswith("~$"):   # skip Excel lock files
                inspect_file(f)
    elif target.exists():
        inspect_file(target)
    else:
        print(f"  Not found: {target}")
        sys.exit(1)

    print()
