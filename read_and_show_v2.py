"""
R2R Read and Show v2 — No LLM, Pure pandas + difflib

1. Reads bank_statement.xlsx and bank_gl_entries.xlsx from data/
2. Auto-detects columns — no fixed column names needed
3. Matches bank transactions against GL entries by amount + reference
4. Classifies: MATCHED / MISMATCH / TIMING_DIFFERENCE / MISSING
5. Logs every step with correlation ID
6. Saves output to data/output_v2.json

Usage:
  python read_and_show_v2.py
"""
import json
import os
import sys
import uuid
import logging
from pathlib import Path
from datetime import datetime, timezone
from difflib import SequenceMatcher

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
LOGS_DIR = Path(__file__).parent / "logs"

# How similar two narrations must be to count as a match (0.0 to 1.0)
NARRATION_THRESHOLD = 0.4

# Amount must match within this tolerance (handles rounding differences)
AMOUNT_TOLERANCE = 0.01


# ── Logger ─────────────────────────────────────────────────────────────────────

class JSONFormatter(logging.Formatter):
    def __init__(self, correlation_id: str):
        super().__init__()
        self.correlation_id = correlation_id

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "level":          record.levelname,
            "correlation_id": self.correlation_id,
            "station":        getattr(record, "station", "pipeline"),
            "message":        record.getMessage()
        }
        for key, val in record.__dict__.items():
            if key not in ("msg", "args", "levelname", "levelno", "pathname",
                           "filename", "module", "exc_info", "exc_text",
                           "stack_info", "lineno", "funcName", "created",
                           "msecs", "relativeCreated", "thread", "threadName",
                           "processName", "process", "name", "message", "station"):
                entry[key] = val
        return json.dumps(entry)


def setup_logger(correlation_id: str) -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    log_file = LOGS_DIR / f"{correlation_id}.log"

    logger = logging.getLogger(f"r2r.{correlation_id}")
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter(correlation_id))
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(JSONFormatter(correlation_id))
    logger.addHandler(file_handler)

    logger.setLevel(logging.DEBUG)
    return logger


def get_log(station: str, logger: logging.Logger) -> logging.LoggerAdapter:
    return logging.LoggerAdapter(logger, extra={"station": station})


# ── Column detection ───────────────────────────────────────────────────────────

# Keywords to search for in column headers
COLUMN_KEYWORDS = {
    "date":        ["date", "dt", "value date", "posting date", "txn date", "transaction date"],
    "debit":       ["debit", "dr", "withdrawal", "dr amount", "lc amt debit", "withdrawl", "paid out"],
    "credit":      ["credit", "cr", "deposit", "cr amount", "lc amt credit", "paid in"],
    "reference":   ["ref", "reference", "doc", "document", "chq", "cheque", "check", "utr", "txn id", "transaction id"],
    "narration":   ["narration", "description", "particulars", "remarks", "details", "text"],
    "balance":     ["balance", "running balance", "closing balance"]
}


def detect_column(df: pd.DataFrame, field: str, log) -> str | None:
    """Find the column in df that best matches the given field using keyword search."""
    cols_lower = {col: col.lower().strip() for col in df.columns}
    keywords   = COLUMN_KEYWORDS.get(field, [])

    for keyword in keywords:
        for col, col_lower in cols_lower.items():
            if keyword in col_lower:
                log.info(f"Column detected", extra={"field": field, "column": col})
                return col

    log.warning(f"Column not found", extra={"field": field, "searched_in": list(df.columns)})
    return None


def detect_all_columns(df: pd.DataFrame, log) -> dict:
    """Detect all required columns in a dataframe."""
    return {field: detect_column(df, field, log) for field in COLUMN_KEYWORDS}


# ── Read Excel ─────────────────────────────────────────────────────────────────

def read_excel(file_path: str, log) -> dict[str, pd.DataFrame]:
    """
    Read all sheets from Excel.
    Returns dict of {sheet_name: dataframe}.
    Skips entirely empty sheets.
    """
    log.info("Reading Excel", extra={"file": Path(file_path).name})
    xl      = pd.ExcelFile(file_path)
    sheets  = {}

    for sheet in xl.sheet_names:
        df = xl.parse(sheet, header=None)
        df.dropna(how="all", inplace=True)
        if df.empty:
            continue

        # Find the header row — first row where most cells are non-null strings
        header_row = 0
        for i, row in df.iterrows():
            non_null = row.dropna()
            string_count = sum(1 for v in non_null if isinstance(v, str))
            if string_count >= 2:
                header_row = i
                break

        df.columns = df.iloc[header_row]
        df = df.iloc[header_row + 1:].reset_index(drop=True)
        df.dropna(how="all", inplace=True)
        df.columns = [str(c).strip() if c is not None else f"col_{i}" for i, c in enumerate(df.columns)]

        sheets[sheet] = df
        log.info("Sheet loaded", extra={"sheet": sheet, "rows": len(df), "columns": list(df.columns)})

    return sheets


# ── Normalize ──────────────────────────────────────────────────────────────────

def to_float(val) -> float:
    """Safely convert any value to float."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0.0
    try:
        return float(str(val).replace(",", "").replace(" ", ""))
    except (ValueError, TypeError):
        return 0.0


def to_str(val) -> str:
    """Safely convert any value to string."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def normalize_rows(df: pd.DataFrame, cols: dict, source: str) -> list[dict]:
    """
    Convert dataframe rows to a standard list of dicts using detected columns.
    Each row has: date, debit, credit, reference, narration, balance, _source
    """
    rows = []
    for _, row in df.iterrows():
        debit  = to_float(row[cols["debit"]])  if cols.get("debit")     else 0.0
        credit = to_float(row[cols["credit"]]) if cols.get("credit")    else 0.0

        # Skip rows where both debit and credit are 0 — likely blank or header rows
        if debit == 0.0 and credit == 0.0:
            continue

        rows.append({
            "date":      to_str(row[cols["date"]])      if cols.get("date")      else "",
            "debit":     debit,
            "credit":    credit,
            "reference": to_str(row[cols["reference"]]) if cols.get("reference") else "",
            "narration": to_str(row[cols["narration"]]) if cols.get("narration") else "",
            "balance":   to_float(row[cols["balance"]]) if cols.get("balance")   else 0.0,
            "_source":   source
        })
    return rows


# ── Matching ───────────────────────────────────────────────────────────────────

def similarity(a: str, b: str) -> float:
    """Return similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def amounts_match(bank_row: dict, gl_row: dict) -> bool:
    """
    Bank debit = GL credit (bank pays out, GL books as credit)
    Bank credit = GL debit (bank receives, GL books as debit)
    """
    bank_amount = bank_row["debit"] or bank_row["credit"]
    gl_amount   = gl_row["debit"]   or gl_row["credit"]
    return abs(bank_amount - gl_amount) <= AMOUNT_TOLERANCE


def amounts_close(bank_row: dict, gl_row: dict) -> tuple[bool, float]:
    """Check if amounts are close but not exact — possible mismatch."""
    bank_amount = bank_row["debit"] or bank_row["credit"]
    gl_amount   = gl_row["debit"]   or gl_row["credit"]
    diff        = abs(bank_amount - gl_amount)
    # Within 5% of the larger amount
    threshold   = max(bank_amount, gl_amount) * 0.05
    return diff <= threshold and diff > AMOUNT_TOLERANCE, round(diff, 2)


def find_best_gl_match(bank_row: dict, gl_rows: list[dict], used: set) -> tuple[dict | None, str]:
    """
    Find the best matching GL row for a bank row.
    Returns (matched_gl_row, match_type) or (None, "MISSING")
    match_type: MATCHED / MISMATCH
    """
    best_match      = None
    best_score      = 0.0
    best_match_type = "MISSING"

    for i, gl_row in enumerate(gl_rows):
        if i in used:
            continue

        # Narration similarity
        nar_score = max(
            similarity(bank_row["narration"], gl_row["narration"]),
            similarity(bank_row["narration"], gl_row["reference"]),
            similarity(bank_row["reference"], gl_row["reference"])
        )

        if nar_score < NARRATION_THRESHOLD:
            continue

        if amounts_match(bank_row, gl_row):
            if nar_score > best_score:
                best_score      = nar_score
                best_match      = (i, gl_row)
                best_match_type = "MATCHED"
        else:
            is_close, diff = amounts_close(bank_row, gl_row)
            if is_close and nar_score > best_score:
                best_score      = nar_score
                best_match      = (i, gl_row)
                best_match_type = "MISMATCH"

    if best_match:
        return best_match[1], best_match_type, best_match[0]
    return None, "MISSING", -1


def match(bank_rows: list[dict], gl_rows: list[dict], log) -> dict:
    log.info("Starting match", extra={"bank_rows": len(bank_rows), "gl_rows": len(gl_rows)})

    matched            = []
    mismatches         = []
    timing_differences = []
    missing            = []
    used_gl            = set()

    for bank_row in bank_rows:
        gl_row, match_type, gl_idx = find_best_gl_match(bank_row, gl_rows, used_gl)

        bank_amount = bank_row["debit"] or bank_row["credit"]

        if match_type == "MATCHED":
            used_gl.add(gl_idx)
            matched.append({
                "date":        bank_row["date"],
                "ref":         bank_row["reference"] or gl_row["reference"],
                "narration":   bank_row["narration"],
                "bank_amount": bank_amount,
                "gl_amount":   gl_row["debit"] or gl_row["credit"]
            })
            log.info("Matched", extra={"ref": bank_row["reference"], "amount": bank_amount})

        elif match_type == "MISMATCH":
            used_gl.add(gl_idx)
            gl_amount = gl_row["debit"] or gl_row["credit"]
            diff      = round(bank_amount - gl_amount, 2)
            mismatches.append({
                "date":        bank_row["date"],
                "ref":         bank_row["reference"] or gl_row["reference"],
                "narration":   bank_row["narration"],
                "bank_amount": bank_amount,
                "gl_amount":   gl_amount,
                "difference":  diff
            })
            log.warning("Mismatch", extra={"ref": bank_row["reference"], "difference": diff})

        else:
            # Not found in GL — could be timing difference or missing
            # Heuristic: if it's a large round number it's likely a timing difference (cheque not cleared)
            is_round  = bank_amount % 1000 == 0
            match_type = "TIMING_DIFFERENCE" if is_round else "MISSING"

            entry = {
                "date":       bank_row["date"],
                "ref":        bank_row["reference"],
                "narration":  bank_row["narration"],
                "amount":     bank_amount,
                "present_in": "BANK_ONLY"
            }
            if match_type == "TIMING_DIFFERENCE":
                timing_differences.append(entry)
                log.info("Timing difference", extra={"ref": bank_row["reference"], "amount": bank_amount})
            else:
                entry["action"] = "Investigate — no matching GL entry found"
                missing.append(entry)
                log.warning("Missing in GL", extra={"ref": bank_row["reference"], "amount": bank_amount})

    # GL rows not matched against any bank row
    for i, gl_row in enumerate(gl_rows):
        if i not in used_gl:
            gl_amount  = gl_row["debit"] or gl_row["credit"]
            is_round   = gl_amount % 1000 == 0
            match_type = "TIMING_DIFFERENCE" if is_round else "MISSING"

            entry = {
                "date":       gl_row["date"],
                "ref":        gl_row["reference"],
                "narration":  gl_row["narration"],
                "amount":     gl_amount,
                "present_in": "GL_ONLY"
            }
            if match_type == "TIMING_DIFFERENCE":
                timing_differences.append(entry)
                log.info("Timing difference (GL only)", extra={"ref": gl_row["reference"], "amount": gl_amount})
            else:
                entry["action"] = "Investigate — no matching bank entry found"
                missing.append(entry)
                log.warning("Missing in bank", extra={"ref": gl_row["reference"], "amount": gl_amount})

    is_reconciled = len(mismatches) == 0 and len(missing) == 0
    summary = {
        "total_bank_rows":          len(bank_rows),
        "total_gl_rows":            len(gl_rows),
        "total_matched":            len(matched),
        "total_mismatches":         len(mismatches),
        "total_timing_differences": len(timing_differences),
        "total_missing":            len(missing),
        "is_reconciled":            is_reconciled
    }

    log.info("Match complete", extra=summary)
    return {
        "matched":            matched,
        "mismatches":         mismatches,
        "timing_differences": timing_differences,
        "missing":            missing,
        "summary":            summary
    }


# ── Print helpers ──────────────────────────────────────────────────────────────

def print_columns(label: str, cols: dict):
    print(f"\n  {label} — detected columns:")
    for field, col in cols.items():
        status = col if col else "NOT FOUND"
        print(f"    {field:<12} → {status}")


def print_match_result(result: dict):
    summary = result.get("summary", {})
    status  = "RECONCILED" if summary.get("is_reconciled") else "NOT RECONCILED"

    print(f"\n{'='*60}")
    print(f"MATCH RESULT  [{status}]")
    print(f"{'='*60}")
    print(f"  Bank rows            : {summary.get('total_bank_rows', 0)}")
    print(f"  GL rows              : {summary.get('total_gl_rows', 0)}")
    print(f"  Matched              : {summary.get('total_matched', 0)}")
    print(f"  Mismatches           : {summary.get('total_mismatches', 0)}")
    print(f"  Timing differences   : {summary.get('total_timing_differences', 0)}")
    print(f"  Missing entries      : {summary.get('total_missing', 0)}")

    if result.get("matched"):
        print(f"\n  MATCHED ({len(result['matched'])}):")
        for m in result["matched"]:
            print(f"    ✓  {m.get('ref',''):<15} {m.get('narration','')[:40]:<40}  {m.get('bank_amount', 0):>12,.2f}")

    if result.get("mismatches"):
        print(f"\n  MISMATCHES ({len(result['mismatches'])}):")
        for m in result["mismatches"]:
            print(f"    ✗  {m.get('ref',''):<15} {m.get('narration','')[:30]:<30}  Bank:{m.get('bank_amount',0):>10,.2f}  GL:{m.get('gl_amount',0):>10,.2f}  Diff:{m.get('difference',0):>8,.2f}")

    if result.get("timing_differences"):
        print(f"\n  TIMING DIFFERENCES ({len(result['timing_differences'])}):")
        for t in result["timing_differences"]:
            print(f"    ⏳  {t.get('ref',''):<15} {t.get('narration','')[:35]:<35}  {t.get('amount',0):>12,.2f}  ({t.get('present_in','')})")

    if result.get("missing"):
        print(f"\n  MISSING ENTRIES ({len(result['missing'])}):")
        for m in result["missing"]:
            print(f"    ?  {m.get('ref',''):<15} {m.get('narration','')[:35]:<35}  {m.get('amount',0):>12,.2f}  ({m.get('present_in','')})")
            print(f"       Action: {m.get('action','')}")

    print(f"{'='*60}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    correlation_id = f"R2R-V2-{uuid.uuid4().hex[:8]}"
    logger         = setup_logger(correlation_id)
    log_main       = get_log("pipeline",   logger)
    log_read       = get_log("reader",     logger)
    log_match      = get_log("matching",   logger)

    print(f"\n{'='*60}")
    print("R2R v2 — No-LLM Bank Reconciliation")
    print(f"{'='*60}")
    print(f"  Correlation ID : {correlation_id}")

    log_main.info("Pipeline started", extra={"correlation_id": correlation_id})

    # ── Check files ────────────────────────────────────────────────────────────
    bank_file = DATA_DIR / "bank_statement.xlsx"
    gl_file   = DATA_DIR / "bank_gl_entries.xlsx"

    for f in [bank_file, gl_file]:
        if not f.exists():
            print(f"\n  Missing file: {f}")
            log_main.error("File not found", extra={"path": str(f)})
            sys.exit(1)

    # ── Read ───────────────────────────────────────────────────────────────────
    bank_sheets = read_excel(str(bank_file), log_read)
    gl_sheets   = read_excel(str(gl_file),   log_read)

    if not bank_sheets or not gl_sheets:
        print("  No data found in Excel files.")
        sys.exit(1)

    # Use first sheet from each file
    bank_sheet_name = list(bank_sheets.keys())[0]
    gl_sheet_name   = list(gl_sheets.keys())[0]
    bank_df         = bank_sheets[bank_sheet_name]
    gl_df           = gl_sheets[gl_sheet_name]

    print(f"\n  Bank sheet : {bank_sheet_name}  ({len(bank_df)} rows)")
    print(f"  GL sheet   : {gl_sheet_name}  ({len(gl_df)} rows)")

    # ── Detect columns ─────────────────────────────────────────────────────────
    bank_cols = detect_all_columns(bank_df, get_log("col_detect", logger))
    gl_cols   = detect_all_columns(gl_df,   get_log("col_detect", logger))

    print_columns("Bank Statement", bank_cols)
    print_columns("GL Entries",     gl_cols)

    # ── Normalize ──────────────────────────────────────────────────────────────
    bank_rows = normalize_rows(bank_df, bank_cols, "BANK")
    gl_rows   = normalize_rows(gl_df,   gl_cols,   "GL")

    print(f"\n  Bank transactions : {len(bank_rows)}")
    print(f"  GL entries        : {len(gl_rows)}")

    if not bank_rows or not gl_rows:
        print("\n  Could not extract transactions — check column detection above.")
        log_main.error("No rows after normalization", extra={"bank_rows": len(bank_rows), "gl_rows": len(gl_rows)})
        sys.exit(1)

    # ── Match ──────────────────────────────────────────────────────────────────
    result = match(bank_rows, gl_rows, log_match)
    print_match_result(result)

    # ── Save ───────────────────────────────────────────────────────────────────
    output = {
        "correlation_id": correlation_id,
        "bank_sheet":     bank_sheet_name,
        "gl_sheet":       gl_sheet_name,
        "bank_rows":      bank_rows,
        "gl_rows":        gl_rows,
        "match_result":   result
    }

    out_path = DATA_DIR / "output_v2.json"
    out_path.write_text(json.dumps(output, indent=2))

    log_main.info("Pipeline complete", extra={"output": str(out_path)})
    print(f"  Output : {out_path}")
    print(f"  Logs   : {LOGS_DIR}/\n")


if __name__ == "__main__":
    run()
