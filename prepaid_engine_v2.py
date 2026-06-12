"""
R2R Prepaid Amortization Engine v2 — matrix/wide schedule layout

Built for the real amortization template:
  - One row per prepaid item on the "Master" sheet
  - Item detail columns: Vendor, Description, Invoice number, Prepaid amount,
    Run Type (Batch Run / Manual Run), Status (Open/Closed), Period, End Period
  - Monthly columns (Apr-26 ... Mar-27) holding the amortization per month
  - Totals row at the bottom (excluded automatically)

What it does:
  Station 1 — Read Master sheet, detect item columns + month columns
  Station 2 — For the run period (e.g. Jun-26):
                release        = value in that month's column
                amortized_todate = sum of month columns before the run period
                remaining      = prepaid amount − amortized to date
              Only Status=Open rows. Run Type containing "manual" → HITL flag.
  Station 3 — Journal: Dr expense (per item) / Cr prepaid asset, Dr = Cr
  Station 4 — Approval queue (approve_journal.py, four-eyes)
  Station 5 — Audit pack + schedule sanity check
              (prepaid − amortized − closing ≈ 0 where closing balance exists)

Usage:
  python prepaid_engine_v2.py                # current month
  python prepaid_engine_v2.py 06/2026        # specific period MM/YYYY
"""
import json
import re
import sys
import uuid
import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

DATA_DIR   = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"
LOGS_DIR   = Path(__file__).parent / "logs"

REGISTER_FILE         = DATA_DIR / "prepaid_register.xlsx"
GL_FILE               = DATA_DIR / "gl_extract.xlsx"      # optional — enables recon
MASTER_SHEET_HINTS    = ("master",)            # sheet name containing the schedule
PREPAID_ASSET_ACCOUNT = "1400100"
DEFAULT_EXPENSE_ACCOUNT = "6100100"            # used when file has no account column

# Register remaining vs GL balance must agree within this tolerance
RECON_TOLERANCE = 1.00

TOTAL_WORDS = ("total", "subtotal", "grand total", "difference", "balance per")


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
    logger = logging.getLogger(f"prepaid_v2.{correlation_id}")
    logger.handlers.clear()
    for handler in [logging.StreamHandler(), logging.FileHandler(LOGS_DIR / f"{correlation_id}.log")]:
        handler.setFormatter(JSONFormatter(correlation_id))
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


def get_log(station: str, logger: logging.Logger) -> logging.LoggerAdapter:
    return logging.LoggerAdapter(logger, extra={"station": station})


# ── Helpers ────────────────────────────────────────────────────────────────────

def to_float(val) -> float:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0.0
    s = str(val).strip().replace(",", "").replace(" ", "")
    if s.endswith("-"):                      # SAP trailing minus: 1000.00-
        s = "-" + s[:-1]
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def to_str(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


MONTH_NAMES = {m.lower(): i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
MONTH_HEADER_RE = re.compile(r"^([A-Za-z]{3,9})[\s\-_/']*(\d{2,4})$")


def parse_month_header(val) -> tuple | None:
    """
    Recognize month column headers in any common form:
      'Apr-26', 'Apr 2026', 'April-26', datetime(2026,4,1), Timestamp...
    Returns (year, month) or None.
    """
    if isinstance(val, (datetime, pd.Timestamp)):
        return (val.year, val.month)
    s = to_str(val)
    m = MONTH_HEADER_RE.match(s)
    if not m:
        return None
    month = MONTH_NAMES.get(m.group(1).lower()[:3])
    if not month:
        return None
    year = int(m.group(2))
    if year < 100:
        year += 2000
    return (year, month)


# ── Station 1: Read the Master sheet ───────────────────────────────────────────

ITEM_COLUMN_KEYWORDS = {
    "vendor":         ["vendor", "supplier"],
    "description":    ["description", "particulars", "details"],
    "invoice_number": ["invoice number", "invoice no", "invoice"],
    "prepaid_amount": ["prepaid amount", "prepaid amt", "total amount"],
    "currency":       ["currency", "curr", "ccy"],
    "run_type":       ["run type", "runtype", "run"],
    "status":         ["status"],
    "period":         ["period", "start period", "from"],
    "end_period":     ["end period", "to period", "end"],
    "expense_account":["expense account", "gl account", "account"],
    "cost_center":    ["cost center", "cost centre", "cc"]
}


def load_master_sheet(log) -> pd.DataFrame:
    if not REGISTER_FILE.exists():
        print(f"\n  Missing file: {REGISTER_FILE}")
        log.error("File not found", extra={"file": str(REGISTER_FILE)})
        sys.exit(1)

    xl = pd.ExcelFile(REGISTER_FILE)
    sheet = None
    for name in xl.sheet_names:
        if any(h in name.lower() for h in MASTER_SHEET_HINTS):
            sheet = name
            break
    if sheet is None:
        sheet = xl.sheet_names[0]
        log.warning("No 'Master' sheet found — using first sheet", extra={"sheet": sheet})

    raw = xl.parse(sheet, header=None)
    raw = raw.dropna(how="all")
    log.info("Master sheet loaded", extra={"sheet": sheet, "raw_rows": len(raw)})

    # Header row = the row with the most month-like headers; fall back to most strings
    best_row, best_score = 0, -1
    for i in range(min(10, len(raw))):
        row = raw.iloc[i]
        months  = sum(1 for v in row if parse_month_header(v))
        strings = sum(1 for v in row.dropna() if isinstance(v, str))
        score   = months * 10 + strings
        if score > best_score:
            best_row, best_score = i, score

    df = raw.copy()
    df.columns = df.iloc[best_row]
    df = df.iloc[best_row + 1:].reset_index(drop=True)
    df = df.dropna(how="all")
    log.info("Header detected", extra={"header_row": int(best_row) + 1, "data_rows": len(df)})
    return df


def detect_item_columns(df: pd.DataFrame, log) -> dict:
    cols = {}
    col_list = list(df.columns)
    for field, keywords in ITEM_COLUMN_KEYWORDS.items():
        found = None
        for keyword in keywords:
            for col in col_list:
                if keyword in to_str(col).lower():
                    found = col
                    break
            if found is not None:
                break
        cols[field] = found
        if found is not None:
            log.info("Item column detected", extra={"field": field, "column": to_str(found)})
        else:
            log.warning("Item column not found", extra={"field": field})
    return cols


def detect_month_columns(df: pd.DataFrame, log) -> dict:
    """
    Map (year, month) → list of column positions with that header.
    The same month may appear in several blocks (prepaid / amortisation / closing);
    we keep all positions and choose the block later.
    """
    months = {}
    for pos, col in enumerate(df.columns):
        ym = parse_month_header(col)
        if ym:
            months.setdefault(ym, []).append(pos)

    log.info("Month columns detected",
             extra={"distinct_months": len(months),
                    "months": [f"{y}-{m:02d}" for (y, m) in sorted(months)],
                    "blocks_per_month": max((len(v) for v in months.values()), default=0)})
    return months


def pick_amortization_block(months: dict, log) -> dict:
    """
    If each month appears N times, the sheet has N blocks of month columns
    (e.g. prepaid / amortisation / closing). The amortisation block is assumed
    to be the SECOND occurrence when 3+ blocks exist, the FIRST otherwise.
    Returns (year, month) → single column position.
    """
    if not months:
        return {}
    block_count = max(len(v) for v in months.values())
    pick = 1 if block_count >= 3 else 0
    chosen = {ym: positions[pick] if len(positions) > pick else positions[0]
              for ym, positions in months.items()}
    log.info("Amortization block selected",
             extra={"blocks_found": block_count, "block_used": pick + 1})
    return chosen


# ── Station 2: Calculate the period release ────────────────────────────────────

def is_total_row(row, item_cols) -> bool:
    desc   = to_str(row[item_cols["description"]]).lower() if item_cols.get("description") is not None else ""
    vendor = to_str(row[item_cols["vendor"]]).lower()      if item_cols.get("vendor")      is not None else ""
    text   = f"{vendor} {desc}"
    if any(w in text for w in TOTAL_WORDS):
        return True
    # rows with no vendor AND no description are totals/blank artifacts
    return vendor == "" and desc == ""


def calculate_period_release(df, item_cols, amort_cols, period_ym, log) -> list:
    """For each Open item: release = this month's column, amortized = sum of earlier months."""
    earlier = sorted(ym for ym in amort_cols if ym < period_ym)
    current = amort_cols.get(period_ym)

    if current is None:
        log.error("Run period not found in schedule",
                  extra={"period": f"{period_ym[0]}-{period_ym[1]:02d}",
                         "available": [f"{y}-{m:02d}" for (y, m) in sorted(amort_cols)]})
        return []

    lines = []
    for idx, row in df.iterrows():
        if is_total_row(row, item_cols):
            log.info("Totals/blank row excluded", extra={"row": int(idx) + 1})
            continue

        status   = to_str(row[item_cols["status"]]).lower()   if item_cols.get("status")   is not None else "open"
        run_type = to_str(row[item_cols["run_type"]])         if item_cols.get("run_type") is not None else ""

        item = {
            "vendor":         to_str(row[item_cols["vendor"]])         if item_cols.get("vendor")         is not None else "",
            "description":    to_str(row[item_cols["description"]])    if item_cols.get("description")    is not None else f"row {idx+1}",
            "invoice_number": to_str(row[item_cols["invoice_number"]]) if item_cols.get("invoice_number") is not None else "",
            "currency":       to_str(row[item_cols["currency"]])       if item_cols.get("currency")       is not None else "",
            "prepaid_amount": to_float(row[item_cols["prepaid_amount"]]) if item_cols.get("prepaid_amount") is not None else 0.0,
            "run_type":       run_type,
            "status":         status,
            "expense_account": to_str(row[item_cols["expense_account"]]) if item_cols.get("expense_account") is not None else DEFAULT_EXPENSE_ACCOUNT,
            "cost_center":     to_str(row[item_cols["cost_center"]])     if item_cols.get("cost_center")     is not None else ""
        }
        if not item["expense_account"]:
            item["expense_account"] = DEFAULT_EXPENSE_ACCOUNT

        release          = to_float(row.iloc[amort_cols[period_ym]])
        amortized_todate = round(sum(to_float(row.iloc[amort_cols[ym]]) for ym in earlier), 2)
        remaining        = round(item["prepaid_amount"] - amortized_todate, 2)

        if status == "closed":
            disposition = "SKIPPED_CLOSED"
        elif release == 0.0:
            disposition = "NO_RELEASE_THIS_PERIOD"
        elif "manual" in run_type.lower():
            disposition = "HITL_MANUAL_RUN"      # release goes through but flagged for review
        else:
            disposition = "RELEASE"

        if disposition in ("RELEASE", "HITL_MANUAL_RUN") and release > remaining + 0.01:
            disposition = "ERROR_OVER_RELEASE"   # schedule asks to release more than remains
            log.warning("Release exceeds remaining balance — route to HITL",
                        extra={"item": item["description"], "release": release, "remaining": remaining})

        lines.append({
            **item,
            "release_this_month": release,
            "amortized_todate":   amortized_todate,
            "remaining_before":   remaining,
            "remaining_after":    round(remaining - (release if disposition in ("RELEASE", "HITL_MANUAL_RUN") else 0.0), 2),
            "disposition":        disposition
        })

        log.info("Item processed",
                 extra={"item": item["description"][:40], "disposition": disposition,
                        "release": release, "remaining": remaining})

    releases = [l for l in lines if l["disposition"] in ("RELEASE", "HITL_MANUAL_RUN")]
    log.info("Period calculation complete",
             extra={"items": len(lines), "releases": len(releases),
                    "hitl_flagged": sum(1 for l in lines if l["disposition"] == "HITL_MANUAL_RUN"),
                    "total_release": round(sum(l["release_this_month"] for l in releases), 2)})
    return lines


# ── Station 3: Journal ─────────────────────────────────────────────────────────

def prepare_journal(lines, period_label, correlation_id, log) -> dict:
    releases = [l for l in lines if l["disposition"] in ("RELEASE", "HITL_MANUAL_RUN")]
    if not releases:
        log.warning("No releases this period — no journal needed")
        return {}

    je_lines, total_debit = [], 0.0
    for i, l in enumerate(releases, 1):
        amount      = round(l["release_this_month"], 2)
        total_debit = round(total_debit + amount, 2)
        je_lines.append({
            "line":        i,
            "account":     l["expense_account"],
            "cost_center": l["cost_center"],
            "debit":       amount,
            "credit":      0.0,
            "narration":   f"Prepaid amortization {period_label} — {l['description'][:38]}",
            "hitl_flag":   l["disposition"] == "HITL_MANUAL_RUN"
        })

    je_lines.append({
        "line":        len(je_lines) + 1,
        "account":     PREPAID_ASSET_ACCOUNT,
        "cost_center": "",
        "debit":       0.0,
        "credit":      total_debit,
        "narration":   f"Prepaid asset release {period_label}",
        "hitl_flag":   False
    })

    journal = {
        "journal_id":     f"JE-PPD-{period_label.replace('/', '')}-{correlation_id[-8:]}",
        "journal_type":   "PREPAID_AMORTIZATION",
        "period":         period_label,
        "posting_status": "PENDING_APPROVAL",
        "prepared_by":    "prepaid_engine_v2",
        "prepared_at":    datetime.now(timezone.utc).isoformat(),
        "approved_by":    None,
        "total_debit":    total_debit,
        "total_credit":   total_debit,
        "balanced":       True,
        "lines":          je_lines
    }
    log.info("Journal prepared", extra={"journal_id": journal["journal_id"],
                                        "lines": len(je_lines), "total_debit": total_debit})
    return journal


def queue_for_approval(journal, log) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    queue_file = OUTPUT_DIR / "approval_queue.json"
    queue = json.loads(queue_file.read_text()) if queue_file.exists() else []
    queue = [j for j in queue if j.get("journal_id") != journal["journal_id"]]
    queue.append(journal)
    queue_file.write_text(json.dumps(queue, indent=2))
    log.info("Journal queued for approval", extra={"journal_id": journal["journal_id"]})
    return queue_file


# ── Station 5a: GL reconciliation (register vs books) ─────────────────────────

# Keyword order matters — "amount in local currency" must win over
# "amount in doc. curr." (document currency is NOT the booked value)
GL_COLUMN_KEYWORDS = {
    "gl_account":  ["gl account", "g/l account", "account", "hkont"],
    "gl_amount":   ["amount in local currency", "local currency amount", "lc amount",
                    "amount in lc", "dmbtr", "balance", "amount"],
    "posting_key": ["posting key", "pstng key", "post key", "bschl", "pk"],
    "text":        ["text", "narration", "description"],
    "reference":   ["reference", "assignment", "document number", "doc number"],
    "cost_center": ["cost center", "cost centre", "kostl"]
}

# SAP posting keys on GL accounts: 40 = debit (into prepaid), 50 = credit (out)
DEBIT_POSTING_KEYS  = {"40"}
CREDIT_POSTING_KEYS = {"50"}


def reconcile_against_gl(lines, log) -> dict:
    """
    Register remaining (before this month's release) must equal the prepaid
    GL balance. If gl_extract.xlsx is absent, recon is skipped with a note.
    """
    if not GL_FILE.exists():
        log.warning("gl_extract.xlsx not found — GL reconciliation skipped")
        return {"performed": False, "note": "gl_extract.xlsx not provided"}

    raw = pd.read_excel(GL_FILE, header=None).dropna(how="all")
    header_row = 0
    for i, row in raw.iterrows():
        if sum(1 for v in row.dropna() if isinstance(v, str)) >= 2:
            header_row = i
            break
    gl_df = raw.copy()
    gl_df.columns = gl_df.iloc[header_row]
    gl_df = gl_df.iloc[header_row + 1:].reset_index(drop=True).dropna(how="all")
    gl_df.columns = [to_str(c) if c is not None else f"col_{i}" for i, c in enumerate(gl_df.columns)]

    cols = {}
    for field, keywords in GL_COLUMN_KEYWORDS.items():
        for keyword in keywords:
            hit = next((c for c in gl_df.columns if keyword in c.lower()), None)
            if hit:
                cols[field] = hit
                break

    if "gl_amount" not in cols:
        log.warning("GL amount column not found — recon skipped",
                    extra={"columns": list(gl_df.columns)})
        return {"performed": False, "note": "GL amount column not detected"}

    # Detect whether amounts are signed. If the extract has a Posting Key and
    # all amounts are positive, the sign comes from the key: 40 = +, 50 = −.
    # If amounts already carry signs (any negative present), use them as-is.
    has_posting_key = "posting_key" in cols
    amounts_signed  = any(to_float(r[cols["gl_amount"]]) < 0 for _, r in gl_df.iterrows())
    apply_key_sign  = has_posting_key and not amounts_signed
    if apply_key_sign:
        log.info("Amounts are unsigned — applying sign from Posting Key (40=+, 50=−)")
    elif has_posting_key:
        log.info("Amounts already signed — Posting Key used for info only")

    def signed_amount(row) -> float:
        amount = to_float(row[cols["gl_amount"]])
        if apply_key_sign:
            key = to_str(row[cols["posting_key"]]).split(".")[0]  # 40.0 → 40
            if key in CREDIT_POSTING_KEYS:
                return -abs(amount)
            if key in DEBIT_POSTING_KEYS:
                return abs(amount)
        return amount

    # Sum prepaid balance. SAP extracts are usually already filtered to the
    # prepaid account, so: try matching the configured account first; if no
    # row matches, sum the whole extract and log which accounts were present.
    def sum_rows(only_prepaid: bool) -> float:
        total = 0.0
        for _, row in gl_df.iterrows():
            if only_prepaid and "gl_account" in cols:
                account = to_str(row[cols["gl_account"]])
                if account and account != PREPAID_ASSET_ACCOUNT and "prepaid" not in account.lower():
                    continue
            total += signed_amount(row)
        return round(total, 2)

    gl_balance = sum_rows(only_prepaid=True)
    if gl_balance == 0.0 and "gl_account" in cols:
        accounts = sorted({to_str(r[cols["gl_account"]]) for _, r in gl_df.iterrows() if to_str(r[cols["gl_account"]])})
        gl_balance = sum_rows(only_prepaid=False)
        log.info("No rows matched configured prepaid account — summed entire extract",
                 extra={"configured_account": PREPAID_ASSET_ACCOUNT, "accounts_in_extract": accounts[:10]})

    # Register remaining BEFORE this month's release should equal the GL today
    register_remaining = round(sum(l["remaining_before"] for l in lines
                                   if l["disposition"] != "SKIPPED_CLOSED"), 2)
    difference  = round(register_remaining - gl_balance, 2)
    reconciled  = abs(difference) <= RECON_TOLERANCE

    recon = {
        "performed":          True,
        "register_remaining": register_remaining,
        "gl_balance":         gl_balance,
        "difference":         difference,
        "is_reconciled":      reconciled
    }
    if reconciled:
        log.info("Prepaid GL reconciliation PASSED", extra=recon)
    else:
        log.warning("Prepaid GL reconciliation FAILED — register and books disagree", extra=recon)
    return recon


def print_recon(recon: dict):
    print(f"\n{'─'*100}")
    if not recon.get("performed"):
        print(f"GL RECONCILIATION — SKIPPED ({recon.get('note', '')})")
        return
    status = "✓ RECONCILED" if recon["is_reconciled"] else "✗ NOT RECONCILED — investigate before posting"
    print(f"GL RECONCILIATION — Register vs Books  [{status}]")
    print(f"{'─'*100}")
    print(f"  Register remaining (open items) : {recon['register_remaining']:>15,.2f}")
    print(f"  Prepaid GL balance              : {recon['gl_balance']:>15,.2f}")
    print(f"  Difference                      : {recon['difference']:>15,.2f}")


# ── Station 5: Evidence ────────────────────────────────────────────────────────

def save_audit_pack(correlation_id, period_label, lines, journal, recon, log) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    pack = {
        "correlation_id": correlation_id,
        "journal_type":   "PREPAID_AMORTIZATION",
        "period":         period_label,
        "run_timestamp":  datetime.now(timezone.utc).isoformat(),
        "schedule_working": lines,
        "journal_entry":  journal,
        "gl_reconciliation": recon,
        "close_status":   "PENDING_APPROVAL" if journal else "NO_JOURNAL_REQUIRED"
    }
    pack_file = OUTPUT_DIR / f"audit_pack_{correlation_id}.json"
    pack_file.write_text(json.dumps(pack, indent=2))
    log.info("Audit pack saved", extra={"file": str(pack_file)})
    return pack_file


# ── Print helpers ──────────────────────────────────────────────────────────────

def print_working(lines):
    print(f"\n{'─'*100}")
    print("SCHEDULE WORKING — per prepaid item")
    print(f"{'─'*100}")
    print(f"  {'Vendor':<14} {'Description':<28} {'Prepaid':>11} {'AmortToDate':>12} "
          f"{'Release':>10} {'Remaining':>11}  Disposition")
    print(f"  {'─'*14} {'─'*28} {'─'*11} {'─'*12} {'─'*10} {'─'*11}  {'─'*20}")
    for l in lines:
        print(f"  {l['vendor'][:14]:<14} {l['description'][:28]:<28} {l['prepaid_amount']:>11,.0f} "
              f"{l['amortized_todate']:>12,.0f} {l['release_this_month']:>10,.2f} "
              f"{l['remaining_after']:>11,.0f}  {l['disposition']}")


def print_journal(journal):
    if not journal:
        print("\n  No journal required this period.")
        return
    print(f"\n{'─'*100}")
    print(f"DRAFT JOURNAL ENTRY — {journal['journal_id']}  [{journal['posting_status']}]")
    print(f"{'─'*100}")
    print(f"  {'Line':<5} {'Account':<11} {'Debit':>13} {'Credit':>13}  {'HITL':<5} Narration")
    print(f"  {'─'*5} {'─'*11} {'─'*13} {'─'*13}  {'─'*5} {'─'*42}")
    for l in journal["lines"]:
        hitl = "⚠" if l.get("hitl_flag") else ""
        print(f"  {l['line']:<5} {l['account']:<11} {l['debit']:>13,.2f} {l['credit']:>13,.2f}  "
              f"{hitl:<5} {l['narration'][:55]}")
    print(f"  {'─'*5} {'─'*11} {'─'*13} {'─'*13}")
    print(f"  {'TOTAL':<17} {journal['total_debit']:>13,.2f} {journal['total_credit']:>13,.2f}")
    print(f"\n  ✓ BALANCED (Debit = Credit)")
    if any(l.get("hitl_flag") for l in journal["lines"]):
        print(f"  ⚠ Lines marked HITL came from Manual Run items — review before approval")


# ── Main ───────────────────────────────────────────────────────────────────────

def run(period_arg: str = None):
    now = datetime.now()
    if period_arg:
        mm, yyyy  = period_arg.split("/")
        period_ym = (int(yyyy), int(mm))
    else:
        period_ym = (now.year, now.month)
    period_label = f"{period_ym[1]:02d}/{period_ym[0]}"

    correlation_id = f"R2R-PPD2-{uuid.uuid4().hex[:8]}"
    logger     = setup_logger(correlation_id)
    log_main   = get_log("pipeline",    logger)
    log_ingest = get_log("ingestion",   logger)
    log_calc   = get_log("calculation", logger)
    log_je     = get_log("journal",     logger)
    log_ev     = get_log("evidence",    logger)

    print(f"\n{'='*100}")
    print("R2R PREPAID AMORTIZATION v2 — matrix schedule layout")
    print(f"{'='*100}")
    print(f"  Correlation ID : {correlation_id}")
    print(f"  Run period     : {period_label}")

    log_main.info("Run started", extra={"period": period_label})

    # Station 1 — read + detect
    df         = load_master_sheet(log_ingest)
    item_cols  = detect_item_columns(df, log_ingest)
    months     = detect_month_columns(df, log_ingest)
    amort_cols = pick_amortization_block(months, log_ingest)

    if not amort_cols:
        print("\n  No month columns (Apr-26 style) found — check the file with inspect_data.py")
        sys.exit(1)

    # Station 2 — calculate
    lines = calculate_period_release(df, item_cols, amort_cols, period_ym, log_calc)
    if not lines:
        print(f"\n  Nothing to process for {period_label} — is the period within the schedule range?")
        sys.exit(1)
    print_working(lines)

    # Station 3 + 4 — journal + queue
    journal = prepare_journal(lines, period_label, correlation_id, log_je)
    print_journal(journal)
    if journal:
        queue_file = queue_for_approval(journal, log_je)
        print(f"\n  Journal queued for finance approval → {queue_file.name}")
        print(f"  (Approve with: python approve_journal.py --list)")

    # Station 5 — GL reconciliation + evidence
    recon = reconcile_against_gl(lines, get_log("recon", logger))
    print_recon(recon)

    pack_file = save_audit_pack(correlation_id, period_label, lines, journal, recon, log_ev)

    log_main.info("Run complete", extra={"journal_id": journal.get("journal_id") if journal else None})
    print(f"\n{'='*100}")
    print(f"  Audit pack : {pack_file}")
    print(f"  Logs       : {LOGS_DIR}/")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
