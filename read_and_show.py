"""
R2R Read and Show — Extract + Match

1. Reads bank_statement.xlsx and bank_gl_entries.xlsx from data/
2. Extracts structured JSON from each using GPT-4o
3. Matches both lists — finds matched, mismatches, timing differences, missing entries
4. Logs every step with correlation ID

Usage:
  python read_and_show.py
"""
import json
import os
import sys
import time
import uuid
from pathlib import Path
from openai import AzureOpenAI
from dotenv import load_dotenv
import openpyxl
import logging
from datetime import datetime, timezone

load_dotenv()
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01"
)
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

PROMPTS_DIR = Path(__file__).parent / "prompts"
DATA_DIR    = Path(__file__).parent / "data"


# ── Logger setup ───────────────────────────────────────────────────────────────

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


LOGS_DIR = Path(__file__).parent / "logs"


def setup_logger(correlation_id: str) -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    log_file = LOGS_DIR / f"{correlation_id}.log"

    logger = logging.getLogger("r2r")
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter(correlation_id))
    logger.addHandler(console_handler)

    # File handler — every run gets its own log file named by correlation ID
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(JSONFormatter(correlation_id))
    logger.addHandler(file_handler)

    logger.setLevel(logging.DEBUG)
    logger.info(f"Log file: {log_file}")
    return logger


def get_log(station: str, logger: logging.Logger) -> logging.LoggerAdapter:
    return logging.LoggerAdapter(logger, extra={"station": station})


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_prompt(file_type: str) -> str:
    prompt_file = PROMPTS_DIR / f"{file_type}.txt"
    if not prompt_file.exists():
        print(f"Prompt file not found: {prompt_file}")
        sys.exit(1)
    return prompt_file.read_text().strip()


CHUNK_SIZE = 300  # rows per GPT call — stays well within token limit


def read_excel_rows(file_path: str, log) -> tuple:
    """
    Read Excel and return:
      - header_text : first 20 non-empty rows as plain text (captures company, period, balances)
      - all_rows    : list of all non-empty row strings
    """
    log.info("Reading Excel file", extra={"file": Path(file_path).name})
    wb = openpyxl.load_workbook(file_path, data_only=True)
    all_rows = []

    for sheet in wb.sheetnames:
        ws = wb[sheet]
        all_rows.append(f"=== Sheet: {sheet} ===")
        for row in ws.iter_rows(values_only=True):
            if any(c is not None for c in row):
                all_rows.append(" | ".join(str(c) if c is not None else "" for c in row))

    header_text = "\n".join(all_rows[:20])
    log.info(
        "Excel read complete",
        extra={"file": Path(file_path).name, "total_rows": len(all_rows), "chunk_size": CHUNK_SIZE}
    )
    return header_text, all_rows


def extract_chunk(chunk_text: str, file_type: str, chunk_no: int, total_chunks: int, log) -> dict:
    """Send one chunk of rows to GPT and return extracted JSON."""
    log.info(
        "Extracting chunk",
        extra={"file_type": file_type, "chunk": f"{chunk_no}/{total_chunks}"}
    )
    start = time.time()

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": load_prompt(file_type)},
            {"role": "user",   "content": f"Raw Excel data (chunk {chunk_no} of {total_chunks}):\n\n{chunk_text}\n\nReturn valid JSON only."}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    duration_ms = round((time.time() - start) * 1000)
    usage = response.usage
    log.info(
        "Chunk extraction complete",
        extra={
            "chunk":         f"{chunk_no}/{total_chunks}",
            "input_tokens":  usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "duration_ms":   duration_ms
        }
    )
    return json.loads(response.choices[0].message.content)


def merge_results(results: list, file_type: str) -> dict:
    """
    Merge multiple chunk results into one final JSON.
    Header fields come from the first chunk.
    Transaction/entry lists are concatenated from all chunks.
    """
    if not results:
        return {}

    # Start with first chunk as base (has header fields)
    merged = results[0].copy()

    # Transaction key differs by file type
    tx_key = {
        "bank_statement": "transactions",
        "bank_gl":        "entries"
    }.get(file_type, "transactions")

    # Collect all transactions from all chunks
    all_tx = []
    for result in results:
        all_tx.extend(result.get(tx_key, []))

    merged[tx_key] = all_tx
    return merged


def extract(file_path: str, file_type: str, log) -> dict:
    """
    Full extraction with chunking.
    Reads the Excel, splits into chunks of CHUNK_SIZE rows,
    extracts each chunk, merges results into one final JSON.
    """
    header_text, all_rows = read_excel_rows(file_path, log)

    # If small enough to fit in one call — send directly
    if len(all_rows) <= CHUNK_SIZE:
        log.info("File fits in single call — no chunking needed", extra={"rows": len(all_rows)})
        return extract_chunk("\n".join(all_rows), file_type, 1, 1, log)

    # Split into chunks — always include header rows in every chunk for context
    chunks = []
    data_rows = all_rows[20:]  # rows after header
    for i in range(0, len(data_rows), CHUNK_SIZE):
        chunk = header_text + "\n" + "\n".join(data_rows[i:i + CHUNK_SIZE])
        chunks.append(chunk)

    total_chunks = len(chunks)
    log.info(
        "Chunking file for extraction",
        extra={"total_rows": len(all_rows), "chunks": total_chunks, "chunk_size": CHUNK_SIZE}
    )

    results = []
    total_input_tokens = 0
    total_output_tokens = 0

    for i, chunk in enumerate(chunks, 1):
        result = extract_chunk(chunk, file_type, i, total_chunks, log)
        results.append(result)

    merged = merge_results(results, file_type)

    tx_key = "transactions" if file_type == "bank_statement" else "entries"
    log.info(
        "Extraction complete — all chunks merged",
        extra={
            "file_type":          file_type,
            "chunks_processed":   total_chunks,
            "total_transactions": len(merged.get(tx_key, []))
        }
    )
    return merged


# ── Matching ───────────────────────────────────────────────────────────────────

def match(bank_data: dict, gl_data: dict, log) -> dict:
    log.info("Starting matching — bank statement vs GL entries")
    start = time.time()

    prompt = f"""You are a bank reconciliation expert.

Compare these two extracted datasets:
1. Bank Statement — transactions recorded by the bank
2. GL Entries — transactions recorded in our accounting system

Match each transaction by reference number, amount and narration.
Use fuzzy matching for narrations — the same transaction will look different in each system.
For example: "NEFT CR INFOSYS LIMITED" matches "Vendor payment — Infosys Ltd"

For every transaction classify it as one of:
- MATCHED          : same transaction, same amount on both sides
- MISMATCH         : same transaction but different amount — note the difference
- TIMING_DIFFERENCE: exists on one side only but expected (cheque not cleared, payment in transit)
- MISSING          : exists on one side only with no clear explanation — needs investigation

Return JSON only:
{{
  "matched": [
    {{
      "ref": "transaction reference",
      "description": "what this transaction is",
      "gl_amount": 0.00,
      "bank_amount": 0.00
    }}
  ],
  "mismatches": [
    {{
      "ref": "...",
      "description": "...",
      "gl_amount": 0.00,
      "bank_amount": 0.00,
      "difference": 0.00,
      "likely_reason": "why this difference might exist"
    }}
  ],
  "timing_differences": [
    {{
      "ref": "...",
      "description": "...",
      "amount": 0.00,
      "present_in": "GL_ONLY or BANK_ONLY",
      "reason": "why this is likely a timing difference"
    }}
  ],
  "missing": [
    {{
      "ref": "...",
      "description": "...",
      "amount": 0.00,
      "present_in": "GL_ONLY or BANK_ONLY",
      "action": "what finance team should do"
    }}
  ],
  "summary": {{
    "total_matched": 0,
    "total_mismatches": 0,
    "total_timing_differences": 0,
    "total_missing": 0,
    "is_reconciled": true,
    "commentary": "one line plain English summary of the reconciliation result"
  }}
}}

Bank Statement:
{json.dumps(bank_data, indent=2)}

GL Entries:
{json.dumps(gl_data, indent=2)}"""

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a bank reconciliation expert. Return valid JSON only."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    duration_ms = round((time.time() - start) * 1000)
    usage = response.usage
    result = json.loads(response.choices[0].message.content)
    summary = result.get("summary", {})

    log.info(
        "Matching complete",
        extra={
            "matched":             summary.get("total_matched", 0),
            "mismatches":          summary.get("total_mismatches", 0),
            "timing_differences":  summary.get("total_timing_differences", 0),
            "missing":             summary.get("total_missing", 0),
            "is_reconciled":       summary.get("is_reconciled"),
            "input_tokens":        usage.prompt_tokens,
            "output_tokens":       usage.completion_tokens,
            "duration_ms":         duration_ms
        }
    )

    if summary.get("total_mismatches", 0) > 0:
        for m in result.get("mismatches", []):
            log.warning(
                "Mismatch found",
                extra={"ref": m.get("ref"), "difference": m.get("difference"), "reason": m.get("likely_reason")}
            )

    if summary.get("total_missing", 0) > 0:
        for m in result.get("missing", []):
            log.warning(
                "Missing entry",
                extra={"ref": m.get("ref"), "present_in": m.get("present_in"), "action": m.get("action")}
            )

    for t in result.get("timing_differences", []):
        log.info(
            "Timing difference noted",
            extra={"ref": t.get("ref"), "present_in": t.get("present_in"), "reason": t.get("reason")}
        )

    return result


# ── Print helpers ──────────────────────────────────────────────────────────────

def print_extraction(label: str, data: dict):
    print(f"\n{'─'*60}")
    print(f"EXTRACTED — {label}")
    print(f"{'─'*60}")
    print(json.dumps(data, indent=2))


def print_match_result(result: dict):
    summary = result.get("summary", {})
    print(f"\n{'─'*60}")
    print("MATCHING RESULT")
    print(f"{'─'*60}")
    print(f"  Matched             : {summary.get('total_matched', 0)}")
    print(f"  Mismatches          : {summary.get('total_mismatches', 0)}")
    print(f"  Timing differences  : {summary.get('total_timing_differences', 0)}")
    print(f"  Missing entries     : {summary.get('total_missing', 0)}")
    print(f"  Reconciled          : {summary.get('is_reconciled')}")
    print(f"  Commentary          : {summary.get('commentary')}")

    if result.get("matched"):
        print(f"\n  MATCHED ({len(result['matched'])}):")
        for m in result["matched"]:
            print(f"    ✓  {m.get('ref',''):<12} {m.get('description','')[:45]:<45}  {m.get('gl_amount', 0):>12,.0f}")

    if result.get("mismatches"):
        print(f"\n  MISMATCHES ({len(result['mismatches'])}):")
        for m in result["mismatches"]:
            print(f"    ✗  {m.get('ref',''):<12} {m.get('description','')[:35]:<35}  GL: {m.get('gl_amount',0):>10,.0f}  Bank: {m.get('bank_amount',0):>10,.0f}  Diff: {m.get('difference',0):>8,.0f}")
            print(f"       Reason: {m.get('likely_reason','')}")

    if result.get("timing_differences"):
        print(f"\n  TIMING DIFFERENCES ({len(result['timing_differences'])}):")
        for t in result["timing_differences"]:
            print(f"    ⏳  {t.get('ref',''):<12} {t.get('description','')[:40]:<40}  {t.get('amount',0):>12,.0f}  ({t.get('present_in','')})")
            print(f"        Reason: {t.get('reason','')}")

    if result.get("missing"):
        print(f"\n  MISSING ENTRIES ({len(result['missing'])}):")
        for m in result["missing"]:
            print(f"    ?  {m.get('ref',''):<12} {m.get('description','')[:40]:<40}  {m.get('amount',0):>12,.0f}  ({m.get('present_in','')})")
            print(f"       Action: {m.get('action','')}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    correlation_id = f"R2R-DEMO-{uuid.uuid4().hex[:8]}"
    logger = setup_logger(correlation_id)

    log_main    = get_log("pipeline",    logger)
    log_extract = get_log("extraction",  logger)
    log_match   = get_log("matching",    logger)

    print(f"\n{'='*60}")
    print("R2R — Read, Extract and Match")
    print(f"{'='*60}")
    print(f"  Correlation ID : {correlation_id}")

    log_main.info("Pipeline started", extra={"correlation_id": correlation_id})

    start = time.time()

    # ── Step 1: Extract bank statement ────────────────────────────────────────
    bank_file = DATA_DIR / "bank_statement.xlsx"
    if not bank_file.exists():
        log_main.error("bank_statement.xlsx not found", extra={"path": str(bank_file)})
        print(f"\n  Place your bank statement Excel at: {bank_file}")
        sys.exit(1)

    bank_data = extract(str(bank_file), "bank_statement", log_extract)
    print_extraction("BANK STATEMENT", bank_data)

    # ── Step 2: Extract GL entries ────────────────────────────────────────────
    gl_file = DATA_DIR / "bank_gl_entries.xlsx"
    if not gl_file.exists():
        log_main.error("bank_gl_entries.xlsx not found", extra={"path": str(gl_file)})
        print(f"\n  Place your GL entries Excel at: {gl_file}")
        sys.exit(1)

    gl_data = extract(str(gl_file), "bank_gl", log_extract)
    print_extraction("GL ENTRIES", gl_data)

    # ── Step 3: Match ─────────────────────────────────────────────────────────
    match_result = match(bank_data, gl_data, log_match)
    print_match_result(match_result)

    # ── Step 4: Save output ───────────────────────────────────────────────────
    duration_ms = round((time.time() - start) * 1000)

    output = {
        "correlation_id":  correlation_id,
        "bank_statement":  bank_data,
        "gl_entries":      gl_data,
        "match_result":    match_result
    }

    out_path = DATA_DIR / "output.json"
    out_path.write_text(json.dumps(output, indent=2))

    log_main.info(
        "Pipeline complete",
        extra={"duration_ms": duration_ms, "output": str(out_path)}
    )

    print(f"\n{'='*60}")
    print(f"  Duration : {duration_ms}ms")
    print(f"  Output   : {out_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run()
