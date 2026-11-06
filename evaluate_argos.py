"""
Argos-Action Evaluation & Metrics Verification Harness
Author: Pedro Lucas Reis (@pedroreoli)
Repository: PedroReoli/argos-action
License: Apache-2.0

Description:
Audits, measures, and benchmarks the Argos-Action dataset and model outputs:
1. Syntactic Schema Conformance (Strict JSON with thought, tool_call, verification_predicate)
2. Tool Name & Argument Completeness
3. Dataset Token Length Census (Input, AXTree, ChatML)
4. Clearly differentiates Training Loss vs Action Accuracy vs Task Success
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List

REQUIRED_SCHEMA_KEYS = {"thought", "tool_call", "verification_predicate"}
REQUIRED_TOOL_CALL_KEYS = {"tool", "arguments"}

def audit_split(file_path: Path) -> Dict[str, Any]:
    print(f"\n--- Auditing Split: {file_path.name} ---")
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    total_rows = 0
    valid_json_rows = 0
    schema_conformant_rows = 0
    tool_counter = Counter()
    domain_counter = Counter()
    input_char_lengths = []
    output_char_lengths = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total_rows += 1

            try:
                record = json.loads(line)
                domain_counter[record.get("domain", "unknown")] += 1
                input_char_lengths.append(len(record.get("input", "")))

                # Parse output string
                raw_out = record.get("output", "")
                output_char_lengths.append(len(raw_out))
                parsed_out = json.loads(raw_out)
                valid_json_rows += 1

                # Check schema
                if REQUIRED_SCHEMA_KEYS.issubset(parsed_out.keys()):
                    tc = parsed_out.get("tool_call", {})
                    if REQUIRED_TOOL_CALL_KEYS.issubset(tc.keys()):
                        tool_counter[tc.get("tool", "unknown")] += 1
                        schema_conformant_rows += 1
            except Exception as e:
                print(f"[ERROR] Line {line_num} failed validation: {e}")

    avg_input_chars = sum(input_char_lengths) / len(input_char_lengths) if input_char_lengths else 0
    avg_output_chars = sum(output_char_lengths) / len(output_char_lengths) if output_char_lengths else 0

    metrics = {
        "file": file_path.name,
        "total_samples": total_rows,
        "valid_json_rate": (valid_json_rows / total_rows) * 100 if total_rows else 0.0,
        "schema_conformance_rate": (schema_conformant_rows / total_rows) * 100 if total_rows else 0.0,
        "tool_distribution": dict(tool_counter),
        "domain_distribution": dict(domain_counter),
        "avg_input_characters": round(avg_input_chars, 1),
        "avg_output_characters": round(avg_output_chars, 1),
    }

    print(f"Total Rows: {total_rows}")
    print(f"JSON Valid: {valid_json_rows}/{total_rows} ({metrics['valid_json_rate']}%)")
    print(f"Schema Conformance: {schema_conformant_rows}/{total_rows} ({metrics['schema_conformance_rate']}%)")
    print(f"Tools Detected: {dict(tool_counter)}")
    return metrics

def run_evaluation():
    base_dir = Path(__file__).resolve().parent / "data"
    train_path = base_dir / "train.jsonl"
    test_path = base_dir / "test.jsonl"

    print("==================================================")
    print("       ARGOS-ACTION METRICS & AUDIT SUITE         ")
    print("==================================================")

    train_metrics = audit_split(train_path)
    test_metrics = audit_split(test_path)

    summary = {
        "evaluation_protocol": "Argos-Action-Offline-Audit",
        "train": train_metrics,
        "test_heldout": test_metrics,
        "metric_taxonomy": {
            "training_loss": "Sequence SFT cross-entropy loss over ChatML tokens",
            "schema_conformance": "Strict output syntax matching {thought, tool_call: {tool, arguments}, verification_predicate}",
            "task_success": "End-to-end goal achievement in runtime browser/OS (requires live CDP/UIA harness)"
        }
    }

    out_file = base_dir / "evaluation_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Evaluation summary exported to: {out_file}")

if __name__ == "__main__":
    run_evaluation()
