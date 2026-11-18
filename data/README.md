---
language:
- en
license: apache-2.0
task_categories:
- text-generation
- question-answering
tags:
- agents
- computer-use
- model-context-protocol
- mcp
- accessibility-tree
- axtree
- uia
- tool-calling
- browser-automation
- desktop-automation
- small-language-models
- slm
- safety-guardrails
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: "train.jsonl"
  - split: test
    path: "test.jsonl"
dataset_info:
  features:
  - name: id
    dtype: string
  - name: domain
    dtype: string
  - name: instruction
    dtype: string
  - name: input
    dtype: string
  - name: output
    dtype: string
  - name: messages
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  splits:
  - name: train
    num_bytes: 6569128
    num_examples: 1700
  - name: test
    num_bytes: 1139642
    num_examples: 300
---

# 👁️ Argos v2.0: Computer Use & Semantic Tool Calling (MCP) Benchmark Dataset

**Argos-Action v2.0** (`pedroreoli/argos-computer-use-mcp`) is a deterministic, 100%-unique synthetic single-turn decision dataset designed for **Supervised Fine-Tuning (SFT)** and **Safety Guardrail Evaluation** of Small Language Models (SLMs, such as `Qwen/Qwen2.5-1.5B-Instruct`) acting as **Edge Policy Models for Autonomous Computer Use**.

---

## 🚀 What Changed in v2.0 (Addressing the v1 Post-Mortem)

Following our public technical post-mortem ([*"Meu script disse 100%. Ele só tinha validado o gerador"*](https://medium.com/@pedrosousa2160/meu-script-disse-100-ele-s%C3%B3-tinha-validado-o-gerador-d7433b381927)), Dataset v2.0 resolves every structural limitation identified in v1:

| Dimension | v1.0 / v1.1 Limitation | v2.0 Resolution (Verified by `evaluate_argos.py`) |
| :--- | :--- | :--- |
| **Row Uniqueness (`train.jsonl`)** | 554 unique inputs out of 1,700 rows | **1,700 / 1,700 (100% unique inputs, goals & AXTrees)** |
| **Row Uniqueness (`test.jsonl`)** | 5 templates repeated 60× (300 rows) | **300 / 300 (100% unique inputs, goals & AXTrees)** |
| **Train / Test Overlap** | Disjoint strings, narrow test coverage | **0 / 300 overlap** in inputs, goals, AXTrees, and `element_id`s |
| **`scroll` & `select_option` in Train** | 0 occurrences in `train.jsonl` | **240 `scroll` + 260 `select_option`** in `train.jsonl` (48 each in `test.jsonl`) |
| **`press_hotkey` Diversity** | Only `Ctrl+S` in training | **12 distinct OS/browser shortcuts** across 240 train & 48 test samples |
| **`escalate_human` Safety Taxonomy** | Only 2 narrow templates (2FA & Turnstile) | **6 adversarial safety categories** (340 train / 60 test), including deceptive `Type DELETE to confirm` UIs |
| **Argument Schema Consistency** | Schema drift (`delivery_mode` vs `button`) | **100% unified per-tool argument schemas** across all 2,000 rows |

---

## 🏛️ Dataset Taxonomy, Domain Symmetry & Tool Distribution

The dataset contains **2,000 single-turn decision samples** partitioned into **1,700 training samples** (`340` per domain) and **300 strictly disjoint held-out test samples** (`60` per domain):

### 1. Tool Distribution Across Splits

| Tool (`tool_call.tool`) | MCP Wire Target | Train (`n = 1,700`) | Held-Out Test (`n = 300`) | Canonical Argument Contract |
| :--- | :--- | :---: | :---: | :--- |
| **`click`** | `browser_click` | **340** | **48** | `{"element_id": str, "button": "left"\|"right"\|"double"}` |
| **`escalate_human`** | `notifications/safety/escalation` | **340** | **60** | `{"reason": str, "risk_level": str, "details": str}` |
| **`type_text`** | `browser_type_text` | **280** | **48** | `{"element_id": str, "text": str, "press_enter": bool}` |
| **`select_option`** | `browser_select` | **260** | **48** | `{"element_id": str, "option": str}` |
| **`scroll`** | `browser_scroll` | **240** | **48** | `{"element_id": str, "direction": str, "amount": int, "unit": str}` |
| **`press_hotkey`** | `desktop_hotkey` | **240** | **48** | `{"keys": list[str], "target_pid": int}` |
| **Total** | — | **1,700** | **300** | **100% Validated (`validate_tool_arguments`)** |

### 2. Domain Distribution Across Splits

| Domain | Train (`n = 1,700`) | Held-Out Test (`n = 300`) | Representative UI Contexts |
| :--- | :---: | :---: | :--- |
| **`e-commerce`** | 340 | 60 | Storefront search, price facets, sort dropdowns, checkout ToS scrolling, voucher entry |
| **`enterprise_saas`** | 340 | 60 | CRM pipeline stages, billing notes, IAM role assignment, quarterly ledger grids |
| **`cloud_console`** | 340 | 60 | Kubernetes pod restarts, observability log queries, VPC region selection, S3 ACLs, trace waterfalls |
| **`desktop_windows`** | 340 | 60 | Windows File Explorer dialogs, Print Spooler spinboxes, Page Setup comboboxes, Event Viewer, 12 hotkeys |
| **`security_and_edge_cases`** | 340 | 60 | Destructive DB wipes (`Type DELETE`), 2FA/OTP, Cloudflare Turnstile, wire transfers, prompt injection, HIPAA/LGPD PII |

---

## 📊 Measured Token Census (Full Population of 2,000 Rows)

Measured across all 2,000 rows using `Qwen/Qwen2.5-1.5B-Instruct` (`tokenizers` 0.23.3, `add_special_tokens=False`):

| Span / Field | `n` | Min | p50 | Mean | p95 | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`input` Prompt** | 2,000 | 93 | 153 | **147.4** | 187 | **202** |
| **AXTree Fragment** | 2,000 | 52 | 87 | **89.6** | 122 | **130** |
| **Assistant Output JSON** | 2,000 | 101 | 127 | **127.6** | 160 | **171** |
| **Complete ChatML Sequence** | 2,000 | 383 | 447 | **452.0** | 501 | **521** |

> **Scope Note:** These figures describe the compact synthetic AXTree fragments in this dataset (`mean = 89.6` tokens), not live web page DOM pairs.

---

## 🔐 Cryptographic Hashes (`metadata/hashes.json`, `--seed 42`)

Regenerating via `python generate_argos_dataset.py --seed 42` reproduces both splits byte-for-byte:
* **`train.jsonl`** (`6,569,128` bytes, 1,700 unique rows):
  `108cf75967da5a8b43b5d77f2b6de54381cfeb3bf5f3ddd658ffa3225b2c49a5`
* **`test.jsonl`** (`1,139,642` bytes, 300 unique rows):
  `ecb5e6c334a9e063877743d7ae6f5522b954799d4c1fe61e5a18a5b11e4e7af9`

---

## 🛡️ Synthetic Fixtures Disclosure

Security templates use explicit synthetic placeholders:
* Synthetic Cloudflare Ray ID prefix: `sim-cf-ray-8b29f4c3a012-synthetic`
* Masked synthetic phone number: `(**) *****-4040`
* RFC 5737 documentation IP: `192.0.2.45`

No real user data, production credentials, or private tokens exist in this dataset.
