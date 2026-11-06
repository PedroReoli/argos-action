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
    num_bytes: 4669033
    num_examples: 1700
  - name: test
    num_bytes: 836940
    num_examples: 300
---

# 👁️ Argos: Computer Use & Semantic Tool Calling (MCP) Benchmark

**Argos-Action** is a synthetic single-turn decision dataset designed for research into **Supervised Fine-Tuning (SFT)** of Small Language Models (SLMs, e.g., Qwen 2.5 1.5B, Llama 3.2) into specialized **Edge Policy Agents for Autonomous Computer Use**.

---

## 🔬 Motivation & Research Scope

Frontier general-purpose LLMs (Claude 3.5 Sonnet, GPT-4o) provide strong reasoning but face operational challenges in continuous GUI execution loops:
1. **High Latency Overhead:** Multi-second roundtrips per interaction slow down interactive desktop/web control.
2. **Context Inflation in Real-World DOMs:** In live web pages, raw HTML DOM trees often reach 40,000 to 80,000 tokens per screen.
3. **Structured Tool Parsing:** Unconstrained chat models can occasionally emit markdown clutter instead of valid JSON tool calls.

**Theoretical vs. Synthetic Dataset Representation:**
* *Theoretical Contrast (Real-World DOM vs. AXTree):* In live scrapers (e.g. Achilles CDP), pruning raw DOM (~45,000 tokens) into accessibility trees (~3,500 tokens) yields an arithmetic reduction of ~92.2% ($1 - 3500/45000$).
* *This Synthetic Dataset:* Contains compact, pre-structured accessibility tree fragments designed for lightweight syntax and action policy fine-tuning (averaging **87.2 tokens** of AXTree per sample).

---

## 🏛️ Dataset Taxonomy & Tool Distribution

The dataset contains **2,000 single-turn decision samples** partitioned into **1,700 training samples** and **300 strictly held-out evaluation samples** across five domains:

| Domain | Scope & Scenarios | Train Actions | Test Actions |
| :--- | :--- | :--- | :--- |
| **E-Commerce & Retail** | Search bars, price filter facets, cart manipulation, scroll discovery | `click`, `type_text` | `scroll` (60) |
| **Enterprise SaaS & CRM** | Data grids, pipeline kanban transitions, role selection menus | `click`, `type_text` | `select_option` (60) |
| **DevOps & Cloud Consoles** | Cluster pod restart, pull request review & squash merge, S3 bucket ACL | `click`, `type_text` | `click` (60) |
| **Native Windows UIA** | File Explorer picker dialogs, print dialog duplex & copy settings | `click`, `press_hotkey` | `type_text` (60) |
| **Safety & Human Escalation** | 2FA/SMS verification, Cloudflare Turnstile anti-bot, database wipe guard | `escalate_human` | `escalate_human` (60) |

> [!NOTE]
> **Tool Distribution Note:** In this dataset version (`ae274fe`), `scroll` (60 samples) and `select_option` (60 samples) are present exclusively in the held-out test split to test generalization on out-of-distribution action schemas. In the training split (1,700 samples), the distribution is: `click` (1,035), `escalate_human` (340), `type_text` (165), and `press_hotkey` (160).

---

## 📊 Measured Token Census (Full Population of 2,000 Rows)

Census measured directly across all 2,000 rows using the `Qwen/Qwen2.5-1.5B-Instruct` tokenizer (`tokenizers` 0.23.3, `add_special_tokens=False`):

| Span / Field | n | Min | p50 | Mean | p95 | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`input` Prompt** | 2,000 | 90 | 127 | **128.4** | 160 | **164** |
| **AXTree Fragment** | 2,000 | 59 | 83 | **87.2** | 114 | **116** |
| **Assistant Output JSON** | 2,000 | 71 | 79 | **85.0** | 112 | **112** |
| **Complete ChatML Sequence** | 2,000 | 262 | 304 | **307.4** | 366 | **370** |

* Training sequence token total (1,700 rows $\times$ 3 epochs): **1,567,626 tokens**.

### SHA-256 Checksums (`metadata/hashes.json`)
* **`train.jsonl`** (`4,669,033` bytes, 1,700 rows): `382f0f953fcf7aad758147a2009b6b615d0a7fd2881cbc278828fa28d9a6476a`
* **`test.jsonl`** (`836,940` bytes, 300 rows): `45f2cf222247abb2536ea2f848c5f59fb0b9c94bf6d7115871c2202247ba50fc`

### Holdout Independence:
* Exact overlap between the 300 test inputs/goals and the training set: **0/300**.
* The test set consists of 5 disjoint templates (each instantiated with variations 60 times) that never appear in the training partition.

---

## 📋 Data Format & Structure

Records include dual formats: **Alpaca Format** (`instruction`, `input`, `output`) and **OpenAI/ChatML Format** (`messages`):

```json
{
  "id": "argos_train_00042",
  "domain": "enterprise_saas",
  "instruction": "You are Argos-Action, an expert autonomous Computer Use policy model...",
  "input": "### User Objective:\nUpdate opportunity status for 'Acme Corp' to 'Closed Won'\n\n### Current Screen State (Semantic Accessibility Tree):\nWindow \"Enterprise CRM - Pipeline\" bounds=(0,0,1920,1080)\n  [Card: \"Acme Corp Deal - $120,000\" id=\"card_deal_acme\"]\n  [Tab: \"Negotiation\" state=\"active\"]\n  [Button: \"Closed Won\" id=\"btn_stage_closed_won\" actions=[click]]\n  [Button: \"Closed Lost\" id=\"btn_stage_closed_lost\"]\n\n### Execution History:\n- opened deal detail for client Acme Corp",
  "output": "{\n  \"thought\": \"The user requested marking Acme Corp's deal as 'Closed Won'. The stage button is visible in the pipeline bar.\",\n  \"tool_call\": {\n    \"tool\": \"click\",\n    \"arguments\": {\n      \"element_id\": \"btn_stage_closed_won\",\n      \"button\": \"left\"\n    }\n  },\n  \"verification_predicate\": {\n    \"expected_element\": \"modal_confirm_won\",\n    \"timeout_ms\": 2000\n  }\n}",
  "messages": [
    {"role": "system", "content": "You are Argos-Action, an expert autonomous Computer Use policy model..."},
    {"role": "user", "content": "### User Objective:\n..."},
    {"role": "assistant", "content": "{\"thought\": ...}"}
  ]
}
```

---

## 🛡️ Synthetic Fixtures Disclosure
Security templates include explicit synthetic placeholders:
* Synthetic Cloudflare Ray ID: `sim-cf-ray-8b29f4c3a012-synthetic` (165 occurrences).
* Masked synthetic phone number: `(**) *****-4040` (350 occurrences).
No real user data, production IP addresses, credentials, or private tokens are present.
