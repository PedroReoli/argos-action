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

**Argos-Action** is a curated dataset designed for **Supervised Fine-Tuning (SFT)** of Small Language Models (SLMs, e.g., Qwen 2.5 1.5B, Llama 3.2 1B/3B) into specialized **Edge Policy Agents for Autonomous Computer Use**.

---

## 🔬 Motivation & Theoretical Framework

Frontier general-purpose LLMs (Claude 3.5 Sonnet, GPT-4o) provide strong reasoning but suffer from operational bottlenecks in high-frequency computer use loops:
1. **Latency Overhead:** Inference times of 2.5s – 5.0s per step make interactive GUI control sluggish.
2. **Context Inflation in Real-World DOMs:** Injecting full raw HTML DOM can consume 40,000 to 80,000 tokens per interaction.
3. **Structured Tool Contracts:** Chat-oriented models require robust alignment to consistently emit valid Model Context Protocol (MCP) JSON schemas without markdown clutter.

**Theoretical vs. Dataset Representation:**
* *Theoretical Literature & In-the-Wild Scrape:* Pruning raw web DOM into clean Accessibility Trees (AXTree) reduces token bloat by ~90% in dense web portals.
* *This Dataset:* Presents concise, pre-pruned semantic accessibility snapshots (averaging ~122 tokens of AXTree per sample) for fast, lightweight policy fine-tuning.

---

## 🏛️ Dataset Taxonomy & Domains

The dataset contains **2,000 structured samples** partitioned into **1,700 training samples** and **300 strictly held-out evaluation samples** across five core operational environments:

| Domain | Scope & Workflows | Target Actions |
| :--- | :--- | :--- |
| **E-Commerce & Retail** | Search bars, multi-attribute filter facets, cart manipulation, checkout flows | `click`, `type_text`, `select_option`, `scroll` |
| **Enterprise SaaS & CRM** | Data grids, pipeline kanban transitions, issue creation, reporting exports | `click`, `type_text`, `select_option` |
| **DevOps & Cloud Consoles** | Cluster pod lifecycle, pull request review & squash merge, bucket ACL enforcement | `click`, `type_text`, `verify_state` |
| **Native Windows UIA** | File Explorer picker dialogs, desktop settings toggles, print subsystem | `click`, `type_text`, `press_hotkey` |
| **Safety & Human Escalation** | 2FA/SMS verification, Cloudflare Turnstile CAPTCHA, irreversible data wipe | `escalate_human` |

---

## 📊 Measured Token Census & File Manifest

Measured across the 2,000 samples using the `Qwen/Qwen2.5-1.5B-Instruct` tokenizer:

| Metric | Average | Maximum |
| :--- | :---: | :---: |
| **Input Tokens (`input`)** | 166.3 tokens | 226 tokens |
| **AXTree Fragment Tokens** | 122.2 tokens | 174 tokens |
| **Complete ChatML Tokens** | 355.5 tokens | 412 tokens |
| **Training `max_length` Window** | 1,536 tokens | *(Zero sample truncation)* |

### SHA-256 Checksums (`data/hashes.json`)
* **`train.jsonl`** (`4,669,033` bytes, 1,700 rows): `382f0f953fcf7aad758147a2009b6b615d0a7fd2881cbc278828fa28d9a6476a`
* **`test.jsonl`** (`836,940` bytes, 300 rows): `45f2cf222247abb2536ea2f848c5f59fb0b9c94bf6d7115871c2202247ba50fc`

> **Data Leakage Prevention:** The test split (`test.jsonl`) consists of disjoint templates and scenarios (including destructive database deletions, multi-select menus, and pagination scrolls) that never appear in the training partition, providing an honest benchmark for out-of-distribution evaluation.

---

## 📋 Data Format & Structure

Every record includes dual schemas for universal training compatibility: **Alpaca Format** (`instruction`, `input`, `output`) and **OpenAI/ChatML Format** (`messages`):

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
Security challenge templates include mock identifiers (e.g., `sim-cf-ray-8b29f4c3a012-synthetic` and RFC 5737 documentation IP `192.0.2.45`) strictly generated as synthetic test fixtures for training safe human escalation routines. Zero live user data, credentials, or proprietary tokens are present.
