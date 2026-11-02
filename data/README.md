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
    num_bytes: 2600000
    num_examples: 1700
  - name: test
    num_bytes: 460000
    num_examples: 300
---

# 👁️ Argos: Computer Use & Semantic Tool Calling (MCP) Benchmark

**Argos-Action** is a curated, high-precision dataset designed for **Supervised Fine-Tuning (SFT)** of Small Language Models (SLMs, e.g., Qwen 2.5 1.5B, Llama 3.2 1B/3B, Gemma 2 2B) into specialized **Edge Policy Agents for Autonomous Computer Use**.

---

## 🔬 Motivation & Theoretical Framework

Frontier general-purpose LLMs (Claude 3.5 Sonnet, GPT-4o) provide strong reasoning but suffer from critical bottlenecks when deployed in autonomous computer use loops:
1. **Latency Overhead:** Average inference times of 2.5s – 5.0s per step make interactive GUI control sluggish.
2. **Token Inflation:** Raw HTML DOM injection consumes 40,000 to 80,000 tokens per interaction, incurring high operational costs and context degradation.
3. **Determinism Gap:** General chat models occasionally hallucinate non-existent CSS selectors or emit unstructured markdown instead of strict JSON tool calls.

**Argos** bridges this gap by grounding agents on **Semantic Accessibility Trees (AXTree / Windows UIA)** — reducing token consumption by up to **92.2%** — and training compact models to emit deterministic **Model Context Protocol (MCP)** tool executions with sub-150ms latency.

---

## 🏛️ Dataset Taxonomy & Domains

The dataset contains **2,000 structured trajectories** partitioned into **1,700 training samples** and **300 held-out evaluation samples** across five core operational environments:

| Domain | Scope & Workflows | Target Actions |
| :--- | :--- | :--- |
| **E-Commerce & Retail** | Search bars, multi-attribute filter facets, cart manipulation, checkout flows | `click`, `type_text`, `select_option`, `scroll` |
| **Enterprise SaaS & CRM** | Data grids, pipeline kanban transitions, issue creation, reporting exports | `click`, `type_text`, `select_option`, `verify_state` |
| **DevOps & Cloud Consoles** | Cluster pod lifecycle, pull request review & squash merge, secrets rotation | `click`, `type_text`, `hotkey`, `verify_state` |
| **Native Windows UIA** | File Explorer picker dialogs, desktop settings toggles, print subsystem | `click`, `type_text`, `double_click` |
| **Safety & Human Escalation** | 2FA/SMS verification, Cloudflare Turnstile CAPTCHA, irreversible data wipe | `escalate_human`, `verify_state` |

---

## 📋 Data Format & Structure

Every record includes dual schemas for universal training compatibility: **Alpaca Format** (`instruction`, `input`, `output`) and **OpenAI/ChatML Format** (`messages`):

```json
{
  "id": "argos_act_01242",
  "domain": "enterprise_saas",
  "instruction": "You are Argos-Action, an expert autonomous Computer Use policy model...",
  "input": "### User Objective:\nUpdate deal status from 'In Negotiation' to 'Closed Won' for client Acme Corp\n\n### Current Screen State (Semantic Accessibility Tree):\nWindow \"CloudCRM - Deal: Acme Corp Enterprise Tier\" bounds=(0,0,1920,1080)...",
  "output": {
    "thought": "The user wants to mark Acme Corp as 'Closed Won'. I will click the 'stage_closed_won' pill button to transition the pipeline stage.",
    "tool_call": {
      "tool": "click",
      "arguments": {
        "element_id": "stage_closed_won",
        "delivery_mode": "background"
      }
    },
    "verification_predicate": {
      "expected_state": {
        "stage_closed_won": "active=true",
        "btn_save_deal": "disabled=false"
      },
      "timeout_ms": 1500
    }
  }
}
```

---

## 🚀 Quickstart & Usage

```python
from datasets import load_dataset

dataset = load_dataset("pedroreoli/argos-computer-use-mcp")
print(f"Train split: {len(dataset['train'])} samples")
print(f"Test split:  {len(dataset['test'])} samples")
```

---

## 🔗 Ecosystem Integration

* **Achilles CDP Agent:** Connects directly with [Achilles CDP Agent](https://github.com/PedroReoli/Achilles-CDP-Agent) and Windows UI Automation runtimes.
* **Model Context Protocol (MCP):** Output contracts adhere 100% to Anthropic's open MCP tool specification.

## 👨‍💻 Author & Citation
* **Pedro Lucas Reis** ([LinkedIn](https://www.linkedin.com/in/pedro-lucas-reis-a93945171/) | [GitHub](https://github.com/PedroReoli))
* Project Argos: High-Throughput Edge Agents for Computer Use (2026).
