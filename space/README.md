---
title: Argos Action 1.5B - MCP Protocol & Guardrail Playground
emoji: 👁️
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
license: apache-2.0
short_description: Static MCP Protocol & Guardrail Playground for Argos-Action
---

# 👁️ Argos-Action-1.5B v2.0 (Static MCP Protocol & Guardrail Playground)

> **Transparency Notice:** This is a static protocol and safety guardrail playground served on Hugging Face Static Spaces (`sdk: static`). It inspects how **Argos-Action-1.5B** structures single-turn Accessibility Tree (AXTree / Windows UIA) predictions into Anthropic **Model Context Protocol (MCP) JSON-RPC 2.0** wire messages, and how the v2.0 **Deterministic Pre-Execution Safety Guardrail (`SafetyPolicyEngine`)** blocks destructive or unsafe actions (`RULE_01`–`RULE_06`) even if a model fails to emit `escalate_human`. No live GPU backend runs on this static page.

- **Model Weights (LoRA):** [`pedroreoli/argos-action-1.5b`](https://huggingface.co/pedroreoli/argos-action-1.5b)
- **Dataset v2.0 (2,000 Unique SFT Samples, 6 Tools):** [`pedroreoli/argos-computer-use-mcp`](https://huggingface.co/datasets/pedroreoli/argos-computer-use-mcp)
- **GitHub Repository:** [`PedroReoli/argos-action`](https://github.com/PedroReoli/argos-action)
- **Technical Post-Mortem & Holdout Evaluation:** [Medium Article](https://medium.com/@pedrosousa2160/meu-script-disse-100-ele-s%C3%B3-tinha-validado-o-gerador-d7433b381927) • [Kaggle Notebook](https://www.kaggle.com/code/pedroreoli/argos-holdout-v2-eval-of-a-1-5b-ui-action-model)
- **Author:** Pedro Lucas Reis ([@pedroreoli](https://huggingface.co/pedroreoli))
