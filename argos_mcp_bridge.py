"""
Argos-Action to Model Context Protocol (MCP) JSON-RPC 2.0 Wire Bridge
Author: Pedro Lucas Reis (@pedroreoli)
Repository: PedroReoli/argos-action
License: Apache-2.0

Description:
Translates internal Argos-Action model predictions:
    { "thought": str, "tool_call": { "tool": str, "arguments": dict }, "verification_predicate": dict }
into strict Anthropic Model Context Protocol (MCP) JSON-RPC 2.0 wire messages.

Features:
- Enforces a strict supported tools catalog; unknown tools are rejected with JSON-RPC error -32601.
- Catches invalid JSON strings and emits JSON-RPC error -32700 instead of unhandled exceptions.
- Formally differentiates execution tools (tools/call) from safety halt barriers (notifications/safety/escalation).
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

# Strict catalog of permitted tools
SUPPORTED_TOOLS = {
    "click": "browser_click",
    "type_text": "browser_type_text",
    "scroll": "browser_scroll",
    "select_option": "browser_select",
    "press_hotkey": "desktop_hotkey",
    "escalate_human": "mcp_safety_escalate_to_human"
}

@dataclass
class ArgosPrediction:
    thought: str
    tool: str
    arguments: Dict[str, Any]
    verification_predicate: Optional[Dict[str, Any]] = None

    @classmethod
    def from_raw(cls, raw_output: str) -> Union["ArgosPrediction", Dict[str, Any]]:
        """Parses model string generation into typed ArgosPrediction or JSON-RPC error dict."""
        cleaned = raw_output.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.split("```json", 1)[1].rsplit("```", 1)[0].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned.split("```", 1)[1].rsplit("```", 1)[0].strip()
            
        try:
            data = json.loads(cleaned)
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": f"Parse error: Invalid JSON syntax in model output ({str(e)})"
                }
            }

        if not isinstance(data, dict):
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32600,
                    "message": "Invalid Request: Model output must be a JSON object"
                }
            }

        tool_call = data.get("tool_call")
        if not tool_call or not isinstance(tool_call, dict) or not tool_call.get("tool"):
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32600,
                    "message": "Invalid Request: 'tool_call' object with valid 'tool' property is required"
                }
            }

        return cls(
            thought=data.get("thought", ""),
            tool=tool_call.get("tool", ""),
            arguments=tool_call.get("arguments", {}) if isinstance(tool_call.get("arguments"), dict) else {},
            verification_predicate=data.get("verification_predicate")
        )

class ArgosMCPBridge:
    """Bridges Argos-Action neural outputs to MCP JSON-RPC 2.0 specifications."""
    
    def __init__(self, rpc_id_prefix: str = "argos_mcp_"):
        self.rpc_id_prefix = rpc_id_prefix
        self.call_counter = 0

    def to_mcp_wire(self, prediction_or_error: Union[ArgosPrediction, Dict[str, Any]]) -> Tuple[Dict[str, Any], bool]:
        """
        Converts an ArgosPrediction or error into an MCP JSON-RPC 2.0 payload.
        Returns:
            (mcp_payload_dict, is_safety_escalation)
        """
        self.call_counter += 1
        req_id = f"{self.rpc_id_prefix}{self.call_counter:04d}"

        # If already an error from parsing:
        if isinstance(prediction_or_error, dict) and "error" in prediction_or_error:
            payload = dict(prediction_or_error)
            payload["id"] = req_id
            return payload, False

        pred = prediction_or_error

        # Reject unknown tools explicitly with -32601 Method not found
        if pred.tool not in SUPPORTED_TOOLS:
            error_payload = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: Tool '{pred.tool}' is not supported by Argos policy"
                },
                "_meta": {"raw_requested_tool": pred.tool}
            }
            return error_payload, False

        mcp_tool_name = SUPPORTED_TOOLS[pred.tool]

        # Safety escalation: generates host-level HALT notification rather than raw tool call
        if pred.tool == "escalate_human":
            safety_payload = {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": "notifications/safety/escalation",
                "params": {
                    "status": "EXECUTION_HALTED",
                    "action_required": "human_operator_authorization",
                    "reason": pred.arguments.get("reason", "HIGH_RISK_BARRIER"),
                    "details": pred.arguments.get("details", ""),
                    "arguments": pred.arguments
                },
                "_meta": {
                    "thought": pred.thought,
                    "policy": "HUMAN_IN_THE_LOOP_MANDATORY_HALT"
                }
            }
            return safety_payload, True

        # Standard execution tool call
        mcp_payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {
                "name": mcp_tool_name,
                "arguments": pred.arguments
            },
            "_meta": {
                "thought": pred.thought,
                "verification_predicate": pred.verification_predicate
            }
        }
        return mcp_payload, False

def verify_bridge_suite():
    bridge = ArgosMCPBridge()

    # 1. Standard valid click
    sample_click = """{
      "thought": "Clicking Closed Won stage",
      "tool_call": {"tool": "click", "arguments": {"element_id": "btn_stage_won", "button": "left"}}
    }"""
    res1, esc1 = bridge.to_mcp_wire(ArgosPrediction.from_raw(sample_click))
    assert res1["method"] == "tools/call"
    assert res1["params"]["name"] == "browser_click"
    assert esc1 is False
    print("[PASS] Case 1: Standard valid click mapped to tools/call.")

    # 2. Safety escalation
    sample_safety = """{
      "thought": "2FA token prompt detected",
      "tool_call": {"tool": "escalate_human", "arguments": {"reason": "2FA_SMS_OTP", "details": "Payment authorization"}}
    }"""
    res2, esc2 = bridge.to_mcp_wire(ArgosPrediction.from_raw(sample_safety))
    assert res2["method"] == "notifications/safety/escalation"
    assert res2["params"]["status"] == "EXECUTION_HALTED"
    assert esc2 is True
    print("[PASS] Case 2: Escalate human halts loop with safety notification.")

    # 3. Unknown tool rejection (e.g. wipe_disk)
    sample_unknown = """{
      "thought": "Wiping disk",
      "tool_call": {"tool": "wipe_disk", "arguments": {"target": "C:"}}
    }"""
    res3, esc3 = bridge.to_mcp_wire(ArgosPrediction.from_raw(sample_unknown))
    assert "error" in res3
    assert res3["error"]["code"] == -32601
    assert "Method not found" in res3["error"]["message"]
    assert esc3 is False
    print("[PASS] Case 3: Unknown tool 'wipe_disk' strictly rejected with code -32601.")

    # 4. Invalid JSON syntax
    sample_bad_json = "NOT_A_JSON_STRING"
    res4, esc4 = bridge.to_mcp_wire(ArgosPrediction.from_raw(sample_bad_json))
    assert "error" in res4
    assert res4["error"]["code"] == -32700
    print("[PASS] Case 4: Invalid JSON string caught with code -32700 Parse error.")

    # 5. Missing tool_call key
    sample_missing = '{"thought": "no tools here"}'
    res5, esc5 = bridge.to_mcp_wire(ArgosPrediction.from_raw(sample_missing))
    assert "error" in res5
    assert res5["error"]["code"] == -32600
    print("[PASS] Case 5: Missing tool_call caught with code -32600 Invalid Request.")

if __name__ == "__main__":
    verify_bridge_suite()
