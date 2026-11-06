"""
Argos-Action to Model Context Protocol (MCP) JSON-RPC 2.0 Wire Bridge
Author: Pedro Lucas Reis (@pedroreoli)
Repository: PedroReoli/argos-action
License: Apache-2.0

Description:
Translates internal Argos-Action model predictions:
    { "thought": str, "tool_call": { "tool": str, "arguments": dict }, "verification_predicate": dict }
into strict Anthropic Model Context Protocol (MCP) wire format:
    { "jsonrpc": "2.0", "id": str, "method": "tools/call", "params": { "name": str, "arguments": dict } }
"""

import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# Mapping internal Argos tools to standard MCP tool identifiers
ARGOS_TO_MCP_TOOL_MAP = {
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
    def from_raw(cls, raw_output: str) -> "ArgosPrediction":
        """Parses model string generation into typed ArgosPrediction."""
        cleaned = raw_output.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.split("```json", 1)[1].rsplit("```", 1)[0].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned.split("```", 1)[1].rsplit("```", 1)[0].strip()
            
        data = json.loads(cleaned)
        tool_call = data.get("tool_call", {})
        return cls(
            thought=data.get("thought", ""),
            tool=tool_call.get("tool", ""),
            arguments=tool_call.get("arguments", {}),
            verification_predicate=data.get("verification_predicate")
        )

class ArgosMCPBridge:
    """Bridges Argos-Action neural outputs to MCP JSON-RPC 2.0 specifications."""
    
    def __init__(self, rpc_id_prefix: str = "argos_mcp_"):
        self.rpc_id_prefix = rpc_id_prefix
        self.call_counter = 0

    def to_mcp_wire(self, prediction: ArgosPrediction) -> Tuple[Dict[str, Any], bool]:
        """
        Converts an ArgosPrediction into an MCP JSON-RPC 2.0 request.
        Returns:
            (mcp_request_dict, is_safety_escalation)
        """
        self.call_counter += 1
        req_id = f"{self.rpc_id_prefix}{self.call_counter:04d}"
        
        mcp_tool_name = ARGOS_TO_MCP_TOOL_MAP.get(prediction.tool, f"custom_{prediction.tool}")
        is_escalation = (prediction.tool == "escalate_human")
        
        mcp_payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {
                "name": mcp_tool_name,
                "arguments": prediction.arguments
            },
            "_meta": {
                "thought": prediction.thought,
                "verification_predicate": prediction.verification_predicate
            }
        }
        return mcp_payload, is_escalation

    def format_wire_json(self, prediction: ArgosPrediction, indent: int = 2) -> str:
        payload, _ = self.to_mcp_wire(prediction)
        return json.dumps(payload, indent=indent, ensure_ascii=False)

def verify_bridge_example():
    sample_model_output = """{
      "thought": "The user wants to mark Acme Corp as 'Closed Won'. Clicking the stage button.",
      "tool_call": {
        "tool": "click",
        "arguments": {
          "element_id": "btn_stage_closed_won",
          "button": "left"
        }
      },
      "verification_predicate": {
        "expected_element": "modal_confirm_won",
        "timeout_ms": 2000
      }
    }"""
    
    pred = ArgosPrediction.from_raw(sample_model_output)
    bridge = ArgosMCPBridge()
    wire_req, is_safety = bridge.to_mcp_wire(pred)
    
    assert wire_req["jsonrpc"] == "2.0"
    assert wire_req["method"] == "tools/call"
    assert wire_req["params"]["name"] == "browser_click"
    assert wire_req["params"]["arguments"]["element_id"] == "btn_stage_closed_won"
    assert is_safety is False
    print("[PASS] ArgosMCPBridge wire translation verified successfully.")
    print(json.dumps(wire_req, indent=2))

if __name__ == "__main__":
    verify_bridge_example()
