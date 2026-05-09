"""
agent/graph.py
--------------
Builds and runs the LangGraph stateful agent.
Four nodes (tools) connected in a directed state graph.
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from agent.tools import (
    tool_log_analyzer,
    tool_threat_classifier,
    tool_mitre_rag,
    tool_report_generator,
)
import pandas as pd


class AgentState(TypedDict):
    """State that flows through all four LangGraph nodes."""
    raw_row: dict                        # Original CSV row as dict
    parsed_log: dict                     # Output of Tool 1 (text + stats + flags)
    classification: dict                 # Output of Tool 2
    mitre_techniques: list               # Output of Tool 3
    report: dict                         # Output of Tool 4
    error: Optional[str]                 # Any error encountered


# ── Node functions ─────────────────────────────────────────────────────────────

def node_log_analyzer(state: AgentState) -> AgentState:
    try:
        row = pd.Series(state["raw_row"])
        state["parsed_log"] = tool_log_analyzer(row)
    except Exception as e:
        state["error"] = f"Log analyzer failed: {e}"
        state["parsed_log"] = {"text": "Error parsing flow.", "stats": {}, "flags": []}
    return state


def node_threat_classifier(state: AgentState) -> AgentState:
    try:
        parsed_text = state["parsed_log"].get("text", "")
        state["classification"] = tool_threat_classifier(parsed_text)
    except Exception as e:
        state["error"] = f"Classifier failed: {e}"
        state["classification"] = {
            "label": "BENIGN", "confidence": "Low",
            "reason": str(e), "severity": "None"
        }
    return state


def node_mitre_rag(state: AgentState) -> AgentState:
    try:
        label = state["classification"].get("label", "BENIGN")
        state["mitre_techniques"] = tool_mitre_rag(label)
    except Exception as e:
        state["error"] = f"MITRE RAG failed: {e}"
        state["mitre_techniques"] = []
    return state


def node_report_generator(state: AgentState) -> AgentState:
    try:
        state["report"] = tool_report_generator(
            state["parsed_log"].get("text", ""),
            state["classification"],
            state["mitre_techniques"],
        )
    except Exception as e:
        state["error"] = f"Report generator failed: {e}"
        state["report"] = {
            "full_report": f"Report generation failed: {e}",
            "summary": "", "behavior": "",
            "mitre_mapping": "", "recommended_actions": ""
        }
    return state


# ── Build and compile agent ────────────────────────────────────────────────────

def build_agent():
    """Builds and compiles the LangGraph four-tool agent."""
    graph = StateGraph(AgentState)

    graph.add_node("log_analyzer", node_log_analyzer)
    graph.add_node("threat_classifier", node_threat_classifier)
    graph.add_node("mitre_rag", node_mitre_rag)
    graph.add_node("report_generator", node_report_generator)

    graph.set_entry_point("log_analyzer")
    graph.add_edge("log_analyzer", "threat_classifier")
    graph.add_edge("threat_classifier", "mitre_rag")
    graph.add_edge("mitre_rag", "report_generator")
    graph.add_edge("report_generator", END)

    return graph.compile()


def run_agent(row: pd.Series) -> AgentState:
    """
    Runs the full NetGuardAgent pipeline on a single network flow record.

    Args:
        row: A pandas Series representing one CICIDS-2017 CSV row

    Returns:
        Final AgentState with all tool outputs populated
    """
    agent = build_agent()
    initial_state: AgentState = {
        "raw_row": row.to_dict(),
        "parsed_log": {},
        "classification": {},
        "mitre_techniques": [],
        "report": {},
        "error": None,
    }
    return agent.invoke(initial_state)
