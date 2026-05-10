"""
agent/graph.py
--------------
LangGraph stateful agent — four nodes connected in sequence.
The rule-based hint from Tool 1 is passed to Tool 2 to guide classification.
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
    raw_row: dict
    parsed_log: dict
    classification: dict
    mitre_techniques: list
    report: dict
    error: Optional[str]


def node_log_analyzer(state: AgentState) -> AgentState:
    try:
        row = pd.Series(state["raw_row"])
        state["parsed_log"] = tool_log_analyzer(row)
    except Exception as e:
        state["error"] = f"Log analyzer: {e}"
        state["parsed_log"] = {"text": "Error parsing flow.", "stats": {}, "flags": [], "hint": "UNKNOWN"}
    return state


def node_threat_classifier(state: AgentState) -> AgentState:
    try:
        parsed = state["parsed_log"]
        state["classification"] = tool_threat_classifier(
            parsed.get("text", ""),
            hint=parsed.get("hint", "UNKNOWN")
        )
    except Exception as e:
        state["error"] = f"Classifier: {e}"
        hint = state["parsed_log"].get("hint", "BENIGN")
        state["classification"] = {
            "label": hint if hint != "UNKNOWN" else "BENIGN",
            "confidence": "Low",
            "reason": str(e),
            "severity": "Unknown"
        }
    return state


def node_mitre_rag(state: AgentState) -> AgentState:
    try:
        label = state["classification"].get("label", "BENIGN")
        state["mitre_techniques"] = tool_mitre_rag(label)
    except Exception as e:
        state["error"] = f"MITRE RAG: {e}"
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
        state["error"] = f"Report generator: {e}"
        state["report"] = {
            "full_report": f"Report generation failed: {e}",
            "summary": "", "behavior": "",
            "mitre_mapping": "", "recommended_actions": ""
        }
    return state


def build_agent():
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
    agent = build_agent()
    initial: AgentState = {
        "raw_row": row.to_dict(),
        "parsed_log": {},
        "classification": {},
        "mitre_techniques": [],
        "report": {},
        "error": None,
    }
    return agent.invoke(initial)