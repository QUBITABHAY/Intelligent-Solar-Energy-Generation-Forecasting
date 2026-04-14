import os
from typing import TypedDict
import pandas as pd
from langgraph.graph import StateGraph, END

from src.tools import analyze_forecast, identify_risks, retrieve_guidelines

class AgentState(TypedDict):
    forecast_df: pd.DataFrame
    user_query: str
    forecast_summary: dict
    risks: list
    guidelines: str
    recommendation: str

def analyze_node(state: AgentState) -> dict:
    """Analyze forecast data."""
    summary = analyze_forecast(state["forecast_df"])
    return {"forecast_summary": summary}

def risk_check_node(state: AgentState) -> dict:
    """Identify risk periods in forecast."""
    risks = identify_risks(state["forecast_df"])
    return {"risks": risks}

def retrieve_node(state: AgentState) -> dict:
    """Retrieve relevant guidelines using RAG."""
    query = state.get("user_query", "solar energy optimization and grid balancing")
    guidelines = retrieve_guidelines(query)
    return {"guidelines": guidelines}

def recommend_node(state: AgentState) -> dict:
    """Generate a structured local report from forecast data and knowledge clips."""
    
    summary = state.get("forecast_summary", {})
    risks = state.get("risks", [])
    guidelines = state.get("guidelines", "No specific guidelines retrieved.")
    
    report = f"""### 📊 Local Grid Optimization Report

#### 1. Solar Generation Forecast Summary
- **Average Generation**: {summary.get('mean', 0):.2f} kW
- **Peak Generation**: {summary.get('max', 0):.2f} kW
- **Minimum Generation**: {summary.get('min', 0):.2f} kW
- **Data Confidence**: High (Local Analysis)

#### 2. Variability and Risk Assessment
Found **{len(risks)}** significant variability periods:
"""
    
    if not risks:
        report += "- No high-risk low-generation periods identified.\n"
    else:
        for i, risk in enumerate(risks[:5], 1):
            report += f"- Period {i}: Index {risk['start_index']} to {risk['end_index']} ({risk['risk_level']} Risk, Avg: {risk['avg_power']:.2f} kW)\n"

    report += f"""
#### 3. Grid Balancing & Knowledge Insights
Based on the retrieved guidelines:
---
{guidelines}
---

#### 4. Supporting References
- Source models: Random Forest Regressor & all-MiniLM-L6-v2
- Local knowledge base query: '{state.get('user_query', 'General Optimization')}'
"""
    
    return {"recommendation": report}


graph_builder = StateGraph(AgentState)

graph_builder.add_node("analyze", analyze_node)
graph_builder.add_node("risk_check", risk_check_node)
graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("recommend", recommend_node)

graph_builder.add_edge("analyze", "risk_check")
graph_builder.add_edge("risk_check", "retrieve")
graph_builder.add_edge("retrieve", "recommend")
graph_builder.add_edge("recommend", END)

graph_builder.set_entry_point("analyze")

agent = graph_builder.compile()
