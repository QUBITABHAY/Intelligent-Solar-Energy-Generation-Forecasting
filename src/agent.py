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
    """Generate recommendations using Groq API."""
    try:
        from groq import Groq
    except ImportError:
        return {"recommendation": "Error: groq not installed. Install with: pip install groq"}
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return {"recommendation": "Error: GROQ_API_KEY environment variable not set"}
    
    client = Groq(api_key=api_key)
    
    prompt = f"""
    Please generate a comprehensive grid management report based on the following context.
    
    User Query: {state.get('user_query', 'Please provide a general solar generation report.')}
    
    Forecast Summary:
    {state['forecast_summary']}
    
    Identified Risks:
    {state['risks']}
    
    Grid Guidelines:
    {state['guidelines']}
    
    Your report MUST include EXACTLY these 5 defined sections:
    - Solar generation forecast summary
    - Identified variability and risk periods
    - Grid balancing and storage recommendations
    - Energy utilization optimization strategies
    - Supporting references
    """
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama2-70b-4096"
        )
        
        if response.choices and response.choices[0].message.content:
            return {"recommendation": response.choices[0].message.content}
        return {"recommendation": "Error: No response from API"}
    except Exception as e:
        return {"recommendation": f"Error generating report: {str(e)}"}


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
