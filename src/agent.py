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
    ROLE: You are an Intelligent Solar Energy Grid Optimization Assistant. Your expertise is STRICTLY limited to solar generation forecasting, grid management, battery storage, and energy optimization.
    
    GUARDRAILS:
    - ONLY answer questions related to solar energy, grid optimization, and the provided forecast data.
    - If the USER QUERY is unrelated to your role (e.g., general programming, HTML, recipes, etc.), politely decline and explain that you are a specialized assistant for solar grid optimization.
    
    USER QUERY: {state.get('user_query', 'Please provide a general solar generation report.')}
    
    CONTEXT DATA:
    - Forecast Summary: {state['forecast_summary']}
    - Identified Risks: {state['risks']}
    - Grid Management Guidelines (RAG): {state['guidelines']}
    
    INSTRUCTIONS:
    1. Check if the user query is on-topic. If not, follow the GUARDRAILS.
    2. If on-topic, directly answer the USER QUERY using the provided CONTEXT DATA.
    3. If the user asks for a "report" or "comprehensive analysis", provide a structured response with these sections:
       - Solar generation forecast summary
       - Identified variability and risk periods
       - Grid balancing and storage recommendations
       - Energy utilization optimization strategies
       - Supporting references
    4. Ground all technical claims in the provided data.
    """
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-120b"
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
