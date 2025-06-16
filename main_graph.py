# main_graph.py
from langgraph.graph import StateGraph, END
from state_schema import AgentState
from agents.query_classification_agent import run_classification_pipeline 
from agents.fallback_agent import fallback_agent
from config.settings import TOGETHER_API_KEY,TOGETHER_MODEL
import json
# สร้าง Graph ด้วย AgentState เป็น schema หลัก
builder = StateGraph(AgentState)
builder.add_node("classify_query", run_classification_pipeline)
builder.add_node("fallback_response", fallback_agent)
builder.set_entry_point("classify_query") #จุดเริ่มต้น 
builder.add_edge("classify_query", END)

# Compile graph
graph = builder.compile()

if __name__ == "__main__":
    initial_state = AgentState(user_query="ต้องทำยังไงดี")
    try:
        raw_state = graph.invoke(initial_state)
        final_state = AgentState(**raw_state)
        print(json.dumps(final_state.model_dump(), indent=2, ensure_ascii=False))
    except Exception as e:
        print("❌ Error during classification:", str(e))
