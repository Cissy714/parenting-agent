from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from state import ParentingAgentState
from nodes import load_baby_profile, load_memory, call_model, summarize_conversation, tool_node
from langchain_core.messages import AIMessage


def build_workflow():
    """构建并编译工作流图"""
    workflow = StateGraph(ParentingAgentState)

    workflow.add_node("load_profile", load_baby_profile)
    workflow.add_node("load_memory", load_memory)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_node("summarize", summarize_conversation)

    workflow.add_edge(START, "load_profile")
    workflow.add_edge("load_profile", "load_memory")
    workflow.add_edge("load_memory", "agent")

    # Agent 条件路由：有 tool_calls → tools，否则 → summarize
    def route_after_agent(state: ParentingAgentState) -> str:
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            return "tools"
        return "summarize"

    workflow.add_conditional_edges("agent", route_after_agent)
    workflow.add_edge("tools", "agent")
    workflow.add_edge("summarize", END)

    return workflow.compile(checkpointer=MemorySaver())


# 编译后的应用实例
app = build_workflow()
