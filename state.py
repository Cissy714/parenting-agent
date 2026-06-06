from typing import TypedDict, List, Optional, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


class ParentingAgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    current_baby_id: Optional[int]
    baby_profile: Optional[dict]
    episodic_context: Optional[str]    # 从情景记忆检索到的历史事件
    semantic_context: Optional[str]    # 从语义记忆检索到的用户偏好
