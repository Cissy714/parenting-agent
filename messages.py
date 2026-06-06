import json
from typing import List, Any
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage


def convert_to_dict_messages(messages: List[Any]) -> List[dict]:
    """将 LangChain 消息转换为 dict，保留 reasoning_content"""
    result = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            d = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"])
                        }
                    } for tc in msg.tool_calls
                ]
            # 保留 reasoning_content
            if msg.additional_kwargs.get("reasoning_content"):
                d["reasoning_content"] = msg.additional_kwargs["reasoning_content"]
            result.append(d)
        elif isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, ToolMessage):
            result.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content
            })
        else:
            # HumanMessage 或其他
            result.append({"role": "user", "content": msg.content})
    return result


def convert_to_ai_message(msg_dict: dict) -> AIMessage:
    """将 API 响应 dict 转换为 AIMessage，保留 reasoning_content"""
    kwargs = {"content": msg_dict.get("content", "")}
    additional = {}

    # 保留 reasoning_content
    if "reasoning_content" in msg_dict:
        additional["reasoning_content"] = msg_dict["reasoning_content"]

    # 处理 tool_calls
    if msg_dict.get("tool_calls"):
        tool_calls = []
        for tc in msg_dict["tool_calls"]:
            func = tc.get("function", {})
            args_str = func.get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({
                "id": tc["id"],
                "name": func.get("name", ""),
                "args": args
            })
        kwargs["tool_calls"] = tool_calls

    if additional:
        kwargs["additional_kwargs"] = additional

    return AIMessage(**kwargs)
