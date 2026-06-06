"""
育儿智能助手 - Streamlit Demo（流式实时展示）
"""
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from graph import app
from memory.db import get_all_babies
from logger import get_logger
import uuid

logger = get_logger("demo")

st.set_page_config(page_title="育儿智能助手", page_icon="🧒", layout="wide")

# ── Session State ────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"demo-{uuid.uuid4().hex[:8]}"
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []
if "first_turn" not in st.session_state:
    st.session_state.first_turn = True


def reset_session():
    st.session_state.thread_id = f"demo-{uuid.uuid4().hex[:8]}"
    st.session_state.display_messages = []
    st.session_state.first_turn = True


NODE_LABELS = {
    "load_profile": "🔍 识别宝宝档案",
    "load_memory": "🧠 检索历史记忆",
    "agent": "🤖 AI 思考中",
    "tools": "🔧 执行工具",
    "summarize": "💾 保存对话记忆",
}

TOOL_LABELS = {
    "parenting_knowledge": "📚 检索知识库",
    "manage_baby_profile": "👶 管理宝宝档案",
}


# ── 侧边栏 ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧒 育儿智能助手")

    st.subheader("📋 宝宝档案")
    babies = get_all_babies()
    if babies:
        baby_options = {b["name"]: b for b in babies}
        selected_name = st.selectbox("选择宝宝", list(baby_options.keys()), key="baby_selector")
        if selected_name:
            baby = baby_options[selected_name]
            st.info(f"""
            **名字**: {baby['name']}
            **出生日期**: {baby.get('birth_date') or '未填写'}
            **过敏史**: {baby.get('allergy') or '无'}
            **备注**: {baby.get('notes') or '无'}
            """)
    else:
        st.warning("暂无宝宝档案，在对话中告诉助手名字即可自动创建。")

    st.divider()
    st.subheader("⚙️ 会话")
    st.caption(f"`{st.session_state.thread_id}`")
    if st.button("🔄 重置对话", use_container_width=True):
        reset_session()
        st.rerun()

    st.divider()
    st.subheader("ℹ️ 关于")
    st.caption("RAG + Agent + Memory 育儿知识问答\n- BM25 + 向量混合检索\n- LangGraph 状态图\n- 情景 + 语义记忆\n- 输出层安全检测")


# ── 主聊天区 ─────────────────────────────────────────────────────
st.title("🧒 育儿智能助手 Demo")

# 欢迎消息
if not st.session_state.display_messages:
    with st.chat_message("assistant"):
        st.markdown("""
        你好！我是育儿智能助手。

        **开始使用：** 告诉我宝宝的名字、月龄等基本信息，我会自动创建档案。
        例如：「宝宝叫小豆子，3个月，无过敏」
        """)

# 历史消息
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        if msg.get("thinking"):
            with st.expander("💭 思考过程", expanded=False):
                st.text(msg["thinking"][:800])
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                label = TOOL_LABELS.get(tc["name"], tc["name"])
                with st.expander(f"🔧 {label}", expanded=False):
                    st.caption("**参数**")
                    st.json(tc.get("args", {}))
                    if tc.get("result"):
                        st.caption("**检索结果**")
                        st.markdown(tc["result"])
        if msg.get("content"):
            st.markdown(msg["content"])


# ── 输入处理（流式实时展示）─────────────────────────────────────
if prompt := st.chat_input("请输入育儿问题..."):
    st.session_state.display_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    thread_config = {
        "configurable": {
            "thread_id": st.session_state.thread_id,
            "stream_callback": None,  # will be set below
        }
    }

    # 准备图输入
    if st.session_state.first_turn:
        invoke_input = {
            "messages": [HumanMessage(content=prompt)],
            "current_baby_id": None,
            "baby_profile": None,
        }
        st.session_state.first_turn = False
    else:
        invoke_input = {"messages": [HumanMessage(content=prompt)]}

    # ── 实时流式展示区域 ──
    with st.chat_message("assistant"):
        # 阶段指示器
        phase_placeholder = st.empty()

        # 思考过程折叠区
        thinking_expander = st.expander("💭 思考过程", expanded=True)
        thinking_placeholder = thinking_expander.empty()

        # 工具调用区
        tool_placeholder = st.empty()
        tool_display_buf = []  # 累积展示内容，避免覆盖

        # 回复文本流
        response_placeholder = st.empty()

        # 流式状态（用 dict 避免 nonlocal 问题）
        stream_state = {
            "thinking_buf": [],
            "response_buf": [],
            "current_tool": {"name": "", "args": ""},
            "tool_calls_done": [],
            "agent_round": 0,
        }
        shown_nodes = set()

        # ── 流式回调（被 call_model 在 LLM 流式输出时调用）──
        def stream_callback(event):
            etype = event.get("event")

            if etype == "thinking":
                stream_state["thinking_buf"].append(event["text"])
                display = "".join(stream_state["thinking_buf"][-200:])
                thinking_placeholder.markdown(f"```\n{display}\n```")

            elif etype == "text":
                stream_state["response_buf"].append(event["text"])
                response_placeholder.markdown("".join(stream_state["response_buf"]))

            elif etype == "tool":
                stream_state["current_tool"]["name"] = event["name"]
                stream_state["current_tool"]["args"] = event["args"]
                label = TOOL_LABELS.get(event["name"], event["name"])
                try:
                    import json
                    args_preview = json.dumps(json.loads(event["args"]), ensure_ascii=False)
                except Exception:
                    args_preview = event["args"]
                # 累积显示：先展示工具调用参数
                entry = f"🔧 **{label}**\n```json\n{args_preview}\n```"
                if entry not in tool_display_buf:
                    tool_display_buf.append(entry)
                tool_placeholder.markdown("\n\n---\n\n".join(tool_display_buf))

            elif etype == "done":
                ct = stream_state["current_tool"]
                if ct["name"]:
                    stream_state["tool_calls_done"].append(dict(ct))
                    ct["name"] = ""
                    ct["args"] = ""

            elif etype == "error":
                phase_placeholder.error(f"❌ 调用失败: {event['text']}")

        thread_config["configurable"]["stream_callback"] = stream_callback

        # ── 使用 stream() 执行图，获取节点级事件 ──
        try:
            for chunk in app.stream(invoke_input, thread_config, stream_mode="updates"):
                for node_name, node_output in chunk.items():
                    shown_nodes.add(node_name)

                    if node_name == "agent":
                        stream_state["agent_round"] += 1
                        r = stream_state["agent_round"]
                        phase_placeholder.info(
                            f"🤖 **第 {r} 轮推理**"
                            + (" (调用工具中...)" if r == 1 else " (生成回复中...)")
                        )

                    elif node_name == "tools":
                        phase_placeholder.info("🔧 **执行工具中...**")
                        msgs = node_output.get("messages", [])
                        if not isinstance(msgs, list):
                            msgs = [msgs]
                        for m in msgs:
                            if isinstance(m, ToolMessage):
                                result_text = m.content if hasattr(m, "content") else str(m)
                                tool_name = getattr(m, "name", "unknown")
                                for tc in stream_state["tool_calls_done"]:
                                    if tc["name"] == tool_name and not tc.get("result"):
                                        tc["result"] = result_text
                                        break
                                label = TOOL_LABELS.get(tool_name, tool_name)
                                # 累积追加结果（完整展示，不截断）
                                tool_display_buf.append(
                                    f"✅ **{label} 返回结果**\n\n{result_text}"
                                )
                                tool_placeholder.markdown("\n\n---\n\n".join(tool_display_buf))

                    elif node_name == "load_profile":
                        phase_placeholder.info("🔍 正在识别宝宝档案...")
                    elif node_name == "load_memory":
                        phase_placeholder.info("🧠 正在检索历史记忆...")
                    elif node_name == "summarize":
                        phase_placeholder.info("💾 正在保存对话记忆...")

            # ── 流式完成，整理最终展示 ──
            phase_placeholder.success("✅ 完成")

            if not stream_state["thinking_buf"]:
                thinking_expander.label = "💭 思考过程（无）"

            display_entry = {
                "role": "assistant",
                "content": "".join(stream_state["response_buf"]) if stream_state["response_buf"] else "",
            }
            if stream_state["thinking_buf"]:
                display_entry["thinking"] = "".join(stream_state["thinking_buf"])
            if stream_state["tool_calls_done"]:
                display_entry["tool_calls"] = stream_state["tool_calls_done"]

            st.session_state.display_messages.append(display_entry)

        except GraphRecursionError:
            phase_placeholder.error("❌ 达到最大推理步数，请重置对话")
        except Exception as e:
            phase_placeholder.error(f"❌ 运行错误: {e}")
            logger.error(f"Graph执行失败: {e}")

    st.rerun()
