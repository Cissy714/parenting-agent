import re
from typing import Literal
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from config import Config, openai_client, openai_client_raw, API_TIMEOUT, retry_with_backoff
from memory.episodic import search_episodic_memory, store_episodic_memory
from memory.semantic import (
    get_user_preferences,
    get_preferences_by_query,
    extract_and_store_preferences
)
from state import ParentingAgentState
from memory.db import find_baby_by_name, get_baby, get_all_babies
from tools import tools, manage_baby_profile
from messages import convert_to_dict_messages, convert_to_ai_message
from datetime import datetime
from logger import get_logger

logger = get_logger("nodes")

# 初始化 LangChain 模型
model = ChatOpenAI(
    model=Config.LLM_MODEL_ID,
    api_key=Config.LLM_API_KEY,
    base_url=Config.LLM_BASE_URL,
    temperature=1,
    request_timeout=120,
    max_retries=5
).bind_tools(tools)

# 工具节点
tool_node = ToolNode(tools)


def load_baby_profile(state: ParentingAgentState):
    """加载宝宝档案：优先使用 current_baby_id，否则从对话中自动识别名字"""
    baby_id = state.get("current_baby_id")
    profile = None

    if baby_id:
        profile = get_baby(baby_id)
        logger.debug(f"从 state 加载宝宝档案: baby_id={baby_id}")
    else:
        # 扫描最近3条用户消息，尝试提取已知宝宝名字
        messages = state.get("messages", [])
        known_babies = get_all_babies()
        known_names = [b['name'] for b in known_babies]

        for msg in reversed(messages):
            content = None
            role = None
            # 处理常见的消息类型
            if isinstance(msg, (tuple, list)):
                if len(msg) >= 2:
                    role, content = msg[0], msg[1]
            elif isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content")
            elif hasattr(msg, "role") and hasattr(msg, "content"):
                role = msg.role
                content = msg.content
            elif hasattr(msg, "type") and hasattr(msg, "content"):
                role = msg.type
                content = msg.content

            if role in ("user", "human") and content:
                for name in known_names:
                    if name in content:
                        baby = find_baby_by_name(name)
                        if baby:
                            baby_id = baby['id']
                            profile = baby
                            logger.info(f"从对话中识别到宝宝: {name}, baby_id={baby_id}")
                            break
                if profile:
                    break

    logger.debug(f"加载宝宝档案结果: baby_id={baby_id}, profile={profile is not None}")
    return {
        "current_baby_id": baby_id or state.get("current_baby_id"),
        "baby_profile": profile or state.get("baby_profile")
    }


# ── 输出层危险动作检测 ────────────────────────────────────────

DANGEROUS_ACTION_PATTERNS = [
    # 催吐（现代中毒急救指南禁止）
    (r"催吐", "禁止家庭催吐，误服药物应立即就医"),
    (r"刺激.*咽喉.*吐", "禁止刺激咽喉催吐"),
    (r"抠.*嗓子.*吐", "禁止抠嗓子催吐"),
    # 倒挂控水（溺水急救禁止）
    (r"倒挂", "溺水急救禁止倒挂控水，应进行心肺复苏"),
    (r"控水", "溺水急救禁止控水，应进行心肺复苏"),
    # 挑破水泡
    (r"挑破.*水泡", "烫伤水泡禁止挑破，应就医处理"),
    (r"挑破.*水疱", "烫伤水泡禁止挑破，应就医处理"),
    (r"把水泡.*挑", "烫伤水泡禁止挑破，应就医处理"),
    # 肥皂条
    (r"肥皂条.*塞|塞.*肥皂", "禁止用肥皂条塞肛，可能损伤肠道黏膜"),
    # 奶水洗眼
    (r"奶水.*洗.*眼|奶水.*冲.*眼", "禁止用奶水冲洗眼睛，可能引起感染"),
    (r"母乳.*洗.*眼|母乳.*冲.*眼", "禁止用母乳冲洗眼睛，可能引起感染"),
    # 偏方
    (r"牙膏.*烫|酱油.*烫|醋.*烫", "禁止在烫伤处涂抹牙膏/酱油/醋"),
    # 自行用药（危险情境）
    (r"可以.*(?:试试|尝试).*(?:艾灸|中药|偏方)", "禁止推荐未经证实的婴幼儿疗法"),
    (r"自行.*(?:吃药|用药|服药|喂药)", "禁止建议自行用药，应遵医嘱"),
    # 来源不明药
    (r"(?:可以|能|建议).*(?:试试|尝试).*(?:国外|代购|海淘).*(?:药|退烧)", "禁止推荐来源不明的药物"),
]

SAFE_FALLBACK_RESPONSE = (
    "根据安全规则，我无法提供该建议。涉及宝宝健康和安全的问题，"
    "请立即就医咨询专业儿科医生，或拨打120急救电话。"
    "在紧急情况下，切勿自行处理或尝试未经验证的方法。"
)


def _check_dangerous_advice(response_text: str) -> list[str]:
    """检查回复中是否包含危险动作建议，返回命中的警告列表"""
    violations = []
    for pattern, warning in DANGEROUS_ACTION_PATTERNS:
        if re.search(pattern, response_text):
            violations.append(warning)
    return violations


def _apply_output_safety_filter(
    messages: list,
    dict_messages: list,
    tools_def: list,
    state: ParentingAgentState,
) -> dict:
    """
    输出层安全拦截：检测危险建议 → 打回重生成一次 → 仍违规则替换。
    仅在非工具调用路径（最终回复）上执行。
    """
    last_msg = dict_messages[-1]
    content = last_msg.get("content", "") if isinstance(last_msg, dict) else ""

    if not content:
        return {}

    violations = _check_dangerous_advice(content)
    if not violations:
        return {}

    logger.warning(f"输出层检测到危险建议，共 {len(violations)} 条违规: {violations}")

    # 第一次违规：注入违规提示并要求重新生成
    correction_prompt = (
        f"【系统安全检查】你上一轮回复包含以下违规内容，请立即修正：\n"
        + "\n".join(f"  - {v}" for v in violations)
        + "\n\n请重新生成一个安全、正确的回答。移除所有违规建议。"
        + "\n对于危急情况，必须明确说'请立即就医'或'请拨打120'。"
    )
    dict_messages.append({"role": "user", "content": correction_prompt})

    try:
        response = openai_client.chat_completions_create(
            model=Config.LLM_MODEL_ID,
            messages=dict_messages,
            tools=tools_def,
            temperature=1,
            timeout=API_TIMEOUT,
        )
        choice = response.choices[0].message
        retry_content = choice.content or ""

        # 第二次检查
        retry_violations = _check_dangerous_advice(retry_content)
        if retry_violations:
            logger.warning(f"二次生成仍包含危险建议，启用兜底文案。违规: {retry_violations}")
            # 替换为安全兜底文案
            return {
                "messages": AIMessage(content=SAFE_FALLBACK_RESPONSE),
                "current_baby_id": state.get("current_baby_id"),
                "baby_profile": state.get("baby_profile"),
            }

        logger.info("二次生成通过安全检查")
        return {
            "messages": AIMessage(content=retry_content),
            "current_baby_id": state.get("current_baby_id"),
            "baby_profile": state.get("baby_profile"),
        }

    except Exception as e:
        logger.error(f"安全重生成失败: {e}")
        return {
            "messages": AIMessage(content=SAFE_FALLBACK_RESPONSE),
            "current_baby_id": state.get("current_baby_id"),
            "baby_profile": state.get("baby_profile"),
        }


def load_memory(state: ParentingAgentState):
    """加载记忆节点：语义检索情景记忆和用户偏好，写入 state"""
    baby_id = state.get("current_baby_id")
    messages = state.get("messages", [])

    if not baby_id:
        logger.debug("无 baby_id，跳过记忆加载")
        return {
            "episodic_context": None,
            "semantic_context": None,
        }

    # 提取最后一条用户消息作为检索锚点
    user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == 'human'):
            user_query = msg.content
            break

    if not user_query:
        logger.debug("未找到用户消息，跳过记忆加载")
        return {
            "episodic_context": None,
            "semantic_context": None,
        }

    episodic = search_episodic_memory(baby_id, user_query, k=3)
    semantic = get_preferences_by_query(baby_id, user_query, k=3)

    logger.debug(f"记忆加载完成: episodic={'hit' if episodic else 'miss'}, semantic={'hit' if semantic else 'miss'}")

    return {
        "episodic_context": episodic or None,
        "semantic_context": semantic or None,
    }


# ── 流式 LLM 调用辅助函数 ──────────────────────────────────────


def _stream_llm(dict_messages: list, tools_def: list, stream_callback=None) -> dict:
    """
    调用 LLM，支持流式和非流式两种模式。

    流式模式下通过 stream_callback 实时推送事件:
      {"event": "thinking", "text": "..."}   - 思考过程（Kimi thinking trace）
      {"event": "text", "text": "..."}        - 回复文本
      {"event": "tool", "name": "...", "args": "..."}  - 工具调用
      {"event": "done"}                       - 完成
    """
    if stream_callback is None:
        response = openai_client.chat_completions_create(
            model=Config.LLM_MODEL_ID,
            messages=dict_messages,
            tools=tools_def,
            temperature=1,
            timeout=API_TIMEOUT
        )
        choice = response.choices[0].message
        result = {"role": "assistant", "content": choice.content or ""}
        if hasattr(choice, 'reasoning_content') and choice.reasoning_content:
            result["reasoning_content"] = choice.reasoning_content
        if choice.tool_calls:
            result["tool_calls"] = [tc.model_dump() for tc in choice.tool_calls]
        return result

    # 流式模式
    accumulated = {"role": "assistant", "content": "", "reasoning_content": ""}
    tool_call_buf = {}  # index → {id, function_name, arguments}

    try:
        response = openai_client_raw.chat.completions.create(
            model=Config.LLM_MODEL_ID,
            messages=dict_messages,
            tools=tools_def,
            temperature=1,
            stream=True,
            timeout=API_TIMEOUT,
        )

        for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue

            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                accumulated["reasoning_content"] += delta.reasoning_content
                stream_callback({"event": "thinking", "text": delta.reasoning_content})

            if delta.content:
                accumulated["content"] += delta.content
                stream_callback({"event": "text", "text": delta.content})

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_call_buf:
                        tool_call_buf[idx] = {"id": "", "name": "", "arguments": ""}
                    buf = tool_call_buf[idx]
                    if tc_delta.id:
                        buf["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            buf["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            buf["arguments"] += tc_delta.function.arguments
                            stream_callback({"event": "tool", "name": buf["name"], "args": buf["arguments"]})

    except Exception as e:
        logger.error(f"流式LLM调用失败: {e}")
        stream_callback({"event": "error", "text": str(e)})
        return accumulated if accumulated["content"] else None

    stream_callback({"event": "done"})

    # 组装 tool_calls（按 index 排序）
    if tool_call_buf:
        accumulated["tool_calls"] = [
            {
                "id": buf["id"],
                "type": "function",
                "function": {"name": buf["name"], "arguments": buf["arguments"]},
            }
            for _idx, buf in sorted(tool_call_buf.items())
        ]

    return accumulated


def call_model(state: ParentingAgentState, config: RunnableConfig = None):
    """LLM 推理节点 - 使用原生 OpenAI 客户端保留 reasoning_content"""
    messages = state["messages"]
    profile = state.get("baby_profile")
    baby_id = state.get("current_baby_id")
    logger.info(f"开始模型调用: baby_id={baby_id}, 消息数={len(messages)}")

    # 提取流式回调（如有）
    stream_callback = None
    if config and config.get("configurable", {}).get("stream_callback"):
        stream_callback = config["configurable"]["stream_callback"]

    # 构建系统提示
    system_content = """你是一个专业的育儿助手，并且具备记忆能力，能为每个宝宝建立专属档案。

    【重要规则】
    1. 回答任何育儿问题前，必须先调用 parenting_knowledge 工具检索知识库。
    2. 当用户告诉你宝宝的名字（或小名）时，你必须立即调用 manage_baby_profile 工具，并将用户说的真实名字作为 name 参数。绝不能使用"宝宝"这种泛称，必须用具体的名字，例如"小豆子"、"乐乐"等。如果用户只说了"我家宝宝"，这说明他还没告诉你名字，你应该询问名字，而不是直接记录"宝宝"。
    3. 如果用户同时提供了宝宝的月龄、出生日期、过敏史等额外信息，也要一并传入工具对应的参数中（birth_date、allergy、notes）。
    4. 如果当前还没有宝宝档案，而用户问了一个需要个性化回答的问题，你应该先询问宝宝的名字、月龄、过敏史等基本信息，再给出建议。

    【安全规则 — 必须遵守】
    5. 回答涉及症状、疾病、外伤、用药的问题时，第一段必须先给出明确结论："是否需要立即就医"以及紧急程度。
    6. 以下情况属于 CRITICAL（危急），必须明确说"立即就医/拨打120/去急诊"，不能说"观察""等一等""如果加重再去"：
       - 体温超过40°C（超高热）
       - 抽搐、惊厥、翻白眼
       - 误服药物/药物过量/药物中毒
       - 头部外伤后持续呕吐或嗜睡
       - 严重过敏反应（全身皮疹+嘴唇肿胀+呼吸困难）
       - 溺水（即使已救上来）
       - 嘴唇/指甲发紫（发绀）
       - 呼吸困难伴肋骨凹陷/鼻翼煽动（三凹征）
    7. 以下行为绝对禁止推荐，即使知识库中有提及也不能建议：
       - 催吐（现代中毒急救指南明确禁止家庭催吐）
       - 倒挂控水（溺水急救禁止）
       - 挑破烫伤水泡
       - 给3个月以下婴儿自行用药（包括退烧药）
       - 用肥皂条/奶水/牙膏/酱油等偏方处理医疗问题
       - 推荐来源不明的药物（包括国外代购药、纯天然退烧药）
       - 推荐艾灸、中药等缺乏婴幼儿安全性证据的疗法
    8. 回答结构：对于有症状的问题，按以下顺序组织回答：
       a. 是否需要就医（结论前置）
       b. 就医前的紧急处理（如适用）
       c. 原因分析/科普知识
       d. 日常护理建议
    """
    if profile:
        system_content += f"\n当前宝宝档案：{profile}"
    episodic = state.get("episodic_context")
    semantic = state.get("semantic_context")
    if episodic:
        system_content += f"\n\n【与该宝宝相关的历史事件】\n{episodic}\n\n请参考这些历史事件，确保建议的一致性和延续性。"
    if semantic:
        system_content += f"\n\n【用户的育儿偏好与风格】\n{semantic}\n\n请尊重用户的偏好和价值观，调整建议的表达方式。"
    if not any(isinstance(m, SystemMessage) for m in messages):
        system_prompt = SystemMessage(content=system_content)
        messages = [system_prompt] + messages

    dict_messages = convert_to_dict_messages(messages)

    tools_def = [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.args_schema.schema() if t.args_schema else {"type": "object", "properties": {}}
            }
        } for t in tools
    ]

    result = _stream_llm(dict_messages, tools_def, stream_callback)

    if result is None:
        error_message = AIMessage(content="抱歉，我暂时无法连接到AI服务，请稍后再试。")
        return {
            "messages": error_message,
            "current_baby_id": state.get("current_baby_id"),
            "baby_profile": state.get("baby_profile")
        }

    msg_dict = {
        "role": "assistant",
        "content": result.get("content", ""),
    }
    if result.get("reasoning_content"):
        msg_dict["reasoning_content"] = result["reasoning_content"]
    if result.get("tool_calls"):
        msg_dict["tool_calls"] = result["tool_calls"]

    ai_message = convert_to_ai_message(msg_dict)
    new_baby_id = state.get("current_baby_id")
    new_profile = state.get("baby_profile")

    if ai_message.tool_calls:
        logger.info(f"模型调用了工具: {[tc['name'] for tc in ai_message.tool_calls]}")
        for tc in ai_message.tool_calls:
            if tc["name"] == "manage_baby_profile":
                baby_name = tc["args"].get("name")
                if baby_name:
                    profile = find_baby_by_name(baby_name)
                    if profile:
                        new_baby_id = profile["id"]
                        new_profile = profile
                        logger.info(f"工具更新宝宝档案: {baby_name}, baby_id={new_baby_id}")

        return {
            "messages": ai_message,
            "current_baby_id": new_baby_id,
            "baby_profile": new_profile
        }

    # 输出层安全拦截（无工具调用 = 最终回复）
    safety_result = _apply_output_safety_filter(
        messages, dict_messages, tools_def, state
    )
    if safety_result:
        return safety_result

    return {
        "messages": ai_message,
        "current_baby_id": new_baby_id,
        "baby_profile": new_profile
    }


def should_continue(state: ParentingAgentState) -> Literal["tools", "__end__"]:
    """判断是否继续执行工具"""
    last_message = state["messages"][-1]
    has_tool_calls = bool(last_message.tool_calls) if isinstance(last_message, AIMessage) else False
    return "tools" if has_tool_calls else "__end__"

def summarize_conversation(state: ParentingAgentState):
    """对话结束时生成摘要并存入记忆库"""
    messages = state["messages"]
    baby_id = state.get("current_baby_id")

    # 如果没有关联宝宝，不生成记忆
    if not baby_id:
        logger.debug("未关联宝宝，跳过后续总结")
        return {}

    logger.info(f"开始对话总结: baby_id={baby_id}")
    
    # 取最近几轮对话（避免过长）
    recent_msgs = messages[-6:]  # 最近3轮（一问一答 + 可能工具消息）
    
    # 构造结构化摘要提示
    summary_prompt = SystemMessage(content=(
        "你是一个育儿对话摘要器。请用以下JSON格式总结这轮对话：\n"
        "{\n"
        '  "topics": ["话题1", "话题2"],\n'
        '  "key_decisions": ["关键决策或给出的方案"],\n'
        '  "user_state": "用户的情绪状态或态度（如焦虑、放心、犹豫）",\n'
        '  "follow_up": ["需要后续关注的事项"],\n'
        '  "baby_snapshot": "宝宝当前信息摘要（名字、月龄、本次涉及的症状/情况）"\n'
        "}\n\n"
        "规则：\n"
        "1. 直接输出JSON，不要加```json```标记\n"
        "2. topics 只列1-3个核心话题\n"
        "3. key_decisions 只记录明确的决策或方案，不要推测\n"
        "4. follow_up 如果确实没有，输出空数组 []\n"
        "5. 整体控制在200字以内"
    ))
    summary_messages = [summary_prompt] + recent_msgs

    # 转换消息为 dict 格式
    dict_messages = convert_to_dict_messages(summary_messages)

    # 工具定义
    tools_def = [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.args_schema.schema() if t.args_schema else {"type": "object", "properties": {}}
            }
        } for t in tools
    ]

    # 使用带重试的客户端调用
    try:
        response = openai_client.chat_completions_create(
            model=Config.LLM_MODEL_ID,
            messages=dict_messages,
            tools=tools_def,
            temperature=1,
            timeout=60  # 摘要生成
        )
        # 将响应转换回 AIMessage
        choice = response.choices[0].message
        summary_text = choice.content
        # 存储情景记忆
        store_episodic_memory(baby_id, summary_text)
        logger.info(f"情景记忆存储成功: {summary_text[:50]}...")
    except Exception as e:
        logger.error(f"摘要生成失败: {e}")
        summary_text = "对话摘要生成失败"

    # 提取并存储语义记忆（用户偏好）
    extract_model = ChatOpenAI(
        model=Config.LLM_MODEL_ID,
        api_key=Config.LLM_API_KEY,
        base_url=Config.LLM_BASE_URL,
        temperature=1,
        request_timeout=60,
        max_retries=3
    )
    extracted_prefs = extract_and_store_preferences(
        recent_msgs,
        baby_id,
        llm_client=extract_model
    )
    if extracted_prefs:
        logger.info(f"语义记忆提取: 发现 {len(extracted_prefs)} 条新偏好")

    return {}  # 不修改状态