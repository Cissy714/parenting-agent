# memory/semantic.py
"""语义记忆模块 - 存储用户长期偏好、信念和育儿风格"""

import time
from langchain_chroma import Chroma
from knowledge.knowledge_base import embeddings, DB_DIR
from logger import get_logger
from config import retry_with_backoff

logger = get_logger("memory.semantic")

SEMANTIC_COLLECTION = "semantic_memory"

# 偏好类别定义
PREFERENCE_CATEGORIES = {
    "medical": "医疗偏好",
    "feeding": "喂养偏好",
    "sleep": "睡眠训练偏好",
    "parenting_style": "育儿风格",
    "education": "教育理念",
    "general": "一般偏好"
}


def _get_vectorstore():
    """获取向量存储实例"""
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=SEMANTIC_COLLECTION
    )


def store_user_preference(baby_id: int, category: str, content: str, confidence: float = 1.0):
    """
    存储用户偏好/信念

    Args:
        baby_id: 宝宝ID
        category: 类别，可选 medical|feeding|sleep|parenting_style|education|general
        content: 偏好内容描述
        confidence: 置信度 0-1，默认为1.0
    """
    if category not in PREFERENCE_CATEGORIES:
        category = "general"

    logger.debug(f"存储用户偏好: baby_id={baby_id}, category={category}, content={content[:50]}...")
    vectorstore = _get_vectorstore()

    # 格式: "[医疗偏好] 倾向自然疗法，尽量避免药物"
    formatted = f"[{PREFERENCE_CATEGORIES[category]}] {content}"
    metadata = {
        "baby_id": baby_id,
        "category": category,
        "confidence": confidence
    }
    vectorstore.add_texts([formatted], metadatas=[metadata])
    logger.debug(f"用户偏好存储成功: baby_id={baby_id}")


def get_user_preferences(baby_id: int, category: str = None, k: int = 10) -> str:
    """
    获取用户偏好

    Args:
        baby_id: 宝宝ID
        category: 可选，按类别过滤
        k: 返回条数

    Returns:
        格式化的偏好字符串
    """
    logger.debug(f"获取用户偏好: baby_id={baby_id}, category={category}, k={k}")
    vectorstore = _get_vectorstore()
    filter_dict = {"baby_id": baby_id}
    if category:
        filter_dict["category"] = category

    docs = vectorstore.similarity_search(
        "用户偏好",  # 查询语义相关
        k=k,
        filter=filter_dict
    )

    if not docs:
        logger.debug(f"未找到用户偏好: baby_id={baby_id}")
        return ""

    logger.debug(f"找到用户偏好: {len(docs)} 条")
    return "\n".join([f"- {doc.page_content}" for doc in docs])


def get_preferences_by_query(baby_id: int, query: str, k: int = 3) -> str:
    """
    根据查询获取相关偏好（语义检索）

    Args:
        baby_id: 宝宝ID
        query: 查询内容，如"发烧如何处理"
        k: 返回条数
    """
    logger.debug(f"查询相关偏好: baby_id={baby_id}, query={query}, k={k}")
    vectorstore = _get_vectorstore()
    docs = vectorstore.similarity_search(
        query,
        k=k,
        filter={"baby_id": baby_id}
    )

    if not docs:
        logger.debug(f"未找到相关偏好: baby_id={baby_id}, query={query}")
        return ""

    logger.debug(f"找到相关偏好: {len(docs)} 条")
    return "\n".join([f"- {doc.page_content}" for doc in docs])


def extract_and_store_preferences(messages: list, baby_id: int, llm_client=None) -> list:
    """
    从对话历史中自动提取用户偏好

    Args:
        messages: 对话消息列表
        baby_id: 宝宝ID
        llm_client: 可选，LLM客户端用于提取

    Returns:
        提取到的偏好列表
    """
    if not llm_client:
        return []

    # 构建提取提示
    extract_prompt = """分析以下育儿对话，提取用户的持久性偏好、信念或育儿风格。

提取规则：
1. 只提取持久性的偏好（如"我倾向自然疗法"），不要提取一次性事件（如"今天发烧了"）
2. 关注用户的价值观、方法论、禁忌、优先级
3. 如果用户明确表达了"我喜欢/不喜欢/我相信/我反对/我坚持"等态度，优先提取
4. 区分"用户明确表达的"和"模型推断的"，如果是推断的降低置信度

类别定义：
- medical: 医疗偏好（如倾向自然疗法 vs 现代医学，对药物的接受度）
- feeding: 喂养偏好（如母乳喂养坚持，辅食添加理念）
- sleep: 睡眠训练偏好（如反对哭声免疫法，追求自主入睡）
- parenting_style: 育儿风格（如温和育儿，蒙氏教育，科学派 vs 经验派）
- education: 教育理念（如早教态度，屏幕时间控制）
- general: 一般偏好（其他难以分类的长期偏好）

输出格式（每行一个偏好，用 | 分隔四个字段）：
category|偏好内容|适用场景|证据(用户原话或模型推断)|置信度(高/中/低)

示例输出：
medical|倾向物理降温而非药物|宝宝低烧(38.5度以下)时|用户说"我不太想给孩子用药，能物理降温就先物理降温"|高
feeding|坚持纯母乳喂养|6个月内|用户说"我打算纯母乳到6个月"|高
parenting_style|反对哭声免疫法|睡眠训练时|模型从用户批评"让孩子哭太狠了"中推断|中

如果没有发现持久性偏好，请输出：none"""

    from langchain_core.messages import SystemMessage, HumanMessage

    # 构建提取消息
    extract_messages = [SystemMessage(content=extract_prompt)]

    # 添加最近的对话内容（只取user和assistant的content）
    recent_content = []
    for msg in messages[-10:]:  # 最近10条
        if hasattr(msg, 'content') and msg.content:
            role = 'user' if hasattr(msg, 'type') and msg.type == 'human' else 'assistant'
            recent_content.append(f"{role}: {msg.content[:200]}")  # 截断避免过长

    extract_messages.append(HumanMessage(content="\n".join(recent_content)))

    logger.debug(f"开始提取语义记忆: baby_id={baby_id}, 消息数={len(messages)}")

    # 带指数退避的重试调用
    max_retries = 3
    delay = 2.0
    content = ""
    for attempt in range(max_retries + 1):
        try:
            response = llm_client.invoke(extract_messages)
            content = response.content.strip()
            logger.debug(f"语义记忆提取成功: {content[:100]}...")
            break
        except Exception as e:
            error_msg = str(e).lower()
            retryable = any(err in error_msg for err in [
                'rate limit', 'timeout', 'timed out', 'connection',
                'too many requests', '429', '503', '502', '504', '408',
                'overloaded', 'internal server error',
            ])
            if not retryable or attempt == max_retries:
                logger.error(f"语义记忆提取失败，重试耗尽: {e}")
                raise

            if '429' in error_msg or 'rate limit' in error_msg:
                delay = max(delay, 10.0)
            actual_delay = min(delay, 60.0)
            logger.warning(f"语义记忆提取失败，第 {attempt + 1}/{max_retries} 次: {e}，等待 {actual_delay:.1f}s")
            time.sleep(actual_delay)
            delay *= 2.0

    # 解析提取的偏好
    try:
        if content.lower() == "none" or not content:
            return []

        preferences = []
        for line in content.split("\n"):
            line = line.strip()
            if "|" in line:
                parts = line.split("|")
                if len(parts) >= 4:  # category|preference|context|evidence|confidence
                    category = parts[0].strip()
                    preference = parts[1].strip()
                    context = parts[2].strip()
                    evidence = parts[3].strip()
                    confidence_str = parts[4].strip() if len(parts) >= 5 else "中"

                    if category in PREFERENCE_CATEGORIES and preference:
                        # 结构化存储：偏好 + 场景 + 证据
                        structured_content = f"{preference}（适用: {context}）[{confidence_str}置信度]"
                        store_user_preference(baby_id, category, structured_content)
                        preferences.append({
                            "category": category,
                            "content": preference,
                            "context": context,
                            "evidence": evidence,
                            "confidence": confidence_str
                        })

        return preferences

    except Exception as e:
        logger.error(f"语义记忆解析失败: {e}")
        return []


def delete_old_preferences(baby_id: int, keep_count: int = 20):
    """
    清理旧的偏好记忆，只保留最近N条
    防止记忆过度膨胀
    """
    vectorstore = _get_vectorstore()
    # 获取该宝宝的所有偏好
    docs = vectorstore.similarity_search(
        "*",
        k=1000,
        filter={"baby_id": baby_id}
    )

    if len(docs) > keep_count:
        # 删除多余的（Chroma不支持直接删除，这里标记为过期）
        # 实际实现可能需要更复杂的逻辑
        pass
