# knowledge_base.py
import os
import sys
import json
import re
from typing import List, Tuple

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from data.parenting_articles import ARTICLES
from langchain_huggingface import HuggingFaceEmbeddings
from logger import get_logger

# ── 新增：多路召回、融合、查询处理 ─────────────────────────────
from knowledge.bm25_retriever import bm25_retriever
from knowledge.fusion import fuse_and_deduplicate
from knowledge.query_processor import process_query, rewrite_query, classify_intent

logger = get_logger("knowledge")

# ── Embeddings 初始化 ────────────────────────────────────────


def _init_embeddings():
    """初始化 embeddings，带错误处理和降级方案"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("HuggingFace Embeddings 初始化成功")
        return embeddings
    except Exception as e:
        logger.critical(f"HuggingFace Embeddings 初始化失败: {e}")
        logger.critical("请确保已安装 sentence-transformers: pip install sentence-transformers")
        sys.exit(1)


embeddings = _init_embeddings()

# 文本分块器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
)

DB_DIR = "chroma_parenting_db"
KNOWLEDGE_COLLECTION = "parenting_books"

# ── 向量库单例 ────────────────────────────────────────────────

_vectorstores: dict[str, Chroma] = {}


def _get_vectorstore(collection_name: str = KNOWLEDGE_COLLECTION) -> Chroma:
    """获取向量库实例（模块级缓存，避免重复创建连接）"""
    if collection_name not in _vectorstores:
        _vectorstores[collection_name] = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
    return _vectorstores[collection_name]


# ── 安全关键词 ────────────────────────────────────────────────

SAFETY_KEYWORDS = [
    "过敏", "窒息", "中毒", "急救", "急诊", "危险", "严重", "紧急",
    "禁用", "禁止", "不能", "危险", "死亡", "抽搐", "昏迷", "呼吸困难",
    "高烧", "高烧不退", "40度", "41度", "血便", "脱水",
    "药", "药物", "吃药", "用药", "剂量", "喂药"
]

# ── 安全风险分级 ───────────────────────────────────────────────

SAFETY_RISK_PATTERNS = {
    "critical": {
        "keywords": [
            "41度", "40度", "超高烧", "抽搐", "翻白眼", "惊厥",
            "发绀", "嘴唇发紫", "甲床发紫", "手指发紫",
            "三凹征", "肋骨凹陷", "鼻翼煽动", "呼吸困难",
            "过敏性休克", "全身起疹.*嘴唇肿", "喘不上气.*蜇",
            "溺水", "误吃.*药", "误服.*药", "吃了.*降压药",
            "摔.*呕吐", "颅内", "脑震荡.*呕吐",
        ],
        "boost": 8.0,
        "expand_to": 20,
        "top_k": 6,
    },
    "high": {
        "keywords": [
            "小月龄.*发烧", "3个月.*发烧", "新生儿.*发烧",
            "黄疸.*不退", "满月.*黄疸", "眼白.*黄",
            "腹泻.*脱水", "拉肚子.*脱水", "尿布.*干",
            "摔到.*头", "头上.*包", "嗜睡", "头部外伤",
            "持续.*高烧", "发热.*3天", "反复.*烧",
            "出血点", "紫癜", "按压.*不褪色",
            "烫伤.*水泡", "开水.*烫",
            "胎动.*减少", "脐.*渗血", "脐.*臭",
            "狗咬", "猫抓", "艾灸", "中药.*退烧",
            "退烧药.*过量", "吃了.*退烧药.*又烧",
            "呛奶.*脸.*紫", "呛到.*脸.*紫",
            "来源不明.*药", "纯天然.*退烧",
        ],
        "boost": 5.0,
        "expand_to": 15,
        "top_k": 5,
    },
    "medium": {
        "keywords": [
            "发烧.*38", "疫苗.*发烧", "咳嗽.*痰",
            "流鼻涕.*黄", "眼.*出血.*奶水",
            "鸡蛋.*过敏", "化妆品.*吃",
            "趴睡", "摔.*后脑勺", "餐椅.*摔",
            "肥皂条", "便秘.*婆婆",
            "退烧药.*冰箱", "储存.*退烧药",
        ],
        "boost": 2.0,
        "expand_to": 12,
        "top_k": 4,
    },
}

SAFETY_TOPICS = {
    "发烧": ["退烧药", "物理降温", "就医指征", "高热惊厥"],
    "咳嗽": ["窒息风险", "止咳药", "蜂蜜"],
    "腹泻": ["脱水", "补液", "电解质"],
    "睡眠": ["猝死", "仰睡", "婴儿床安全"],
    "辅食": ["窒息", "过敏", "食物禁忌"],
    "疫苗": ["禁忌症", "不良反应", "接种后护理"]
}


def classify_safety_risk(query: str) -> str:
    """按 CRITICAL/HIGH/MEDIUM/LOW 分级评估查询的安全风险等级"""
    import re as _re
    query_lower = query.lower()
    for level in ["critical", "high", "medium"]:
        for pattern in SAFETY_RISK_PATTERNS[level]["keywords"]:
            if _re.search(pattern, query_lower):
                return level
    return "low"


def _is_safety_critical(query: str) -> bool:
    """检测是否为安全关键查询"""
    query_lower = query.lower()
    for keyword in SAFETY_KEYWORDS:
        if keyword in query_lower:
            return True
    return False


def _get_safety_related_queries(query: str) -> List[str]:
    """获取安全相关的扩展查询"""
    related = [query]
    query_lower = query.lower()
    for topic, related_terms in SAFETY_TOPICS.items():
        if topic in query_lower:
            for term in related_terms:
                related.append(f"{topic} {term}")
    return related


# ── 精排 ─────────────────────────────────────────────────────


def _simple_rerank(query: str, docs_with_scores: List[Tuple], top_k: int = 3) -> List:
    """
    简单重排序策略（基于向量相似度 + 关键词匹配 + 安全优先）
    保留用于 use_hybrid=False 的向后兼容路径。
    """
    query_keywords = set(re.findall(r'\w+', query.lower()))
    scored_docs = []
    seen_content = set()

    for doc, similarity_score in docs_with_scores:
        content = doc.page_content.lower()
        content_hash = hash(content[:100])
        if content_hash in seen_content:
            continue
        seen_content.add(content_hash)

        score = (similarity_score + 1) / 2 * 10
        doc_keywords = set(re.findall(r'\w+', content))
        keyword_match = len(query_keywords & doc_keywords)
        score += keyword_match * 0.5

        for safety_kw in SAFETY_KEYWORDS[:10]:
            if safety_kw in content:
                score += 3.0
                break

        scored_docs.append((score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]


def _llm_rerank(query: str, docs: List, top_k: int = 3) -> List:
    """使用 LLM 进行重排序（如果可用）"""
    try:
        from config import Config, openai_client

        if len(docs) <= top_k:
            return docs

        doc_texts = []
        for i, doc in enumerate(docs[:10]):
            title = doc.metadata.get('title', '未知')
            content = doc.page_content[:200]
            doc_texts.append(f"[{i}] 【{title}】{content}")

        docs_text = "\n".join(doc_texts)
        rerank_prompt = f"""你是一个育儿知识检索专家。用户查询："{query}"

以下是检索到的候选文档，请选出最相关的{top_k}个，按相关度排序（最相关排第一）。
只返回文档编号，格式：[0, 3, 1]

候选文档：
{docs_text}

请只返回编号列表，不要解释："""

        response = openai_client.chat_completions_create(
            model=Config.LLM_MODEL_ID,
            messages=[{"role": "user", "content": rerank_prompt}],
            temperature=1,
            timeout=10
        )

        content = response.choices[0].message.content.strip()
        numbers = re.findall(r'\d+', content)
        indices = [int(n) for n in numbers if int(n) < len(docs)]
        return [docs[i] for i in indices[:top_k]] if indices else docs[:top_k]

    except Exception as e:
        logger.warning(f"LLM重排序失败，使用简单重排序: {e}")
        return _simple_rerank(query, docs, top_k)


# ── 安全关键词后置增强 ───────────────────────────────────────


def _boost_safety_docs(docs: list[tuple], query: str, risk_level: str = "low") -> list[tuple]:
    """
    对融合后的文档按安全风险等级做后置加分。
    输入/输出: [(content, metadata, score), ...]
    """
    if risk_level == "low":
        return docs

    risk_cfg = SAFETY_RISK_PATTERNS.get(risk_level, {})
    boost = risk_cfg.get("boost", 3.0)

    boosted = []
    for content, metadata, score in docs:
        bonus = 0.0
        for safety_kw in SAFETY_KEYWORDS[:10]:
            if safety_kw in content:
                bonus += boost
                break
        boosted.append((content, metadata, score + bonus))

    boosted.sort(key=lambda x: x[2], reverse=True)
    return boosted


# ── 混合检索核心 ──────────────────────────────────────────────


def _hybrid_search(
    query: str,
    k: int = 3,
    initial_k: int = 10,
    query_variants: list[str] | None = None,
) -> list[tuple]:
    """
    多路混合检索：BM25 + 向量 → RRF 融合 → 去重。

    Args:
        query: 原始查询
        k: 最终返回文档数
        initial_k: 每路初始召回数
        query_variants: 查询改写变体列表（含原始查询）

    Returns:
        [(content, metadata, fused_score), ...] 融合后的文档
    """
    if query_variants is None:
        query_variants = [query]

    vectorstore = _get_vectorstore()
    result_lists: list[list] = []

    for variant in query_variants:
        # 路径 1：BM25 关键词检索
        bm25_results = bm25_retriever.search(variant, k=initial_k)
        if bm25_results:
            result_lists.append(bm25_results)

        # 路径 2：向量语义检索（带分数）
        try:
            vector_results = vectorstore.similarity_search_with_score(variant, k=initial_k)
            if vector_results:
                # 转为统一格式: [(content, score, metadata), ...]
                formatted = [
                    (doc.page_content, score, doc.metadata)
                    for doc, score in vector_results
                ]
                result_lists.append(formatted)
        except Exception as e:
            logger.error(f"向量检索失败 [{variant}]: {e}")

    if not result_lists:
        return []

    # RRF 融合 + Jaccard 去重
    fused = fuse_and_deduplicate(result_lists, rrf_k=60, dedup_threshold=0.80)
    return fused[:max(k * 2, initial_k)]  # 返回略多于 k 供后续安全增强


# ── 主检索接口 ────────────────────────────────────────────────


def search_knowledge(
    query: str,
    k: int = 3,
    initial_k: int = 10,
    use_llm_rerank: bool = False,
    enable_safety_boost: bool = True,
    use_hybrid: bool = True,
    llm_client=None,
) -> str:
    """
    知识检索主入口 —— 支持混合检索（多路召回 + RRF 融合）。

    Args:
        query: 查询字符串
        k: 最终返回的文档数
        initial_k: 每路召回的文档数
        use_llm_rerank: 是否使用 LLM 重排序（仅 use_hybrid=False 时生效）
        enable_safety_boost: 是否启用安全关键词检测和高召回模式
        use_hybrid: 是否使用混合检索（BM25 + 向量 + 查询改写 + RRF 融合）
        llm_client: 查询改写用的 LLM 客户端（LangChain ChatModel 或 None）

    Returns:
        格式化后的知识字符串
    """
    logger.info(f"知识库查询: query='{query}', k={k}, hybrid={use_hybrid}")

    # 阶段 0：安全风险分级
    safety_risk = classify_safety_risk(query) if enable_safety_boost else "low"
    is_safety = safety_risk != "low"
    if is_safety:
        risk_cfg = SAFETY_RISK_PATTERNS.get(safety_risk, {})
        sb = risk_cfg.get("boost", 3.0)
        expand = risk_cfg.get("expand_to", 15)
        tk = risk_cfg.get("top_k", 5)
        initial_k = max(initial_k, expand)
        k = max(k, tk)
        logger.warning(f"安全查询检测: risk={safety_risk}, boost={sb}, expand_to={expand}, top_k={tk}")

    # ── 混合检索路径 ──────────────────────────────────────
    if use_hybrid:
        # 查询改写：优先使用传入的客户端，其次尝试原生客户端，最后降级
        query_variants = [query]
        if llm_client is not None:
            query_variants = rewrite_query(query, llm_client)
        elif is_safety:
            query_variants = _get_safety_related_queries(query)
        else:
            try:
                from config import openai_client as _native_client
                query_variants = rewrite_query(query, _native_client)
            except Exception:
                pass  # 降级为仅原始查询

        # 多路检索 + 融合 + 去重
        fused_docs = _hybrid_search(
            query,
            k=k,
            initial_k=initial_k,
            query_variants=query_variants,
        )

        if not fused_docs:
            logger.warning(f"混合检索未找到结果: {query}")
            return "未找到相关育儿知识，建议咨询儿科医生。"

        # 安全增强（按风险等级）
        if enable_safety_boost:
            fused_docs = _boost_safety_docs(fused_docs, query, risk_level=safety_risk)

        # 取 top-k
        selected = fused_docs[:k]
        logger.debug(f"混合检索完成: {len(fused_docs)} 条融合后 → 返回 {len(selected)} 条")

    # ── 传统检索路径（向后兼容）───────────────────────────
    else:
        vectorstore = _get_vectorstore()

        try:
            if is_safety:
                related_queries = _get_safety_related_queries(query)
                all_docs_with_scores = []
                seen_ids = set()

                for q in related_queries[:3]:
                    docs_with_scores = vectorstore.similarity_search_with_score(q, k=initial_k // 2)
                    for doc, score in docs_with_scores:
                        doc_id = doc.metadata.get('chunk_id', 0) + hash(doc.page_content[:50])
                        if doc_id not in seen_ids:
                            all_docs_with_scores.append((doc, score))
                            seen_ids.add(doc_id)

                all_docs_with_scores.sort(key=lambda x: x[1], reverse=True)
                docs_with_scores = all_docs_with_scores[:initial_k]
            else:
                docs_with_scores = vectorstore.similarity_search_with_score(query, k=initial_k)

        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return "知识检索服务暂时不可用，建议咨询儿科医生。"

        if not docs_with_scores:
            logger.warning(f"知识库未找到结果: {query}")
            return "未找到相关育儿知识，建议咨询儿科医生。"

        # 精排
        if use_llm_rerank:
            docs_only = [doc for doc, _ in docs_with_scores]
            selected = _llm_rerank(query, docs_only, top_k=k)
        else:
            selected = _simple_rerank(query, docs_with_scores, top_k=k)

        # 转为统一格式供后续格式化
        selected = [(doc.page_content, doc.metadata, 0.0) for doc in selected]

    # ── 格式化输出（两路共用）──────────────────────────────
    results = []
    for item in selected:
        content, metadata, _score = item
        source = metadata.get("source", metadata.get("book", "未知来源"))
        title = metadata.get("title", metadata.get("chapter", ""))
        results.append(f"【{title}】（来源：{source}）\n{content}")

    safety_notice = ""
    if safety_risk == "critical":
        safety_notice = "🚨 紧急提醒：此为危急情况，请立即就医！以下知识仅供参考，不能替代专业急救。\n\n"
    elif safety_risk == "high":
        safety_notice = "⚠️ 安全提醒：此情况存在较高风险，建议尽快就医。以下信息仅供参考。\n\n"
    elif safety_risk == "medium":
        safety_notice = "⚠️ 注意：以下信息仅供参考，如有疑虑请咨询医生。\n\n"

    # 拼接查询改写信息（供 demo 展示内部检索过程）
    query_info = ""
    if use_hybrid and len(query_variants) > 1:
        variants_str = "\n".join(f"  · {v}" for v in query_variants)
        query_info = f"[检索查询改写]\n{variants_str}\n\n"

    return query_info + safety_notice + "\n\n---\n".join(results)


# ── 知识库构建 ───────────────────────────────────────────────


def build_vector_store(collection_name: str = "parenting_knowledge"):
    """构建向量数据库（首次运行）"""
    logger.info("开始构建向量数据库...")
    documents = []
    for article in ARTICLES:
        chunks = text_splitter.split_text(article["content"])
        for i, chunk in enumerate(chunks):
            documents.append({
                "page_content": chunk,
                "metadata": {
                    "source": article["source"],
                    "title": article["title"],
                    "chunk_id": i
                }
            })

    texts = [doc["page_content"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]

    if os.path.exists(DB_DIR):
        logger.info(f"加载现有向量数据库: {DB_DIR}")
        return Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings,
            collection_name=collection_name
        )
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=DB_DIR,
        collection_name=collection_name
    )

    logger.info(f"向量数据库构建完成，共存储 {len(texts)} 个文本块")
    return vectorstore


# ── 评估工具 ─────────────────────────────────────────────────


def evaluate_retrieval(test_queries: List[Tuple[str, List[str]]]) -> dict:
    """评估检索效果（离线测试）"""
    metrics = {
        "total": len(test_queries),
        "recall_at_3": 0,
        "recall_at_5": 0,
        "safety_detected": 0,
        "avg_docs_retrieved": 0
    }

    for query, expected_titles in test_queries:
        result = search_knowledge(query, k=5, initial_k=10, use_hybrid=True)
        metrics["avg_docs_retrieved"] += result.count("【")

        if _is_safety_critical(query):
            metrics["safety_detected"] += 1

        for title in expected_titles:
            if title in result:
                metrics["recall_at_5"] += 1
                break

    metrics["recall_at_5"] /= max(metrics["total"], 1)
    metrics["avg_docs_retrieved"] /= max(metrics["total"], 1)

    logger.info(f"检索评估结果: {metrics}")
    return metrics
