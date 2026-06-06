"""
多路召回结果融合

算法：
1. RRF (Reciprocal Rank Fusion) —— 不依赖原始分数量纲，基于排名融合多路结果
2. Jaccard 去重 —— 基于字符 n-gram 的文本相似度去重
"""

from logger import get_logger

logger = get_logger("knowledge.fusion")

# ── 文档标识 ─────────────────────────────────────────────────


def _make_doc_id(content: str) -> int:
    """基于内容前 100 字符生成文档 ID（用于跨路径去重）"""
    return hash(content[:100])


# ── RRF 融合 ─────────────────────────────────────────────────


def rrf_fusion(result_lists: list[list], k: int = 60) -> list[tuple]:
    """
    Reciprocal Rank Fusion —— 融合多路检索结果。

    RRF 公式：score(d) = Σ_i 1 / (k + rank_i(d))

    Args:
        result_lists: 多路检索结果。
            每路格式: [(content, score, metadata), ...]
            content 为文本，score 为该路内部的分数，metadata 为 dict。
            列表按相关性降序排列（最佳排第一，rank=1）。
        k: RRF 参数，默认 60（经典值）。

    Returns:
        [(content, metadata, rrf_score), ...] 按 RRF 分数降序排列。
    """
    if not result_lists:
        return []

    # 检测输入格式：LangChain Document 对象 vs 纯文本
    scores: dict[int, tuple[str, dict, float]] = {}  # doc_id -> (content, metadata, accumulated_rrf)

    for results in result_lists:
        if not results:
            continue

        for rank, item in enumerate(results, start=1):
            # 兼容两种输入格式
            if hasattr(item, "page_content"):
                # LangChain Document 对象（来自向量检索）
                content = item.page_content
                metadata = item.metadata if hasattr(item, "metadata") else {}
            elif isinstance(item, (tuple, list)) and len(item) >= 1:
                # 元组格式 (content, score, metadata) 或 (content, metadata)
                content = item[0]
                metadata = item[2] if len(item) >= 3 else (item[1] if len(item) == 2 and isinstance(item[1], dict) else {})
            else:
                continue

            doc_id = _make_doc_id(content)
            rrf_contribution = 1.0 / (k + rank)

            if doc_id in scores:
                prev_content, prev_meta, prev_score = scores[doc_id]
                # 保留原始分数更高的 metadata（通常是向量检索的更可靠）
                scores[doc_id] = (prev_content, prev_meta, prev_score + rrf_contribution)
            else:
                scores[doc_id] = (content, metadata, rrf_contribution)

    # 按 RRF 分数降序排列
    sorted_docs = sorted(scores.values(), key=lambda x: x[2], reverse=True)

    if not sorted_docs:
        return []

    logger.debug(
        f"RRF 融合: {sum(len(r) for r in result_lists if r)} 条输入 → "
        f"{len(sorted_docs)} 条去重输出, "
        f"top-1 分数={sorted_docs[0][2]:.4f}"
    )
    return sorted_docs


# ── Jaccard 去重 ─────────────────────────────────────────────


def _char_ngrams(text: str, n: int = 3) -> frozenset[str]:
    """生成字符 n-gram 集合，截取前 600 字符加速"""
    truncated = text[:600]
    if len(truncated) < n:
        return frozenset([truncated])
    return frozenset(truncated[i:i + n] for i in range(len(truncated) - n + 1))


def _jaccard_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def deduplicate_docs(
    docs: list[tuple],
    threshold: float = 0.80,
) -> list[tuple]:
    """
    基于 Jaccard 字符 n-gram 相似度的文档去重。

    保留 RRF 分数更高的文档（假设输入已按分数降序排列）。

    Args:
        docs: [(content, metadata, score), ...] 已按分数降序排列
        threshold: 相似度阈值，超过此值视为重复

    Returns:
        去重后的文档列表
    """
    if len(docs) <= 1:
        return docs

    kept: list[tuple] = []
    kept_ngrams: list[frozenset[str]] = []

    for doc_tuple in docs:
        content = doc_tuple[0]
        ngrams = _char_ngrams(content)

        is_dup = False
        for existing_ngrams in kept_ngrams:
            if _jaccard_similarity(ngrams, existing_ngrams) > threshold:
                is_dup = True
                break

        if not is_dup:
            kept.append(doc_tuple)
            kept_ngrams.append(ngrams)

    if len(kept) < len(docs):
        logger.debug(f"Jaccard 去重: {len(docs)} → {len(kept)} (threshold={threshold})")

    return kept


# ── 融合入口 ─────────────────────────────────────────────────


def fuse_and_deduplicate(
    result_lists: list[list],
    rrf_k: int = 60,
    dedup_threshold: float = 0.80,
) -> list[tuple]:
    """
    RRF 融合 + Jaccard 去重 的一站式入口。

    Args:
        result_lists: 多路检索结果列表
        rrf_k: RRF 参数
        dedup_threshold: 去重阈值

    Returns:
        [(content, metadata, score), ...] 融合去重后的结果
    """
    fused = rrf_fusion(result_lists, k=rrf_k)
    deduped = deduplicate_docs(fused, threshold=dedup_threshold)
    return deduped
