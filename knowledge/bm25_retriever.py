"""
BM25 关键词检索器 —— 精确匹配召回路径

与向量检索互补：
- 向量检索擅长语义近似（"宝宝发烧" ≈ "婴儿体温升高"）
- BM25 擅长精确匹配（"对乙酰氨基酚"、"轮状病毒"等专有名词）

单例模式：首次搜索时自动从 ChromaDB 构建索引并缓存。
"""

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma

from logger import get_logger

logger = get_logger("knowledge.bm25")

# 导入 embeddings 和 DB_DIR（延迟导入避免循环引用）
_embeddings = None
_DB_DIR = "chroma_parenting_db"


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        from knowledge.knowledge_base import embeddings as _e
        _embeddings = _e
    return _embeddings


class BM25Retriever:
    """BM25 关键词检索器（单例）"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.corpus_texts: list[str] = []
        self.corpus_metadatas: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] | None = None
        self._built = False

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """中文分词，过滤空白"""
        return [w.strip() for w in jieba.cut(text) if w.strip()]

    def build_from_chroma(self, collection_name: str = "parenting_books", force_rebuild: bool = False):
        """从 ChromaDB 读取全部文档并构建 BM25 索引"""
        if self._built and not force_rebuild:
            logger.debug("BM25 索引已存在，跳过构建")
            return

        logger.info(f"从 ChromaDB 构建 BM25 索引，collection={collection_name}...")

        try:
            vectorstore = Chroma(
                persist_directory=_DB_DIR,
                embedding_function=_get_embeddings(),
                collection_name=collection_name,
            )
            result = vectorstore.get(include=["documents", "metadatas"])
        except Exception as e:
            logger.error(f"无法读取 ChromaDB: {e}")
            self._built = True  # 标记为已尝试，避免反复失败
            return

        if not result or not result.get("documents"):
            logger.warning("ChromaDB 中无文档，BM25 索引为空")
            self._built = True
            return

        self.corpus_texts = result["documents"]
        self.corpus_metadatas = result["metadatas"] or [{}] * len(self.corpus_texts)

        # 分词 + 建索引
        self._tokenized_corpus = [self.tokenize(text) for text in self.corpus_texts]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        self._built = True

        logger.info(f"BM25 索引构建完成: {len(self.corpus_texts)} 个文档")

    def search(self, query: str, k: int = 10) -> list[tuple[str, float, dict]]:
        """
        BM25 检索

        Returns:
            [(doc_content, bm25_score, metadata), ...] 按分数降序排列
        """
        if not self._bm25:
            self.build_from_chroma()
        if not self._bm25:
            return []

        tokens = self.tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        # scores 是 numpy array，值越大越相关
        # 归一化到 [0, 1] 便于后续融合
        max_score = np.max(scores) if len(scores) > 0 else 1.0
        if max_score > 0:
            scores = scores / max_score

        # 取 top-k
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 只返回有匹配的
                results.append((
                    self.corpus_texts[idx],
                    float(scores[idx]),
                    self.corpus_metadatas[idx],
                ))

        return results


# 全局单例
bm25_retriever = BM25Retriever()
