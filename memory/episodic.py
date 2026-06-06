# memory/episodic.py
from langchain_chroma import Chroma
from knowledge.knowledge_base import embeddings, DB_DIR   # 复用嵌入模型和数据库目录
from datetime import datetime
from logger import get_logger

logger = get_logger("memory.episodic")

EPISODIC_COLLECTION = "episodic_memory"


def store_episodic_memory(baby_id, summary):
    """存储情景记忆"""
    logger.debug(f"存储情景记忆: baby_id={baby_id}, summary={summary[:50]}...")
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=EPISODIC_COLLECTION
    )
    metadata = {"baby_id": baby_id, "timestamp": datetime.now().isoformat()}
    vectorstore.add_texts([summary], metadatas=[metadata])
    logger.debug(f"情景记忆存储成功: baby_id={baby_id}")


def search_episodic_memory(baby_id, query, k=3):
    """搜索情景记忆"""
    logger.debug(f"搜索情景记忆: baby_id={baby_id}, query={query}")
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=EPISODIC_COLLECTION
    )
    docs = vectorstore.similarity_search(query, k=k, filter={"baby_id": baby_id})
    if not docs:
        logger.debug(f"未找到情景记忆: baby_id={baby_id}")
        return ""
    results = [doc.page_content for doc in docs]
    logger.debug(f"找到情景记忆: {len(docs)} 条")
    return "\n---\n".join(results)
