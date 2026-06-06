"""检索质量评估模块"""

from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from logger import get_logger

logger = get_logger("evaluation.retrieval")


@dataclass
class RetrievalTestCase:
    """检索测试用例"""
    query: str
    expected_doc_ids: List[str]  # 期望召回的文档ID
    expected_keywords: List[str]  # 期望包含的关键词
    is_safety_critical: bool = False  # 是否安全关键查询
    description: str = ""  # 测试描述


class RetrievalEvaluator:
    """检索质量评估器"""

    # 标准测试集
    STANDARD_TESTS = [
        RetrievalTestCase(
            query="宝宝发烧怎么办",
            expected_doc_ids=[],
            expected_keywords=["发烧", "体温", "退烧"],
            is_safety_critical=True,
            description="基础疾病查询"
        ),
        RetrievalTestCase(
            query="母乳喂养注意事项",
            expected_doc_ids=[],
            expected_keywords=["母乳", "喂养", "哺乳"],
            is_safety_critical=False,
            description="喂养指导查询"
        ),
        RetrievalTestCase(
            query="宝宝高烧40度可以吃什么药",
            expected_doc_ids=[],
            expected_keywords=["发烧", "药物", "用药", "剂量"],
            is_safety_critical=True,
            description="安全关键-用药查询"
        ),
        RetrievalTestCase(
            query="辅食添加时间",
            expected_doc_ids=[],
            expected_keywords=["辅食", "添加", "6个月"],
            is_safety_critical=False,
            description="辅食查询"
        ),
        RetrievalTestCase(
            query="宝宝睡眠不好怎么办",
            expected_doc_ids=[],
            expected_keywords=["睡眠", "哭闹", "哄睡"],
            is_safety_critical=False,
            description="睡眠问题查询"
        ),
        RetrievalTestCase(
            query="宝宝误食了药物怎么办",
            expected_doc_ids=[],
            expected_keywords=["误食", "中毒", "急救", "就医"],
            is_safety_critical=True,
            description="紧急情况-误食处理"
        ),
    ]

    def __init__(self):
        self.results = []

    def evaluate_recall(self, results: List[Any], expected_ids: List[str]) -> Dict[str, float]:
        """
        计算召回率指标

        Args:
            results: 检索结果文档列表
            expected_ids: 期望的文档ID列表

        Returns:
            {"recall@k": float, "precision@k": float, "f1@k": float}
        """
        if not expected_ids:
            return {"recall@k": 0.0, "precision@k": 0.0, "f1@k": 0.0}

        result_ids = [str(getattr(r, 'id', r.metadata.get('chunk_id', i))) for i, r in enumerate(results)]

        # 计算不同k值的召回率
        metrics = {}
        for k in [3, 5, 10]:
            top_k_results = result_ids[:k]
            hits = len(set(top_k_results) & set(expected_ids))

            recall = hits / len(expected_ids) if expected_ids else 0.0
            precision = hits / k if k > 0 else 0.0
            f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0

            metrics[f"recall@{k}"] = recall
            metrics[f"precision@{k}"] = precision
            metrics[f"f1@{k}"] = f1

        return metrics

    def evaluate_mrr(self, results: List[Any], expected_ids: List[str]) -> float:
        """
        计算MRR (Mean Reciprocal Rank)

        第一个相关文档的排名倒数
        """
        if not expected_ids:
            return 0.0

        result_ids = [str(getattr(r, 'id', r.metadata.get('chunk_id', i))) for i, r in enumerate(results)]

        for rank, doc_id in enumerate(result_ids, 1):
            if doc_id in expected_ids:
                return 1.0 / rank

        return 0.0

    def evaluate_keyword_coverage(self, result_text: str, expected_keywords: List[str]) -> Dict[str, Any]:
        """
        评估关键词覆盖率

        检查检索结果中是否包含期望的关键词
        """
        result_lower = result_text.lower()
        covered = []
        missed = []

        for keyword in expected_keywords:
            if keyword.lower() in result_lower:
                covered.append(keyword)
            else:
                missed.append(keyword)

        coverage = len(covered) / len(expected_keywords) if expected_keywords else 0.0

        return {
            "coverage": coverage,
            "covered_keywords": covered,
            "missed_keywords": missed
        }

    def evaluate_safety_detection(self, query: str, is_detected: bool) -> bool:
        """评估安全检测是否正确"""
        from knowledge.knowledge_base import _is_safety_critical

        expected = _is_safety_critical(query)
        correct = (expected == is_detected)

        if not correct:
            logger.warning(f"安全检测偏差: query='{query}', expected={expected}, got={is_detected}")

        return correct

    def run_single_test(self, test_case: RetrievalTestCase) -> Dict[str, Any]:
        """运行单个测试用例"""
        logger.info(f"测试: {test_case.description} - '{test_case.query}'")

        from knowledge.knowledge_base import search_knowledge, _is_safety_critical

        # 执行检索
        result_text = search_knowledge(test_case.query, k=5, initial_k=10)

        # 获取带分数的原始结果用于指标计算
        # 注意：这里需要直接调用向量检索
        from knowledge.knowledge_base import embeddings, DB_DIR
        from langchain_chroma import Chroma

        vectorstore = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings,
            collection_name="parenting_books"
        )
        raw_results = vectorstore.similarity_search_with_score(test_case.query, k=10)

        # 计算指标
        results = [doc for doc, _ in raw_results]

        recall_metrics = self.evaluate_recall(results, test_case.expected_doc_ids)
        mrr = self.evaluate_mrr(results, test_case.expected_doc_ids)
        keyword_eval = self.evaluate_keyword_coverage(result_text, test_case.expected_keywords)

        # 安全检测评估
        is_detected = _is_safety_critical(test_case.query)
        safety_correct = self.evaluate_safety_detection(test_case.query, is_detected)

        test_result = {
            "query": test_case.query,
            "description": test_case.description,
            "is_safety_critical": test_case.is_safety_critical,
            "recall_metrics": recall_metrics,
            "mrr": mrr,
            "keyword_coverage": keyword_eval,
            "safety_detection_correct": safety_correct,
            "result_length": len(result_text)
        }

        logger.debug(f"测试结果: recall@3={recall_metrics.get('recall@3', 0):.2%}, "
                    f"MRR={mrr:.3f}, keyword_coverage={keyword_eval['coverage']:.2%}")

        return test_result

    def run_standard_tests(self) -> Dict[str, Any]:
        """运行标准测试集"""
        logger.info(f"运行检索评估，共 {len(self.STANDARD_TESTS)} 个测试用例")

        all_results = []
        for test_case in self.STANDARD_TESTS:
            result = self.run_single_test(test_case)
            all_results.append(result)

        # 汇总指标
        summary = {
            "total_tests": len(all_results),
            "avg_recall@3": sum(r["recall_metrics"].get("recall@3", 0) for r in all_results) / len(all_results),
            "avg_recall@5": sum(r["recall_metrics"].get("recall@5", 0) for r in all_results) / len(all_results),
            "avg_mrr": sum(r["mrr"] for r in all_results) / len(all_results),
            "avg_keyword_coverage": sum(r["keyword_coverage"]["coverage"] for r in all_results) / len(all_results),
            "safety_detection_accuracy": sum(r["safety_detection_correct"] for r in all_results) / len(all_results),
            "detailed_results": all_results
        }

        logger.info(f"检索评估完成: recall@3={summary['avg_recall@3']:.2%}, "
                   f"recall@5={summary['avg_recall@5']:.2%}, MRR={summary['avg_mrr']:.3f}")

        return summary


def evaluate_retrieval(test_queries: List[Tuple[str, List[str]]] = None) -> Dict[str, Any]:
    """
    快速评估检索效果（兼容旧接口）

    Args:
        test_queries: [(查询, [相关文档ID列表]), ...]

    Returns:
        评估结果字典
    """
    evaluator = RetrievalEvaluator()

    if test_queries:
        # 使用自定义测试集
        test_cases = [
            RetrievalTestCase(
                query=q,
                expected_doc_ids=ids,
                expected_keywords=[],
                is_safety_critical=False
            )
            for q, ids in test_queries
        ]
        evaluator.STANDARD_TESTS = test_cases

    return evaluator.run_standard_tests()
