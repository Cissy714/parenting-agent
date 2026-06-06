"""
章节级检索评估（粗粒度）

匹配规则：只要 retrieved chunk 的章节在 expected_chapters 列表中，就算命中
"""

import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from logger import get_logger

logger = get_logger("evaluation.retrieval_chapter")


@dataclass
class ChapterTestCase:
    """章节级测试用例"""
    query: str
    expected_source: str  # 期望来源书名
    expected_chapters: List[str]  # 期望章节关键词列表
    expected_keywords: List[str]
    is_safety_critical: bool = False


class ChapterLevelEvaluator:
    """章节级检索评估器"""

    def __init__(self):
        self.results = []

    def evaluate_recall(self, results: List[Any], expected_chapters: List[str]) -> Dict[str, float]:
        """
        计算章节级召回率

        只要 retrieved chunk 的章节名包含任意 expected_chapter 子串，就算命中
        """
        if not expected_chapters:
            return {"recall@k": 0.0, "precision@k": 0.0, "f1@k": 0.0}

        metrics = {}
        for k in [3, 5, 10]:
            top_k_results = results[:k]
            hits = 0

            for doc in top_k_results:
                chapter = doc.metadata.get('chapter', '')
                source = doc.metadata.get('source', '')
                # 检查是否匹配期望的任意章节
                for exp_ch in expected_chapters:
                    if exp_ch in chapter or exp_ch in source:
                        hits += 1
                        break

            recall = hits / min(k, len(top_k_results)) if top_k_results else 0.0
            precision = hits / k if k > 0 else 0.0
            f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0

            metrics[f"recall@{k}"] = recall
            metrics[f"precision@{k}"] = precision
            metrics[f"f1@{k}"] = f1

        return metrics

    def evaluate_mrr(self, results: List[Any], expected_chapters: List[str]) -> float:
        """计算章节级 MRR"""
        if not expected_chapters:
            return 0.0

        for rank, doc in enumerate(results, 1):
            chapter = doc.metadata.get('chapter', '')
            source = doc.metadata.get('source', '')
            for exp_ch in expected_chapters:
                if exp_ch in chapter or exp_ch in source:
                    return 1.0 / rank
        return 0.0

    def evaluate_keyword_coverage(self, result_text: str, expected_keywords: List[str]) -> Dict[str, Any]:
        """评估关键词覆盖率"""
        result_lower = result_text.lower()
        covered = []
        missed = []

        for kw in expected_keywords:
            if kw.lower() in result_lower:
                covered.append(kw)
            else:
                missed.append(kw)

        coverage = len(covered) / len(expected_keywords) if expected_keywords else 0.0

        return {
            "coverage": coverage,
            "covered_keywords": covered,
            "missed_keywords": missed
        }

    def run_single_test(self, test_case: ChapterTestCase) -> Dict[str, Any]:
        """运行单个测试用例"""
        logger.info(f"测试: '{test_case.query[:40]}...'")

        from knowledge.knowledge_base import search_knowledge, _is_safety_critical
        from knowledge.knowledge_base import embeddings, DB_DIR
        from langchain_chroma import Chroma

        # 执行检索
        result_text = search_knowledge(test_case.query, k=5, initial_k=10)

        # 获取原始结果
        vectorstore = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings,
            collection_name="parenting_books"
        )
        raw_results = vectorstore.similarity_search(test_case.query, k=10)

        # 计算指标
        recall_metrics = self.evaluate_recall(raw_results, test_case.expected_chapters)
        mrr = self.evaluate_mrr(raw_results, test_case.expected_chapters)
        keyword_eval = self.evaluate_keyword_coverage(result_text, test_case.expected_keywords)

        is_detected = _is_safety_critical(test_case.query)
        safety_correct = (is_detected == test_case.is_safety_critical)

        return {
            "query": test_case.query,
            "is_safety_critical": test_case.is_safety_critical,
            "recall_metrics": recall_metrics,
            "mrr": mrr,
            "keyword_coverage": keyword_eval,
            "safety_detection_correct": safety_correct,
            "result_length": len(result_text)
        }

    def run_tests(self, test_cases: List[ChapterTestCase]) -> Dict[str, Any]:
        """运行全部测试"""
        logger.info(f"运行章节级检索评估，共 {len(test_cases)} 个测试用例")

        all_results = []
        for test_case in test_cases:
            result = self.run_single_test(test_case)
            all_results.append(result)

        # 汇总
        summary = {
            "total_tests": len(all_results),
            "avg_recall@3": sum(r["recall_metrics"].get("recall@3", 0) for r in all_results) / len(all_results),
            "avg_recall@5": sum(r["recall_metrics"].get("recall@5", 0) for r in all_results) / len(all_results),
            "avg_recall@10": sum(r["recall_metrics"].get("recall@10", 0) for r in all_results) / len(all_results),
            "avg_mrr": sum(r["mrr"] for r in all_results) / len(all_results),
            "avg_keyword_coverage": sum(r["keyword_coverage"]["coverage"] for r in all_results) / len(all_results),
            "safety_detection_accuracy": sum(r["safety_detection_correct"] for r in all_results) / len(all_results),
            "detailed_results": all_results
        }

        logger.info(f"评估完成: recall@3={summary['avg_recall@3']:.2%}, "
                   f"recall@5={summary['avg_recall@5']:.2%}, MRR={summary['avg_mrr']:.3f}")

        return summary


def main():
    """主函数：运行评估"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    # 加载测试集
    test_path = Path(__file__).parent.parent / "data" / "retrieval_test_set_v2_100_corrected.json"
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 构造测试用例
    test_cases = []
    for item in test_data:
        tc = ChapterTestCase(
            query=item['query'],
            expected_source=item['expected_source'],
            expected_chapters=item['expected_chapters'],
            expected_keywords=item['expected_keywords'],
            is_safety_critical=item['is_safety_critical']
        )
        test_cases.append(tc)

    # 运行评估
    evaluator = ChapterLevelEvaluator()
    results = evaluator.run_tests(test_cases)

    # 打印报告
    print("\n" + "="*60)
    print("章节级检索评估报告 (94 queries)")
    print("="*60)
    print(f"总测试数: {results['total_tests']}")
    print(f"平均 recall@3:  {results['avg_recall@3']:.2%}")
    print(f"平均 recall@5:  {results['avg_recall@5']:.2%}")
    print(f"平均 recall@10: {results['avg_recall@10']:.2%}")
    print(f"平均 MRR:       {results['avg_mrr']:.3f}")
    print(f"关键词覆盖率:   {results['avg_keyword_coverage']:.2%}")
    print(f"安全检测准确率: {results['safety_detection_accuracy']:.2%}")

    # 分组统计
    safety_results = [r for r in results['detailed_results'] if r['is_safety_critical']]
    normal_results = [r for r in results['detailed_results'] if not r['is_safety_critical']]

    if safety_results:
        print(f"\n安全关键 ({len(safety_results)}条):")
        print(f"  recall@3: {sum(r['recall_metrics']['recall@3'] for r in safety_results)/len(safety_results):.2%}")
        print(f"  MRR:      {sum(r['mrr'] for r in safety_results)/len(safety_results):.3f}")

    if normal_results:
        print(f"\n普通查询 ({len(normal_results)}条):")
        print(f"  recall@3: {sum(r['recall_metrics']['recall@3'] for r in normal_results)/len(normal_results):.2%}")
        print(f"  MRR:      {sum(r['mrr'] for r in normal_results)/len(normal_results):.3f}")

    # 分布
    recall_3_dist = [r['recall_metrics']['recall@3'] for r in results['detailed_results']]
    print(f"\nRecall@3 分布:")
    print(f"  0%:   {sum(1 for v in recall_3_dist if v == 0)} 条")
    print(f"  33%:  {sum(1 for v in recall_3_dist if 0.3 < v <= 0.4)} 条")
    print(f"  50%:  {sum(1 for v in recall_3_dist if 0.4 < v <= 0.6)} 条")
    print(f"  67%:  {sum(1 for v in recall_3_dist if 0.6 < v <= 0.7)} 条")
    print(f"  100%: {sum(1 for v in recall_3_dist if v == 1.0)} 条")

    # 保存详细结果
    output_path = Path(__file__).parent.parent / "data" / "retrieval_eval_chapter_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存: {output_path}")


if __name__ == "__main__":
    main()
