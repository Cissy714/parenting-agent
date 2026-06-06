"""
统一评估运行器 - 运行全部四项评估并输出报告
用法: python evaluation/run_all_eval.py [--skip-retrieval] [--skip-safety] [--skip-agent] [--skip-memory]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
from datetime import datetime
from logger import get_logger

logger = get_logger("evaluation.run_all")


def run_retrieval():
    """运行检索评估（章节级，不需要LLM调用）"""
    print("\n" + "=" * 60)
    print("  1/4  检索评估 (Retrieval Evaluation)")
    print("=" * 60)

    from evaluation.runners.retrieval_eval_chapter import ChapterLevelEvaluator, ChapterTestCase

    # 加载测试集
    test_path = Path(__file__).parent / "data" / "retrieval_test_set_v2_100_corrected.json"
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

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

    evaluator = ChapterLevelEvaluator()
    results = evaluator.run_tests(test_cases)

    print(f"  总用例: {results['total_tests']}")
    print(f"  recall@3: {results['avg_recall@3']:.2%}")
    print(f"  recall@5: {results['avg_recall@5']:.2%}")
    print(f"  recall@10: {results['avg_recall@10']:.2%}")
    print(f"  MRR: {results['avg_mrr']:.4f}")
    print(f"  关键词覆盖: {results['avg_keyword_coverage']:.2%}")
    print(f"  安全检测准确率: {results['safety_detection_accuracy']:.2%}")

    # 0% recall 统计
    recall_3_dist = [r['recall_metrics']['recall@3'] for r in results['detailed_results']]
    zero_count = sum(1 for v in recall_3_dist if v == 0)
    full_count = sum(1 for v in recall_3_dist if v == 1.0)
    print(f"  recall@3=0: {zero_count} 条, recall@3=100%: {full_count} 条")

    # 安全 vs 普通
    safety_results = [r for r in results['detailed_results'] if r['is_safety_critical']]
    if safety_results:
        sr3 = sum(r['recall_metrics']['recall@3'] for r in safety_results) / len(safety_results)
        print(f"  安全关键({len(safety_results)}条) recall@3: {sr3:.2%}")

    return {"dimension": "retrieval", "raw": results}


def run_safety():
    """运行安全评估（需要LLM调用）"""
    print("\n" + "=" * 60)
    print("  2/4  安全评估 (Safety Evaluation)")
    print("=" * 60)
    from evaluation.runners.safety_eval import SafetyEvaluator
    evaluator = SafetyEvaluator()
    results = evaluator.run_standard_tests()
    print(f"  总用例: {results['total_tests']}")
    print(f"  就医建议准确率: {results['referral_accuracy']:.2%}")
    print(f"  危险建议率: {results['dangerous_advice_rate']:.2%}")
    print(f"  安全提示覆盖率: {results['avg_safety_coverage']:.2%}")
    print(f"  幻觉率: {results['avg_hallucination_rate']:.2%}")
    print(f"  CRITICAL通过率: {results['critical_pass_rate']:.2%}")
    return {"dimension": "safety", "metrics": results}


def run_agent():
    """运行Agent评估（需要LLM调用）"""
    print("\n" + "=" * 60)
    print("  3/4  Agent评估 (Agent Evaluation)")
    print("=" * 60)
    from evaluation.runners.agent_eval import AgentEvaluator
    evaluator = AgentEvaluator()
    results = evaluator.run_standard_tests()
    print(f"  总用例: {results['total_tests']}")
    print(f"  成功执行: {results['successful_tests']}")
    print(f"  工具准确率: {results['avg_tool_accuracy']:.2%}")
    print(f"  工具F1: {results['avg_tool_f1']:.2%}")
    print(f"  回答质量: {results['avg_response_quality']:.2%}")
    print(f"  禁止词违规: {results['forbidden_violations']} 条")
    print(f"  档案正确率: {results['profile_correct_rate']:.2%}")
    return {"dimension": "agent", "metrics": results}


def run_memory():
    """运行记忆评估（需要LLM调用）"""
    print("\n" + "=" * 60)
    print("  4/4  记忆评估 (Memory Evaluation)")
    print("=" * 60)
    from evaluation.runners.memory_eval import MemoryEvaluator
    evaluator = MemoryEvaluator()
    results = evaluator.run_standard_tests()
    print(f"  总用例: {results['total_tests']}")
    print(f"  档案回忆率: {results['avg_profile_recall']:.2%}")
    print(f"  偏好学习率: {results['preference_learning_rate']:.2%}")
    print(f"  档案持久化率: {results['profile_persistence_rate']:.2%}")
    return {"dimension": "memory", "metrics": results}


def main():
    skip = set()
    for arg in sys.argv[1:]:
        if arg.startswith("--skip-"):
            skip.add(arg.replace("--skip-", ""))

    start_time = time.time()
    print(f"\n{'#' * 60}")
    print(f"  育儿Agent四维评估")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 60}")

    all_results = []

    # 1. Retrieval (no LLM needed, fastest)
    if "retrieval" not in skip:
        try:
            all_results.append(run_retrieval())
        except Exception as e:
            print(f"  FAILED: {e}")
            all_results.append({"dimension": "retrieval", "error": str(e)})

    # 2. Safety
    if "safety" not in skip:
        try:
            all_results.append(run_safety())
        except Exception as e:
            print(f"  FAILED: {e}")
            all_results.append({"dimension": "safety", "error": str(e)})

    # 3. Agent
    if "agent" not in skip:
        try:
            all_results.append(run_agent())
        except Exception as e:
            print(f"  FAILED: {e}")
            all_results.append({"dimension": "agent", "error": str(e)})

    # 4. Memory
    if "memory" not in skip:
        try:
            all_results.append(run_memory())
        except Exception as e:
            print(f"  FAILED: {e}")
            all_results.append({"dimension": "memory", "error": str(e)})

    elapsed = time.time() - start_time

    # Print summary
    print(f"\n{'#' * 60}")
    print(f"  评估总结")
    print(f"{'#' * 60}")
    print(f"  耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print()

    for r in all_results:
        dim = r["dimension"]
        if "error" in r:
            print(f"  [{dim}] ERROR: {r['error']}")
        elif dim == "retrieval":
            m = r["raw"]
            print(f"  [{dim}] recall@3={m['avg_recall@3']:.2%}  recall@5={m['avg_recall@5']:.2%}  MRR={m['avg_mrr']:.3f}  kw_cov={m['avg_keyword_coverage']:.2%}")
        else:
            m = r["metrics"]
            print(f"  [{dim}] ", end="")
            if dim == "safety":
                print(f"referral_acc={m['referral_accuracy']:.2%}  danger_rate={m['dangerous_advice_rate']:.2%}  critical_pass={m['critical_pass_rate']:.2%}")
            elif dim == "agent":
                print(f"tool_f1={m['avg_tool_f1']:.2%}  quality={m['avg_response_quality']:.2%}  profile={m['profile_correct_rate']:.2%}")
            elif dim == "memory":
                print(f"recall={m['avg_profile_recall']:.2%}  preference={m['preference_learning_rate']:.2%}  persist={m['profile_persistence_rate']:.2%}")

    # Save detailed report
    report_path = Path(__file__).parent / "reports" / f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n详细报告已保存: {report_path}")


if __name__ == "__main__":
    main()
