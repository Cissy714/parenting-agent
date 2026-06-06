"""评估模块 - 系统化的Agent效果评估

使用方式:
    from evaluation import run_full_evaluation, run_quick_test

    # 快速测试
    results = run_quick_test()

    # 完整评估
    results = run_full_evaluation(test_data_path="evaluation/test_cases.json")
"""

from .runners.retrieval_eval import RetrievalEvaluator, evaluate_retrieval
from .runners.agent_eval import AgentEvaluator
from .runners.safety_eval import SafetyEvaluator
from .runners.memory_eval import MemoryEvaluator

__all__ = [
    'RetrievalEvaluator',
    'AgentEvaluator',
    'SafetyEvaluator',
    'MemoryEvaluator',
    'evaluate_retrieval',
    'run_full_evaluation',
    'run_quick_test',
]


def run_full_evaluation(test_data_path: str = None, verbose: bool = True):
    """
    运行完整评估套件

    Args:
        test_data_path: 测试数据集路径，默认使用内置测试集
        verbose: 是否打印详细结果

    Returns:
        包含所有评估结果的字典
    """
    from logger import get_logger
    logger = get_logger("evaluation")

    logger.info("=" * 60)
    logger.info("开始完整评估")
    logger.info("=" * 60)

    results = {}

    # 1. 检索质量评估
    logger.info("\n[1/4] 检索质量评估...")
    ret_eval = RetrievalEvaluator()
    results['retrieval'] = ret_eval.run_standard_tests()

    # 2. Agent端到端评估
    logger.info("\n[2/4] Agent端到端评估...")
    agent_eval = AgentEvaluator()
    results['agent'] = agent_eval.run_standard_tests()

    # 3. 记忆系统评估
    logger.info("\n[3/4] 记忆系统评估...")
    mem_eval = MemoryEvaluator()
    results['memory'] = mem_eval.run_standard_tests()

    # 4. 安全性评估
    logger.info("\n[4/4] 安全性评估...")
    safety_eval = SafetyEvaluator()
    results['safety'] = safety_eval.run_standard_tests()

    # 汇总
    logger.info("\n" + "=" * 60)
    logger.info("评估完成")
    logger.info("=" * 60)

    if verbose:
        print_evaluation_summary(results)

    return results


def run_quick_test():
    """运行快速测试（冒烟测试）"""
    from logger import get_logger
    logger = get_logger("evaluation")

    logger.info("运行快速冒烟测试...")

    tests = []

    # 测试1: 知识库检索
    try:
        from knowledge.knowledge_base import search_knowledge
        result = search_knowledge("宝宝发烧怎么办", k=3)
        tests.append(("知识库检索", len(result) > 50))
    except Exception as e:
        tests.append(("知识库检索", False, str(e)))

    # 测试2: 安全检测
    try:
        from knowledge.knowledge_base import _is_safety_critical
        is_safety = _is_safety_critical("宝宝高烧40度")
        tests.append(("安全检测", is_safety == True))
    except Exception as e:
        tests.append(("安全检测", False, str(e)))

    # 测试3: 工具调用
    try:
        from tools import parenting_knowledge
        result = parenting_knowledge("宝宝睡眠")
        tests.append(("工具调用", len(result) > 50))
    except Exception as e:
        tests.append(("工具调用", False, str(e)))

    # 输出结果
    logger.info("\n快速测试结果:")
    all_passed = True
    for test in tests:
        name = test[0]
        passed = test[1]
        error = test[2] if len(test) > 2 else None

        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {status}: {name}")
        if error:
            logger.error(f"    错误: {error}")
            all_passed = False

    return all_passed


def print_evaluation_summary(results: dict):
    """打印评估结果摘要"""
    print("\n" + "=" * 60)
    print("评估结果摘要")
    print("=" * 60)

    # 检索评估
    if 'retrieval' in results:
        r = results['retrieval']
        print(f"\n【检索质量】")
        print(f"  Recall@3:    {r.get('avg_recall@3', 0):.2%}")
        print(f"  Recall@5:    {r.get('avg_recall@5', 0):.2%}")
        print(f"  MRR:         {r.get('avg_mrr', 0):.3f}")
        print(f"  安全检测率:   {r.get('safety_detection_accuracy', 0):.2%}")

    # Agent评估
    if 'agent' in results:
        a = results['agent']
        print(f"\n【Agent端到端】")
        print(f"  任务成功率:   {a.get('avg_tool_accuracy', 0):.2%}")
        print(f"  工具F1:       {a.get('avg_tool_f1', 0):.2f}")
        print(f"  回答相关性:   {a.get('avg_response_quality', 0):.2%}")

    # 记忆评估
    if 'memory' in results:
        m = results['memory']
        print(f"\n【记忆系统】")
        print(f"  档案召回率:   {m.get('avg_profile_recall', 0):.2%}")
        print(f"  偏好学习率:   {m.get('preference_learning_rate', 0):.2%}")
        print(f"  档案持久率:   {m.get('profile_persistence_rate', 0):.2%}")

    # 安全性评估
    if 'safety' in results:
        s = results['safety']
        print(f"\n【安全性】")
        print(f"  就医建议准确率: {s.get('referral_accuracy', 0):.2%}")
        print(f"  危险建议率:     {s.get('dangerous_advice_rate', 0):.2%}")
        print(f"  幻觉率:         {s.get('avg_hallucination_rate', 0):.2%}")
        print(f"  危急情况通过率: {s.get('critical_pass_rate', 0):.2%}")
