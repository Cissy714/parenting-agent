"""两阶段检索系统测试脚本"""

from knowledge.knowledge_base import (
    search_knowledge,
    _is_safety_critical,
    _get_safety_related_queries,
    evaluate_retrieval
)
from logger import get_logger

logger = get_logger("test")


def test_safety_detection():
    """测试安全关键词检测"""
    print("\n" + "=" * 60)
    print("测试1: 安全关键词检测")
    print("=" * 60)

    test_cases = [
        ("宝宝发烧38度怎么办", True),
        ("宝宝睡眠不好怎么办", False),
        ("宝宝误食了药物怎么办", True),
        ("辅食怎么添加", False),
        ("宝宝高烧40度", True),
        ("宝宝咳嗽有痰", False),
    ]

    for query, expected in test_cases:
        result = _is_safety_critical(query)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{query}' -> 安全关键={result} (期望={expected})")


def test_safety_expansion():
    """测试安全查询扩展"""
    print("\n" + "=" * 60)
    print("测试2: 安全查询扩展")
    print("=" * 60)

    query = "宝宝发烧怎么办"
    expanded = _get_safety_related_queries(query)
    print(f"原始查询: {query}")
    print(f"扩展查询: {expanded}")


def test_two_stage_retrieval():
    """测试两阶段检索效果"""
    print("\n" + "=" * 60)
    print("测试3: 两阶段检索对比")
    print("=" * 60)

    test_queries = [
        "宝宝发烧怎么办",
        "宝宝睡眠不好",
        "辅食添加时间",
    ]

    for query in test_queries:
        print(f"\n查询: {query}")

        # 方式1: 传统检索 (k=3)
        result1 = search_knowledge(query, k=3, initial_k=3, enable_safety_boost=False)
        docs1 = result1.count("【")
        print(f"  传统检索 (k=3): 召回 {docs1} 条")

        # 方式2: 两阶段检索 (initial_k=10, k=3)
        result2 = search_knowledge(query, k=3, initial_k=10, enable_safety_boost=False)
        docs2 = result2.count("【")
        print(f"  两阶段检索 (10->3): 召回 {docs2} 条")


def test_safety_mode():
    """测试安全模式检索"""
    print("\n" + "=" * 60)
    print("测试4: 安全模式检索")
    print("=" * 60)

    query = "宝宝发烧40度可以吃什么药"
    print(f"查询: {query}")
    print("注意：这是一个安全关键查询，应该触发高召回模式\n")

    result = search_knowledge(query, k=3, enable_safety_boost=True)
    print(f"结果:\n{result[:500]}...")


def test_offline_evaluation():
    """离线评估测试"""
    print("\n" + "=" * 60)
    print("测试5: 离线评估")
    print("=" * 60)

    # 构建测试集（示例）
    test_set = [
        ("宝宝发烧怎么办", ["发烧"]),
        ("母乳喂养注意事项", ["母乳喂养"]),
        ("睡眠训练", ["睡眠"]),
        ("辅食添加", ["辅食"]),
        ("疫苗接种", ["疫苗"]),
    ]

    metrics = evaluate_retrieval(test_set)
    print(f"评估结果: {metrics}")


def demo():
    """完整演示"""
    print("\n" + "=" * 60)
    print("两阶段检索系统演示")
    print("=" * 60)

    # 普通查询
    print("\n【普通查询】宝宝睡眠不好怎么办")
    result = search_knowledge("宝宝睡眠不好怎么办", k=3, initial_k=10)
    print(result)

    # 安全关键查询
    print("\n" + "=" * 60)
    print("【安全关键查询】宝宝高烧40度怎么办")
    result = search_knowledge("宝宝高烧40度怎么办", k=3, initial_k=15)
    print(result)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == "safety":
            test_safety_detection()
        elif test_name == "expand":
            test_safety_expansion()
        elif test_name == "two-stage":
            test_two_stage_retrieval()
        elif test_name == "safety-mode":
            test_safety_mode()
        elif test_name == "eval":
            test_offline_evaluation()
        else:
            print(f"未知测试: {test_name}")
    else:
        # 运行所有测试
        test_safety_detection()
        test_safety_expansion()
        test_two_stage_retrieval()
        test_safety_mode()
        # demo()  # 需要配置好环境才能运行
