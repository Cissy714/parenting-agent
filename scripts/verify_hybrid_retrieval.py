"""
混合检索验收脚本 —— 对比 hybrid (BM25+向量+RRF) vs legacy (纯向量)
使用 evaluation/data/retrieval_test_set_v2_100_corrected.json 测试集
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from logger import get_logger

logger = get_logger("verify_hybrid")

# ── 加载测试集 ─────────────────────────────────────────────


def load_test_set(path: str = None) -> list[dict]:
    if path is None:
        path = Path(__file__).parent.parent / "evaluation" / "data" / "retrieval_test_set_v2_100_corrected.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── 关键词覆盖计算 ──────────────────────────────────────────


def keyword_coverage(result_text: str, expected_keywords: list[str]) -> tuple[float, list, list]:
    """返回 (覆盖率, 命中列表, 未命中列表)"""
    text_lower = result_text.lower()
    covered, missed = [], []
    for kw in expected_keywords:
        if kw.lower() in text_lower:
            covered.append(kw)
        else:
            missed.append(kw)
    return len(covered) / len(expected_keywords) if expected_keywords else 1.0, covered, missed


def count_unique_sources(result_text: str) -> int:
    """统计结果中出现了几个不同的书籍来源"""
    sources = set()
    for line in result_text.split("\n"):
        if "来源：" in line:
            source = line.split("来源：")[1].strip("）")
            sources.add(source)
    return len(sources)


# ── 运行对比 ───────────────────────────────────────────────


def run_comparison(test_cases: list[dict]) -> dict:
    from knowledge.knowledge_base import search_knowledge
    from knowledge import knowledge_base as _kb
    # 必须 patch knowledge_base 模块内的 rewrite_query 引用（search_knowledge 内部用到）
    _original_rewrite = _kb.rewrite_query
    _kb.rewrite_query = lambda q, *a, **kw: [q]

    results = {
        "legacy": [],
        "hybrid": [],
    }

    safety_cases = []
    normal_cases = []

    total = len(test_cases)
    print(f"开始对比评估，共 {total} 条测试用例\n")

    for i, case in enumerate(test_cases):
        query = case["query"]
        expected_keywords = case.get("expected_keywords", [])
        is_safety = case.get("is_safety_critical", False)

        if not expected_keywords:
            continue

        # Legacy (纯向量 + 规则打分)
        result_legacy = search_knowledge(query, k=3, initial_k=10, use_hybrid=False)
        cov_legacy, covered_l, missed_l = keyword_coverage(result_legacy, expected_keywords)
        sources_l = count_unique_sources(result_legacy)

        # Hybrid (BM25 + 向量 + RRF, 无查询改写)
        result_hybrid = search_knowledge(query, k=3, initial_k=10, use_hybrid=True)
        cov_hybrid, covered_h, missed_h = keyword_coverage(result_hybrid, expected_keywords)
        sources_h = count_unique_sources(result_hybrid)

        delta = cov_hybrid - cov_legacy

        entry = {
            "query": query,
            "legacy_coverage": cov_legacy,
            "hybrid_coverage": cov_hybrid,
            "delta": delta,
            "legacy_sources": sources_l,
            "hybrid_sources": sources_h,
            "is_safety": is_safety,
            "covered_legacy": covered_l,
            "covered_hybrid": covered_h,
            "missed_legacy": missed_l,
            "missed_hybrid": missed_h,
            "result_legacy": result_legacy,
            "result_hybrid": result_hybrid,
        }

        results["legacy"].append(cov_legacy)
        results["hybrid"].append(cov_hybrid)

        if is_safety:
            safety_cases.append(entry)
        else:
            normal_cases.append(entry)

        # 进度
        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{total}")

    # 恢复
    _kb.rewrite_query = _original_rewrite

    # 汇总
    all_entries = safety_cases + normal_cases

    avg_legacy = sum(results["legacy"]) / len(results["legacy"])
    avg_hybrid = sum(results["hybrid"]) / len(results["hybrid"])

    avg_safety_legacy = sum(r["legacy_coverage"] for r in safety_cases) / len(safety_cases) if safety_cases else 0
    avg_safety_hybrid = sum(r["hybrid_coverage"] for r in safety_cases) / len(safety_cases) if safety_cases else 0

    avg_normal_legacy = sum(r["legacy_coverage"] for r in normal_cases) / len(normal_cases) if normal_cases else 0
    avg_normal_hybrid = sum(r["hybrid_coverage"] for r in normal_cases) / len(normal_cases) if normal_cases else 0

    avg_sources_legacy = sum(r["legacy_sources"] for r in all_entries) / len(all_entries)
    avg_sources_hybrid = sum(r["hybrid_sources"] for r in all_entries) / len(all_entries)

    wins = sum(1 for e in all_entries if e["delta"] > 0.01)
    losses = sum(1 for e in all_entries if e["delta"] < -0.01)
    ties = sum(1 for e in all_entries if abs(e["delta"]) <= 0.01)

    # 打印报告
    print("\n" + "=" * 60)
    print("混合检索 vs 传统检索 对比报告")
    print("=" * 60)

    print(f"\n{'指标':<25} {'Legacy':>10} {'Hybrid':>10} {'变化':>10}")
    print("-" * 55)
    print(f"{'平均关键词覆盖率':<25} {avg_legacy:>10.2%} {avg_hybrid:>10.2%} {avg_hybrid - avg_legacy:>+10.2%}")
    print(f"{'  安全查询子集':<25} {avg_safety_legacy:>10.2%} {avg_safety_hybrid:>10.2%} {avg_safety_hybrid - avg_safety_legacy:>+10.2%}")
    print(f"{'  普通查询子集':<25} {avg_normal_legacy:>10.2%} {avg_normal_hybrid:>10.2%} {avg_normal_hybrid - avg_normal_legacy:>+10.2%}")
    print(f"{'平均来源数(Top-3)':<25} {avg_sources_legacy:>10.2f} {avg_sources_hybrid:>10.2f} {avg_sources_hybrid - avg_sources_legacy:>+10.2f}")
    print(f"{'Hybrid更优/持平/更差':<25} {wins:>5}/{ties:>5}/{losses:>5}")

    print(f"\n安全查询({len(safety_cases)}条) / 普通查询({len(normal_cases)}条)")

    # Top 5 改进最大
    print(f"\n{'─' * 60}")
    print("Hybrid 改进最大的 5 条:")
    print(f"{'─' * 60}")
    improved = sorted(all_entries, key=lambda x: x["delta"], reverse=True)[:5]
    for i, e in enumerate(improved, 1):
        print(f"\n  [{i}] {e['query'][:60]}")
        print(f"      Legacy: {e['legacy_coverage']:.0%} | Hybrid: {e['hybrid_coverage']:.0%} (Δ={e['delta']:+.0%})")
        print(f"      Legacy 未命中: {e['missed_legacy']}")
        print(f"      Hybrid 未命中: {e['missed_hybrid']}")
        print(f"      Legacy 来源数: {e['legacy_sources']}  Hybrid 来源数: {e['hybrid_sources']}")

    # Top 5 退化最大
    print(f"\n{'─' * 60}")
    print("Hybrid 退化最大的 5 条:")
    print(f"{'─' * 60}")
    degraded = sorted(all_entries, key=lambda x: x["delta"])[:5]
    for i, e in enumerate(degraded, 1):
        print(f"\n  [{i}] {e['query'][:60]}")
        print(f"      Legacy: {e['legacy_coverage']:.0%} | Hybrid: {e['hybrid_coverage']:.0%} (Δ={e['delta']:+.0%})")
        print(f"      Legacy 未命中: {e['missed_legacy']}")
        print(f"      Hybrid 未命中: {e['missed_hybrid']}")
        print(f"      Legacy 来源数: {e['legacy_sources']}  Hybrid 来源数: {e['hybrid_sources']}")

    return {
        "avg_legacy": avg_legacy,
        "avg_hybrid": avg_hybrid,
        "avg_safety_legacy": avg_safety_legacy,
        "avg_safety_hybrid": avg_safety_hybrid,
        "avg_normal_legacy": avg_normal_legacy,
        "avg_normal_hybrid": avg_normal_hybrid,
        "avg_sources_legacy": avg_sources_legacy,
        "avg_sources_hybrid": avg_sources_hybrid,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "total": len(all_entries),
        "top_improved": improved,
        "top_degraded": degraded,
    }


def main():
    start_time = time.time()

    test_cases = load_test_set()
    results = run_comparison(test_cases)

    elapsed = time.time() - start_time
    print(f"\n验收耗时: {elapsed:.0f}s")

    # 输出结论
    print(f"\n结论:")
    delta = results["avg_hybrid"] - results["avg_legacy"]
    if delta > 0.02:
        print(f"  混合检索整体优于传统检索 (关键词覆盖率 +{delta:.1%})")
    elif delta < -0.02:
        print(f"  混合检索整体弱于传统检索 (关键词覆盖率 {delta:.1%})，需排查")
    else:
        print(f"  混合检索与传统检索持平，多路召回增益有限")

    if results["avg_sources_hybrid"] > results["avg_sources_legacy"] + 0.1:
        print(f"  混合检索来源多样性更高 ({results['avg_sources_hybrid']:.2f} vs {results['avg_sources_legacy']:.2f})，验证了多路互补")
    else:
        print(f"  混合检索来源多样性无显著提升，BM25和向量检索结果高度重叠")

    return results


if __name__ == "__main__":
    main()
