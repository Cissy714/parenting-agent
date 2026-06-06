"""
检索三阶段对比: Base(纯向量) vs V1(+关键词精排) vs V2(多路BM25+向量+RRF)
使用 search_knowledge 获取 V1/V2 的真实行为，Base 用原始向量检索。
"""
import sys, json, re, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_test_set():
    p = Path(__file__).parent.parent / "evaluation" / "data" / "retrieval_test_set_v2_100_corrected.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def extract_chapters(result_text: str) -> list[str]:
    return re.findall(r"【(.+?)】（来源：", result_text)


def compute_metrics(chapters: list[str], expected: list[str]) -> dict:
    """
    Binary recall: 至少一个期望章节出现在 Top-3
    Coverage recall: 被覆盖的期望章节数 / 总期望章节数
    MRR: 1 / 第一个命中排名
    """
    if not expected or not chapters:
        return {"binary_recall": 0.0, "coverage_recall": 0.0, "mrr": 0.0}

    top3 = chapters[:3]
    covered_expected = set()
    first_hit_rank = None

    for rank, ch in enumerate(top3, 1):
        for exp in expected:
            if exp.lower() in ch.lower() or ch.lower() in exp.lower():
                covered_expected.add(exp)
                if first_hit_rank is None:
                    first_hit_rank = rank

    binary = 1.0 if covered_expected else 0.0
    coverage = len(covered_expected) / len(expected)
    mrr = 1.0 / first_hit_rank if first_hit_rank else 0.0
    return {"binary_recall": binary, "coverage_recall": coverage, "mrr": mrr}


def main():
    from knowledge.knowledge_base import search_knowledge, _get_vectorstore
    from knowledge import knowledge_base as _kb
    _original_rewrite = _kb.rewrite_query
    _kb.rewrite_query = lambda q, *a, **kw: [q]

    test_cases = load_test_set()

    store = {
        "base": {"binary_recall": [], "coverage_recall": [], "mrr": []},
        "v1":   {"binary_recall": [], "coverage_recall": [], "mrr": []},
        "v2":   {"binary_recall": [], "coverage_recall": [], "mrr": []},
    }

    total = len(test_cases)
    print(f"开始三阶段对比，共 {total} 条\n")

    for i, case in enumerate(test_cases):
        query = case["query"]
        expected = case.get("expected_chapters", [])
        if not expected:
            continue

        # ── Base: 纯向量 Top-3 ──
        vectorstore = _get_vectorstore()
        raw_docs = vectorstore.similarity_search_with_score(query, k=3)
        base_chapters = [doc.metadata.get("chapter", "") for doc, _ in raw_docs]
        m_base = compute_metrics(base_chapters, expected)

        # ── V1: 向量 + 关键词精排 + 安全优先 (legacy path) ──
        result_v1 = search_knowledge(query, k=3, initial_k=10, use_hybrid=False, enable_safety_boost=True)
        v1_chapters = extract_chapters(result_v1)
        m_v1 = compute_metrics(v1_chapters, expected)

        # ── V2: BM25 + 向量 + RRF 融合 + 安全增强 (hybrid path) ──
        result_v2 = search_knowledge(query, k=3, initial_k=10, use_hybrid=True, enable_safety_boost=True)
        v2_chapters = extract_chapters(result_v2)
        m_v2 = compute_metrics(v2_chapters, expected)

        for key, m in [("base", m_base), ("v1", m_v1), ("v2", m_v2)]:
            store[key]["binary_recall"].append(m["binary_recall"])
            store[key]["coverage_recall"].append(m["coverage_recall"])
            store[key]["mrr"].append(m["mrr"])

        if (i + 1) % 30 == 0:
            print(f"  进度: {i+1}/{total}")

    _kb.rewrite_query = _original_rewrite

    # ── 输出 ──
    n = len(store["base"]["binary_recall"])
    print(f"\n{'='*75}")
    print(f"三阶段检索对比 (共 {total} 条, 有章节标注 {n} 条)")
    print(f"{'='*75}")

    labels = [
        ("base", "Base (纯向量 Top-3)"),
        ("v1",   "V1 (向量粗召回 + 关键词精排 + 安全优先)"),
        ("v2",   "V2 (BM25+向量多路召回 + RRF融合 + 分级安全增强)"),
    ]

    prev_bin, prev_cov, prev_mrr = None, None, None
    for key, label in labels:
        avg_bin = sum(store[key]["binary_recall"]) / n
        avg_cov = sum(store[key]["coverage_recall"]) / n
        avg_mrr = sum(store[key]["mrr"]) / n

        deltas = ""
        if prev_bin is not None:
            db = avg_bin - prev_bin
            dc = avg_cov - prev_cov
            dm = avg_mrr - prev_mrr
            deltas = f"  (Δ={db:+.2%} / {dc:+.2%} / {dm:+.4f})"

        print(f"\n  {label}")
        print(f"    Binary Recall@3:  {avg_bin:.2%}{' ←' + deltas if deltas else ''}")
        print(f"    Coverage Recall:  {avg_cov:.2%}")
        print(f"    MRR:              {avg_mrr:.4f}")

        prev_bin, prev_cov, prev_mrr = avg_bin, avg_cov, avg_mrr

    # 退化分析
    print(f"\n{'─'*75}")
    all_results = []
    for i, case in enumerate(test_cases):
        expected = case.get("expected_chapters", [])
        if not expected:
            continue
        q = case["query"]
        idx = len(all_results)
        b_bin = store["base"]["binary_recall"][idx]
        v1_bin = store["v1"]["binary_recall"][idx]
        v2_bin = store["v2"]["binary_recall"][idx]
        all_results.append((q, expected, b_bin, v1_bin, v2_bin))

    v1_wins = [r for r in all_results if r[3] > r[2]]
    v1_loses = [r for r in all_results if r[3] < r[2]]
    v2_wins = [r for r in all_results if r[4] > r[3]]
    v2_loses = [r for r in all_results if r[4] < r[3]]

    print(f"V1 vs Base: 改进 {len(v1_wins)} 条, 退化 {len(v1_loses)} 条, 持平 {n - len(v1_wins) - len(v1_loses)} 条")
    print(f"V2 vs V1:   改进 {len(v2_wins)} 条, 退化 {len(v2_loses)} 条, 持平 {n - len(v2_wins) - len(v2_loses)} 条")

    if v1_loses:
        print(f"\n{'─'*75}")
        print(f"V1 vs Base 退化案例 (Base 命中但 V1 未命中, {len(v1_loses)}条):")
        for q, exp, _, _, _ in v1_loses[:8]:
            print(f"  - [{exp}] {q[:60]}")


if __name__ == "__main__":
    main()
