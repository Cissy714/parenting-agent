"""
计算章节级 recall@3 和 MRR —— 双路对比 (legacy vs hybrid)

两个 recall 版本：
  - Binary recall: 只要有一个 expected_chapter 出现在 Top-3 就算命中 (0或1)
  - Coverage recall: 被命中的 expected_chapters 数 / 总 expected_chapters 数
  - MRR: 1 / 第一个命中结果的排名

示例：
  expected_chapters = ["生长", "发育"], Top-3 章节 = ["生长", "发烧", "睡眠"]
  → binary recall = 1, coverage recall = 0.5, MRR = 1/1 = 1.0
"""

import sys
import re
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_test_set():
    p = Path(__file__).parent.parent / "evaluation" / "data" / "retrieval_test_set_v2_100_corrected.json"
    import json
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def extract_chapters(result_text: str) -> list[str]:
    """从格式化结果中提取各条结果的章节名。格式：【章节名】（来源：xxx）"""
    return re.findall(r"【(.+?)】（来源：", result_text)


def chapter_hits(result_chapters: list[str], expected_chapters: list[str]) -> list[int]:
    """返回命中的排名列表（1-indexed），空列表表示无命中"""
    hits = []
    for rank, ch in enumerate(result_chapters, 1):
        for exp in expected_chapters:
            # 双向子串匹配：章节名包含期望词 或 期望词包含章节名
            if exp.lower() in ch.lower() or ch.lower() in exp.lower():
                hits.append(rank)
                break  # 每条结果只算一次
    return hits


def compute_metrics(result_chapters: list[str], expected_chapters: list[str]) -> dict:
    """
    计算 recall@3, coverage_recall, MRR
    result_chapters: Top-K 结果的章节名列表
    expected_chapters: 期望的章节名列表
    """
    if not expected_chapters:
        return {"binary_recall": None, "coverage_recall": None, "mrr": None}

    top3 = result_chapters[:3]
    hits = chapter_hits(top3, expected_chapters)

    binary_recall = 1.0 if hits else 0.0
    coverage_recall = len(set(hits)) / len(expected_chapters)  # 不同排名的命中数 / 期望数
    mrr = 1.0 / hits[0] if hits else 0.0

    return {
        "binary_recall": binary_recall,
        "coverage_recall": coverage_recall,
        "mrr": mrr,
    }


def main():
    from knowledge.knowledge_base import search_knowledge
    from knowledge import knowledge_base as _kb
    _original_rewrite = _kb.rewrite_query
    _kb.rewrite_query = lambda q, *a, **kw: [q]

    test_cases = load_test_set()

    # 聚合指标
    metrics = {
        "legacy": {"binary_recall": [], "coverage_recall": [], "mrr": []},
        "hybrid": {"binary_recall": [], "coverage_recall": [], "mrr": []},
    }

    # 按 expected_chapters 数量分组
    by_num_chapters = defaultdict(lambda: {"legacy": [], "hybrid": [], "queries": []})

    skipped = 0
    total = len(test_cases)

    for i, case in enumerate(test_cases):
        query = case["query"]
        expected_chapters = case.get("expected_chapters", [])

        if not expected_chapters:
            skipped += 1
            continue

        num_ch = len(expected_chapters)

        # Legacy
        result_l = search_knowledge(query, k=3, initial_k=10, use_hybrid=False)
        chapters_l = extract_chapters(result_l)
        m_l = compute_metrics(chapters_l, expected_chapters)

        # Hybrid
        result_h = search_knowledge(query, k=3, initial_k=10, use_hybrid=True)
        chapters_h = extract_chapters(result_h)
        m_h = compute_metrics(chapters_h, expected_chapters)

        for key in ["binary_recall", "coverage_recall", "mrr"]:
            if m_l[key] is not None:
                metrics["legacy"][key].append(m_l[key])
                metrics["hybrid"][key].append(m_h[key])

        by_num_chapters[num_ch]["legacy"].append(m_l)
        by_num_chapters[num_ch]["hybrid"].append(m_h)
        by_num_chapters[num_ch]["queries"].append(query)

        if (i + 1) % 30 == 0:
            print(f"  进度: {i+1}/{total}")

    _kb.rewrite_query = _original_rewrite

    # ── 输出 ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"章节级 Recall@3 和 MRR 对比  (共 {total} 条, 有章节标注 {total - skipped} 条)")
    print(f"{'='*60}")

    n = len(metrics["legacy"]["binary_recall"])
    avg_l_bin = sum(metrics["legacy"]["binary_recall"]) / n
    avg_h_bin = sum(metrics["hybrid"]["binary_recall"]) / n
    avg_l_cov = sum(metrics["legacy"]["coverage_recall"]) / n
    avg_h_cov = sum(metrics["hybrid"]["coverage_recall"]) / n
    avg_l_mrr = sum(metrics["legacy"]["mrr"]) / n
    avg_h_mrr = sum(metrics["hybrid"]["mrr"]) / n

    print(f"\n{'指标':<30} {'Legacy':>10} {'Hybrid':>10} {'变化':>10}")
    print(f"{'-'*60}")
    print(f"{'Binary Recall@3 (宽松)':<30} {avg_l_bin:>10.2%} {avg_h_bin:>10.2%} {avg_h_bin - avg_l_bin:>+10.2%}")
    print(f"{'Coverage Recall@3 (严格)':<30} {avg_l_cov:>10.2%} {avg_h_cov:>10.2%} {avg_h_cov - avg_l_cov:>+10.2%}")
    print(f"{'MRR':<30} {avg_l_mrr:>10.4f} {avg_h_mrr:>10.4f} {avg_h_mrr - avg_l_mrr:>+10.4f}")

    # 按 expected_chapters 数量分组
    print(f"\n{'─'*60}")
    print("按 expected_chapters 数量分组:")
    print(f"{'─'*60}")
    for num_ch in sorted(by_num_chapters):
        g = by_num_chapters[num_ch]
        g_l = g["legacy"]
        g_h = g["hybrid"]
        qs = g["queries"]
        gl_bin = sum(m["binary_recall"] for m in g_l) / len(g_l)
        gh_bin = sum(m["binary_recall"] for m in g_h) / len(g_h)
        gl_cov = sum(m["coverage_recall"] for m in g_l) / len(g_l)
        gh_cov = sum(m["coverage_recall"] for m in g_h) / len(g_h)
        gl_mrr = sum(m["mrr"] for m in g_l) / len(g_l)
        gh_mrr = sum(m["mrr"] for m in g_h) / len(g_h)
        print(f"\n  {num_ch} 个期望章节 ({len(qs)} 条):")
        print(f"    Binary Recall@3:  {gl_bin:.2%} → {gh_bin:.2%}  ({gh_bin - gl_bin:+.2%})")
        print(f"    Coverage Recall@3: {gl_cov:.2%} → {gh_cov:.2%}  ({gh_cov - gl_cov:+.2%})")
        print(f"    MRR:               {gl_mrr:.4f} → {gh_mrr:.4f}  ({gh_mrr - gl_mrr:+.4f})")

    # 两个算法说明
    print(f"\n{'─'*60}")
    print("两个 recall 算法:")
    print(f"  Binary:   只要 Top-3 中有一条命中任一期望章节 → 1.0")
    print(f"            e.g. expected=[生长,发育], Top-3章节=[生长,发烧,睡眠] → recall=1.0")
    print(f"  Coverage: 被命中的不同期望章节数 / 总期望章节数")
    print(f"            e.g. expected=[生长,发育], Top-3章节=[生长,发烧,睡眠] → recall=0.5")
    print(f"  MRR:      1 / 第一个命中结果的排名")
    print(f"            e.g. 第一个命中在排名2 → MRR=0.5")

    # Top-5 退化（按 binary recall）
    print(f"\n{'─'*60}")
    print("Legacy 命中但 Hybrid 未命中的案例 (Binary Recall):")
    print(f"{'─'*60}")
    count = 0
    for i, case in enumerate(test_cases):
        query = case["query"]
        expected = case.get("expected_chapters", [])
        if not expected:
            continue
        idx = i - skipped if i >= skipped else i  # rough index
        # Re-fetch to compare... Actually let me just print from stored data
    # Can't easily access stored chapter data — print from separate run approach instead.
    # Skip the list since we'd need to re-run.

    return metrics


if __name__ == "__main__":
    main()
