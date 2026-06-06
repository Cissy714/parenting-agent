# -*- coding: utf-8 -*-
"""快速验证修正后的测试集效果"""

import json
import sys
from pathlib import Path
sys.path.insert(0, '.')

from evaluation.runners.retrieval_eval_chapter import ChapterLevelEvaluator, ChapterTestCase

# 读取修正后的测试集
with open(Path(__file__).parent / 'data' / 'retrieval_test_set_v2_100_corrected.json', 'r', encoding='utf-8') as f:
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

# 运行评估
evaluator = ChapterLevelEvaluator()
results = evaluator.run_tests(test_cases)

# 输出报告
report = []
report.append("\n" + "="*60)
report.append("修正后评估报告 (94 queries)")
report.append("="*60)
report.append(f"总测试数: {results['total_tests']}")
report.append(f"平均 recall@3:  {results['avg_recall@3']:.2%}")
report.append(f"平均 recall@5:  {results['avg_recall@5']:.2%}")
report.append(f"平均 recall@10: {results['avg_recall@10']:.2%}")
report.append(f"平均 MRR:       {results['avg_mrr']:.3f}")
report.append(f"关键词覆盖率:   {results['avg_keyword_coverage']:.2%}")
report.append(f"安全检测准确率: {results['safety_detection_accuracy']:.2%}")

# 分布
recall_3_dist = [r['recall_metrics']['recall@3'] for r in results['detailed_results']]
report.append(f"\nRecall@3 分布:")
report.append(f"  0%:   {sum(1 for v in recall_3_dist if v == 0)} 条")
report.append(f"  33%:  {sum(1 for v in recall_3_dist if 0 < v <= 0.4)} 条")
report.append(f"  50%:  {sum(1 for v in recall_3_dist if 0.4 < v <= 0.6)} 条")
report.append(f"  67%:  {sum(1 for v in recall_3_dist if 0.6 < v <= 0.7)} 条")
report.append(f"  100%: {sum(1 for v in recall_3_dist if v == 1.0)} 条")

# 安全 vs 普通
safety_results = [r for r in results['detailed_results'] if r['is_safety_critical']]
normal_results = [r for r in results['detailed_results'] if not r['is_safety_critical']]
if safety_results:
    report.append(f"\n安全关键 ({len(safety_results)}条):")
    report.append(f"  recall@3: {sum(r['recall_metrics']['recall@3'] for r in safety_results)/len(safety_results):.2%}")
    report.append(f"  MRR:      {sum(r['mrr'] for r in safety_results)/len(safety_results):.3f}")
if normal_results:
    report.append(f"\n普通查询 ({len(normal_results)}条):")
    report.append(f"  recall@3: {sum(r['recall_metrics']['recall@3'] for r in normal_results)/len(normal_results):.2%}")
    report.append(f"  MRR:      {sum(r['mrr'] for r in normal_results)/len(normal_results):.3f}")

# 对比原始结果
with open(Path(__file__).parent / 'data' / 'retrieval_eval_chapter_results.json', 'r', encoding='utf-8') as f:
    old_results = json.load(f)

report.append(f"\n{'='*60}")
report.append("修正前后对比")
report.append("="*60)
report.append(f"{'指标':<20} {'修正前':>10} {'修正后':>10} {'变化':>10}")
report.append("-"*50)
report.append(f"{'recall@3':<20} {old_results['avg_recall@3']:>10.2%} {results['avg_recall@3']:>10.2%} {results['avg_recall@3']-old_results['avg_recall@3']:>+10.2%}")
report.append(f"{'recall@5':<20} {old_results['avg_recall@5']:>10.2%} {results['avg_recall@5']:>10.2%} {results['avg_recall@5']-old_results['avg_recall@5']:>+10.2%}")
report.append(f"{'recall@10':<20} {old_results['avg_recall@10']:>10.2%} {results['avg_recall@10']:>10.2%} {results['avg_recall@10']-old_results['avg_recall@10']:>+10.2%}")
report.append(f"{'MRR':<20} {old_results['avg_mrr']:>10.3f} {results['avg_mrr']:>10.3f} {results['avg_mrr']-old_results['avg_mrr']:>+10.3f}")
report.append(f"{'关键词覆盖':<20} {old_results['avg_keyword_coverage']:>10.2%} {results['avg_keyword_coverage']:>10.2%} {results['avg_keyword_coverage']-old_results['avg_keyword_coverage']:>+10.2%}")

old_zero = sum(1 for r in old_results['detailed_results'] if r['recall_metrics']['recall@3'] == 0)
new_zero = sum(1 for v in recall_3_dist if v == 0)
report.append(f"\n0% Recall条数: {old_zero} --> {new_zero} (减少 {old_zero - new_zero} 条)")

print('\n'.join(report))

# 保存
report_path = Path(__file__).parent / 'reports' / 'verification_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))
print(f"\n\n验证报告已保存: {report_path}")
