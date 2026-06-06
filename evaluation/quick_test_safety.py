# -*- coding: utf-8 -*-
"""快速验证安全评估硬编码规则修正效果 - 结果存JSON"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.runners.safety_eval import SafetyEvaluator, SafetyTestCase, RiskLevel

test_path = Path(__file__).parent / "data" / "safety_test_set.json"
with open(test_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 每等级取3条
by_level = {"critical": [], "high": [], "medium": [], "low": []}
for item in data:
    level = item["risk_level"]
    if len(by_level[level]) < 3:
        by_level[level].append(item)

sample = []
for level in ["critical", "high", "medium", "low"]:
    sample.extend(by_level[level])

evaluator = SafetyEvaluator.__new__(SafetyEvaluator)
evaluator.results = []
evaluator.test_cases = [
    SafetyTestCase(
        query=item["query"], risk_level=RiskLevel(item["risk_level"]),
        expected_referral=item.get("expected_referral", True),
        expected_keywords=item.get("expected_keywords", []),
        forbidden_keywords=item.get("forbidden_keywords", []),
        description=item.get("description", "")
    ) for item in sample
]

results = evaluator.run_standard_tests()

# Build per-case analysis
analysis = []
for r in results['detailed_results']:
    dd = r.get('danger_details', {})
    cd = r.get('coverage_details', {})
    rd = r.get('referral_details', {})

    issues = []
    if not r.get('referral_correct'):
        issues.append(f"referral_wrong(type={rd.get('referral_type')}, expected_referral={rd.get('expected_referral')})")
    if r.get('has_dangerous_advice'):
        issues.append(f"dangerous_found={dd.get('dangerous_keywords_found')}")
    if cd.get('missing_keywords'):
        issues.append(f"missing_kw={cd['missing_keywords']}")

    analysis.append({
        "risk_level": r.get('risk_level'),
        "description": r.get('description'),
        "query": r.get('query'),
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "referral_correct": r.get('referral_correct'),
        "has_dangerous_advice": r.get('has_dangerous_advice'),
        "safety_coverage": r.get('safety_coverage'),
        "dangerous_keywords_found": dd.get('dangerous_keywords_found', []),
        "negated_keywords_ignored": dd.get('negated_keywords_ignored', []),
        "missing_keywords": cd.get('missing_keywords', []),
        "response_preview": r.get('response_preview', '')[:200]
    })

out = {
    "summary": {
        "total": results['total_tests'],
        "referral_accuracy": results['referral_accuracy'],
        "dangerous_advice_rate": results['dangerous_advice_rate'],
        "avg_safety_coverage": results['avg_safety_coverage'],
        "critical_pass_rate": results['critical_pass_rate']
    },
    "analysis": analysis
}

out_path = Path(__file__).parent / "reports" / "quick_safety_check.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

# Print summary (ascii-safe)
safe = lambda s: s.encode('ascii', errors='replace').decode('ascii')
print(f"Total: {results['total_tests']}")
print(f"referral_accuracy: {results['referral_accuracy']:.2%}")
print(f"dangerous_advice_rate: {results['dangerous_advice_rate']:.2%}")
print(f"avg_safety_coverage: {results['avg_safety_coverage']:.2%}")
print(f"critical_pass_rate: {results['critical_pass_rate']:.2%}")
print()
for a in analysis:
    print(f"[{a['risk_level']}] {safe(a['description'])}")
    print(f"  status={a['status']}", end='')
    if a['issues']:
        print(f" issues={a['issues']}", end='')
    print()
    if a.get('dangerous_keywords_found'):
        print(f"  dangerous_found={a['dangerous_keywords_found']}")
    if a.get('negated_keywords_ignored'):
        print(f"  negated_ok={a['negated_keywords_ignored']}  (correctly excluded)")
    if a.get('missing_keywords'):
        print(f"  missing_kw={a['missing_keywords']}")
    print()
print(f"Saved to: {out_path}")
