"""Agent端到端评估模块

评估完整的对话流程，包括工具调用准确性、任务完成度、回答质量等
"""

import json
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from logger import get_logger

logger = get_logger("evaluation.agent")

TEST_SET_PATH = Path(__file__).parent.parent / "data" / "agent_test_set.json"

NEGATION_WORDS = [
    "不要", "不能", "不可", "不建议", "不推荐", "不应",
    "避免", "切勿", "禁止", "不是", "并不", "无需",
    "不需要", "不一定", "不需", "没有",
    "暂时不需要", "暂时不要", "先不要", "现在不要",
    "暂时不用", "先别", "不用立即", "不需要立即",
    "没太大必要", "不一定需要",
]


@dataclass
class AgentTestCase:
    """Agent测试用例"""
    name: str
    conversation: List[tuple]  # [(role, message), ...]
    expected_tools: List[str]  # 期望调用的工具
    expected_profile_update: Dict[str, Any] = None  # 期望更新的档案字段
    expected_contains: List[str] = None  # 回答中应包含的内容
    forbidden_contains: List[str] = None  # 回答中不应包含的内容


class AgentEvaluator:
    """Agent端到端评估器"""

    def __init__(self, test_set_path: str = None):
        self.results = []
        self.test_cases = self._load_test_set(test_set_path)

    @staticmethod
    def _load_test_set(path: str = None) -> List[AgentTestCase]:
        path = Path(path) if path else TEST_SET_PATH
        if not path.exists():
            logger.warning(f"测试集文件不存在: {path}，使用空测试集")
            return []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        test_cases = []
        for item in data:
            # conversation 是 [[role, content], ...] 格式
            conversation = [tuple(c) for c in item["conversation"]]
            test_cases.append(AgentTestCase(
                name=item["name"],
                conversation=conversation,
                expected_tools=item.get("expected_tools", []),
                expected_profile_update=item.get("expected_profile_update"),
                expected_contains=item.get("expected_contains", []),
                forbidden_contains=item.get("forbidden_contains", [])
            ))
        logger.info(f"从 {path.name} 加载了 {len(test_cases)} 条Agent测试用例")
        return test_cases

    def evaluate_tool_calls(self, actual_tools: List[str], expected_tools: List[str]) -> Dict[str, Any]:
        """评估工具调用准确性"""
        if not expected_tools:
            return {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}

        actual_set = set(actual_tools)
        expected_set = set(expected_tools)

        correct = len(actual_set & expected_set)
        precision = correct / len(actual_set) if actual_set else 0.0
        recall = correct / len(expected_set) if expected_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": correct / max(len(actual_set), len(expected_set)),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "expected": list(expected_set),
            "actual": list(actual_set),
            "missing": list(expected_set - actual_set),
            "extra": list(actual_set - expected_set)
        }

    @staticmethod
    def _is_negated(text: str, keyword: str, window: int = 50) -> bool:
        """检查关键词在回答中是否全部处于否定语境中"""
        lower_text = text.lower()
        lower_kw = keyword.lower()
        pos = 0
        while True:
            idx = lower_text.find(lower_kw, pos)
            if idx == -1:
                break
            context_start = max(0, idx - window)
            context = text[context_start:idx]
            if not any(nw in context for nw in NEGATION_WORDS):
                return False  # 此出现未被否定
            pos = idx + 1
        return True  # 所有出现都被否定

    def evaluate_response_quality(self, response: str, test_case: AgentTestCase) -> Dict[str, Any]:
        """评估回答质量"""
        quality_metrics = {
            "has_expected": True,
            "has_forbidden": False,
            "expected_coverage": 0.0,
            "response_length": len(response)
        }

        response_lower = response.lower()

        # 检查必须包含的内容
        if test_case.expected_contains:
            found = []
            missing = []
            for keyword in test_case.expected_contains:
                if keyword.lower() in response_lower:
                    found.append(keyword)
                else:
                    missing.append(keyword)

            quality_metrics["expected_found"] = found
            quality_metrics["expected_missing"] = missing
            quality_metrics["expected_coverage"] = len(found) / len(test_case.expected_contains)
            quality_metrics["has_expected"] = len(missing) == 0

        # 检查禁止包含的内容（含否定检测）
        if test_case.forbidden_contains:
            found_forbidden = []
            negated_ignored = []
            for keyword in test_case.forbidden_contains:
                if keyword.lower() not in response_lower:
                    continue
                if self._is_negated(response, keyword):
                    negated_ignored.append(keyword)
                    continue
                found_forbidden.append(keyword)

            quality_metrics["forbidden_found"] = found_forbidden
            quality_metrics["forbidden_negated_ignored"] = negated_ignored
            quality_metrics["has_forbidden"] = len(found_forbidden) > 0

        return quality_metrics

    def evaluate_profile_update(self, profile: Dict, expected_update: Dict) -> Dict[str, Any]:
        """评估档案更新是否正确"""
        if not expected_update:
            return {"correct": True, "coverage": 1.0}

        correct_fields = []
        incorrect_fields = []

        for key, expected_value in expected_update.items():
            actual_value = profile.get(key)
            if actual_value == expected_value or (expected_value and expected_value in str(actual_value)):
                correct_fields.append(key)
            else:
                incorrect_fields.append({"key": key, "expected": expected_value, "actual": actual_value})

        coverage = len(correct_fields) / len(expected_update)

        return {
            "correct": len(incorrect_fields) == 0,
            "coverage": coverage,
            "correct_fields": correct_fields,
            "incorrect_fields": incorrect_fields
        }

    def run_single_test(self, test_case: AgentTestCase) -> Dict[str, Any]:
        """运行单个Agent测试"""
        logger.info(f"Agent测试: {test_case.name}")

        from graph import app
        from langchain_core.messages import HumanMessage

        # 构建对话状态
        messages = []
        for role, content in test_case.conversation:
            if role == "user":
                messages.append(HumanMessage(content=content))

        # 执行对话
        try:
            result = app.invoke({
                "messages": messages,
                "current_baby_id": None,
                "baby_profile": None
            })

            messages = result.get("messages", [])
            if not messages or messages[-1] is None:
                logger.warning(f"Agent测试 API返回空消息: {test_case.name}")
                return {
                    "name": test_case.name,
                    "error": "API returned empty or None message (likely timeout/429)",
                    "tool_accuracy": 0.0,
                    "response_quality": 0.0
                }

            final_message = messages[-1]
            response_content = final_message.content if hasattr(final_message, 'content') else str(final_message)

            # 提取工具调用记录（从消息历史中提取）
            actual_tools = []
            for msg in messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc is None:
                            continue
                        actual_tools.append(tc.get('name', tc.get('function', {}).get('name', 'unknown')))

            # 评估各项指标
            tool_eval = self.evaluate_tool_calls(actual_tools, test_case.expected_tools)
            quality_eval = self.evaluate_response_quality(response_content, test_case)

            profile_eval = {"correct": True, "coverage": 1.0}
            if test_case.expected_profile_update:
                profile = result.get("baby_profile", {})
                profile_eval = self.evaluate_profile_update(profile, test_case.expected_profile_update)

            test_result = {
                "name": test_case.name,
                "tool_accuracy": tool_eval["accuracy"],
                "tool_f1": tool_eval["f1"],
                "response_quality": quality_eval["expected_coverage"],
                "has_forbidden": quality_eval["has_forbidden"],
                "profile_correct": profile_eval["correct"],
                "tool_details": tool_eval,
                "quality_details": quality_eval,
                "profile_details": profile_eval,
                "response_preview": response_content[:200] if response_content else ""
            }

            logger.debug(f"测试结果: tool_f1={test_result['tool_f1']:.2f}, "
                        f"quality={test_result['response_quality']:.2%}")

            return test_result

        except Exception as e:
            logger.error(f"测试执行失败: {e}")
            return {
                "name": test_case.name,
                "error": str(e),
                "tool_accuracy": 0.0,
                "response_quality": 0.0
            }

    def run_standard_tests(self) -> Dict[str, Any]:
        """运行标准Agent测试集"""
        if not self.test_cases:
            logger.warning("没有测试用例可运行")
            return {"total_tests": 0, "error": "no test cases loaded"}

        logger.info(f"运行Agent评估，共 {len(self.test_cases)} 个测试用例")

        all_results = []
        for test_case in self.test_cases:
            result = self.run_single_test(test_case)
            all_results.append(result)
            time.sleep(random.uniform(0.5, 1.5))  # 避免触发限流

        # 过滤掉执行失败的
        successful_results = [r for r in all_results if "error" not in r]

        summary = {
            "total_tests": len(all_results),
            "successful_tests": len(successful_results),
            "avg_tool_accuracy": sum(r["tool_accuracy"] for r in successful_results) / max(len(successful_results), 1),
            "avg_tool_f1": sum(r["tool_f1"] for r in successful_results) / max(len(successful_results), 1),
            "avg_response_quality": sum(r["response_quality"] for r in successful_results) / max(len(successful_results), 1),
            "forbidden_violations": sum(r["has_forbidden"] for r in successful_results),
            "profile_correct_rate": sum(r["profile_correct"] for r in successful_results) / max(len(successful_results), 1),
            "detailed_results": all_results
        }

        logger.info(f"Agent评估完成: tool_f1={summary['avg_tool_f1']:.2f}, "
                   f"response_quality={summary['avg_response_quality']:.2%}, "
                   f"forbidden_violations={summary['forbidden_violations']}")

        return summary
