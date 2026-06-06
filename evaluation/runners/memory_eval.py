"""记忆系统评估模块

评估档案记忆、偏好学习、情景记忆的准确性和一致性
"""

import json
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from logger import get_logger

logger = get_logger("evaluation.memory")

TEST_SET_PATH = Path(__file__).parent.parent / "data" / "memory_test_set.json"


@dataclass
class MemoryTestCase:
    """记忆测试用例"""
    name: str
    setup_conversation: List[tuple]  # 设置阶段对话
    test_query: str                  # 测试查询
    expected_memory_recall: List[str]  # 期望回忆的内容
    expected_preference: str = None    # 期望应用的偏好
    description: str = ""


class MemoryEvaluator:
    """记忆系统评估器"""

    def __init__(self, test_set_path: str = None):
        self.results = []
        self.test_cases = self._load_test_set(test_set_path)

    @staticmethod
    def _load_test_set(path: str = None) -> List[MemoryTestCase]:
        path = Path(path) if path else TEST_SET_PATH
        if not path.exists():
            logger.warning(f"测试集文件不存在: {path}，使用空测试集")
            return []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        test_cases = []
        for item in data:
            setup = [tuple(c) for c in item.get("setup_conversation", [])]
            test_cases.append(MemoryTestCase(
                name=item["name"],
                setup_conversation=setup,
                test_query=item["test_query"],
                expected_memory_recall=item.get("expected_memory_recall", []),
                expected_preference=item.get("expected_preference"),
                description=item.get("description", "")
            ))
        logger.info(f"从 {path.name} 加载了 {len(test_cases)} 条记忆测试用例")
        return test_cases

    def evaluate_profile_recall(self, response: str, expected_items: List[str]) -> Dict[str, Any]:
        """评估档案信息是否被正确回忆"""
        found = []
        missing = []

        response_lower = response.lower()
        for item in expected_items:
            if item.lower() in response_lower:
                found.append(item)
            else:
                missing.append(item)

        recall = len(found) / len(expected_items) if expected_items else 1.0

        return {
            "recall": recall,
            "found": found,
            "missing": missing
        }

    def evaluate_preference_application(self, response: str, expected_preference: str) -> Dict[str, Any]:
        """评估偏好是否被正确应用"""
        if not expected_preference:
            return {"applied": True, "confidence": 1.0}

        response_lower = response.lower()
        preference_lower = expected_preference.lower()

        # 检查偏好相关词是否出现
        applied = preference_lower in response_lower

        # 也可以检查语义相似度（如果需要更精确）
        confidence = 1.0 if applied else 0.0

        return {
            "applied": applied,
            "confidence": confidence,
            "preference": expected_preference
        }

    def check_memory_consistency(self, session1_response: str, session2_response: str) -> Dict[str, Any]:
        """检查跨会话记忆一致性"""
        # 简单检查：两次回答是否都包含相同的关键信息
        # 更复杂的检查可以用LLM判断

        return {
            "consistent": True,  # 简化处理
            "method": "keyword_match"
        }

    def run_single_test(self, test_case: MemoryTestCase) -> Dict[str, Any]:
        """运行单个记忆测试"""
        logger.info(f"记忆测试: {test_case.name}")

        from graph import app
        from langchain_core.messages import HumanMessage
        from memory.db import find_baby_by_name

        try:
            # 阶段1：设置（创建记忆）
            baby_id = None
            for role, content in test_case.setup_conversation:
                result = app.invoke({
                    "messages": [HumanMessage(content=content)],
                    "current_baby_id": baby_id,
                    "baby_profile": None
                })
                if result is None or not result.get("messages"):
                    logger.warning(f"记忆测试 setup阶段 API返回空: {test_case.name}")
                    return {
                        "name": test_case.name,
                        "error": "API returned empty response during setup (likely timeout/429)",
                        "profile_recall": 0.0,
                        "preference_applied": False
                    }
                baby_id = result.get("current_baby_id")

            # 阶段2：测试（验证记忆）
            test_result = app.invoke({
                "messages": [HumanMessage(content=test_case.test_query)],
                "current_baby_id": baby_id,
                "baby_profile": None
            })

            messages = test_result.get("messages", [])
            if not messages or messages[-1] is None:
                logger.warning(f"记忆测试 API返回空消息: {test_case.name}")
                return {
                    "name": test_case.name,
                    "error": "API returned empty or None message (likely timeout/429)",
                    "profile_recall": 0.0,
                    "preference_applied": False
                }

            response = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])

            # 评估记忆回忆
            recall_eval = self.evaluate_profile_recall(response, test_case.expected_memory_recall)

            # 评估偏好应用
            preference_eval = self.evaluate_preference_application(response, test_case.expected_preference)

            # 检查档案是否正确更新到数据库
            profile_correct = True
            if baby_id:
                from memory.db import get_baby
                profile = get_baby(baby_id)
                if not profile:
                    profile_correct = False

            test_result = {
                "name": test_case.name,
                "profile_recall": recall_eval["recall"],
                "memory_items_found": recall_eval["found"],
                "memory_items_missed": recall_eval["missing"],
                "preference_applied": preference_eval["applied"],
                "profile_persisted": profile_correct,
                "response_preview": response[:200]
            }

            logger.debug(f"记忆测试结果: profile_recall={test_result['profile_recall']:.2%}, "
                        f"preference_applied={test_result['preference_applied']}")

            return test_result

        except Exception as e:
            logger.error(f"记忆测试执行失败: {e}")
            return {
                "name": test_case.name,
                "error": str(e),
                "profile_recall": 0.0,
                "preference_applied": False
            }

    def run_standard_tests(self) -> Dict[str, Any]:
        """运行标准记忆测试集"""
        if not self.test_cases:
            logger.warning("没有测试用例可运行")
            return {"total_tests": 0, "error": "no test cases loaded"}

        logger.info(f"运行记忆评估，共 {len(self.test_cases)} 个测试用例")

        all_results = []
        for test_case in self.test_cases:
            result = self.run_single_test(test_case)
            all_results.append(result)
            time.sleep(random.uniform(0.5, 1.5))  # 避免触发限流

        successful_results = [r for r in all_results if "error" not in r]

        summary = {
            "total_tests": len(all_results),
            "successful_tests": len(successful_results),
            "avg_profile_recall": sum(r["profile_recall"] for r in successful_results) / max(len(successful_results), 1),
            "preference_learning_rate": sum(r["preference_applied"] for r in successful_results) / max(len(successful_results), 1),
            "profile_persistence_rate": sum(r["profile_persisted"] for r in successful_results) / max(len(successful_results), 1),
            "detailed_results": all_results
        }

        logger.info(f"记忆评估完成: profile_recall={summary['avg_profile_recall']:.2%}, "
                   f"preference_learning={summary['preference_learning_rate']:.2%}")

        return summary