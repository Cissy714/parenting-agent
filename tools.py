from langchain_core.tools import tool
from knowledge.knowledge_base import search_knowledge
from memory.db import upsert_baby, get_baby, find_baby_by_name
from logger import get_logger

logger = get_logger("tools")


@tool
def parenting_knowledge(query: str) -> str:
    """查询育儿知识库，获取关于喂养、睡眠、哭闹、辅食、疫苗等专业建议。

    Args:
        query: 要查询的问题，例如 "宝宝睡眠问题" 或 "母乳喂养注意事项"
    """
    logger.info(f"知识库查询: {query}")
    result = search_knowledge(query)
    logger.debug(f"知识库返回: {len(result)} 字符")
    return result


@tool
def manage_baby_profile(name: str, birth_date: str = None,
                        allergy: str = None, notes: str = None) -> str:
    """创建或更新宝宝的专属档案。调用此工具时，必须使用用户提供的真实名字作为 name 参数。

    Args:
        name: 宝宝的真实名字或小名（如"小豆子"），不能使用"宝宝"这类泛称。
        birth_date: 出生日期，格式 YYYY-MM-DD，如果用户只说月龄可不填。
        allergy: 过敏史，如"牛奶蛋白过敏"，无则留空。
        notes: 其他备注，如"3个月，纯母乳喂养"。
    """
    logger.info(f"管理宝宝档案: name={name}, birth_date={birth_date}, allergy={allergy is not None}")
    baby_id = upsert_baby(name, birth_date, allergy, notes)
    profile = get_baby(baby_id)
    logger.info(f"宝宝档案更新成功: baby_id={baby_id}, name={name}")
    return f"已更新 {name} 的档案：当前信息为 {profile}"


# 工具列表
tools = [parenting_knowledge, manage_baby_profile]
