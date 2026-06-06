"""
查询理解与改写模块

功能：
1. 意图分类 —— 基于关键词快速识别用户意图（医疗/喂养/睡眠/发育/教育）
2. 查询改写 —— LLM 将口语化查询改写为 2-3 个检索友好的书面变体
3. 实体抽取 —— 提取月龄、症状、药名等关键实体
"""

import re
from logger import get_logger

logger = get_logger("knowledge.query")

# ── 意图分类关键词 ───────────────────────────────────────────
INTENT_KEYWORDS: dict[str, list[str]] = {
    "medical": [
        "发烧", "咳嗽", "腹泻", "呕吐", "过敏", "湿疹", "感冒", "流感", "肺炎",
        "疫苗", "接种", "体检", "就医", "医院", "急诊", "药", "症状", "疼痛",
        "出血", "红肿", "皮疹", "便秘", "黄疸", "惊厥", "抽搐", "哮喘", "窒息",
        "中毒", "烫伤", "摔伤", "磕碰", "骨折", "溺水", "异物", "呛到",
        "鼻塞", "流涕", "喷嚏", "喉咙", "嗓子", "肚子疼", "拉肚子",
    ],
    "feeding": [
        "母乳", "奶粉", "配方奶", "喂养", "喂奶", "辅食", "断奶", "夜奶",
        "厌奶", "吐奶", "溢奶", "拍嗝", "奶瓶", "吸奶器", "储奶", "奶量",
        "奶嘴", "奶粉冲调", "转奶", "水解奶粉", "深度水解",
    ],
    "sleep": [
        "睡眠", "睡觉", "入睡", "夜醒", "哄睡", "抱睡", "奶睡", "自主入睡",
        "早醒", "晚睡", "作息", "小睡", "睡眠倒退", "睡前程序",
        "白天觉", "并觉", "落地醒", "惊醒", "噩梦", "梦游",
    ],
    "development": [
        "翻身", "坐立", "爬行", "走路", "说话", "长牙", "身高", "体重",
        "生长曲线", "发育", "大运动", "精细动作", "认知", "语言",
        "抬头", "抓握", "站立", "独走", "跑", "跳", "出牙", "换牙",
    ],
    "education": [
        "早教", "绘本", "玩具", "游戏", "动画", "屏幕", "电视", "手机",
        "看书", "学习", "幼儿园", "社交", "启蒙", "英语", "识字",
        "画画", "音乐", "运动", "感统", "专注力", "习惯",
    ],
}

# 优先级：medical > feeding > sleep > development > education
INTENT_PRIORITY = ["medical", "feeding", "sleep", "development", "education"]


def classify_intent(query: str) -> str:
    """基于关键词的快速意图分类，不调用 API"""
    query_lower = query.lower()
    scores: dict[str, int] = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        scores[intent] = sum(1 for kw in keywords if kw in query_lower)

    # 按优先级返回得分最高的意图
    best_intent = "general"
    best_score = 0
    for intent in INTENT_PRIORITY:
        if scores[intent] > best_score:
            best_score = scores[intent]
            best_intent = intent
    return best_intent


# ── 查询改写 ─────────────────────────────────────────────────

REWRITE_PROMPT = """将以下育儿相关的口语化问题，改写为 2~3 个简洁、专业的书面检索查询词。
要求：
- 每个查询词 5~15 字
- 覆盖不同检索角度（如症状+原因、症状+处理、西医学名）
- 每行一个，不要编号，不要解释

用户问题：{query}

改写结果："""


def rewrite_query(query: str, llm_client=None) -> list[str]:
    """
    LLM 查询改写，生成 2-3 个检索变体。

    支持两种客户端接口：
    - 原生 OpenAI: client.chat_completions_create(model=..., messages=[...])
    - LangChain: client.invoke(prompt) → response.content

    Args:
        query: 原始用户查询
        llm_client: LLM 客户端，为 None 时降级为原查询

    Returns:
        查询变体列表，含原始查询
    """
    if llm_client is None:
        logger.debug("无 LLM 客户端，查询改写降级为原始查询")
        return [query]

    try:
        prompt = REWRITE_PROMPT.format(query=query)

        # 原生 OpenAI 客户端
        if hasattr(llm_client, 'chat_completions_create'):
            from config import Config
            response = llm_client.chat_completions_create(
                model=Config.LLM_MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,  # Kimi K2.6 仅支持 temperature=1
                timeout=15,
            )
            content = response.choices[0].message.content
        # LangChain ChatModel
        elif hasattr(llm_client, 'invoke'):
            response = llm_client.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
        else:
            logger.warning("LLM 客户端接口不兼容，降级为原始查询")
            return [query]

        if not content:
            return [query]

        variants = []
        for line in content.strip().split("\n"):
            line = line.strip().lstrip("0123456789.、-•·) ")
            if line and len(line) >= 3:
                variants.append(line)

        if not variants:
            return [query]

        # 去重 + 限制数量（原始查询始终在第一位）
        seen = {query}
        unique_variants = [query]
        for v in variants:
            if v not in seen and len(unique_variants) < 4:
                seen.add(v)
                unique_variants.append(v)

        logger.debug(f"查询改写: '{query}' → {unique_variants}")
        return unique_variants

    except Exception as e:
        logger.warning(f"查询改写失败，降级为原始查询: {e}")
        return [query]


# ── 实体抽取 ─────────────────────────────────────────────────

# 月龄模式
MONTH_AGE_PATTERNS = [
    re.compile(r"(\d{1,2})\s*(个?月|个月|月龄)"),
    re.compile(r"(\d{1,2})\s*(岁|周岁)"),
    re.compile(r"刚?\s*(满月|满\s*\d+\s*个?月|新生儿|初生)"),
]

# 常见症状实体
SYMPTOM_ENTITIES = [
    "发烧", "发热", "高烧", "低烧", "咳嗽", "干咳", "湿咳", "腹泻", "拉肚子",
    "呕吐", "便秘", "湿疹", "红疹", "皮疹", "过敏", "鼻塞", "流鼻涕",
    "哭闹", "烦躁", "拒奶", "厌奶", "吐奶", "溢奶", "肠绞痛", "胀气",
    "抽搐", "惊厥", "黄疸", "脱水", "食欲不振", "体重不增",
]

# 常见药物实体
MEDICINE_ENTITIES = [
    "布洛芬", "对乙酰氨基酚", "泰诺林", "美林", "妈咪爱", "蒙脱石散",
    "益生菌", "维生素D", "维生素AD", "伊可新", "退烧药", "退热药",
    "抗生素", "消炎药", "止咳药", "止泻药",
]


def extract_entities(query: str) -> dict[str, list[str]]:
    """
    从查询中提取关键实体。

    Returns:
        {"month_ages": [...], "symptoms": [...], "medicines": [...]}
    """
    result: dict[str, list[str]] = {
        "month_ages": [],
        "symptoms": [],
        "medicines": [],
    }

    # 月龄
    for pattern in MONTH_AGE_PATTERNS:
        for m in pattern.finditer(query):
            result["month_ages"].append(m.group(0))

    # 症状
    query_lower = query.lower()
    for symptom in SYMPTOM_ENTITIES:
        if symptom in query_lower:
            result["symptoms"].append(symptom)

    # 药物
    for med in MEDICINE_ENTITIES:
        if med in query:
            result["medicines"].append(med)

    return result


# ── 查询预处理入口 ───────────────────────────────────────────

def process_query(query: str, llm_client=None) -> dict:
    """
    查询预处理：意图分类 + 改写 + 实体抽取。

    Returns:
        {
            "original": str,
            "variants": [str, ...],
            "intent": str,
            "entities": dict,
        }
    """
    return {
        "original": query,
        "variants": rewrite_query(query, llm_client),
        "intent": classify_intent(query),
        "entities": extract_entities(query),
    }
