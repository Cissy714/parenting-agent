# 育儿 Agent 记忆系统架构文档

**日期**: 2026-04-25  
**版本**: v1.0  
**作者**: Claude Code

---

## 一、架构概述

本系统实现了**四层记忆架构**，模拟人类记忆的运作方式：

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1: Working Memory (工作记忆)                      │
│  └── 当前对话上下文 (state["messages"])                    │
│  └── 容量有限，只保留当前 session                          │
├─────────────────────────────────────────────────────────┤
│  Layer 2: Fact Memory (事实记忆)                         │
│  └── 宝宝基础档案 (SQLite)                               │
│  └── 姓名、生日、过敏史等静态信息                          │
├─────────────────────────────────────────────────────────┤
│  Layer 3: Episodic Memory (情景记忆)                     │
│  └── 具体事件时间线 (Chroma向量库)                        │
│  └── "2024-01-15 打疫苗后发烧"                            │
├─────────────────────────────────────────────────────────┤
│  Layer 4: Semantic Memory (语义记忆)                     │
│  └── 抽象偏好与风格 (Chroma向量库)                        │
│  └── "偏好自然疗法"、"反对哭声免疫法"                      │
└─────────────────────────────────────────────────────────┘
```

---

## 二、各层详解

### 2.1 Working Memory (工作记忆)

**用途**: 当前对话的短期上下文  
**存储**: `state["messages"]`  
**生命周期**: Session 结束后丢弃  

**特点**:
- 随着对话自动累积
- 超出 LLM 上下文窗口时会被截断
- 无需额外实现，由 LangGraph 自动维护

---

### 2.2 Fact Memory (事实记忆)

**用途**: 宝宝基础档案信息  
**存储**: SQLite (`parenting_memory.db`)  
**文件**: `memory/db.py`

**数据表**:
```sql
-- 宝宝信息表
CREATE TABLE babies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,           -- 姓名
    birth_date TEXT,              -- 出生日期 YYYY-MM-DD
    allergy TEXT,                 -- 过敏史
    notes TEXT,                   -- 额外备注
    created_at TEXT,
    updated_at TEXT
);
```

**API**:
| 函数 | 说明 |
|------|------|
| `create_baby(name, ...)` | 创建宝宝档案 |
| `get_baby(baby_id)` | 获取宝宝信息 |
| `update_baby(baby_id, **kwargs)` | 更新档案 |
| `find_baby_by_name(name)` | 模糊查找 |
| `upsert_baby(name, ...)` | 创建或更新 |

---

### 2.3 Episodic Memory (情景记忆)

**用途**: 存储**具体事件**的时间线  
**特征**:
- 有时间戳
- 具体、详细、情境化
- "何时何地发生了什么"

**存储**: Chroma 向量库  
**Collection**: `episodic_memory`  
**文件**: `memory/episodic.py`

**示例**:
```
"2024-01-15: 小豆子打疫苗后出现低烧37.8度，
用户询问是否正常，Agent建议物理降温并观察"
```

**API**:
| 函数 | 说明 |
|------|------|
| `store_episodic_memory(baby_id, summary)` | 存储事件摘要 |
| `search_episodic_memory(baby_id, query, k=3)` | 语义检索历史事件 |

**元数据**:
```python
{
    "baby_id": 1,
    "timestamp": "2024-01-15T10:30:00"
}
```

---

### 2.4 Semantic Memory (语义记忆)

**用途**: 存储**抽象知识**和**用户偏好**  
**特征**:
- 无时间戳（持久有效）
- 概括、抽象、去情境化
- "用户喜欢什么/讨厌什么/相信什么"

**存储**: Chroma 向量库  
**Collection**: `semantic_memory`  
**文件**: `memory/semantic.py` ⭐ 新增

**偏好类别**:
| 类别 | 说明 | 示例 |
|------|------|------|
| `medical` | 医疗偏好 | 倾向自然疗法，尽量避免药物 |
| `feeding` | 喂养偏好 | 坚持纯母乳喂养 |
| `sleep` | 睡眠训练偏好 | 反对哭声免疫法 |
| `parenting_style` | 育儿风格 | 认同温和育儿法 |
| `education` | 教育理念 | 重视早期阅读 |
| `general` | 一般偏好 | 其他难以分类的偏好 |

**API**:
| 函数 | 说明 |
|------|------|
| `store_user_preference(baby_id, category, content)` | 存储偏好 |
| `get_user_preferences(baby_id, category=None)` | 获取偏好列表 |
| `get_preferences_by_query(baby_id, query)` | 语义检索偏好 |
| `extract_and_store_preferences(messages, baby_id, llm_client)` | 自动提取偏好 |

**元数据**:
```python
{
    "baby_id": 1,
    "category": "medical",
    "confidence": 1.0
}
```

---

## 三、工作流程

### 3.1 对话时的记忆加载流程

```
用户输入
    ↓
[Working Memory] 加载当前 messages 上下文
    ↓
[Fact Memory] 加载宝宝档案 (年龄、过敏等)
    ↓
[Episodic Memory] 检索相关历史事件
    ↓
[Semantic Memory] 检索相关偏好（NEW）
    ↓
LLM 生成回答（结合以上所有上下文）
```

### 3.2 对话结束时的记忆存储流程

```
对话结束
    ↓
生成对话摘要
    ↓
    ├──→ 存入 [Episodic Memory]（具体事件）
    └──→ 提取偏好 → 存入 [Semantic Memory]（抽象偏好）
```

---

## 四、文件修改记录

### 4.1 新增文件

#### `memory/semantic.py` ⭐ NEW
- **功能**: 语义记忆完整实现
- **包含**:
  - `store_user_preference()` - 存储偏好
  - `get_user_preferences()` - 获取偏好
  - `get_preferences_by_query()` - 语义检索
  - `extract_and_store_preferences()` - 自动提取偏好

### 4.2 修改文件

#### `memory/episodic.py`
**修改内容**:
```python
# Before
from knowledge.knowledge_base import embeddings
DB_DIR = "./chroma_data"

# After
from knowledge.knowledge_base import embeddings, DB_DIR
# 统一使用 knowledge_base.py 中的 DB_DIR ("chroma_parenting_db")
```

**原因**: 确保情景记忆和语义记忆使用同一个数据库目录

#### `nodes.py`
**修改内容**:

1. **导入语义记忆模块**:
```python
from memory.semantic import (
    get_user_preferences,
    get_preferences_by_query,
    extract_and_store_preferences
)
```

2. **`call_model` 节点**: 增加语义记忆检索
```python
# 检索语义记忆（用户偏好和风格）
semantic_context = get_preferences_by_query(baby_id, user_query, k=3)
if semantic_context:
    system_content += f"\n\n【用户的育儿偏好与风格】\n{semantic_context}"
```

3. **`summarize_conversation` 节点**: 增加自动提取偏好
```python
# 提取并存储语义记忆（用户偏好）
extract_model = ChatOpenAI(..., temperature=0.3)
extracted_prefs = extract_and_store_preferences(
    recent_msgs, baby_id, llm_client=extract_model
)
```

---

## 五、使用示例

### 5.1 存储用户偏好（手动）

```python
from memory.semantic import store_user_preference

# 记录用户的医疗偏好
store_user_preference(
    baby_id=1,
    category="medical",
    content="倾向自然疗法，发烧优先考虑物理降温"
)

# 记录用户的喂养偏好
store_user_preference(
    baby_id=1,
    category="feeding",
    content="坚持纯母乳喂养，对配方奶有顾虑"
)
```

### 5.2 获取用户偏好

```python
from memory.semantic import get_user_preferences

# 获取所有偏好
all_prefs = get_user_preferences(baby_id=1)
print(all_prefs)
# 输出:
# - [医疗偏好] 倾向自然疗法，发烧优先考虑物理降温
# - [喂养偏好] 坚持纯母乳喂养，对配方奶有顾虑

# 按类别获取
medical_prefs = get_user_preferences(baby_id=1, category="medical")
```

### 5.3 语义检索偏好

```python
from memory.semantic import get_preferences_by_query

# 查询与"发烧"相关的偏好
relevant_prefs = get_preferences_by_query(
    baby_id=1,
    query="发烧如何处理",
    k=3
)
```

### 5.4 自动提取偏好

```python
from memory.semantic import extract_and_store_preferences
from langchain_openai import ChatOpenAI

# 从对话中自动提取
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
preferences = extract_and_store_preferences(
    messages=recent_messages,
    baby_id=1,
    llm_client=llm
)
# 返回提取到的偏好列表
```

---

## 六、系统提示词整合示例

当所有记忆加载后，系统提示词结构如下：

```
你是一个专业的育儿助手...

【重要规则】
1. 回答任何育儿问题前，必须先调用 parenting_knowledge 工具检索知识库。
2. ...

当前宝宝档案：{'name': '小豆子', 'birth_date': '2024-01-01', 'allergy': None}

【与该宝宝相关的历史事件】
2024-04-20: 小豆子3个月时进行睡眠训练，采用渐进式等待法，
一周后成功实现自主入睡...

【用户的育儿偏好与风格】
- [医疗偏好] 倾向自然疗法，发烧优先考虑物理降温，尽量避免药物
- [睡眠偏好] 反对哭声免疫法，追求温和育儿方式

请尊重用户的偏好和价值观，调整建议的表达方式。
```

---

## 七、未来优化方向

| 优先级 | 优化项 | 说明 |
|--------|--------|------|
| 🟡 中 | 记忆去重 | 存入前检查相似度，避免重复 |
| 🟡 中 | 记忆重要性评分 | 只存储有价值的对话 |
| 🟢 低 | 定期清理旧记忆 | >90天的记忆自动归档 |
| 🟢 低 | 记忆层次压缩 | 日摘要 → 周摘要 → 月摘要 |

---

## 八、数据库文件说明

| 文件/目录 | 用途 |
|-----------|------|
| `parenting_memory.db` | SQLite 数据库，存储宝宝档案 |
| `chroma_parenting_db/` | Chroma 向量数据库 |
| `chroma_parenting_db/episodic_memory` | 情景记忆 Collection |
| `chroma_parenting_db/semantic_memory` | 语义记忆 Collection |
| `chroma_parenting_db/` | 育儿知识库 Collection |

---

**文档结束**
