# 两阶段检索系统文档

## 概述

实现了基于"粗召回 + 精排"的两阶段检索系统，在保证高召回率的同时提升精确度，特别针对育儿场景中的安全关键查询进行了优化。

## 系统架构

```
用户查询
    ↓
[阶段0] 安全检测
    ↓ (如果是安全查询，扩大召回)
[阶段1] 粗召回 (向量检索，initial_k=10-15)
    ↓
[阶段2] 精排 (重排序算法，选出top_k=3-5)
    ↓
格式化输出 (+ 安全提示)
```

## 核心功能

### 1. 安全关键词检测

```python
SAFETY_KEYWORDS = [
    "过敏", "窒息", "中毒", "急救", "急诊", "危险", "严重", "紧急",
    "禁用", "禁止", "不能", "死亡", "抽搐", "昏迷", "呼吸困难",
    "高烧", "高烧不退", "40度", "41度", "血便", "脱水",
    "药", "药物", "吃药", "用药", "剂量", "喂药"
]
```

检测到安全关键词时：
- 扩大召回数量（initial_k=15）
- 返回更多结果（k=5）
- 添加安全提示

### 2. 查询扩展（安全模式）

针对安全查询，自动生成相关查询：

```python
SAFETY_TOPICS = {
    "发烧": ["退烧药", "物理降温", "就医指征", "高热惊厥"],
    "咳嗽": ["窒息风险", "止咳药", "蜂蜜"],
    "腹泻": ["脱水", "补液", "电解质"],
    ...
}
```

### 3. 重排序策略

#### 简单重排序（默认）
- 关键词匹配加分
- 安全关键词优先
- 内容去重

#### LLM重排序（可选）
- 使用大模型判断相关性
- 更精确但耗时
- 通过 `use_llm_rerank=True` 启用

## 使用方法

### 基本用法

```python
from knowledge.knowledge_base import search_knowledge

# 默认参数：两阶段检索，自动安全检测
result = search_knowledge("宝宝发烧怎么办")

# 自定义参数
result = search_knowledge(
    query="宝宝发烧怎么办",
    k=5,                    # 最终结果数
    initial_k=10,           # 粗召回数量
    use_llm_rerank=False,   # 是否使用LLM重排
    enable_safety_boost=True  # 是否启用安全检测
)
```

### 评估检索效果

```python
from knowledge.knowledge_base import evaluate_retrieval

test_set = [
    ("宝宝发烧怎么办", ["发烧护理指南"]),
    ("母乳喂养", ["母乳喂养指南"]),
]

metrics = evaluate_retrieval(test_set)
print(metrics)
# {'total': 2, 'recall_at_3': 0.5, 'recall_at_5': 0.8, ...}
```

## API 接口

### search_knowledge

```python
def search_knowledge(
    query: str,
    k: int = 3,                    # 最终返回文档数
    initial_k: int = 10,           # 粗召回文档数
    use_llm_rerank: bool = False,  # 是否使用LLM重排
    enable_safety_boost: bool = True  # 是否启用安全检测
) -> str:
```

### _is_safety_critical

```python
def _is_safety_critical(query: str) -> bool:
    """检测查询是否包含安全关键词"""
```

### _get_safety_related_queries

```python
def _get_safety_related_queries(query: str) -> List[str]:
    """获取安全相关的扩展查询列表"""
```

## 测试

运行测试脚本：

```bash
# 测试安全关键词检测
python test_retrieval.py safety

# 测试查询扩展
python test_retrieval.py expand

# 测试两阶段检索对比
python test_retrieval.py two-stage

# 测试安全模式
python test_retrieval.py safety-mode

# 运行所有测试
python test_retrieval.py
```

## 优化建议

### 1. 调整召回参数

- **普通查询**: `initial_k=10, k=3`
- **安全查询**: `initial_k=15, k=5`

### 2. 扩展安全词库

根据实际业务场景，补充更多安全关键词：

```python
SAFETY_KEYWORDS += ["你的新关键词", "..."]
```

### 3. 添加领域特定重排

针对不同主题（喂养、睡眠、疾病）实现专门的重排逻辑。

## 性能考虑

| 模式 | 向量检索次数 | 平均延迟 | 适用场景 |
|------|------------|---------|---------|
| 传统 (k=3) | 1 | ~50ms | 简单查询 |
| 两阶段 (10→3) | 1 | ~50ms | 平衡场景 |
| 安全模式 (15→5) | 1-3 | ~100ms | 安全关键查询 |
| +LLM重排 | 1 | ~500ms | 高精度要求 |

## 日志输出

检索过程会记录详细日志：

```
[INFO] 知识库查询: query='宝宝发烧40度怎么办', k=3, initial_k=15
[WARNING] 安全关键查询检测，扩大召回至 initial_k=15
[DEBUG] 安全模式召回: 12 条（来自 3 个查询）
[DEBUG] 阶段1-粗召回: 12 条候选
[DEBUG] 阶段2-精排后: 5 条
```

## 覆盖文件

- `knowledge/knowledge_base.py` - 主要实现
- `tools.py` - 工具接口（无需修改，兼容）
- `test_retrieval.py` - 测试脚本（新增）
