# Parenting Agent — 育儿智能助手

基于 LLM + RAG + 四层记忆架构的智能育儿助手，支持多宝宝档案管理、混合知识检索、长期偏好学习和三层安全兜底。

## 功能特性

- **混合知识检索**：BM25 关键词 + 向量语义多路召回，RRF 融合 + Jaccard 去重，覆盖专有名词和口语化表达
- **四层记忆系统**：工作记忆（上下文）/ 事实记忆（档案）/ 情景记忆（事件）/ 语义记忆（偏好），跨会话持久化
- **多宝宝档案**：自动识别宝宝名字并创建档案，支持月龄、过敏史、出生日期等信息管理
- **三层安全机制**：检索层风险分级扩召回 → 生成层结构化约束 → 输出层正则拦截 + 打回重生成
- **流式对话**：Streamlit Web 界面，实时展示思考过程、工具调用和回复生成
- **量化评估**：209 条标准化测试集，覆盖检索/安全/Agent/记忆四个维度

## 项目结构

```
parenting-agents/
├── main.py                     # CLI 交互入口
├── demo.py                     # Streamlit Web 界面
├── agent.py                    # Agent 主入口（向后兼容导出）
├── graph.py                    # LangGraph 状态图定义
├── nodes.py                    # 工作流节点（档案加载、LLM推理、记忆、安全检测）
├── state.py                    # 状态数据结构
├── tools.py                    # 工具定义（知识检索、档案管理）
├── messages.py                 # 消息格式转换（保留 reasoning_content）
├── config.py                   # 配置管理、启动校验、指数退避重试
├── logger.py                   # 统一日志系统（按天 + 按错误分级）
├── knowledge/                  # 知识检索子系统
│   ├── knowledge_base.py       # 混合检索主入口、安全分级、精排
│   ├── bm25_retriever.py       # BM25 关键词检索器（单例）
│   ├── fusion.py               # RRF 融合 + Jaccard 去重
│   └── query_processor.py      # 意图分类、查询改写、实体抽取
├── memory/                     # 记忆子系统
│   ├── db.py                   # SQLite 宝宝档案 CRUD
│   ├── episodic.py             # 情景记忆（ChromaDB 向量存储）
│   └── semantic.py             # 语义记忆（偏好提取与检索）
├── evaluation/                 # 评估子系统
│   ├── run_all_eval.py         # 四维评估统一入口
│   ├── runners/                # 评估运行器（检索/安全/Agent/记忆）
│   ├── generators/             # 测试集生成脚本
│   └── data/                   # 209 条 JSON 测试用例
└── scripts/                    # 数据构建脚本
    ├── build_knowledge_base.py # 书籍清洗、分块、向量化
    └── build_db.py             # 数据库初始化
```

## 核心架构

```
用户输入 → LangGraph 状态图
  ├── load_profile     → 自动识别/加载宝宝档案
  ├── load_memory      → 语义检索情景记忆 + 用户偏好
  ├── agent ⇄ tools    → LLM 推理 + 工具调用循环
  │   ├── parenting_knowledge   → 混合检索（BM25 + 向量 → RRF 融合）
  │   └── manage_baby_profile   → 宝宝档案 CRUD
  └── summarize        → 对话摘要 → 情景记忆 + 语义偏好持久化
```

**检索流程**：查询改写（LLM）→ 多路召回（BM25 + 向量）→ RRF 融合 → Jaccard 去重 → 安全增强排序

**记忆流程**：每轮对话结束后，LLM 异步提取两类信息——具体事件存入情景记忆（ChromaDB），长期偏好存入语义记忆（ChromaDB），下次对话时以用户 query 为锚点语义检索注入 System Prompt。

## 快速开始

### 环境要求

- Python 3.10+
- 兼容 OpenAI 协议的 LLM API（默认使用 Kimi K2.6）

### 安装

```bash
# 克隆项目
git clone <repo-url>
cd parenting-agents

# 安装依赖
pip install -r requirements.txt
```

### 配置

复制环境变量模板并填入你的 API Key：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
LLM_API_KEY="sk-your-key-here"
LLM_MODEL_ID="kimi-k2.6"
LLM_BASE_URL="https://api.moonshot.cn/v1"
SERPAPI_API_KEY="your-serpapi-key"      # 可选
TAVILY_API_KEY="your-tavily-key"        # 可选
```

启动时 `config.py` 会自动校验配置合法性（Key 格式、URL 协议、Model ID 非空），校验失败会阻止启动。

### 构建知识库

首次使用需要构建向量数据库：

```bash
python scripts/build_knowledge_base.py
```

### 运行

**命令行模式：**

```bash
python main.py
```

**Streamlit Web 界面：**

```bash
streamlit run demo.py
# 或双击 run_demo.bat (Windows)
```

打开 http://localhost:8501 ，界面实时展示：
- 思考过程（Kimi thinking trace）
- 工具调用及返回结果
- 流式回复生成

## 使用示例

```
> 宝宝叫小豆子，3个月，牛奶蛋白过敏
已更新 小豆子 的档案...

> 小豆子今天发烧38.5度怎么办
[检索知识库：婴儿发烧处理]
[加载情景记忆：上次疫苗后低烧...]
[加载语义记忆：倾向物理降温...]

根据小豆子3个月月龄和牛奶蛋白过敏的情况，建议...
```

对话结束后自动存储为情景记忆和语义偏好，下次对话时自动加载。

## 评估体系

209 条标准化测试集，覆盖四个维度：

| 维度 | 测试数 | 核心指标 | 当前值 |
|------|--------|----------|--------|
| 检索质量 | 100 | recall@3 / MRR | 72.70% / 0.894 |
| 安全性 | 45 | 就医准确率 / CRITICAL 通过率 | 93.33% / 100% |
| Agent | 39 | 工具调用 F1 / 回答质量 | 94.12% / 73.77% |
| 记忆 | 25 | 档案回忆率 / 偏好学习率 | 70.00% / 84.00% |

运行评估：

```bash
python evaluation/run_all_eval.py           # 全量评估
python evaluation/run_all_eval.py --skip-memory  # 跳过指定维度
```

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| Agent 框架 | LangGraph + LangChain |
| LLM | Kimi K2.6（兼容 OpenAI 协议） |
| 向量数据库 | ChromaDB（HNSW 索引） |
| 嵌入模型 | BAAI/bge-large-zh-v1.5 |
| 关键词检索 | BM25（jieba 分词） |
| 结构化存储 | SQLite |
| Web 界面 | Streamlit |

## License

MIT
