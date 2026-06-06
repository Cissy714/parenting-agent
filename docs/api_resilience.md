---
name: API容错与重试机制实现
description: 为项目添加了API容错、重试机制和配置校验功能
type: project
originSessionId: 30ad37f5-eeb6-44df-9331-eced65f2ca0e
---
## 已实现的改进

### 1. 配置校验 (config.py)
- 添加了 `ConfigError` 异常类
- 实现了 `Config.validate()` 方法，校验：
  - LLM_API_KEY 是否设置且格式正确
  - LLM_BASE_URL 是否为有效的 HTTP/HTTPS URL
  - LLM_MODEL_ID 是否为空
- 启动时自动校验，校验失败会打印具体错误并抛出异常

### 2. 重试装饰器 (config.py)
- 实现了 `retry_with_backoff` 装饰器，支持：
  - 指数退避重试（可配置初始延迟、最大延迟、指数基数）
  - 智能识别可重试错误（rate limit、timeout、connection、5xx错误等）
  - 随机抖动避免惊群效应
  - 默认 3 次重试，最大延迟 60 秒

### 3. API 客户端包装器 (config.py)
- 创建了 `APIClientWithRetry` 类包装 OpenAI 客户端
- `chat_completions_create` 方法带重试和超时控制（默认 60 秒）

### 4. 模型调用改进 (nodes.py)
- ChatOpenAI 模型添加 `request_timeout=60` 和 `max_retries=3`
- `call_model` 节点添加 try-except 包装，API 失败返回友好错误消息
- `summarize_conversation` 节点添加错误处理，摘要失败不中断流程

### 5. 语义记忆改进 (memory/semantic.py)
- `extract_and_store_preferences` 添加重试逻辑（最多 2 次）
- 导入 `time` 模块支持延迟

### 6. 知识库初始化改进 (knowledge/knowledge_base.py)
- 添加 `_init_embeddings()` 函数包装 HuggingFaceEmbeddings 初始化
- 初始化失败打印明确错误信息并退出程序

## 配置示例

```python
# 重试装饰器使用
@retry_with_backoff(max_retries=3, initial_delay=1.0, max_delay=60.0)
def my_api_call():
    pass

# API 客户端使用（自动带重试）
from config import openai_client
response = openai_client.chat_completions_create(..., timeout=60)
```

## 覆盖的文件
- `config.py` - 配置校验 + 重试机制
- `nodes.py` - API 调用错误处理
- `memory/semantic.py` - 重试逻辑
- `knowledge/knowledge_base.py` - 初始化错误处理
