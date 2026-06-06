# 日志系统文档

## 概述

项目使用 Python 标准库 `logging` 实现统一日志系统，支持控制台输出和文件存储。

## 日志文件位置

```
logs/
├── YYYYMMDD.log          # 完整日志（DEBUG及以上）
└── YYYYMMDD_error.log    # 仅错误日志（ERROR及以上）
```

## 使用方法

### 1. 获取日志记录器

```python
from logger import get_logger

logger = get_logger("module_name")
```

### 2. 记录日志

```python
logger.debug("调试信息")
logger.info("一般信息")
logger.warning("警告信息")
logger.error("错误信息")
logger.critical("严重错误")
```

## 日志记录器命名规范

| 模块 | 记录器名称 |
|------|-----------|
| config.py | config |
| nodes.py | nodes |
| tools.py | tools |
| memory/db.py | memory.db |
| memory/episodic.py | memory.episodic |
| memory/semantic.py | memory.semantic |
| knowledge/knowledge_base.py | knowledge |
| main.py | main |

## 日志级别

- **DEBUG**: 详细调试信息（如函数入参、返回值）
- **INFO**: 重要操作记录（如启动、完成、状态变化）
- **WARNING**: 警告信息（如重试、降级处理）
- **ERROR**: 错误信息（如 API 调用失败、数据异常）
- **CRITICAL**: 严重错误（如配置错误、系统无法启动）

## 日志格式

### 控制台输出
```
HH:MM:SS - LEVEL - [logger_name] message
```

### 文件输出
```
YYYY-MM-DD HH:MM:SS - LEVEL - [logger_name] - [filename:lineno] message
```

## 关键日志点

### config.py
- 配置校验结果
- API 调用重试记录
- 配置错误（CRITICAL）

### nodes.py
- 模型调用开始/完成
- 工具调用记录
- 宝宝档案加载/更新
- API 错误

### tools.py
- 知识库查询
- 宝宝档案创建/更新

### memory/db.py
- 数据库初始化
- 宝宝档案 CRUD 操作
- 查询结果统计

### memory/episodic.py & semantic.py
- 记忆存储/检索
- 语义偏好提取

### knowledge/knowledge_base.py
- Embeddings 初始化
- 知识库查询及结果统计
