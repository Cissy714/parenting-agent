import os
from dotenv import load_dotenv
from functools import wraps
import time
import random
import httpx
from openai import OpenAI
from logger import get_logger

load_dotenv()

logger = get_logger("config")

# 精细超时: 连接 15s, 读取 300s (给 kimi-k2.6 思考留足时间)
API_TIMEOUT = httpx.Timeout(connect=15.0, read=300.0, write=60.0, pool=15.0)


class ConfigError(Exception):
    """配置错误异常"""
    pass


class Config:
    """配置类，包含配置校验"""
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "gpt-4o-mini")

    @classmethod
    def validate(cls):
        """校验配置是否有效"""
        errors = []

        # 校验 API KEY
        if not cls.LLM_API_KEY or cls.LLM_API_KEY.strip() == "":
            errors.append("LLM_API_KEY 未设置或为空")
        elif not cls.LLM_API_KEY.startswith("sk-") and len(cls.LLM_API_KEY) < 10:
            errors.append("LLM_API_KEY 格式似乎不正确")

        # 校验 BASE_URL
        if not cls.LLM_BASE_URL:
            errors.append("LLM_BASE_URL 不能为空")
        elif not (cls.LLM_BASE_URL.startswith("http://") or cls.LLM_BASE_URL.startswith("https://")):
            errors.append(f"LLM_BASE_URL 必须以 http:// 或 https:// 开头，当前值: {cls.LLM_BASE_URL}")

        # 校验 MODEL_ID
        if not cls.LLM_MODEL_ID or cls.LLM_MODEL_ID.strip() == "":
            errors.append("LLM_MODEL_ID 不能为空")

        if errors:
            for error in errors:
                logger.error(f"配置校验失败: {error}")
            raise ConfigError("配置校验失败:\n" + "\n".join(f"  - {e}" for e in errors))

        logger.info("配置校验通过")
        return True


# 重试装饰器
def retry_with_backoff(max_retries=5, initial_delay=2.0, max_delay=90.0, exponential_base=2.0):
    """
    指数退避重试装饰器

    Args:
        max_retries: 最大重试次数
        initial_delay: 初始延迟（秒）
        max_delay: 最大延迟（秒）
        exponential_base: 指数基数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # 判断是否应该重试
                    error_msg = str(e).lower()
                    retryable_errors = [
                        'rate limit', 'timeout', 'timed out', 'timedout',
                        'connection', 'temporarily', 'overloaded',
                        'internal server error', 'service unavailable', 'too many requests',
                        '503', '502', '504', '429', '408',
                        'remote protocol error', 'read timeout',
                    ]
                    should_retry = any(err in error_msg for err in retryable_errors)

                    if not should_retry and attempt < max_retries:
                        # 非可重试错误，直接抛出
                        raise

                    if attempt == max_retries:
                        # 已用完所有重试次数
                        logger.error(f"API调用失败，已重试 {max_retries} 次: {func.__name__} - {e}")
                        raise

                    # 429 限流用更长的初始等待
                    if '429' in error_msg or 'rate limit' in error_msg or 'too many requests' in error_msg:
                        delay = max(delay, 10.0)

                    # 添加随机抖动避免惊群效应
                    jitter = random.uniform(0, 0.1 * delay)
                    actual_delay = min(delay + jitter, max_delay)

                    logger.warning(f"API调用失败，第 {attempt + 1}/{max_retries} 次: {func.__name__} - {e}")
                    logger.info(f"等待 {actual_delay:.2f} 秒后重试...")

                    time.sleep(actual_delay)
                    delay *= exponential_base

            raise last_exception
        return wrapper
    return decorator


# API 调用包装器
class APIClientWithRetry:
    """带重试机制的 API 客户端包装器"""

    def __init__(self, client):
        self.client = client

    @retry_with_backoff(max_retries=2, initial_delay=2.0, max_delay=30.0)
    def chat_completions_create(self, *args, timeout=None, **kwargs):
        """带重试的 chat.completions.create"""
        kwargs['timeout'] = timeout or API_TIMEOUT
        return self.client.chat.completions.create(*args, **kwargs)


# 首次导入时校验配置
try:
    Config.validate()
    logger.info("配置校验通过，应用启动中...")
except ConfigError as e:
    logger.critical(f"配置错误: {e}")
    raise

# 创建原生 OpenAI 客户端
openai_client_raw = OpenAI(
    api_key=Config.LLM_API_KEY,
    base_url=Config.LLM_BASE_URL,
    timeout=API_TIMEOUT,
)

# 包装为带重试的客户端
openai_client = APIClientWithRetry(openai_client_raw)
