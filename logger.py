"""日志配置模块 - 统一的日志系统"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# 日志目录
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# 日志格式
CONSOLE_FORMAT = "%(asctime)s - %(levelname)s - [%(name)s] %(message)s"
FILE_FORMAT = "%(asctime)s - %(levelname)s - [%(name)s] - [%(filename)s:%(lineno)d] %(message)s"

# 日志级别映射
LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def setup_logger(name: str = None, level: str = "info") -> logging.Logger:
    """
    设置并获取日志记录器

    Args:
        name: 日志记录器名称，默认 None 使用根记录器
        level: 日志级别，可选 debug/info/warning/error/critical

    Returns:
        logging.Logger 实例
    """
    logger = logging.getLogger(name)

    # 避免重复配置
    if logger.handlers:
        return logger

    # 设置级别
    log_level = LEVELS.get(level.lower(), logging.INFO)
    logger.setLevel(log_level)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(CONSOLE_FORMAT, datefmt="%H:%M:%S")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 文件处理器
    log_file = LOG_DIR / f"{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # 文件记录更详细的日志
    file_formatter = logging.Formatter(FILE_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 错误日志单独文件
    error_file = LOG_DIR / f"{datetime.now().strftime('%Y%m%d')}_error.log"
    error_handler = logging.FileHandler(error_file, encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    logger.addHandler(error_handler)

    return logger


# 默认日志记录器
logger = setup_logger("parenting_agent")


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的日志记录器"""
    return setup_logger(name)
