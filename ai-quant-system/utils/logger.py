"""
AI量化交易系统 - 日志模块
支持文件日志和控制台日志
"""
import os
import sys
from loguru import logger
from pathlib import Path


def setup_logger(log_dir: str = "../logs", log_level: str = "INFO") -> None:
    """配置日志系统"""
    # 移除默认handler
    logger.remove()

    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 控制台日志
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )

    # 文件日志 - 主日志
    logger.add(
        log_path / "quant_system_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="00:00",
        retention="30 days",
        compression="zip",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    )

    # 文件日志 - 交易日志
    logger.add(
        log_path / "trades_{time:YYYY-MM-DD}.log",
        level="INFO",
        rotation="00:00",
        retention="90 days",
        compression="zip",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
    )

    # 文件日志 - 错误日志
    logger.add(
        log_path / "errors_{time:YYYY-MM-DD}.log",
        level="ERROR",
        rotation="00:00",
        retention="365 days",
        compression="zip",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}",
    )


def get_logger(name: str = "quant") -> object:
    """获取日志实例"""
    return logger.bind(name=name)
