"""
日志模块。

职责：
1. 提供终端彩色日志格式。
2. 统一创建项目 logger（控制台 + 可选文件）。
3. 默认把日志写到 `logs/app_YYYY-MM-DD.log`。
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """给终端输出增加颜色，便于快速区分日志级别。"""

    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
    }
    RESET = '\033[0m'
    
    def format(self, record):
        """在原始格式化前，先给 level 名称套上 ANSI 颜色。"""
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = "MacWatermarkRemover",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    创建并返回项目 logger。

    关键点：
    - 已创建过 handler 时直接复用，避免重复打印同一条日志。
    - `log_file` 不为空时再额外启用文件日志。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    console_format = ColoredFormatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def get_default_log_file() -> str:
    """生成当天日志文件路径，并确保 `logs` 目录存在。"""
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    today = datetime.now().strftime('%Y-%m-%d')
    return str(log_dir / f"app_{today}.log")


logger = setup_logger(log_file=get_default_log_file())
