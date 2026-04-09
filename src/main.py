#!/usr/bin/env python3
"""
程序主入口。

给初学者的理解方式：
- 这里负责解析命令行参数（例如 `--debug`）。
- 然后启动 GUI 窗口。
- 如果启动失败，会记录日志并以非 0 状态退出。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gui.window import run_app
from src.utils.logger import logger


def main():
    # 解析启动参数：是否调试、版本号输出等。
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Mac Watermark Remover - 视频水印智能识别与去除软件'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='Mac Watermark Remover v1.0.0'
    )
    
    args = parser.parse_args()
    
    logger.info("Starting Mac Watermark Remover...")
    
    try:
        # 真正启动 GUI 应用。
        run_app(debug=args.debug)
    except KeyboardInterrupt:
        # 用户手动中断（例如 Ctrl+C）。
        logger.info("Application terminated by user")
    except Exception as e:
        # 兜底异常：记录错误并返回失败退出码。
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
