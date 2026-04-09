"""
GUI 层导出模块。

这个文件只做一件事：统一导出窗口类和前后端桥接 API，
让其他模块可以通过 `src.gui` 直接导入。
"""

from .window import MainWindow
from .api import API

# 显式声明对外可见符号，减少 `from src.gui import *` 的歧义。
__all__ = ['MainWindow', 'API']
