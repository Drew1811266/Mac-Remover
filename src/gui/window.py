"""
桌面主窗口封装（PyWebView）。

职责：
1. 创建并管理应用主窗口。
2. 把 Python API 暴露给前端页面。
3. 在窗口生命周期中处理进度推送、资源回收和配置保存。
"""

import os
import json
import webview
from pathlib import Path
from typing import Optional

from .api import API
from ..config import get_config, save_config
from ..utils.logger import logger
from ..utils.memory_cleanup import release_unified_memory


class MainWindow:
    """应用主窗口对象，负责 UI 壳层和生命周期管理。"""

    def __init__(
        self,
        title: str = "Mac Watermark Remover",
        width: int = 1400,
        height: int = 900,
        resizable: bool = True,
        fullscreen: bool = False,
        min_size: tuple = (1200, 700)
    ):
        """
        初始化窗口配置与 API 实例。

        参数是窗口的基础外观设置，不影响业务逻辑。
        """
        self.title = title
        self.width = width
        self.height = height
        self.resizable = resizable
        self.fullscreen = fullscreen
        self.min_size = min_size
        
        self.window: Optional[webview.Window] = None
        self.api = API()
        self.config = get_config()
        
        self._template_path = self._resolve_template_path()

    def _resolve_template_path(self) -> Path:
        """
        决定加载哪个前端入口文件。

        优先使用 React 构建产物 `templates/dist/index.html`，
        若不存在再回退到旧模板页面。
        """
        templates_dir = Path(__file__).parent / "templates"
        dist_path = templates_dir / "dist" / "index.html"
        legacy_path = templates_dir / "index.html"
        # Manual-annotation-only runtime uses React dist as primary UI.
        if dist_path.exists():
            return dist_path
        return legacy_path
    
    def _get_html_path(self) -> str:
        """返回要加载的本地 HTML 路径。"""
        return str(self._template_path)
    
    def _on_loaded(self):
        """窗口加载完成后，注册进度回调到 API。"""
        logger.info("Window loaded")
        
        self.api.set_progress_callback(self._on_progress)
    
    def _on_closing(self):
        """
        窗口关闭前的清理流程。

        包括：停止视频处理、取消模型下载、关闭原生播放器、
        清理会话临时数据并落盘配置。
        """
        logger.info("Window closing")
        
        if self.api.processor and self.api.processor.is_processing():
            self.api.processor.stop_processing()

        try:
            self.api.cancel_model_download()
        except Exception as e:
            logger.warning(f"Cancel model download on close failed: {e}")

        try:
            self.api.cancel_upscale_model_download()
        except Exception as e:
            logger.warning(f"Cancel upscale model download on close failed: {e}")

        try:
            self.api.cancel_upscale_task()
        except Exception as e:
            logger.warning(f"Cancel upscale task on close failed: {e}")
        
        self.api.close_all_native_players()

        try:
            self.api.clear_session_transient_data()
        except Exception as e:
            logger.warning(f"Clear transient session data on close failed: {e}")

        try:
            release_unified_memory("window_closing")
        except Exception as e:
            logger.warning(f"Unified memory cleanup on close failed: {e}")
        
        save_config()
    
    def _on_progress(self, data: dict):
        """
        把后端进度数据安全地转发给前端。

        这里只透传白名单字段，避免把无关内容注入到前端事件。
        """
        if self.window:
            payload_data = {}
            for key in (
                'progress',
                'message',
                'status',
                'processed_frames',
                'total_frames',
                'estimated_time',
                'phase',
                'eta_seconds',
                'throughput_fps',
                'opaque_infer',
            ):
                if key in data and data[key] is not None:
                    payload_data[key] = data[key]
            
            if not payload_data:
                return
            
            payload = json.dumps(payload_data, ensure_ascii=False)
            
            js_code = f"""
            window.dispatchEvent(new CustomEvent('wmr-progress', {{ detail: {payload} }}));
            const root = document.querySelector('[x-data]');
            let app = null;
            if (root) {{
                if (root.__x && root.__x.$data) {{
                    app = root.__x.$data;
                }} else if (Array.isArray(root._x_dataStack) && root._x_dataStack.length > 0) {{
                    app = root._x_dataStack[0];
                }}
            }}
            if (!app && window.__wmrApp) {{
                app = window.__wmrApp;
            }}
            if (app && typeof app.updateProgress === 'function') {{
                app.updateProgress({payload});
            }}
            """
            try:
                self.window.evaluate_js(js_code)
            except Exception as e:
                logger.warning(f"Push progress to UI failed: {e}")
    
    def _get_device_info_str(self) -> str:
        """拼出设备与内存占用的简短文本，便于状态展示。"""
        from ..utils.device import get_device_info, get_memory_usage
        
        device_info = get_device_info()
        used, total = get_memory_usage()
        
        return f"{device_info.name} ({device_info.device_type}) | {used:.1f}GB / {total:.1f}GB"
    
    def create(self):
        """
        创建 pywebview 窗口并绑定事件。

        返回值是底层 `webview.Window` 对象。
        """
        self.window = webview.create_window(
            title=self.title,
            url=self._get_html_path(),
            js_api=self.api,
            width=self.width,
            height=self.height,
            resizable=self.resizable,
            fullscreen=self.fullscreen,
            min_size=self.min_size,
            text_select=False,
            confirm_close=True
        )

        if hasattr(self.api, "bind_window"):
            self.api.bind_window(self.window)
        
        self.window.events.loaded += self._on_loaded
        self.window.events.closing += self._on_closing
        
        return self.window
    
    def show(self, debug: bool = False, http_server: bool = False):
        """启动窗口事件循环。首次调用会先自动创建窗口。"""
        if self.window is None:
            self.create()
        
        webview.start(
            debug=debug,
            http_server=http_server,
            gui='webkitgtk' if os.name != 'darwin' else None
        )
    
    def evaluate_js(self, code: str):
        """在窗口上下文执行一段 JS，并返回执行结果。"""
        if self.window:
            return self.window.evaluate_js(code)
        return None
    
    def show_message(self, message: str, title: str = "提示"):
        """弹出简单提示框（当前由前端 `alert` 实现）。"""
        if self.window:
            self.window.evaluate_js(f"alert('{message}');")
    
    def update_progress(self, progress: float, message: str = ""):
        """供后端主动触发进度展示的简化入口。"""
        self._on_progress({
            'progress': progress,
            'message': message
        })


def create_window(**kwargs) -> MainWindow:
    """工厂函数：按传入参数创建 `MainWindow`。"""
    return MainWindow(**kwargs)


def run_app(debug: bool = False):
    """脚本模式入口：直接创建并显示主窗口。"""
    window = MainWindow()
    window.show(debug=debug)


if __name__ == "__main__":
    run_app(debug=True)
