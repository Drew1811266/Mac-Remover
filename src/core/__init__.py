"""
核心能力聚合入口。

为了减少包导入开销，这里使用懒加载（`__getattr__`）：
- 只有真正访问某个类时，才去导入对应模块。
"""

__all__ = ['WatermarkRemover', 'VideoProcessor', 'ModelRegistry']


def __getattr__(name):
    # 按名称延迟导入，避免一开始就加载所有重依赖。
    if name == 'WatermarkRemover':
        from .remover import WatermarkRemover

        return WatermarkRemover
    if name == 'VideoProcessor':
        from .video_processor import VideoProcessor

        return VideoProcessor
    if name == 'ModelRegistry':
        from .model_registry import ModelRegistry

        return ModelRegistry
    raise AttributeError(name)
