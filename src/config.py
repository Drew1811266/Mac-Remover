"""
配置管理模块（单例）。

这个文件主要做三件事：
1. 定义应用配置的数据结构（数据类）。
2. 从磁盘读取配置并做兼容处理。
3. 对外提供统一的读写入口，避免到处直接操作 JSON。
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List


@dataclass
class RemovalConfig:
    """去水印流程配置。"""
    use_gpu: bool = True
    use_fp16: bool = True
    batch_size: int = 4
    fade_in: float = 0.0
    fade_out: float = 0.0


@dataclass
class OutputConfig:
    """输出相关配置。"""
    format: str = "MP4"
    model_id: str = "lama_roi"
    output_path: str = ""
    filename_suffix: str = "_no_watermark"
    overwrite: bool = False


@dataclass
class AppConfig:
    """应用总配置（语言、主题、输出、最近文件等）。"""
    language: str = "zh"
    theme: str = "light"
    removal: RemovalConfig = field(default_factory=RemovalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    recent_files: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # 如果用户未配置输出目录，给一个可用默认值。
        if not self.output.output_path:
            self.output.output_path = str(Path.home() / "Downloads" / "WatermarkRemover")


class ConfigManager:
    """配置管理器（单例实现）。"""
    _instance = None
    _config: Optional[AppConfig] = None
    
    def __new__(cls):
        # 单例：整个进程只保留一个配置管理器实例。
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # 首次初始化时加载本地配置。
        if self._config is None:
            self._config = AppConfig()
            self._load_config()
    
    @property
    def config(self) -> AppConfig:
        return self._config
    
    def _get_config_path(self) -> Path:
        # 依次尝试可写目录，谁可写就用谁。
        candidates = [
            Path.home() / ".mac_watermark_remover",
            Path.cwd() / ".mac_watermark_remover",
            Path("/tmp") / ".mac_watermark_remover",
        ]
        
        for config_dir in candidates:
            try:
                config_dir.mkdir(parents=True, exist_ok=True)
                return config_dir / "config.json"
            except Exception:
                # 当前目录不可用时继续尝试下一个。
                continue
        
        return Path("/tmp/mac_watermark_remover_config.json")
    
    def _load_config(self):
        # 从 JSON 读取配置，并处理旧版本字段兼容。
        config_path = self._get_config_path()
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    removal_raw = data.get('removal', {}) or {}
                    output_raw = data.get('output', {}) or {}
                    if not isinstance(removal_raw, dict):
                        removal_raw = {}
                    if not isinstance(output_raw, dict):
                        output_raw = {}

                    # Backward compatibility:
                    # - legacy output.path -> output_path
                    # - legacy output.quality -> model_id mapping
                    legacy_quality = str(output_raw.get('quality', '')).lower()
                    if 'model_id' in output_raw:
                        model_id = str(output_raw.get('model_id') or '').strip().lower()
                    elif legacy_quality == 'low':
                        model_id = 'lama_roi'
                    elif legacy_quality == 'medium':
                        # STTN 已移除，旧中质量档迁移到 LaMa。
                        model_id = 'lama_roi'
                    else:
                        model_id = 'lama_roi'

                    # 兼容旧配置中的 sttn_roi，统一迁移为 lama_roi。
                    if model_id == 'sttn_roi':
                        model_id = 'lama_roi'

                    if model_id not in {'lama_roi', 'propainter_roi'}:
                        # 兜底：未知模型统一回到默认模型。
                        model_id = 'lama_roi'

                    output_path = output_raw.get('output_path')
                    if not output_path:
                        output_path = output_raw.get('path', '')

                    self._config = AppConfig(
                        language=data.get('language', 'zh'),
                        theme=data.get('theme', 'light'),
                        removal=RemovalConfig(
                            use_gpu=bool(removal_raw.get('use_gpu', True)),
                            use_fp16=bool(removal_raw.get('use_fp16', True)),
                            batch_size=int(removal_raw.get('batch_size', 4)),
                            fade_in=float(removal_raw.get('fade_in', 0.0)),
                            fade_out=float(removal_raw.get('fade_out', 0.0)),
                        ),
                        output=OutputConfig(
                            format=str(output_raw.get('format', 'MP4') or 'MP4'),
                            model_id=model_id,
                            output_path=str(output_path or ''),
                            filename_suffix=str(output_raw.get('filename_suffix', output_raw.get('suffix', '_no_watermark')) or '_no_watermark'),
                            overwrite=bool(output_raw.get('overwrite', False)),
                        ),
                        recent_files=data.get('recent_files', [])
                    )
            except Exception:
                # 读取失败时回退默认配置，保证应用可启动。
                self._config = AppConfig()
    
    def save_config(self):
        # 将内存中的配置落盘到 JSON。
        config_path = self._get_config_path()
        data = asdict(self._config)
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False
    
    def update_config(self, **kwargs):
        # 只更新已存在字段，避免写入无效键。
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        self.save_config()
    
    def add_recent_file(self, file_path: str):
        # 最近文件列表去重并限制最大条数。
        if file_path in self._config.recent_files:
            self._config.recent_files.remove(file_path)
        self._config.recent_files.insert(0, file_path)
        self._config.recent_files = self._config.recent_files[:10]
        self.save_config()
    
    def clear_recent_files(self):
        # 清空最近文件并立即保存。
        self._config.recent_files = []
        self.save_config()


def get_config() -> AppConfig:
    """获取当前应用配置（快捷函数）。"""
    return ConfigManager().config


def save_config():
    """保存当前配置（快捷函数）。"""
    return ConfigManager().save_config()
