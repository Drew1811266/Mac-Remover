"""
Python 与前端页面之间的桥接 API。

这个模块是 GUI 层的“总入口”，主要负责：
1. 接收前端调用（文件选择、处理视频、保存设置等）。
2. 调用核心处理链并把进度/结果回推给前端。
3. 管理临时资源（预览会话、对话框请求、模型下载任务）。
"""

import os
import subprocess
import platform
import hashlib
import tempfile
import shutil
import base64
import math
import time
import uuid
import threading
from typing import Optional, Dict, Any, List, Set
from pathlib import Path

from ..core.remover import WatermarkRemover
from ..core.video_processor import VideoProcessor
from ..core.progress_estimator import ProgressEstimator
from ..core.annotations import load_sidecar, save_sidecar, delete_sidecar
from ..core.model_registry import SUPPORTED_MODEL_IDS
from ..core.model_downloader import (
    DOWNLOADABLE_MODEL_IDS,
    DownloadCancelled,
    download_model,
    list_model_download_entries,
)
from ..core.seedvr_manifest import (
    LEGACY_REMOVED_MODELS,
    SEEDVR_DEFAULT_MODEL_ID,
    SEEDVR_ENGINE_ID,
    get_seedvr_model_spec,
)
from ..core.realesrgan_manifest import (
    REALESRGAN_DEFAULT_MODEL_ID,
    REALESRGAN_ENGINE_ID,
    get_realesrgan_model_spec,
)
from ..core.upscale_processor import (
    REALESRGAN_MODELS,
    SEEDVR_MODELS,
    UPSCALE_ENGINES,
    UPSCALE_MODES,
    UPSCALE_SAME_RES_STRENGTHS,
    UPSCALE_TARGET_PRESETS,
    UpscaleCancelled,
    UpscaleProcessor,
)
from ..core.upscale_model_downloader import (
    UPSCALE_DOWNLOADABLE_MODEL_IDS,
    download_upscale_model,
    is_upscale_model_installed,
    list_upscale_model_download_entries,
    remove_legacy_upscale_model_files,
)
from ..config import ConfigManager, save_config
from ..utils.device import get_device, get_device_info, get_memory_usage
from ..utils.ffmpeg_runtime import resolve_ffmpeg_path, runtime_ffmpeg_info
from ..utils.logger import logger
from ..utils.memory_cleanup import release_unified_memory
from .main_thread_dispatch import is_main_thread


class API:
    """前后端桥接对象，供 pywebview 暴露给前端调用。"""

    def __init__(self):
        """
        初始化桥接层状态。

        包括：
        - 懒加载的处理器对象（`remover`/`processor`）
        - 预览会话、对话框请求、下载任务等运行时状态
        - 可选的 macOS 原生播放器能力探测
        """
        self.remover: Optional[WatermarkRemover] = None
        self.processor: Optional[VideoProcessor] = None
        self.config_manager = ConfigManager()
        self.config = self.config_manager.config
        self._progress_callback = None
        self._preview_dir = Path(tempfile.gettempdir()) / "mac_watermark_remover_preview"
        self._preview_dir.mkdir(parents=True, exist_ok=True)
        self._video_preview_sessions: Dict[str, Dict[str, Any]] = {}
        self._video_preview_sessions_lock = threading.RLock()
        self._video_preview_session_ttl = 1800
        self._file_dialog_lock = threading.Lock()
        self._dialog_requests: Dict[str, Dict[str, Any]] = {}
        self._dialog_requests_lock = threading.RLock()
        self._dialog_request_ttl = 600
        self._session_video_paths: Set[str] = set()
        self._session_video_paths_lock = threading.RLock()
        self._native_player = None
        self._window = None
        self._model_download_lock = threading.RLock()
        self._model_download_cancel_event: Optional[threading.Event] = None
        self._model_download_thread: Optional[threading.Thread] = None
        self._model_download_task: Dict[str, Any] = self._new_model_download_task()
        self._upscale_model_download_lock = threading.RLock()
        self._upscale_model_download_cancel_event: Optional[threading.Event] = None
        self._upscale_model_download_thread: Optional[threading.Thread] = None
        self._upscale_model_download_task: Dict[str, Any] = self._new_model_download_task()
        self._upscale_processor = UpscaleProcessor()
        self._upscale_task_lock = threading.RLock()
        self._upscale_cancel_event: Optional[threading.Event] = None
        self._upscale_task_thread: Optional[threading.Thread] = None
        self._upscale_task: Dict[str, Any] = self._new_upscale_task()
        self._init_native_player()
        self._log_ffmpeg_runtime()
        self._log_duplicate_av_dylib_warning()
        self._cleanup_removed_upscale_models()

    def bind_window(self, window: Any):
        """绑定 pywebview 窗口对象，后续用于打开文件对话框等能力。"""
        self._window = window

    def _init_native_player(self):
        """
        初始化 macOS 原生播放器后端。

        非 Darwin 平台直接跳过；初始化失败时只记日志，不影响主功能。
        """
        if platform.system() != "Darwin":
            return
        try:
            from .native_player import NativePlayerManager

            self._native_player = NativePlayerManager()
            logger.info("Native AVPlayer backend is available")
        except Exception as e:
            self._native_player = None
            logger.warning(f"Native AVPlayer backend unavailable: {e}")

    @staticmethod
    def _log_ffmpeg_runtime() -> None:
        """记录当前 ffmpeg/ffprobe 的来源、路径和版本信息。"""
        try:
            runtime = runtime_ffmpeg_info()
            ffmpeg_meta = runtime.get('ffmpeg', {})
            ffprobe_meta = runtime.get('ffprobe', {})
            ffmpeg_path = ffmpeg_meta.get('path', '') or 'missing'
            ffprobe_path = ffprobe_meta.get('path', '') or 'missing'
            ffmpeg_source = ffmpeg_meta.get('source', 'missing')
            ffprobe_source = ffprobe_meta.get('source', 'missing')
            ffmpeg_version = ffmpeg_meta.get('version', '')
            ffprobe_version = ffprobe_meta.get('version', '')
            logger.info(
                "FFmpeg runtime resolved: ffmpeg=%s (%s), ffprobe=%s (%s)",
                ffmpeg_path,
                ffmpeg_source,
                ffprobe_path,
                ffprobe_source,
            )
            if ffmpeg_version:
                logger.info("FFmpeg version: %s", ffmpeg_version)
            if ffprobe_version:
                logger.info("FFprobe version: %s", ffprobe_version)
        except Exception as e:
            logger.warning(f"Failed to inspect FFmpeg runtime: {e}")

    @staticmethod
    def _cleanup_removed_upscale_models() -> None:
        """启动时尽力清理已移除模型残留文件（失败仅记录 warning）。"""
        try:
            result = remove_legacy_upscale_model_files()
            removed = result.get('removed') or []
            failed = result.get('failed') or []
            if removed:
                logger.info(f"Removed legacy upscale model files: {removed}")
            for item in failed:
                logger.warning(
                    "Failed to remove legacy upscale model file %s: %s",
                    item.get('path', ''),
                    item.get('error', ''),
                )
        except Exception as e:
            logger.warning(f"Cleanup removed upscale models failed: {e}")

    @staticmethod
    def _log_duplicate_av_dylib_warning() -> None:
        """
        启动时检测 cv2/av 是否同时携带 libavdevice，提示潜在冲突风险。
        """
        import importlib.util

        def _resolve_package_dir(module_name: str) -> Optional[Path]:
            try:
                spec = importlib.util.find_spec(module_name)
            except Exception:
                return None
            if spec is None:
                return None
            locations = list(spec.submodule_search_locations or [])
            if locations:
                try:
                    return Path(locations[0]).resolve()
                except Exception:
                    return None
            origin = str(getattr(spec, "origin", "") or "").strip()
            if not origin:
                return None
            try:
                return Path(origin).resolve().parent
            except Exception:
                return None

        cv2_lib_path = ""
        av_lib_path = ""
        cv2_base = _resolve_package_dir("cv2")
        if cv2_base is not None:
            candidate = cv2_base / ".dylibs" / "libavdevice.61.3.100.dylib"
            if candidate.exists():
                cv2_lib_path = str(candidate)
        av_base = _resolve_package_dir("av")
        if av_base is not None:
            candidate = av_base / ".dylibs" / "libavdevice.61.3.100.dylib"
            if candidate.exists():
                av_lib_path = str(candidate)

        if cv2_lib_path and av_lib_path:
            logger.warning(
                "Potential duplicate libavdevice runtime detected (cv2=%s, av=%s). "
                "This may cause unstable AVFoundation behavior on macOS.",
                cv2_lib_path,
                av_lib_path,
            )

    def _release_watermark_runtime_memory(self, reason: str) -> None:
        """
        去水印任务收尾释放统一内存（容错，不抛异常）。

        包括：
        - 卸载 remover 模型对象；
        - 断开 processor 内部模型注册缓存引用；
        - 额外触发一次统一内存回收兜底。
        """
        safe_reason = str(reason or "unknown")
        if self.remover is not None:
            try:
                self.remover.unload_model()
            except Exception as exc:
                logger.warning(f"Unload watermark remover failed ({safe_reason}): {exc}")

        if self.processor is not None and hasattr(self.processor, "_model_registry"):
            try:
                self.processor._model_registry = None
            except Exception as exc:
                logger.warning(f"Reset model registry failed ({safe_reason}): {exc}")

        try:
            release_unified_memory(f"watermark_task_end:{safe_reason}")
        except Exception as exc:
            logger.warning(f"Watermark memory cleanup failed ({safe_reason}): {exc}")

    def _release_upscale_runtime_memory(self, reason: str) -> None:
        """
        放大任务收尾释放统一内存（容错，不抛异常）。
        """
        safe_reason = str(reason or "unknown")
        try:
            release_unified_memory(f"upscale_task_end:{safe_reason}")
        except Exception as exc:
            logger.warning(f"Upscale memory cleanup failed ({safe_reason}): {exc}")
        try:
            self._upscale_processor.invalidate_capabilities_cache()
        except Exception as exc:
            logger.warning(f"Invalidate upscale capabilities cache failed ({safe_reason}): {exc}")
    
    def _ensure_models(self):
        """
        确保核心处理对象已就绪，并保持 remover/processor 的引用一致。

        当 remover 被替换时，会清空 processor 的模型注册缓存，避免旧状态残留。
        """
        if self.remover is None:
            device = get_device()
            self.remover = WatermarkRemover(device=device)

        if self.processor is None:
            self.processor = VideoProcessor(remover=self.remover)
        else:
            if self.processor.remover is not self.remover:
                self.processor.remover = self.remover
                if hasattr(self.processor, "_model_registry"):
                    self.processor._model_registry = None

    @staticmethod
    def _new_model_download_task() -> Dict[str, Any]:
        """创建下载任务的初始状态字典。"""
        return {
            'state': 'idle',
            'model_id': '',
            'progress': 0.0,
            'downloaded_bytes': 0,
            'total_bytes': 0,
            'speed_bps': 0.0,
            'current_file': '',
            'message': '',
            'error': '',
        }

    def _snapshot_model_download_task(self) -> Dict[str, Any]:
        """
        线程安全地复制下载任务状态，并做类型/范围清洗。

        这样前端拿到的字段格式更稳定，不容易因 None/NaN 崩溃。
        """
        with self._model_download_lock:
            snapshot = dict(self._model_download_task)
        snapshot['progress'] = self._sanitize_progress(snapshot.get('progress', 0.0))
        snapshot['downloaded_bytes'] = int(snapshot.get('downloaded_bytes') or 0)
        snapshot['total_bytes'] = int(snapshot.get('total_bytes') or 0)
        snapshot['speed_bps'] = float(snapshot.get('speed_bps') or 0.0)
        snapshot['state'] = str(snapshot.get('state') or 'idle')
        snapshot['model_id'] = str(snapshot.get('model_id') or '')
        snapshot['current_file'] = str(snapshot.get('current_file') or '')
        snapshot['message'] = str(snapshot.get('message') or '')
        snapshot['error'] = str(snapshot.get('error') or '')
        return snapshot

    @staticmethod
    def _new_upscale_task() -> Dict[str, Any]:
        """创建 AI 放大任务的初始状态。"""
        return {
            'state': 'idle',
            'progress': 0.0,
            'phase': '',
            'message': '',
            'eta_seconds': None,
            'input_path': '',
            'output_path': '',
            'preview_path': '',
            'mode': '',
            'engine': '',
            'effective_engine': '',
            'model_id': '',
            'error': '',
            'warning': '',
            'segment_index': 0,
            'segment_total': 0,
            'scene_split_mode': '',
        }

    def _snapshot_upscale_task(self) -> Dict[str, Any]:
        """线程安全复制并清洗 AI 放大任务状态。"""
        with self._upscale_task_lock:
            snapshot = dict(self._upscale_task)
        snapshot['state'] = str(snapshot.get('state') or 'idle')
        snapshot['progress'] = self._sanitize_progress(snapshot.get('progress', 0.0))
        snapshot['phase'] = str(snapshot.get('phase') or '')
        snapshot['message'] = str(snapshot.get('message') or '')
        eta_raw = snapshot.get('eta_seconds')
        try:
            snapshot['eta_seconds'] = max(0.0, float(eta_raw)) if eta_raw is not None else None
        except (TypeError, ValueError):
            snapshot['eta_seconds'] = None
        snapshot['input_path'] = str(snapshot.get('input_path') or '')
        snapshot['output_path'] = str(snapshot.get('output_path') or '')
        snapshot['preview_path'] = str(snapshot.get('preview_path') or '')
        snapshot['mode'] = str(snapshot.get('mode') or '')
        snapshot['engine'] = str(snapshot.get('engine') or '')
        snapshot['effective_engine'] = str(snapshot.get('effective_engine') or '')
        snapshot['model_id'] = str(snapshot.get('model_id') or '')
        snapshot['error'] = str(snapshot.get('error') or '')
        snapshot['warning'] = str(snapshot.get('warning') or '')
        snapshot['segment_index'] = int(snapshot.get('segment_index') or 0)
        snapshot['segment_total'] = int(snapshot.get('segment_total') or 0)
        snapshot['scene_split_mode'] = str(snapshot.get('scene_split_mode') or '')
        return snapshot

    def _snapshot_upscale_model_download_task(self) -> Dict[str, Any]:
        """线程安全复制并清洗 AI 放大模型下载任务状态。"""
        with self._upscale_model_download_lock:
            snapshot = dict(self._upscale_model_download_task)
        snapshot['progress'] = self._sanitize_progress(snapshot.get('progress', 0.0))
        snapshot['downloaded_bytes'] = int(snapshot.get('downloaded_bytes') or 0)
        snapshot['total_bytes'] = int(snapshot.get('total_bytes') or 0)
        snapshot['speed_bps'] = float(snapshot.get('speed_bps') or 0.0)
        snapshot['state'] = str(snapshot.get('state') or 'idle')
        snapshot['model_id'] = str(snapshot.get('model_id') or '')
        snapshot['current_file'] = str(snapshot.get('current_file') or '')
        snapshot['message'] = str(snapshot.get('message') or '')
        snapshot['error'] = str(snapshot.get('error') or '')
        return snapshot

    @staticmethod
    def _sanitize_progress(progress: Any) -> float:
        """把任意进度值钳制到 `[0.0, 1.0]`，并过滤非法数字。"""
        try:
            value = float(progress)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return min(max(value, 0.0), 1.0)

    def _push_progress(
        self,
        state: Dict[str, float],
        progress: Optional[float] = None,
        message: Optional[str] = None,
        status: Optional[str] = None,
        processed_frames: Optional[int] = None,
        total_frames: Optional[int] = None,
        estimated_time: Optional[str] = None,
        eta_seconds: Optional[float] = None,
        throughput_fps: Optional[float] = None,
        phase: Optional[str] = None,
        opaque_infer: Optional[bool] = None,
        force: bool = False
    ):
        """
        按需组装进度 payload 并推送给前端。

        关键保护：
        - 默认不允许进度回退（除非 `force=True`）。
        - 所有可选字段都先做基本类型清理再下发。
        """
        if not self._progress_callback:
            return

        payload: Dict[str, Any] = {}

        if progress is not None:
            safe_progress = self._sanitize_progress(progress)
            last_progress = float(state.get('last_progress', 0.0))
            if not force and safe_progress < last_progress:
                safe_progress = last_progress
            state['last_progress'] = safe_progress
            payload['progress'] = safe_progress

        if message is not None:
            payload['message'] = message
        if status is not None:
            payload['status'] = status
        if processed_frames is not None:
            payload['processed_frames'] = processed_frames
        if total_frames is not None:
            payload['total_frames'] = total_frames
        if estimated_time is not None:
            payload['estimated_time'] = estimated_time
        if eta_seconds is not None:
            try:
                payload['eta_seconds'] = max(0.0, float(eta_seconds))
            except (TypeError, ValueError):
                pass
        if throughput_fps is not None:
            try:
                payload['throughput_fps'] = max(0.0, float(throughput_fps))
            except (TypeError, ValueError):
                pass
        if phase is not None:
            payload['phase'] = str(phase)
        if opaque_infer is not None:
            payload['opaque_infer'] = bool(opaque_infer)

        if payload:
            self._progress_callback(payload)

    def _track_session_video_path(self, path: Any) -> None:
        """记录本次会话涉及的视频路径，便于退出时清理临时标注文件。"""
        if not path:
            return
        try:
            normalized = str(Path(str(path)).expanduser().resolve())
        except Exception:
            normalized = str(path)
        if not normalized:
            return
        with self._session_video_paths_lock:
            self._session_video_paths.add(normalized)
    
    def select_file(self, _: Any = None) -> Optional[Dict[str, str]]:
        """
        同步打开“选择文件”对话框。

        返回：
        - 选中时 `{'path': ...}`
        - 取消时 `None`
        """
        logger.info("select_file called")

        if not self._file_dialog_lock.acquire(blocking=False):
            logger.warning("select_file ignored because another file dialog is active")
            return None

        try:
            path = self._run_native_file_dialog(select_folder=False)

            if path:
                self._track_session_video_path(path)
                logger.info(f"select_file selected: {path}")
                return {'path': path}

            logger.info("select_file cancelled")
            return None
        finally:
            try:
                self._file_dialog_lock.release()
            except Exception:
                pass
    
    def select_folder(self, _: Any = None) -> Optional[Dict[str, str]]:
        """同步打开“选择目录”对话框。"""
        logger.info("select_folder called")

        if not self._file_dialog_lock.acquire(blocking=False):
            logger.warning("select_folder ignored because another file dialog is active")
            return None

        try:
            path = self._run_native_file_dialog(select_folder=True)

            if path:
                logger.info(f"select_folder selected: {path}")
                return {'path': path}

            logger.info("select_folder cancelled")
            return None
        finally:
            try:
                self._file_dialog_lock.release()
            except Exception:
                pass

    def begin_select_file(self, _: Any = None) -> Dict[str, Any]:
        """异步发起文件选择请求，返回 `request_id` 给前端轮询。"""
        logger.info("begin_select_file called")
        return self._begin_dialog_request(select_folder=False)

    def begin_select_folder(self, _: Any = None) -> Dict[str, Any]:
        """异步发起目录选择请求，返回 `request_id` 给前端轮询。"""
        logger.info("begin_select_folder called")
        return self._begin_dialog_request(select_folder=True)

    def poll_dialog_result(self, payload: Any = None) -> Dict[str, Any]:
        """
        轮询异步文件对话框结果。

        返回协议：
        - `done=False`：仍在等待用户选择。
        - `done=True` 且 `success=True`：完成（可能是取消）。
        - `success=False`：请求无效或执行失败。
        """
        request_id = ""
        if isinstance(payload, dict):
            request_id = str(payload.get('request_id') or "")
        elif payload:
            request_id = str(payload)

        if not request_id:
            return {'success': False, 'done': True, 'error': 'Missing request_id'}

        with self._dialog_requests_lock:
            record = self._dialog_requests.get(request_id)

        if record is None:
            return {'success': False, 'done': True, 'error': 'Dialog request not found'}

        status = str(record.get('status') or 'pending')
        if status == 'pending':
            return {'success': True, 'done': False}
        if status == 'done':
            path = record.get('path')
            return {
                'success': True,
                'done': True,
                'path': path,
                'cancelled': not bool(path)
            }

        error = str(record.get('error') or 'Dialog failed')
        return {'success': False, 'done': True, 'error': error}

    def clear_dialog_result(self, payload: Any = None) -> Dict[str, Any]:
        """清理某个对话框请求记录，避免内存中长期积累。"""
        request_id = ""
        if isinstance(payload, dict):
            request_id = str(payload.get('request_id') or "")
        elif payload:
            request_id = str(payload)

        if not request_id:
            return {'success': False, 'error': 'Missing request_id'}

        with self._dialog_requests_lock:
            self._dialog_requests.pop(request_id, None)

        return {'success': True}
    
    def get_device_info(self, _: Any = None) -> Dict[str, str]:
        """返回设备名称、内存占用文本和 FP16 支持信息。"""
        device_info = get_device_info()
        app_used, total = get_memory_usage()
        
        try:
            import psutil
            vm = psutil.virtual_memory()
            system_used = (vm.total - vm.available) / (1024 ** 3)
            memory_text = f"应用 {app_used:.1f}GB | 系统 {system_used:.1f}GB / {total:.1f}GB"
        except Exception:
            memory_text = f"{app_used:.1f}GB / {total:.1f}GB"
        
        return {
            'device': f"{device_info.name} ({device_info.device_type})",
            'memory': memory_text,
            'supports_fp16': device_info.supports_fp16
        }

    def get_media_info(self, payload: Any) -> Dict[str, Any]:
        """
        读取媒体基础信息（视频或图片）。

        视频返回帧率、总帧数、分辨率和时长；
        图片按单帧媒体返回宽高信息。
        """
        path = payload.get('path') if isinstance(payload, dict) else payload
        if not path:
            return {'success': False, 'error': 'Missing input path'}
        if not os.path.exists(path):
            return {'success': False, 'error': f'File not found: {path}'}

        try:
            if self._is_video_path(path):
                import cv2

                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    return {'success': False, 'error': 'Cannot open video'}

                try:
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                finally:
                    cap.release()

                duration = 0.0
                if fps > 0 and frame_count > 0:
                    duration = frame_count / fps

                return {
                    'success': True,
                    'type': 'video',
                    'fps': fps,
                    'frame_count': frame_count,
                    'duration': duration,
                    'width': width,
                    'height': height
                }

            import cv2

            image = cv2.imread(path)
            if image is None:
                return {'success': False, 'error': 'Cannot read image'}
            height, width = image.shape[:2]
            return {
                'success': True,
                'type': 'image',
                'fps': 0.0,
                'frame_count': 1,
                'duration': 0.0,
                'width': int(width),
                'height': int(height)
            }
        except Exception as e:
            logger.warning(f"Get media info failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def load_annotations(self, payload: Any) -> Dict[str, Any]:
        """
        读取视频对应的标注 sidecar 文件。

        这里区分三种情况：
        - 文件不存在（`exists=False`）
        - 文件存在但有兼容警告（返回 warning）
        - 文件正常（返回 segments/video_meta）
        """
        video_path = payload.get('video_path') if isinstance(payload, dict) else payload
        if not video_path:
            return {'success': False, 'error': 'Missing video_path'}
        if not os.path.exists(video_path):
            return {'success': False, 'error': f'File not found: {video_path}'}
        self._track_session_video_path(video_path)

        try:
            sidecar_path, annotation_payload, warning = load_sidecar(str(video_path))
            if annotation_payload is None:
                return {
                    'success': True,
                    'exists': False,
                    'sidecar_path': str(sidecar_path) if sidecar_path else '',
                    'segments': [],
                    'video_meta': None,
                    'warning': warning or 'Annotation file not found'
                }

            if warning:
                return {
                    'success': True,
                    'exists': True,
                    'sidecar_path': str(sidecar_path),
                    'segments': [],
                    'video_meta': annotation_payload.get('video_meta'),
                    'warning': warning
                }

            return {
                'success': True,
                'exists': True,
                'sidecar_path': str(sidecar_path),
                'segments': annotation_payload.get('segments', []),
                'video_meta': annotation_payload.get('video_meta'),
                'updated_at': annotation_payload.get('updated_at')
            }
        except Exception as e:
            logger.error(f"Load annotations failed: {e}")
            return {'success': False, 'error': str(e)}

    def save_annotations(self, payload: Any) -> Dict[str, Any]:
        """保存前端编辑后的标注数据到 sidecar 文件。"""
        if not isinstance(payload, dict):
            return {'success': False, 'error': 'Invalid payload'}

        video_path = payload.get('video_path')
        segments = payload.get('segments', [])
        video_meta = payload.get('video_meta')

        if not video_path:
            return {'success': False, 'error': 'Missing video_path'}
        if not os.path.exists(video_path):
            return {'success': False, 'error': f'File not found: {video_path}'}
        if not isinstance(segments, list):
            return {'success': False, 'error': 'segments must be a list'}
        self._track_session_video_path(video_path)

        try:
            sidecar_path, saved_payload = save_sidecar(
                video_path=str(video_path),
                segments=segments,
                video_meta=video_meta
            )
            return {
                'success': True,
                'sidecar_path': str(sidecar_path),
                'segments': saved_payload.get('segments', []),
                'video_meta': saved_payload.get('video_meta'),
                'updated_at': saved_payload.get('updated_at')
            }
        except Exception as e:
            logger.error(f"Save annotations failed: {e}")
            return {'success': False, 'error': str(e)}

    def delete_annotations(self, payload: Any) -> Dict[str, Any]:
        """删除某个视频对应的 sidecar 标注文件。"""
        video_path = payload.get('video_path') if isinstance(payload, dict) else payload
        if not video_path:
            return {'success': False, 'error': 'Missing video_path'}
        if not os.path.exists(video_path):
            return {'success': False, 'error': f'File not found: {video_path}'}
        self._track_session_video_path(video_path)

        try:
            delete_sidecar(str(video_path))
            return {'success': True}
        except Exception as e:
            logger.error(f"Delete annotations failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def process_video(
        self,
        input_path: Any,
        output_path: str = "",
        annotation_segments: Optional[List[Dict]] = None,
        settings: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        手动标注流程的视频处理入口。

        输入：
        - 视频路径
        - 标注“标记段”列表（必须）
        - 可选设置（当前只允许 `model_id`）

        输出：
        - 成功时返回输出路径和实际模型信息
        - 失败时返回错误文本
        """
        payload: Dict[str, Any] = {}
        if isinstance(input_path, dict):
            payload = input_path
            input_path = payload.get('input_path', '')
            output_path = payload.get('output_path', output_path)
            annotation_segments = payload.get('annotation_segments', annotation_segments)
            settings = payload.get('settings', settings)
            allowed_keys = {'input_path', 'output_path', 'annotation_segments', 'settings'}
            unknown_keys = sorted([k for k in payload.keys() if k not in allowed_keys])
            if unknown_keys:
                return {
                    'success': False,
                    'error': (
                        f"Unsupported payload fields for manual-only pipeline: {', '.join(unknown_keys)}. "
                        "Only input_path/output_path/annotation_segments/settings are allowed."
                    ),
                }

        if not input_path:
            return {'success': False, 'error': 'Missing input path'}
        if not os.path.exists(input_path):
            return {'success': False, 'error': f'File not found: {input_path}'}
        if not self._is_video_path(str(input_path)):
            return {'success': False, 'error': 'Only video processing is supported'}

        if not annotation_segments or not isinstance(annotation_segments, list):
            return {'success': False, 'error': 'annotation_segments is required and must be a list'}

        settings_payload: Dict[str, Any] = {}
        if settings is not None:
            if not isinstance(settings, dict):
                return {'success': False, 'error': 'settings must be an object'}
            settings_payload = settings

        if 'output_quality' in settings_payload:
            return {
                'success': False,
                'error': (
                    "settings.output_quality is removed. Use settings.model_id "
                    "('lama_roi' | 'propainter_roi')."
                ),
            }

        settings_unknown = sorted(
            [k for k in settings_payload.keys() if k not in {'model_id'}]
        )
        if settings_unknown:
            return {
                'success': False,
                'error': (
                    f"Unsupported settings fields: {', '.join(settings_unknown)}. "
                    "Only model_id is allowed."
                ),
            }

        # 模型优先级：本次请求 > 持久化配置 > 默认值。
        model_id = str(settings_payload.get('model_id') or self.config.output.model_id or 'lama_roi').strip().lower()
        if model_id == 'sttn_roi':
            # STTN 已移除，兼容旧客户端/旧配置自动迁移到 LaMa。
            model_id = 'lama_roi'
        if model_id not in SUPPORTED_MODEL_IDS:
            return {
                'success': False,
                'error': (
                    f"Invalid model_id: {model_id}. "
                    f"Supported values: {', '.join(SUPPORTED_MODEL_IDS)}"
                ),
            }

        import cv2

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 24.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        cap.release()

        self._ensure_models()

        if not output_path:
            output_path = self.config.output.output_path

        if not output_path:
            output_path = str(Path.home() / "Downloads" / "WatermarkRemover")

        os.makedirs(output_path, exist_ok=True)

        input_name = Path(input_path).stem
        output_file = os.path.join(output_path, f"{input_name}_no_watermark.mp4")
        progress_state: Dict[str, float] = {'last_progress': 0.0}
        estimator = ProgressEstimator(
            total_frames=frame_count,
            alpha=0.25,
            sample_window=0.8,
        )
        infer_started = False
        infer_floor_ratio = 0.0
        infer_last_engine_update_at = time.monotonic()
        infer_last_floor_tick_at = infer_last_engine_update_at

        enabled_count = 0
        for seg in annotation_segments:
            if isinstance(seg, dict) and bool(seg.get('enabled', True)):
                enabled_count += 1
        if enabled_count <= 0:
            return {'success': False, 'error': 'No enabled annotation segments provided'}

        def _emit_estimated_progress(
            *,
            message: Optional[str] = None,
            status: Optional[str] = None,
            processed_frames_value: Optional[int] = None,
            total_frames_value: Optional[int] = None,
            estimated_time_override: Optional[str] = None,
            force: bool = False,
        ) -> None:
            """把估算器快照转换成前端可直接消费的进度事件。"""
            snapshot = estimator.snapshot(force_recompute=force)
            eta_display = str(snapshot.get('estimated_time') or '--:--')
            if eta_display == '--:--' and estimated_time_override:
                eta_display = str(estimated_time_override)

            emitted_processed = (
                processed_frames_value
                if processed_frames_value is not None
                else int(snapshot.get('processed_frames') or 0)
            )
            emitted_total = (
                total_frames_value
                if total_frames_value is not None
                else int(snapshot.get('total_frames') or 0)
            )
            eta_seconds_raw = snapshot.get('eta_seconds')
            throughput_raw = snapshot.get('throughput_fps')

            self._push_progress(
                progress_state,
                progress=float(snapshot.get('progress') or 0.0),
                message=message,
                status=status,
                processed_frames=emitted_processed if emitted_total > 0 else None,
                total_frames=emitted_total if emitted_total > 0 else None,
                estimated_time=eta_display,
                eta_seconds=float(eta_seconds_raw) if eta_seconds_raw is not None else None,
                throughput_fps=float(throughput_raw) if throughput_raw is not None else None,
                phase=str(snapshot.get('phase') or ''),
                opaque_infer=bool(snapshot.get('opaque_infer', False)),
                force=force,
            )

        def progress_callback(
            progress: float,
            message: str,
            processed_frames: Optional[int] = None,
            total_frames: Optional[int] = None,
            estimated_time: Optional[str] = None,
            extra: Optional[Dict[str, Any]] = None,
        ):
            """
            核心处理链的细粒度进度回调。

            - 优先使用 processed/total 帧数更新 ETA。
            - 无帧数时回退为阶段进度百分比。
            - `extra` 可携带阶段状态和“opaque_infer”标记。
            """
            nonlocal infer_started, infer_floor_ratio, infer_last_engine_update_at, infer_last_floor_tick_at
            now = time.monotonic()
            phase = ''
            if isinstance(extra, dict):
                phase = str(extra.get('phase') or '').strip().lower()

            if phase == 'infer' and not infer_started:
                infer_started = True
                # 首次收到 infer 结构化回调后，才把前置阶段标为完成。
                estimator.update_phase_progress('prepare', 1.0)
                estimator.update_phase_progress('load_models', 1.0)
                estimator.update_phase_progress('extract', 1.0)
                estimator.update_phase_progress('infer', 0.01)

            fallback_total = (
                int(total_frames)
                if total_frames is not None
                else frame_count
            )

            real_ratio: Optional[float] = None
            if total_frames is not None or processed_frames is not None:
                estimator.update_processed_frames(
                    int(processed_frames or 0),
                    int(fallback_total) if fallback_total is not None else None,
                )
                if fallback_total and fallback_total > 0:
                    real_ratio = min(
                        1.0,
                        max(0.0, float(int(processed_frames or 0)) / float(max(1, int(fallback_total)))),
                    )
            else:
                estimator.update_phase_progress('infer', self._sanitize_progress(progress))

            engine_ratio: Optional[float] = None
            if isinstance(extra, dict):
                if phase in {'prepare', 'load_models', 'extract', 'infer', 'compose', 'finalize'}:
                    if 'step' in extra and 'total' in extra:
                        try:
                            step_value = int(extra.get('step') or 0)
                            total_value = max(1, int(extra.get('total') or 0))
                            engine_ratio = min(1.0, max(0.0, float(step_value) / float(total_value)))
                            estimator.update_phase_step(phase, step_value, total_value)
                        except (TypeError, ValueError):
                            pass
                    elif 'progress' in extra:
                        try:
                            engine_ratio = min(
                                1.0,
                                max(0.0, float(extra.get('progress') or 0.0)),
                            )
                            estimator.update_phase_progress(phase, engine_ratio)
                        except (TypeError, ValueError):
                            pass

                if 'opaque_infer' in extra:
                    estimator.set_opaque_infer(bool(extra.get('opaque_infer')))
            elif phase == 'infer':
                engine_ratio = self._sanitize_progress(progress)

            if phase == 'infer':
                if engine_ratio is not None:
                    infer_last_engine_update_at = now
                    infer_last_floor_tick_at = now
                    infer_floor_ratio = max(infer_floor_ratio, engine_ratio)
                elif now - infer_last_engine_update_at >= 2.0:
                    delta = max(0.0, now - infer_last_floor_tick_at)
                    if delta > 0:
                        infer_floor_ratio = min(0.96, infer_floor_ratio + delta * 0.0025)
                        infer_last_floor_tick_at = now

                if real_ratio is not None:
                    infer_floor_ratio = max(infer_floor_ratio, real_ratio)
                effective_infer_ratio = max(
                    infer_floor_ratio,
                    real_ratio if real_ratio is not None else 0.0,
                    engine_ratio if engine_ratio is not None else 0.0,
                )
                estimator.update_phase_progress('infer', effective_infer_ratio)

            _emit_estimated_progress(
                message=message,
                processed_frames_value=processed_frames,
                total_frames_value=total_frames,
                estimated_time_override=estimated_time,
            )
        
        def status_callback(status: str):
            """
            文本状态回调（粗粒度阶段）。

            用于把“加载模型/抽帧/收尾”等状态映射到估算器阶段，
            让前端即使在早期阶段也有稳定进度反馈。
            """
            lower = status.lower()

            if "preparing task" in lower:
                estimator.update_phase_progress('prepare', 0.05)
            if "loading models" in lower:
                estimator.update_phase_progress('load_models', 0.15)
            elif "extracting frames" in lower:
                estimator.update_phase_progress('extract', 0.20)
            elif "processing frames" in lower:
                estimator.update_phase_progress('infer', 0.01)
                estimator.set_opaque_infer(False)
            elif "finalizing video" in lower:
                estimator.update_phase_progress('infer', 1.0)
                estimator.update_phase_progress('compose', 1.0)
                estimator.update_phase_progress('finalize', 0.4)
                estimator.set_opaque_infer(False)
            elif "complete" in lower:
                estimator.complete_all()
                estimator.set_opaque_infer(False)

            _emit_estimated_progress(
                status=status,
                force=("complete" in lower),
            )
        
        release_reason = "failed"
        try:
            # 先推一次初始状态，避免前端进度条长时间停在 0 且无提示。
            estimator.transition_to('prepare')
            estimator.update_phase_progress('prepare', 0.05)
            _emit_estimated_progress(
                status="Preparing task...",
                processed_frames_value=0 if frame_count > 0 else None,
                total_frames_value=frame_count if frame_count > 0 else None,
                estimated_time_override="--:--" if frame_count > 0 else None,
                force=True,
            )
            result = self.processor.process_video(
                video_path=input_path,
                output_path=output_file,
                annotation_segments=annotation_segments,
                model_id=model_id,
                progress_callback=progress_callback,
                status_callback=status_callback
            )
            release_reason = "cancelled" if bool(result.get("stopped")) else "success"
            estimator.complete_all()
            estimator.update_processed_frames(frame_count if frame_count > 0 else 0, frame_count if frame_count > 0 else None)
            _emit_estimated_progress(
                status="Complete!",
                message="Complete!",
                processed_frames_value=frame_count if frame_count > 0 else None,
                total_frames_value=frame_count if frame_count > 0 else None,
                force=True,
            )
            
            return {
                'success': True,
                'output_path': result.get('output_path', ''),
                'requested_model_id': result.get('requested_model_id', model_id),
                'effective_model_id': result.get('effective_model_id', model_id),
                'model_warning': result.get('model_warning', ''),
            }
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            if "cancel" in str(e).lower():
                release_reason = "cancelled"
            return {'success': False, 'error': str(e)}
        finally:
            self._release_watermark_runtime_memory(release_reason)
    
    def stop_processing(self, _: Any = None) -> Dict[str, bool]:
        """请求停止当前视频处理任务。"""
        if self.processor:
            self.processor.stop_processing()
        return {'success': True}
    
    def open_output_dir(self, _: Any = None) -> Dict[str, bool]:
        """打开输出目录（按操作系统选择 `open/explorer/xdg-open`）。"""
        output_path = self.config.output.output_path
        if not output_path:
            output_path = str(Path.home() / "Downloads" / "WatermarkRemover")
        
        if os.path.exists(output_path):
            if platform.system() == 'Darwin':
                subprocess.Popen(['open', output_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif platform.system() == 'Windows':
                subprocess.Popen(['explorer', output_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.Popen(['xdg-open', output_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        return {'success': True}
    
    def get_settings(self, _: Any = None) -> Dict[str, Any]:
        """读取当前配置并返回给前端设置页。"""
        return {
            'removal': {
                'use_gpu': self.config.removal.use_gpu,
                'use_fp16': self.config.removal.use_fp16,
                'batch_size': self.config.removal.batch_size,
                'fade_in': self.config.removal.fade_in,
                'fade_out': self.config.removal.fade_out
            },
            'output': {
                'format': self.config.output.format,
                'path': self.config.output.output_path,
                'model_id': self.config.output.model_id,
                'suffix': self.config.output.filename_suffix
            },
            'language': self.config.language,
            'theme': self.config.theme
        }
    
    def save_settings(self, settings: Dict[str, Any]) -> Dict[str, bool]:
        """
        保存前端设置到配置对象并持久化。

        注意：`output.model_id` 会做白名单校验，非法值会被忽略。
        """
        try:
            if 'language' in settings:
                self.config.language = settings['language']
            if 'theme' in settings:
                self.config.theme = settings['theme']

            if 'removal' in settings:
                r = settings['removal']
                if 'use_gpu' in r:
                    self.config.removal.use_gpu = r['use_gpu']
                if 'use_fp16' in r:
                    self.config.removal.use_fp16 = r['use_fp16']
                if 'batch_size' in r:
                    self.config.removal.batch_size = r['batch_size']
                if 'fade_in' in r:
                    self.config.removal.fade_in = r['fade_in']
                if 'fade_out' in r:
                    self.config.removal.fade_out = r['fade_out']
            
            if 'output' in settings:
                o = settings['output']
                if 'format' in o:
                    self.config.output.format = o['format']
                if 'path' in o:
                    self.config.output.output_path = o['path']
                if 'model_id' in o:
                    candidate_model = str(o.get('model_id') or '').strip().lower()
                    if candidate_model in SUPPORTED_MODEL_IDS:
                        self.config.output.model_id = candidate_model
                if 'suffix' in o:
                    self.config.output.filename_suffix = o['suffix']
            
            save_config()
            return {'success': True}
        except Exception as e:
            logger.error(f"Save settings failed: {e}")
            return {'success': False, 'error': str(e)}

    def get_model_download_status(self, _: Any = None) -> Dict[str, Any]:
        """返回可下载模型列表和当前下载任务状态。"""
        try:
            models = list_model_download_entries()
            task = self._snapshot_model_download_task()
            return {
                'success': True,
                'models': models,
                'task': task,
            }
        except Exception as e:
            logger.error(f"Get model download status failed: {e}")
            return {
                'success': False,
                'models': [],
                'task': self._snapshot_model_download_task(),
                'error': str(e),
            }

    def start_model_download(self, payload: Any = None) -> Dict[str, Any]:
        """
        启动模型下载后台线程。

        约束：
        - 同一时刻只允许一个下载任务运行。
        - 支持 `force` 参数强制重下。
        """
        if not isinstance(payload, dict):
            payload = {}

        model_id = str(payload.get('model_id') or '').strip().lower()
        if model_id not in DOWNLOADABLE_MODEL_IDS:
            return {
                'success': False,
                'error': (
                    f"Invalid model_id: {model_id}. "
                    f"Supported values: {', '.join(DOWNLOADABLE_MODEL_IDS)}"
                ),
            }

        force = bool(payload.get('force', False))

        with self._model_download_lock:
            if self._model_download_task.get('state') == 'running':
                running_model = str(self._model_download_task.get('model_id') or '')
                return {
                    'success': False,
                    'error': f"Another download is already running ({running_model}).",
                }

            cancel_event = threading.Event()
            self._model_download_cancel_event = cancel_event
            self._model_download_task = {
                'state': 'running',
                'model_id': model_id,
                'progress': 0.0,
                'downloaded_bytes': 0,
                'total_bytes': 0,
                'speed_bps': 0.0,
                'current_file': '',
                'message': 'Preparing download...',
                'error': '',
            }

        def run_download():
            """后台下载线程主体：更新状态、处理取消、处理异常。"""
            try:
                def progress_callback(progress_payload: Dict[str, Any]) -> None:
                    """下载器进度回调：线程安全更新共享任务状态。"""
                    with self._model_download_lock:
                        if self._model_download_task.get('state') != 'running':
                            return

                        previous_progress = self._sanitize_progress(
                            self._model_download_task.get('progress')
                        )
                        next_progress = self._sanitize_progress(progress_payload.get('progress'))
                        if next_progress < previous_progress:
                            next_progress = previous_progress

                        self._model_download_task.update(
                            {
                                'progress': next_progress,
                                'downloaded_bytes': int(progress_payload.get('downloaded_bytes') or 0),
                                'total_bytes': int(progress_payload.get('total_bytes') or 0),
                                'speed_bps': float(progress_payload.get('speed_bps') or 0.0),
                                'current_file': str(progress_payload.get('current_file') or ''),
                                'message': str(progress_payload.get('message') or ''),
                                'error': str(progress_payload.get('error') or ''),
                            }
                        )

                download_model(
                    model_id=model_id,
                    force=force,
                    progress_callback=progress_callback,
                    cancel_event=cancel_event,
                )

                with self._model_download_lock:
                    self._model_download_task.update(
                        {
                            'state': 'success',
                            'progress': 1.0,
                            'speed_bps': 0.0,
                            'message': 'Download complete',
                            'error': '',
                        }
                    )
            except DownloadCancelled:
                with self._model_download_lock:
                    self._model_download_task.update(
                        {
                            'state': 'cancelled',
                            'speed_bps': 0.0,
                            'message': 'Download cancelled',
                            'error': '',
                        }
                    )
            except Exception as exc:
                logger.error(f"Model download failed ({model_id}): {exc}")
                with self._model_download_lock:
                    self._model_download_task.update(
                        {
                            'state': 'failed',
                            'speed_bps': 0.0,
                            'message': 'Download failed',
                            'error': str(exc),
                        }
                    )
            finally:
                with self._model_download_lock:
                    self._model_download_thread = None
                    self._model_download_cancel_event = None

        worker = threading.Thread(
            target=run_download,
            daemon=True,
            name=f"wmr-model-download-{model_id}",
        )

        with self._model_download_lock:
            self._model_download_thread = worker

        worker.start()
        return {'success': True}

    def cancel_model_download(self, _: Any = None) -> Dict[str, Any]:
        """请求取消当前下载任务（如果正在运行）。"""
        with self._model_download_lock:
            state = str(self._model_download_task.get('state') or 'idle')
            cancel_event = self._model_download_cancel_event
            if state != 'running' or cancel_event is None:
                return {'success': True}

            cancel_event.set()
            self._model_download_task.update(
                {
                    'message': 'Cancelling download...',
                    'speed_bps': 0.0,
                }
            )

        return {'success': True}

    def get_upscale_model_download_status(self, _: Any = None) -> Dict[str, Any]:
        """返回 AI 放大模型列表与下载任务状态。"""
        try:
            models = list_upscale_model_download_entries()
            task = self._snapshot_upscale_model_download_task()
            return {'success': True, 'models': models, 'task': task}
        except Exception as e:
            logger.error(f"Get upscale model download status failed: {e}")
            return {
                'success': False,
                'models': [],
                'task': self._snapshot_upscale_model_download_task(),
                'error': str(e),
            }

    def start_upscale_model_download(self, payload: Any = None) -> Dict[str, Any]:
        """启动 AI 放大模型下载后台任务。"""
        if not isinstance(payload, dict):
            payload = {}

        model_id = str(payload.get('model_id') or '').strip()
        if model_id in LEGACY_REMOVED_MODELS:
            return {
                'success': False,
                'error': (
                    f"Model removed: {model_id}. "
                    f"Supported values: {', '.join(UPSCALE_DOWNLOADABLE_MODEL_IDS)}"
                ),
            }
        if model_id not in UPSCALE_DOWNLOADABLE_MODEL_IDS:
            return {
                'success': False,
                'error': (
                    f"Invalid model_id: {model_id}. "
                    f"Supported values: {', '.join(UPSCALE_DOWNLOADABLE_MODEL_IDS)}"
                ),
            }

        force = bool(payload.get('force', False))
        with self._upscale_model_download_lock:
            if self._upscale_model_download_task.get('state') == 'running':
                running_model = str(self._upscale_model_download_task.get('model_id') or '')
                return {'success': False, 'error': f"Another upscale model download is already running ({running_model})."}

            cancel_event = threading.Event()
            self._upscale_model_download_cancel_event = cancel_event
            self._upscale_model_download_task = {
                'state': 'running',
                'model_id': model_id,
                'progress': 0.0,
                'downloaded_bytes': 0,
                'total_bytes': 0,
                'speed_bps': 0.0,
                'current_file': '',
                'message': 'Preparing download...',
                'error': '',
            }

        def run_download() -> None:
            try:
                def progress_callback(progress_payload: Dict[str, Any]) -> None:
                    with self._upscale_model_download_lock:
                        if self._upscale_model_download_task.get('state') != 'running':
                            return
                        previous_progress = self._sanitize_progress(self._upscale_model_download_task.get('progress'))
                        next_progress = self._sanitize_progress(progress_payload.get('progress'))
                        if next_progress < previous_progress:
                            next_progress = previous_progress
                        self._upscale_model_download_task.update(
                            {
                                'progress': next_progress,
                                'downloaded_bytes': int(progress_payload.get('downloaded_bytes') or 0),
                                'total_bytes': int(progress_payload.get('total_bytes') or 0),
                                'speed_bps': float(progress_payload.get('speed_bps') or 0.0),
                                'current_file': str(progress_payload.get('current_file') or ''),
                                'message': str(progress_payload.get('message') or ''),
                                'error': str(progress_payload.get('error') or ''),
                            }
                        )

                download_upscale_model(
                    model_id=model_id,
                    force=force,
                    progress_callback=progress_callback,
                    cancel_event=cancel_event,
                )
                try:
                    self._upscale_processor.invalidate_capabilities_cache()
                except Exception:
                    # 缓存刷新失败不应影响已完成下载任务。
                    pass
                with self._upscale_model_download_lock:
                    self._upscale_model_download_task.update(
                        {
                            'state': 'success',
                            'progress': 1.0,
                            'speed_bps': 0.0,
                            'message': 'Download complete',
                            'error': '',
                        }
                    )
            except DownloadCancelled:
                with self._upscale_model_download_lock:
                    self._upscale_model_download_task.update(
                        {
                            'state': 'cancelled',
                            'speed_bps': 0.0,
                            'message': 'Download cancelled',
                            'error': '',
                        }
                    )
            except Exception as exc:
                logger.error(f"Upscale model download failed ({model_id}): {exc}")
                with self._upscale_model_download_lock:
                    self._upscale_model_download_task.update(
                        {
                            'state': 'failed',
                            'speed_bps': 0.0,
                            'message': 'Download failed',
                            'error': str(exc),
                        }
                    )
            finally:
                with self._upscale_model_download_lock:
                    self._upscale_model_download_thread = None
                    self._upscale_model_download_cancel_event = None

        worker = threading.Thread(
            target=run_download,
            daemon=True,
            name=f"wmr-upscale-model-download-{model_id}",
        )
        with self._upscale_model_download_lock:
            self._upscale_model_download_thread = worker
        worker.start()
        return {'success': True}

    def cancel_upscale_model_download(self, _: Any = None) -> Dict[str, Any]:
        """请求取消当前 AI 放大模型下载任务。"""
        with self._upscale_model_download_lock:
            state = str(self._upscale_model_download_task.get('state') or 'idle')
            cancel_event = self._upscale_model_download_cancel_event
            if state != 'running' or cancel_event is None:
                return {'success': True}
            cancel_event.set()
            self._upscale_model_download_task.update(
                {
                    'message': 'Cancelling download...',
                    'speed_bps': 0.0,
                }
            )
        return {'success': True}

    def get_upscale_capabilities(self, _: Any = None) -> Dict[str, Any]:
        """返回 AI 放大能力信息（模式/引擎/模型/默认值）。"""
        try:
            force_refresh = False
            if isinstance(_, dict):
                force_refresh = bool(_.get('force_refresh', False))
            return self._upscale_processor.get_capabilities(force_refresh=force_refresh)
        except Exception as e:
            logger.error(f"Get upscale capabilities failed: {e}")
            return {'success': False, 'error': str(e)}

    def start_upscale(self, payload: Any = None) -> Dict[str, Any]:
        """
        启动 AI 放大任务（后台线程）。

        严格白名单字段，避免与 manual-only 主流程参数混用。
        """
        if not isinstance(payload, dict):
            return {'success': False, 'error': 'payload must be an object'}

        allowed_keys = {
            'input_path',
            'output_dir',
            'mode',
            'engine',
            'model_id',
            'target_preset',
            'same_res_strength',
            'denoise_strength',
            'keep_audio',
        }
        unknown_keys = sorted([k for k in payload.keys() if k not in allowed_keys])
        if unknown_keys:
            return {
                'success': False,
                'error': (
                    f"Unsupported payload fields for upscale task: {', '.join(unknown_keys)}. "
                    "Only input_path/output_dir/mode/engine/model_id/target_preset/"
                    "same_res_strength/denoise_strength/keep_audio are allowed."
                ),
            }

        input_path = str(payload.get('input_path') or '').strip()
        output_dir = str(payload.get('output_dir') or '').strip()
        mode = str(payload.get('mode') or '').strip()
        engine = str(payload.get('engine') or '').strip()
        model_id = str(payload.get('model_id') or '').strip()
        target_preset = str(payload.get('target_preset') or '').strip() or None
        same_res_strength = str(payload.get('same_res_strength') or '').strip() or 'x2_then_downscale'
        keep_audio = bool(payload.get('keep_audio', True))
        denoise_raw = payload.get('denoise_strength', 0.35)

        try:
            denoise_strength = float(denoise_raw)
        except (TypeError, ValueError):
            return {'success': False, 'error': 'denoise_strength must be a number between 0 and 1'}

        if denoise_strength < 0.0 or denoise_strength > 1.0:
            return {'success': False, 'error': 'denoise_strength must be in range [0, 1]'}

        if not input_path:
            return {'success': False, 'error': 'Missing input_path'}
        if not os.path.exists(input_path):
            return {'success': False, 'error': f'File not found: {input_path}'}
        if not self._is_video_path(input_path):
            return {'success': False, 'error': 'Only video processing is supported'}

        if not mode:
            mode = 'upscale_resolution'
        if mode not in UPSCALE_MODES:
            return {
                'success': False,
                'error': f"Invalid mode: {mode}. Supported values: {', '.join(UPSCALE_MODES)}",
            }

        if not engine:
            engine = REALESRGAN_ENGINE_ID
        if engine not in UPSCALE_ENGINES:
            return {
                'success': False,
                'error': f"Invalid engine: {engine}. Supported values: {', '.join(UPSCALE_ENGINES)}",
            }

        if not model_id:
            model_id = REALESRGAN_DEFAULT_MODEL_ID if engine == REALESRGAN_ENGINE_ID else SEEDVR_DEFAULT_MODEL_ID
        elif model_id in LEGACY_REMOVED_MODELS:
            supported_model_ids = sorted(set(REALESRGAN_MODELS) | set(SEEDVR_MODELS))
            return {
                'success': False,
                'error': (
                    f"Model removed: {model_id}. "
                    f"Supported values: {', '.join(supported_model_ids)}"
                ),
            }

        if engine == SEEDVR_ENGINE_ID:
            valid_models = set(SEEDVR_MODELS)
        elif engine == REALESRGAN_ENGINE_ID:
            valid_models = set(REALESRGAN_MODELS)
        else:
            valid_models = set()
        if model_id not in valid_models:
            return {
                'success': False,
                'error': (
                    f"Invalid model_id for engine {engine}: {model_id}. "
                    f"Supported values: {', '.join(sorted(valid_models))}"
                ),
            }

        # 模型前置统一内存检查，避免进入 running 后快速失败。
        try:
            if engine == SEEDVR_ENGINE_ID:
                spec = get_seedvr_model_spec(model_id)
            else:
                spec = get_realesrgan_model_spec(model_id)
            memory_gb = float(get_device_info().memory_gb)
            if memory_gb < float(spec.min_memory_gb):
                return {
                    'success': False,
                    'error': (
                        f"Selected model {model_id} requires at least {spec.min_memory_gb:.0f}GB unified memory, "
                        f"current device has {memory_gb:.1f}GB."
                    ),
                }
        except Exception as e:
            logger.warning(f"Upscale model preflight check failed: {e}")

        if not is_upscale_model_installed(model_id):
            return {
                'success': False,
                'error': (
                    f"Upscale model is not installed: {model_id}. "
                    "Please download the model first."
                ),
            }

        # 预检查引擎可用性：避免任务先进入 running 再在 3% 附近失败。
        try:
            capabilities = self._upscale_processor.get_capabilities(force_refresh=True)
            engines_payload = capabilities.get('engines', []) if isinstance(capabilities, dict) else []
            engine_map = {
                str(item.get('engine') or ''): {
                    'available': bool(item.get('available')),
                    'reason': str(item.get('reason') or ''),
                }
                for item in engines_payload
                if isinstance(item, dict)
            }
            selected_engine_available = bool(engine_map.get(engine, {}).get('available'))
            if not selected_engine_available:
                reason = str(engine_map.get(engine, {}).get('reason') or '')
                return {
                    'success': False,
                    'error': reason or f"Selected engine is unavailable: {engine}",
                }
        except Exception as e:
            logger.warning(f"Upscale capability preflight check failed: {e}")

        if mode == 'upscale_resolution':
            if target_preset is None:
                target_preset = '1080p'
            if target_preset not in UPSCALE_TARGET_PRESETS:
                return {
                    'success': False,
                    'error': (
                        f"Invalid target_preset: {target_preset}. "
                        f"Supported values: {', '.join(UPSCALE_TARGET_PRESETS)}"
                    ),
                }
            # 固化排它参数，避免前端误传时后端歧义。
            same_res_strength = 'x2_then_downscale'
        else:
            if same_res_strength == 'x4_then_downscale':
                same_res_strength = 'x2_then_downscale'
            if same_res_strength not in UPSCALE_SAME_RES_STRENGTHS:
                return {
                    'success': False,
                    'error': (
                        f"Invalid same_res_strength: {same_res_strength}. "
                        f"Supported values: {', '.join(UPSCALE_SAME_RES_STRENGTHS)}"
                    ),
                }
            target_preset = None

        if not output_dir:
            output_dir = str(self.config.output.output_path or '').strip()
        if not output_dir:
            output_dir = str(Path.home() / "Downloads" / "WatermarkRemover")
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            return {'success': False, 'error': f'Cannot create output_dir: {e}'}

        with self._upscale_task_lock:
            if str(self._upscale_task.get('state') or 'idle') == 'running':
                return {'success': False, 'error': 'Another upscale task is already running.'}

            cancel_event = threading.Event()
            self._upscale_cancel_event = cancel_event
            self._upscale_task = {
                'state': 'running',
                'progress': 0.0,
                'phase': 'prepare',
                'message': 'Preparing upscale task...',
                'eta_seconds': None,
                'input_path': input_path,
                'output_path': '',
                'preview_path': '',
                'mode': mode,
                'engine': engine,
                'effective_engine': '',
                'model_id': model_id,
                'error': '',
                'warning': '',
                'segment_index': 0,
                'segment_total': 0,
                'scene_split_mode': '',
            }

        def worker() -> None:
            try:
                def progress_cb(progress_payload: Dict[str, Any]) -> None:
                    with self._upscale_task_lock:
                        if str(self._upscale_task.get('state') or '') != 'running':
                            return
                        previous_progress = self._sanitize_progress(self._upscale_task.get('progress'))
                        next_progress = self._sanitize_progress(progress_payload.get('progress'))
                        if next_progress < previous_progress:
                            next_progress = previous_progress
                        eta_raw = progress_payload.get('eta_seconds')
                        try:
                            eta_seconds = max(0.0, float(eta_raw)) if eta_raw is not None else None
                        except (TypeError, ValueError):
                            eta_seconds = None
                        try:
                            segment_index = int(progress_payload.get('segment_index') or self._upscale_task.get('segment_index') or 0)
                        except (TypeError, ValueError):
                            segment_index = int(self._upscale_task.get('segment_index') or 0)
                        try:
                            segment_total = int(progress_payload.get('segment_total') or self._upscale_task.get('segment_total') or 0)
                        except (TypeError, ValueError):
                            segment_total = int(self._upscale_task.get('segment_total') or 0)
                        self._upscale_task.update(
                            {
                                'progress': next_progress,
                                'phase': str(progress_payload.get('phase') or self._upscale_task.get('phase') or ''),
                                'message': str(progress_payload.get('message') or self._upscale_task.get('message') or ''),
                                'eta_seconds': eta_seconds,
                                'segment_index': segment_index,
                                'segment_total': segment_total,
                                'scene_split_mode': str(progress_payload.get('scene_split_mode') or self._upscale_task.get('scene_split_mode') or ''),
                            }
                        )

                result = self._upscale_processor.upscale_video(
                    input_path=input_path,
                    output_dir=output_dir,
                    mode=mode,
                    engine=engine,
                    model_id=model_id,
                    target_preset=target_preset,
                    same_res_strength=same_res_strength,
                    denoise_strength=denoise_strength,
                    keep_audio=keep_audio,
                    progress_callback=progress_cb,
                    cancel_event=cancel_event,
                )

                output_path = str(result.get('output_path') or '')
                preview_path = ''
                if output_path and os.path.exists(output_path):
                    prepared = self.prepare_video_preview({'path': output_path})
                    if prepared.get('success') and prepared.get('path'):
                        preview_path = str(prepared.get('path'))

                with self._upscale_task_lock:
                    try:
                        finished_segments = int(result.get('segment_total') or self._upscale_task.get('segment_total') or 0)
                    except (TypeError, ValueError):
                        finished_segments = int(self._upscale_task.get('segment_total') or 0)
                    self._upscale_task.update(
                        {
                            'state': 'success',
                            'progress': 1.0,
                            'phase': 'finalize',
                            'message': 'Upscale completed',
                            'eta_seconds': 0.0,
                            'output_path': output_path,
                            'preview_path': preview_path,
                            'effective_engine': str(result.get('effective_engine') or engine),
                            'warning': str(result.get('warning') or ''),
                            'error': '',
                            'segment_index': finished_segments,
                            'segment_total': finished_segments,
                            'scene_split_mode': str(result.get('scene_split_mode') or self._upscale_task.get('scene_split_mode') or ''),
                        }
                    )
            except UpscaleCancelled:
                with self._upscale_task_lock:
                    self._upscale_task.update(
                        {
                            'state': 'cancelled',
                            'message': 'Upscale cancelled',
                            'eta_seconds': None,
                        }
                    )
            except Exception as exc:
                logger.error(f"Upscale task failed: {exc}")
                with self._upscale_task_lock:
                    self._upscale_task.update(
                        {
                            'state': 'failed',
                            'message': 'Upscale failed',
                            'eta_seconds': None,
                            'error': str(exc),
                        }
                    )
            finally:
                release_reason = "failed"
                with self._upscale_task_lock:
                    task_state = str(self._upscale_task.get('state') or 'idle')
                    if task_state == 'success':
                        release_reason = "success"
                    elif task_state == 'cancelled':
                        release_reason = "cancelled"
                    self._upscale_task_thread = None
                    self._upscale_cancel_event = None
                self._release_upscale_runtime_memory(release_reason)

        thread = threading.Thread(target=worker, daemon=True, name="wmr-upscale-task")
        with self._upscale_task_lock:
            self._upscale_task_thread = thread
        thread.start()
        return {'success': True}

    def get_upscale_task_status(self, _: Any = None) -> Dict[str, Any]:
        """返回当前 AI 放大任务状态。"""
        return {'success': True, 'task': self._snapshot_upscale_task()}

    def cancel_upscale_task(self, _: Any = None) -> Dict[str, Any]:
        """请求取消当前 AI 放大任务。"""
        with self._upscale_task_lock:
            state = str(self._upscale_task.get('state') or 'idle')
            cancel_event = self._upscale_cancel_event
            if state != 'running' or cancel_event is None:
                return {'success': True}

            cancel_event.set()
            self._upscale_task.update(
                {
                    'message': 'Cancelling upscale task...',
                    'phase': self._upscale_task.get('phase') or 'infer',
                }
            )

        return {'success': True}
    
    def get_recent_files(self, _: Any = None) -> List[str]:
        """返回最近打开文件列表。"""
        return self.config.recent_files
    
    def clear_recent_files(self, _: Any = None) -> Dict[str, bool]:
        """清空最近打开文件列表。"""
        self.config_manager.clear_recent_files()
        return {'success': True}

    def clear_session_transient_data(self, _: Any = None) -> Dict[str, Any]:
        """
        清理本次会话产生的临时数据。

        包括：
        - 会话中涉及视频的 sidecar 标注文件
        - 最近文件记录
        - 预览缓存目录
        """
        targets: Set[str] = set()

        with self._session_video_paths_lock:
            targets.update(self._session_video_paths)

        for item in self.config.recent_files or []:
            if not item:
                continue
            try:
                normalized = str(Path(str(item)).expanduser().resolve())
            except Exception:
                normalized = str(item)
            if normalized:
                targets.add(normalized)

        failed: List[Dict[str, str]] = []
        for video_path in sorted(targets):
            try:
                delete_sidecar(video_path)
            except Exception as e:
                logger.warning(f"Clear transient annotation failed ({video_path}): {e}")
                failed.append({'video_path': video_path, 'error': str(e)})

        with self._session_video_paths_lock:
            self._session_video_paths.clear()

        try:
            self.config_manager.clear_recent_files()
        except Exception as e:
            logger.warning(f"Clear recent files failed: {e}")
            failed.append({'video_path': '', 'error': f'clear_recent_files failed: {e}'})

        try:
            if self._preview_dir.exists():
                for child in self._preview_dir.iterdir():
                    try:
                        if child.is_dir():
                            shutil.rmtree(child, ignore_errors=True)
                        else:
                            child.unlink(missing_ok=True)
                    except Exception as e:
                        logger.warning(f"Clear preview cache item failed ({child}): {e}")
        except Exception as e:
            logger.warning(f"Clear preview cache directory failed: {e}")
            failed.append({'video_path': '', 'error': f'clear_preview_cache failed: {e}'})

        return {
            'success': len(failed) == 0,
            'targets': len(targets),
            'failed': failed,
        }

    def native_player_status(self, _: Any = None) -> Dict[str, Any]:
        """查询原生播放器能力是否可用。"""
        return {
            'success': True,
            'available': self._native_player is not None
        }

    def open_native_player(self, payload: Any) -> Dict[str, Any]:
        """打开原生播放器窗口并加载指定视频。"""
        if self._native_player is None:
            return {'success': False, 'error': 'Native AVPlayer backend is not available'}

        if isinstance(payload, dict):
            role = payload.get('role', 'source')
            path = payload.get('path')
            title = payload.get('title')
            autoplay = bool(payload.get('autoplay', False))
        else:
            role = 'source'
            path = str(payload) if payload else None
            title = None
            autoplay = False

        try:
            return self._native_player.open(role=role, path=path, title=title, autoplay=autoplay)
        except Exception as e:
            logger.error(f"Open native player failed: {e}")
            return {'success': False, 'error': str(e)}

    def native_player_play(self, payload: Any) -> Dict[str, Any]:
        """控制原生播放器开始播放。"""
        if self._native_player is None:
            return {'success': False, 'error': 'Native AVPlayer backend is not available'}
        role = payload.get('role', 'source') if isinstance(payload, dict) else 'source'
        try:
            return self._native_player.play(role)
        except Exception as e:
            logger.error(f"Native player play failed: {e}")
            return {'success': False, 'error': str(e)}

    def native_player_pause(self, payload: Any) -> Dict[str, Any]:
        """控制原生播放器暂停。"""
        if self._native_player is None:
            return {'success': False, 'error': 'Native AVPlayer backend is not available'}
        role = payload.get('role', 'source') if isinstance(payload, dict) else 'source'
        try:
            return self._native_player.pause(role)
        except Exception as e:
            logger.error(f"Native player pause failed: {e}")
            return {'success': False, 'error': str(e)}

    def native_player_seek(self, payload: Any) -> Dict[str, Any]:
        """控制原生播放器跳转到指定秒数。"""
        if self._native_player is None:
            return {'success': False, 'error': 'Native AVPlayer backend is not available'}
        if isinstance(payload, dict):
            role = payload.get('role', 'source')
            seconds = float(payload.get('seconds', 0.0))
        else:
            role = 'source'
            seconds = 0.0
        try:
            return self._native_player.seek(role, seconds)
        except Exception as e:
            logger.error(f"Native player seek failed: {e}")
            return {'success': False, 'error': str(e)}

    def native_player_state(self, payload: Any) -> Dict[str, Any]:
        """读取原生播放器当前状态（时间、总长、播放态）。"""
        if self._native_player is None:
            return {'success': False, 'error': 'Native AVPlayer backend is not available'}
        role = payload.get('role', 'source') if isinstance(payload, dict) else 'source'
        try:
            return self._native_player.state(role)
        except Exception as e:
            logger.error(f"Native player state failed: {e}")
            return {'success': False, 'error': str(e)}

    def close_native_player(self, payload: Any) -> Dict[str, Any]:
        """关闭指定角色的原生播放器窗口。"""
        if self._native_player is None:
            return {'success': True}
        role = payload.get('role', 'source') if isinstance(payload, dict) else 'source'
        try:
            return self._native_player.close(role)
        except Exception as e:
            logger.error(f"Close native player failed: {e}")
            return {'success': False, 'error': str(e)}

    def close_all_native_players(self, _: Any = None) -> Dict[str, Any]:
        """关闭所有原生播放器窗口。"""
        if self._native_player is None:
            return {'success': True}
        try:
            return self._native_player.close_all()
        except Exception as e:
            logger.error(f"Close all native players failed: {e}")
            return {'success': False, 'error': str(e)}

    def set_progress_callback(self, callback):
        """注册处理进度回调函数（由窗口层注入）。"""
        self._progress_callback = callback
    
    def _create_file_dialog_compat(self, dialog_type, file_types=None):
        """
        兼容不同 pywebview 版本的 `create_file_dialog` 参数差异。
        """
        import webview

        window = self._window
        if window is None:
            try:
                window = webview.windows[0]
            except Exception as e:
                raise RuntimeError(f"No active webview window: {e}") from e
        kwargs = {}
        if file_types is not None:
            kwargs['file_types'] = file_types
        
        # pywebview signatures differ by version:
        # - some use allow_multiple
        # - some use no explicit flag for single-select
        for key in ('allow_multiple', 'multiple'):
            try_kwargs = dict(kwargs)
            try_kwargs[key] = False
            try:
                return window.create_file_dialog(dialog_type, **try_kwargs)
            except TypeError:
                continue
        
        return window.create_file_dialog(dialog_type, **kwargs)

    def _begin_dialog_request(self, select_folder: bool) -> Dict[str, Any]:
        """
        创建异步文件对话框请求并立即返回 request_id。

        真正的弹窗动作在后台线程里执行，前端通过轮询拿结果。
        """
        self._cleanup_dialog_requests()
        request_id = uuid.uuid4().hex

        record = {
            'id': request_id,
            'status': 'pending',
            'select_folder': bool(select_folder),
            'path': None,
            'error': None,
            'created_at': time.time(),
            'updated_at': time.time(),
        }

        with self._dialog_requests_lock:
            self._dialog_requests[request_id] = record

        def worker():
            """后台执行文件对话框并把结果写回请求表。"""
            path = None
            error = None
            try:
                logger.info(
                    "Dialog request %s start (select_folder=%s, thread=%s, main_thread=%s)",
                    request_id,
                    bool(select_folder),
                    threading.current_thread().name,
                    is_main_thread(),
                )
                with self._file_dialog_lock:
                    path = self._run_native_file_dialog(select_folder=select_folder)
                    if path and not select_folder:
                        self._track_session_video_path(path)
            except Exception as e:
                logger.error(f"Dialog request {request_id} failed: {e}")
                error = str(e)

            if error is None:
                if path:
                    logger.info(f"Dialog request {request_id} selected: {path}")
                else:
                    logger.info(f"Dialog request {request_id} cancelled")

            with self._dialog_requests_lock:
                current = self._dialog_requests.get(request_id)
                if current is None:
                    return
                current['status'] = 'done' if error is None else 'error'
                current['path'] = path
                current['error'] = error
                current['updated_at'] = time.time()

        thread = threading.Thread(target=worker, daemon=True, name=f"wmr-dialog-{request_id[:8]}")
        thread.start()

        return {'success': True, 'request_id': request_id}

    def _cleanup_dialog_requests(self) -> None:
        """清理超时的对话框请求记录。"""
        now = time.time()
        stale_ids: List[str] = []

        with self._dialog_requests_lock:
            for request_id, record in self._dialog_requests.items():
                status = str(record.get('status') or 'pending')
                updated_at = float(record.get('updated_at') or 0.0)
                if status in {'done', 'error'} and (now - updated_at) > self._dialog_request_ttl:
                    stale_ids.append(request_id)

            for request_id in stale_ids:
                self._dialog_requests.pop(request_id, None)

    def _run_native_file_dialog(self, select_folder: bool = False) -> Optional[str]:
        """统一执行系统文件/目录选择对话框。"""
        import webview

        if select_folder:
            dialog_type = getattr(getattr(webview, "FileDialog", None), "FOLDER", None)
            if dialog_type is None:
                dialog_type = webview.FOLDER_DIALOG
            # pywebview/cocoa 已内建主线程派发；这里避免双重同步等待导致超时。
            result = self._create_file_dialog_compat(dialog_type)
            return self._normalize_dialog_result(result)

        file_types = [
            'Video Files (*.mp4;*.avi;*.mov;*.mkv;*.flv;*.wmv;*.webm)',
            'All Files (*.*)'
        ]
        dialog_type = getattr(getattr(webview, "FileDialog", None), "OPEN", None)
        if dialog_type is None:
            dialog_type = webview.OPEN_DIALOG
        # pywebview/cocoa 已内建主线程派发；这里避免双重同步等待导致超时。
        result = self._create_file_dialog_compat(dialog_type, file_types=file_types)
        return self._normalize_dialog_result(result)
    
    @staticmethod
    def _normalize_dialog_result(result) -> Optional[str]:
        """把 pywebview 返回结果标准化成单个路径字符串。"""
        if not result:
            return None
        
        if isinstance(result, (list, tuple)):
            if not result:
                return None
            return str(result[0])
        
        return str(result)
    
    @staticmethod
    def _is_video_path(path: str) -> bool:
        """按后缀判断路径是否属于支持的视频格式。"""
        video_exts = set(VideoProcessor.SUPPORTED_FORMATS)
        return Path(path).suffix.lower() in video_exts
    
    def _build_preview_cache_path(self, path: str) -> Path:
        """根据源路径和文件元数据生成稳定的预览缓存文件名。"""
        p = Path(path)
        stat = p.stat()
        key_src = f"{p.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        digest = hashlib.sha1(key_src.encode("utf-8")).hexdigest()
        return self._preview_dir / f"{digest}.mp4"

    def _cleanup_video_preview_sessions(self):
        """回收超时未访问的视频预览会话。"""
        now = time.time()
        stale_sessions: List[Dict[str, Any]] = []

        with self._video_preview_sessions_lock:
            stale_ids = []
            for sid, session in self._video_preview_sessions.items():
                last_access = session.get('last_access', 0.0)
                if now - last_access > self._video_preview_session_ttl:
                    stale_ids.append(sid)

            for sid in stale_ids:
                session = self._video_preview_sessions.pop(sid, None)
                if session is not None:
                    stale_sessions.append(session)

        for session in stale_sessions:
            self._release_preview_session(session)

    @staticmethod
    def _release_preview_session(session: Dict[str, Any]) -> None:
        """安全释放预览会话里的视频解码器资源。"""
        lock = session.get('lock')
        if lock is None:
            lock = threading.RLock()

        try:
            with lock:
                session['closed'] = True
                cap = session.get('cap')
                if cap is not None:
                    cap.release()
                    session['cap'] = None
        except Exception:
            pass
    
    def prepare_video_preview(self, path: Any) -> Dict[str, Any]:
        """
        准备前端可稳定播放的预览视频文件。

        策略：
        - 先用 ffmpeg 转码到缓存；
        - ffmpeg 不可用或失败时回退 OpenCV；
        - 都失败则回退原文件。
        """
        if isinstance(path, dict):
            path = path.get('path')
        
        if not path:
            return {'success': False, 'error': 'Missing input path'}
        
        if not os.path.exists(path):
            return {'success': False, 'error': f'File not found: {path}'}
        
        if not self._is_video_path(path):
            return {'success': True, 'path': path, 'transcoded': False}
        
        try:
            preview_path = self._build_preview_cache_path(path)
            if preview_path.exists() and preview_path.stat().st_size == 0:
                preview_path.unlink(missing_ok=True)
            
            if preview_path.exists() and preview_path.stat().st_size > 0:
                return {
                    'success': True,
                    'path': str(preview_path),
                    'transcoded': True,
                    'cached': True
                }
            
            ffmpeg_bin = resolve_ffmpeg_path()
            if ffmpeg_bin:
                # 优先使用 ffmpeg，速度更快且兼容性更好。
                cmd = [
                    ffmpeg_bin, '-y',
                    '-i', path,
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart',
                    '-c:a', 'aac',
                    '-b:a', '96k',
                    str(preview_path)
                ]
                
                try:
                    subprocess.run(cmd, capture_output=True, check=True, timeout=1800)
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Preview FFmpeg transcoding failed: {e}. Fallback to OpenCV.")
                except Exception as e:
                    logger.warning(f"Preview FFmpeg transcoding error: {e}. Fallback to OpenCV.")
            else:
                logger.warning("FFmpeg executable not found. Fallback to OpenCV preview transcoding.")
            
            if not (preview_path.exists() and preview_path.stat().st_size > 0):
                # ffmpeg 失败后回退 OpenCV，尽量保证预览可用。
                ok = self._transcode_preview_with_opencv(path, preview_path)
                if not ok:
                    return {
                        'success': True,
                        'path': path,
                        'transcoded': False,
                        'warning': 'Preview transcoding failed (FFmpeg/OpenCV), using original file'
                    }
            
            if preview_path.exists() and preview_path.stat().st_size > 0:
                return {
                    'success': True,
                    'path': str(preview_path),
                    'transcoded': True,
                    'cached': False
                }
            
            return {
                'success': True,
                'path': path,
                'transcoded': False,
                'warning': 'Preview transcoding produced empty output, using original file'
            }
        except Exception as e:
            logger.warning(f"Preview preparation failed: {e}")
            return {
                'success': True,
                'path': path,
                'transcoded': False,
                'warning': str(e)
            }

    def _transcode_preview_with_opencv(self, input_path: str, output_path: Path) -> bool:
        """
        使用 OpenCV 生成预览视频（ffmpeg 失败时的兜底方案）。

        返回值表示是否生成了非空输出文件。
        """
        try:
            import cv2
            
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                return False
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 25.0
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0:
                cap.release()
                return False
            
            # H.264-compatible output is preferred for WebKit playback.
            # Some encoders require even dimensions.
            target_width = width if width % 2 == 0 else width - 1
            target_height = height if height % 2 == 0 else height - 1
            if target_width <= 0 or target_height <= 0:
                target_width, target_height = width, height
            
            codec_candidates = ['avc1', 'H264', 'X264', 'mp4v']
            writer = None
            selected_codec = None
            for codec in codec_candidates:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    probe = cv2.VideoWriter(
                        str(output_path), fourcc, fps, (target_width, target_height)
                    )
                    if probe.isOpened():
                        writer = probe
                        selected_codec = codec
                        break
                    probe.release()
                except Exception:
                    continue
            
            if writer is None:
                cap.release()
                return False
            
            logger.info(f"OpenCV preview transcoding codec selected: {selected_codec}")
            
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[1] != target_width or frame.shape[0] != target_height:
                    frame = cv2.resize(frame, (target_width, target_height))
                writer.write(frame)
            
            writer.release()
            cap.release()
            
            return output_path.exists() and output_path.stat().st_size > 0
        except Exception as e:
            logger.warning(f"OpenCV preview transcoding failed: {e}")
            return False

    def open_video_preview_session(self, path: Any, target_fps: int = 15, max_width: int = 640) -> Dict[str, Any]:
        """
        打开一个预览解码会话。

        会话内会按 `target_fps` 抽样读取帧，并可限制最大宽度，
        这样前端预览更省资源。
        """
        if isinstance(path, dict):
            target_fps = int(path.get('target_fps', target_fps))
            max_width = int(path.get('max_width', max_width))
            path = path.get('path')
        
        if not path:
            return {'success': False, 'error': 'Missing input path'}
        
        if not os.path.exists(path):
            return {'success': False, 'error': f'File not found: {path}'}
        
        if not self._is_video_path(path):
            return {'success': False, 'error': 'Not a video file'}
        
        self._cleanup_video_preview_sessions()
        
        try:
            import cv2
            
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                return {'success': False, 'error': 'Cannot open video'}
            
            src_fps = cap.get(cv2.CAP_PROP_FPS)
            if not src_fps or src_fps <= 0:
                src_fps = 24.0
            
            src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            src_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if src_width <= 0 or src_height <= 0:
                cap.release()
                return {'success': False, 'error': 'Invalid video dimensions'}
            
            target_fps = max(1, min(30, int(target_fps)))
            step = max(1, int(round(src_fps / target_fps)))
            preview_fps = src_fps / step
            preview_total = max(1, int(math.ceil(max(1, src_frame_count) / step)))
            
            width = src_width
            height = src_height
            if max_width > 0 and src_width > max_width:
                ratio = max_width / float(src_width)
                width = int(src_width * ratio)
                height = int(src_height * ratio)
            
            if width % 2 == 1:
                width -= 1
            if height % 2 == 1:
                height -= 1
            if width <= 0 or height <= 0:
                width, height = src_width, src_height
            
            session_id = uuid.uuid4().hex
            session = {
                'cap': cap,
                'path': path,
                'src_fps': src_fps,
                'src_frame_count': src_frame_count,
                'step': step,
                'preview_fps': preview_fps,
                'preview_total': preview_total,
                'preview_cursor': 0,
                'target_width': width,
                'target_height': height,
                'last_access': time.time(),
                'lock': threading.RLock(),
                'closed': False
            }

            with self._video_preview_sessions_lock:
                self._video_preview_sessions[session_id] = session
            
            return {
                'success': True,
                'session_id': session_id,
                'preview_fps': preview_fps,
                'total_preview_frames': preview_total,
                'width': width,
                'height': height
            }
        except Exception as e:
            logger.warning(f"Open preview session failed: {e}")
            return {'success': False, 'error': str(e)}

    def read_video_preview_frame(self, session_id: Any, frame_index: Optional[int] = None) -> Dict[str, Any]:
        """
        从预览会话读取一帧并返回 base64 JPEG。

        - `frame_index=None`：顺序快速读取（低开销）。
        - 指定 `frame_index`：随机访问读取（用于拖动预览）。
        """
        if isinstance(session_id, dict):
            payload = session_id
            session_id = payload.get('session_id')
            frame_index = payload.get('frame_index', frame_index)
        
        if not session_id:
            return {'success': False, 'error': 'Missing session id'}

        with self._video_preview_sessions_lock:
            session = self._video_preview_sessions.get(str(session_id))

        if not session:
            return {'success': False, 'error': 'Preview session not found'}
        
        try:
            import cv2

            lock = session.get('lock')
            if lock is None:
                lock = threading.RLock()
                session['lock'] = lock

            with lock:
                if session.get('closed'):
                    return {'success': False, 'error': 'Preview session has been closed'}

                cap = session.get('cap')
                if cap is None:
                    return {'success': False, 'error': 'Preview session decoder unavailable'}

                step = int(session['step'])
                total = int(session['preview_total'])
                width = int(session['target_width'])
                height = int(session['target_height'])
                src_frame_count = int(session.get('src_frame_count', 0))

                start_ts = time.time()

                if frame_index is None:
                    # 快路径：顺序读取，避免每帧都随机 seek，性能更稳定。
                    frame_index = int(session.get('preview_cursor', 0))
                    if frame_index < 0:
                        frame_index = 0
                    if total > 0:
                        frame_index = frame_index % total

                    ok, frame = cap.read()
                    if not ok:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                        if not ok:
                            return {'success': False, 'error': 'Cannot decode frame'}
                        frame_index = 0

                    for _ in range(max(0, step - 1)):
                        if not cap.grab():
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            break

                    if total > 0:
                        session['preview_cursor'] = (frame_index + 1) % total
                    else:
                        session['preview_cursor'] = frame_index + 1
                else:
                    # 随机访问：用于用户主动跳到某一帧预览。
                    try:
                        frame_index = int(frame_index)
                    except Exception:
                        frame_index = 0

                    if frame_index < 0:
                        frame_index = 0
                    if total > 0:
                        frame_index = frame_index % total

                    source_idx = frame_index * step
                    if src_frame_count > 0:
                        source_idx = min(source_idx, max(0, src_frame_count - 1))

                    cap.set(cv2.CAP_PROP_POS_FRAMES, source_idx)
                    ok, frame = cap.read()
                    if not ok:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                        if not ok:
                            return {'success': False, 'error': 'Cannot decode frame'}
                        frame_index = 0

                    if total > 0:
                        session['preview_cursor'] = (frame_index + 1) % total
                    else:
                        session['preview_cursor'] = frame_index + 1

                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))

                ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 74])
                if not ok:
                    return {'success': False, 'error': 'Failed to encode frame'}

                session['last_access'] = time.time()
                frame_url = 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode('ascii')
                return {
                    'success': True,
                    'frame_index': frame_index,
                    'frame_url': frame_url,
                    'decode_ms': int((time.time() - start_ts) * 1000)
                }
        except Exception as e:
            logger.warning(f"Read preview frame failed: {e}")
            return {'success': False, 'error': str(e)}

    def close_video_preview_session(self, session_id: Any) -> Dict[str, Any]:
        """关闭并释放指定预览会话。"""
        if isinstance(session_id, dict):
            session_id = session_id.get('session_id')
        
        if not session_id:
            return {'success': False, 'error': 'Missing session id'}

        with self._video_preview_sessions_lock:
            session = self._video_preview_sessions.pop(str(session_id), None)

        if session:
            self._release_preview_session(session)
        
        return {'success': True}
