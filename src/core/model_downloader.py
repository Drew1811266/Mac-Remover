"""
模型下载与部署服务。

核心目标：
- 支持按模型（LaMa/ProPainter）下载；
- 支持进度回调与中途取消；
- 通过“临时目录 + 原子替换”避免半下载状态污染正式目录。
"""

from __future__ import annotations

import importlib
import os
import shutil
import tarfile
import threading
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
THIRD_PARTY_DIR = MODELS_DIR / "third_party"

LAMA_DIR = MODELS_DIR / "big-lama"
LAMA_ZIP_URL = "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip"

PROPAINTER_DIR = THIRD_PARTY_DIR / "ProPainter"
PROPAINTER_SCRIPT = PROPAINTER_DIR / "inference_propainter.py"
PROPAINTER_WEIGHTS_DIR = PROPAINTER_DIR / "weights"
PROPAINTER_WEIGHT_URLS: Dict[str, str] = {
    "raft-things.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth",
    "recurrent_flow_completion.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth",
    "ProPainter.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth",
}
PROPAINTER_ARCHIVE_URL = "https://codeload.github.com/sczhou/ProPainter/tar.gz/refs/heads/main"

DOWNLOADABLE_MODEL_IDS = ("lama_roi", "propainter_roi")
UNKNOWN_SIZE_FALLBACK_BYTES = 512 * 1024 * 1024
CHUNK_SIZE = 512 * 1024
HTTP_TIMEOUT_SECONDS = 45


class DownloadCancelled(Exception):
    """Raised when user requests cancellation during model download."""


@dataclass
class DownloadItem:
    """单个下载任务项（来源、暂存名、部署函数）。"""
    key: str
    display_name: str
    source_type: str
    source: str
    stage_filename: str
    deploy: Callable[[Path, Path], None]


def _lama_installed() -> bool:
    """判断 LaMa 资源是否就绪。"""
    if (LAMA_DIR / "config.yaml").exists():
        return True
    if (LAMA_DIR / "big-lama").exists():
        return True
    try:
        return any(LAMA_DIR.glob("**/*.pt"))
    except Exception:
        return False


def _propainter_installed() -> bool:
    """判断 ProPainter 资源是否就绪（脚本 + 全部权重）。"""
    if not PROPAINTER_SCRIPT.exists():
        return False
    for name in PROPAINTER_WEIGHT_URLS:
        if not (PROPAINTER_WEIGHTS_DIR / name).exists():
            return False
    return True


def _model_display_name(model_id: str) -> str:
    """模型 ID -> 人类可读名称。"""
    if model_id == "lama_roi":
        return "LaMa-ROI"
    return "ProPainter-ROI"


def _model_install_hint(model_id: str) -> str:
    """模型安装提示文案。"""
    if model_id == "lama_roi":
        return "基础单帧修复模型。"
    return "高质量时序修复模型（含源码与多权重）。"


def is_model_installed(model_id: str) -> bool:
    """对外安装状态查询入口。"""
    normalized = str(model_id or "").strip().lower()
    if normalized == "lama_roi":
        return _lama_installed()
    if normalized == "propainter_roi":
        return _propainter_installed()
    return False


def list_model_download_entries() -> List[Dict[str, Any]]:
    """列出可下载模型及其当前状态。"""
    entries: List[Dict[str, Any]] = []
    for model_id in DOWNLOADABLE_MODEL_IDS:
        entries.append(
            {
                "model_id": model_id,
                "display_name": _model_display_name(model_id),
                "installed": is_model_installed(model_id),
                "can_redownload": True,
                "install_hint": _model_install_hint(model_id),
            }
        )
    return entries


class _ProgressEmitter:
    """下载进度聚合器：负责计算进度、速度并回调给 UI。"""
    def __init__(
        self,
        items: List[DownloadItem],
        callback: Optional[Callable[[Dict[str, Any]], None]],
    ):
        self._callback = callback
        self._item_expected: Dict[str, int] = {
            item.key: UNKNOWN_SIZE_FALLBACK_BYTES for item in items
        }
        self._total_expected = sum(self._item_expected.values())

        self._completed_expected = 0
        self._completed_actual = 0

        self._current_key: Optional[str] = None
        self._current_file = ""
        self._current_downloaded = 0

        self._last_progress = 0.0
        self._last_speed_ts = time.time()
        self._speed_acc_bytes = 0
        self._speed_bps = 0.0
        self._last_emit_ts = 0.0

    def _emit(
        self,
        message: str,
        *,
        force: bool = False,
        error: str = "",
        progress_override: Optional[float] = None,
        current_file: Optional[str] = None,
    ) -> None:
        # 节流发射，避免回调过于频繁导致 UI 卡顿。
        if not self._callback:
            return

        now = time.time()
        if not force and (now - self._last_emit_ts) < 0.15:
            return

        if progress_override is None:
            estimated_done = self._completed_expected + min(
                self._current_downloaded,
                self._item_expected.get(self._current_key or "", 0),
            )
            denominator = max(self._total_expected, 1)
            progress = float(estimated_done) / float(denominator)
            if progress < self._last_progress:
                progress = self._last_progress
            self._last_progress = min(max(progress, 0.0), 1.0)
        else:
            clamped = min(max(progress_override, 0.0), 1.0)
            if clamped < self._last_progress:
                clamped = self._last_progress
            self._last_progress = clamped

        payload = {
            "progress": self._last_progress,
            "downloaded_bytes": int(self._completed_actual + self._current_downloaded),
            "total_bytes": int(max(self._total_expected, 1)),
            "speed_bps": float(max(self._speed_bps, 0.0)),
            "current_file": str(current_file if current_file is not None else self._current_file),
            "message": str(message),
            "error": str(error or ""),
        }
        self._last_emit_ts = now
        self._callback(payload)

    def start_item(self, key: str, current_file: str, known_total: Optional[int] = None) -> None:
        """开始下载某个子任务。"""
        self._current_key = key
        self._current_file = current_file
        self._current_downloaded = 0

        if known_total is not None and known_total > 0:
            previous = self._item_expected.get(key, UNKNOWN_SIZE_FALLBACK_BYTES)
            self._item_expected[key] = int(known_total)
            self._total_expected = max(1, self._total_expected + int(known_total) - int(previous))

        self._emit(f"Downloading {current_file}...", force=True)

    def add_chunk(self, chunk_size: int) -> None:
        """累计 chunk 大小并更新速度估算。"""
        now = time.time()
        self._current_downloaded += max(0, int(chunk_size))
        self._speed_acc_bytes += max(0, int(chunk_size))

        elapsed = now - self._last_speed_ts
        if elapsed >= 0.2:
            self._speed_bps = float(self._speed_acc_bytes) / max(elapsed, 1e-6)
            self._speed_acc_bytes = 0
            self._last_speed_ts = now

        self._emit(f"Downloading {self._current_file}...")

    def finish_item(self, message: str) -> None:
        """标记当前子任务完成。"""
        if not self._current_key:
            return

        expected = self._item_expected.get(self._current_key, UNKNOWN_SIZE_FALLBACK_BYTES)
        self._completed_expected += expected
        self._completed_actual += self._current_downloaded

        self._current_downloaded = 0
        self._emit(message, force=True)

        self._current_key = None
        self._current_file = ""

    def mark_done(self, message: str) -> None:
        """标记整任务成功完成。"""
        self._current_key = None
        self._current_file = ""
        self._current_downloaded = 0
        self._speed_bps = 0.0
        self._emit(message, force=True, progress_override=1.0)

    def mark_cancelled(self, message: str) -> None:
        """标记任务被取消。"""
        self._speed_bps = 0.0
        self._emit(message, force=True)

    def mark_failed(self, message: str, error: str) -> None:
        """标记任务失败并附带错误信息。"""
        self._speed_bps = 0.0
        self._emit(message, force=True, error=error)


def _assert_not_cancelled(cancel_event: Optional[threading.Event]) -> None:
    """统一取消检查点：检测到取消即抛异常终止流程。"""
    if cancel_event and cancel_event.is_set():
        raise DownloadCancelled("Download cancelled by user")


def _download_http(
    url: str,
    target: Path,
    cancel_event: Optional[threading.Event],
    emitter: _ProgressEmitter,
    item_key: str,
    display_name: str,
) -> None:
    """通过 HTTP 流式下载单个文件。"""
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "MacWatermarkRemover/1.0")

    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
            total_raw = resp.headers.get("Content-Length")
            total = int(total_raw) if total_raw and str(total_raw).isdigit() else None
            emitter.start_item(item_key, display_name, known_total=total)

            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as handle:
                while True:
                    _assert_not_cancelled(cancel_event)
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
                    emitter.add_chunk(len(chunk))
    except DownloadCancelled:
        raise
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP error {exc.code} while downloading {display_name}: {exc.reason}")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error while downloading {display_name}: {exc.reason}")
    except Exception as exc:
        raise RuntimeError(f"Failed downloading {display_name}: {exc}")


def _download_gdrive(
    file_id: str,
    target: Path,
    cancel_event: Optional[threading.Event],
    emitter: _ProgressEmitter,
    item_key: str,
    display_name: str,
) -> None:
    """通过 gdown 下载 Google Drive 文件，并桥接进度回调。"""
    try:
        import gdown  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"gdown is required for Google Drive download: {exc}")

    module = importlib.import_module(gdown.download.__module__)
    original_tqdm = module.tqdm.tqdm

    class ProgressProxy:
        def __init__(self, total=None, initial=0, unit=None, unit_scale=None):
            safe_total: Optional[int] = None
            if total is not None:
                try:
                    safe_total = int(total)
                except Exception:
                    safe_total = None
            emitter.start_item(item_key, display_name, known_total=safe_total)

            initial_bytes = int(initial or 0)
            if initial_bytes > 0:
                emitter.add_chunk(initial_bytes)

        def update(self, n=1):
            _assert_not_cancelled(cancel_event)
            delta = int(n or 0)
            if delta > 0:
                emitter.add_chunk(delta)

        def close(self):
            return None

        def set_description(self, *_args, **_kwargs):
            return None

        def refresh(self):
            return None

    module.tqdm.tqdm = ProgressProxy

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        result = gdown.download(
            id=file_id,
            output=str(target),
            quiet=False,
            fuzzy=True,
            resume=False,
        )
        if cancel_event and cancel_event.is_set():
            raise DownloadCancelled("Download cancelled by user")
        if not result or not target.exists() or target.stat().st_size <= 0:
            raise RuntimeError("Google Drive download did not produce a valid file")
    except DownloadCancelled:
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed downloading {display_name} from Google Drive: {exc}")
    finally:
        module.tqdm.tqdm = original_tqdm


def _replace_file_atomically(staged_file: Path, target_file: Path) -> None:
    """原子替换单文件，避免目标文件处于中间态。"""
    target_file.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target_file.parent / f".{target_file.name}.incoming-{uuid.uuid4().hex}"

    if temp_target.exists():
        if temp_target.is_dir():
            shutil.rmtree(temp_target, ignore_errors=True)
        else:
            temp_target.unlink(missing_ok=True)

    shutil.move(str(staged_file), str(temp_target))
    os.replace(str(temp_target), str(target_file))


def _replace_directory_atomically(staged_dir: Path, target_dir: Path) -> None:
    """
    原子替换目录。

    失败时会尽力回滚到旧目录，避免安装目录损坏。
    """
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    backup_dir = target_dir.parent / f".{target_dir.name}.backup-{uuid.uuid4().hex}"

    if backup_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)

    if target_dir.exists():
        os.replace(str(target_dir), str(backup_dir))

    try:
        os.replace(str(staged_dir), str(target_dir))
    except Exception:
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        if backup_dir.exists() and not target_dir.exists():
            os.replace(str(backup_dir), str(target_dir))
        raise
    else:
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)


def _resolve_extracted_root(extract_root: Path, required_file: str) -> Path:
    """在解压目录中找到包含关键文件的真实根目录。"""
    direct = extract_root / required_file
    if direct.exists():
        return extract_root

    for child in extract_root.iterdir():
        if child.is_dir() and (child / required_file).exists():
            return child

    for candidate in extract_root.rglob(required_file):
        if candidate.is_file():
            return candidate.parent

    raise RuntimeError(f"Cannot locate extracted root with required file: {required_file}")


def _deploy_archive_directory(
    archive_path: Path,
    staging_root: Path,
    target_dir: Path,
    *,
    archive_kind: str,
    required_file: str,
) -> None:
    """解压归档并原子部署到目标目录。"""
    extract_root = staging_root / f"extract-{uuid.uuid4().hex}"
    extract_root.mkdir(parents=True, exist_ok=True)

    if archive_kind == "zip":
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_root)
    elif archive_kind == "tar.gz":
        with tarfile.open(archive_path, "r:gz") as tar_ref:
            tar_ref.extractall(extract_root)
    else:
        raise RuntimeError(f"Unsupported archive kind: {archive_kind}")

    resolved_root = _resolve_extracted_root(extract_root, required_file)
    prepared_dir = staging_root / f"prepared-{target_dir.name}-{uuid.uuid4().hex}"
    shutil.move(str(resolved_root), str(prepared_dir))
    _replace_directory_atomically(prepared_dir, target_dir)


def _deploy_lama(stage_file: Path, staging_root: Path) -> None:
    """部署 LaMa 压缩包。"""
    _deploy_archive_directory(
        stage_file,
        staging_root,
        LAMA_DIR,
        archive_kind="zip",
        required_file="config.yaml",
    )


def _deploy_propainter_repo(stage_file: Path, staging_root: Path) -> None:
    """部署 ProPainter 源码包。"""
    _deploy_archive_directory(
        stage_file,
        staging_root,
        PROPAINTER_DIR,
        archive_kind="tar.gz",
        required_file="inference_propainter.py",
    )


def _deploy_propainter_weight_factory(filename: str) -> Callable[[Path, Path], None]:
    """按权重文件名生成对应部署函数。"""
    target = PROPAINTER_WEIGHTS_DIR / filename

    def _deploy(stage_file: Path, _staging_root: Path) -> None:
        _replace_file_atomically(stage_file, target)

    return _deploy


def _build_download_plan(model_id: str, force: bool) -> List[DownloadItem]:
    """根据模型 ID 与 force 选项生成下载计划。"""
    model_id = str(model_id or "").strip().lower()
    if model_id not in DOWNLOADABLE_MODEL_IDS:
        raise ValueError(f"Unsupported model id: {model_id}")

    items: List[DownloadItem] = []

    if model_id == "lama_roi":
        if force or not _lama_installed():
            items.append(
                DownloadItem(
                    key="lama_bundle",
                    display_name="LaMa bundle",
                    source_type="http",
                    source=LAMA_ZIP_URL,
                    stage_filename="lama_bundle.zip",
                    deploy=_deploy_lama,
                )
            )
        return items

    if force or not PROPAINTER_SCRIPT.exists():
        items.append(
            DownloadItem(
                key="propainter_repo",
                display_name="ProPainter source",
                source_type="http",
                source=PROPAINTER_ARCHIVE_URL,
                stage_filename="propainter_repo.tar.gz",
                deploy=_deploy_propainter_repo,
            )
        )

    for filename, url in PROPAINTER_WEIGHT_URLS.items():
        target_file = PROPAINTER_WEIGHTS_DIR / filename
        if force or not target_file.exists():
            items.append(
                DownloadItem(
                    key=f"propainter_weight_{filename}",
                    display_name=f"ProPainter weight: {filename}",
                    source_type="http",
                    source=url,
                    stage_filename=filename,
                    deploy=_deploy_propainter_weight_factory(filename),
                )
            )

    return items


def download_model(
    model_id: str,
    *,
    force: bool = False,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    """
    下载并部署指定模型（对外主入口）。

    结果：
    - 成功：返回 installed/skipped 状态；
    - 取消：抛 DownloadCancelled；
    - 失败：抛异常并通过进度回调返回错误信息。
    """
    normalized_model = str(model_id or "").strip().lower()
    if normalized_model not in DOWNLOADABLE_MODEL_IDS:
        raise ValueError(
            f"Unsupported model id: {normalized_model}. "
            f"Supported values: {', '.join(DOWNLOADABLE_MODEL_IDS)}"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    THIRD_PARTY_DIR.mkdir(parents=True, exist_ok=True)

    items = _build_download_plan(normalized_model, force=bool(force))
    emitter = _ProgressEmitter(items, progress_callback)

    if not items:
        emitter.mark_done(f"{_model_display_name(normalized_model)} already installed")
        return {
            "model_id": normalized_model,
            "installed": True,
            "skipped": True,
        }

    task_root = MODELS_DIR / ".download_staging" / f"{normalized_model}-{uuid.uuid4().hex}"
    downloads_dir = task_root / "downloads"
    task_root.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    try:
        for item in items:
            _assert_not_cancelled(cancel_event)
            staged_file = downloads_dir / item.stage_filename

            if item.source_type == "http":
                _download_http(
                    item.source,
                    staged_file,
                    cancel_event,
                    emitter,
                    item.key,
                    item.display_name,
                )
            elif item.source_type == "gdrive":
                _download_gdrive(
                    item.source,
                    staged_file,
                    cancel_event,
                    emitter,
                    item.key,
                    item.display_name,
                )
            else:
                raise RuntimeError(f"Unsupported source type: {item.source_type}")

            _assert_not_cancelled(cancel_event)
            item.deploy(staged_file, task_root)
            emitter.finish_item(f"Deployed {item.display_name}")

        emitter.mark_done(f"{_model_display_name(normalized_model)} download complete")
        return {
            "model_id": normalized_model,
            "installed": is_model_installed(normalized_model),
            "skipped": False,
        }
    except DownloadCancelled:
        emitter.mark_cancelled(f"{_model_display_name(normalized_model)} download cancelled")
        raise
    except Exception as exc:
        emitter.mark_failed(
            f"{_model_display_name(normalized_model)} download failed",
            str(exc),
        )
        raise
    finally:
        shutil.rmtree(task_root, ignore_errors=True)


__all__ = [
    "DOWNLOADABLE_MODEL_IDS",
    "DownloadCancelled",
    "download_model",
    "is_model_installed",
    "list_model_download_entries",
]
