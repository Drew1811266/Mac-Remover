"""
AI 放大模型下载与部署服务（SeedVR + Real-ESRGAN）。

覆盖：
1) SeedVR2 运行时与权重下载；
2) Real-ESRGAN 运行时与权重下载；
3) 统一进度回调、取消与原子替换语义；
4) 兼容历史 removed 模型的确定性错误提示。
"""

from __future__ import annotations

import hashlib
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .model_downloader import (
    DownloadCancelled,
    DownloadItem,
    MODELS_DIR,
    _ProgressEmitter,
    _assert_not_cancelled,
    _deploy_archive_directory,
    _download_http,
    _replace_file_atomically,
)
from .realesrgan_manifest import (
    REALESRGAN_MODEL_DIR,
    REALESRGAN_UPSCALE_MODELS,
    get_realesrgan_model_spec,
)
from .realesrgan_runtime import RealESRGANRuntime
from .seedvr_manifest import (
    LEGACY_REMOVED_MODELS,
    SEEDVR_INFERENCE_SCRIPT,
    SEEDVR_MODEL_DIR,
    SEEDVR_REPO_ARCHIVE_URL,
    SEEDVR_REPO_REF,
    SEEDVR_REPO_REF_FILE,
    SEEDVR_REPO_DIR,
    SEEDVR_REQUIREMENTS_FILE,
    SEEDVR_RUNTIME_ROOT,
    SEEDVR_UPSCALE_MODELS,
    get_seedvr_model_spec,
)
from .seedvr_runtime import SeedVRRuntime


UPSCALE_DOWNLOADABLE_MODEL_IDS = tuple([*SEEDVR_UPSCALE_MODELS, *REALESRGAN_UPSCALE_MODELS])
REMOVED_UPSCALE_MODEL_IDS = set(LEGACY_REMOVED_MODELS)
LEGACY_REMOVED_FILE_PATHS: tuple[Path, ...] = (
    SEEDVR_RUNTIME_ROOT / "models" / "SEEDVR2" / "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
)


def _is_seedvr_model(model_id: str) -> bool:
    return str(model_id or "").strip() in set(SEEDVR_UPSCALE_MODELS)


def _is_realesrgan_model(model_id: str) -> bool:
    return str(model_id or "").strip() in set(REALESRGAN_UPSCALE_MODELS)


def _seedvr_repo_installed() -> bool:
    return SEEDVR_INFERENCE_SCRIPT.exists() and SEEDVR_REQUIREMENTS_FILE.exists()


def _read_installed_repo_ref() -> str:
    try:
        return str(SEEDVR_REPO_REF_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        return ""


def _write_repo_ref_marker() -> None:
    try:
        SEEDVR_RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        SEEDVR_REPO_REF_FILE.write_text(f"{SEEDVR_REPO_REF}\n", encoding="utf-8")
    except Exception:
        return


def _seedvr_repo_ref_matches() -> bool:
    if not _seedvr_repo_installed():
        return False
    return _read_installed_repo_ref() == str(SEEDVR_REPO_REF)


def _model_display_name(model_id: str) -> str:
    try:
        if _is_seedvr_model(model_id):
            return get_seedvr_model_spec(model_id).display_name
        if _is_realesrgan_model(model_id):
            return get_realesrgan_model_spec(model_id).display_name
    except Exception:
        pass
    return str(model_id or "Upscale model")


def _model_install_hint(model_id: str) -> str:
    try:
        if _is_seedvr_model(model_id):
            return get_seedvr_model_spec(model_id).install_hint
        if _is_realesrgan_model(model_id):
            return get_realesrgan_model_spec(model_id).install_hint
    except Exception:
        pass
    return "AI upscale model."


def remove_legacy_upscale_model_files() -> Dict[str, Any]:
    """
    删除已下线模型残留文件（尽力而为，不阻塞主流程）。
    """
    removed: list[str] = []
    failed: list[dict[str, str]] = []
    for path in LEGACY_REMOVED_FILE_PATHS:
        try:
            if not path.exists():
                continue
            path.unlink()
            removed.append(str(path))
        except Exception as exc:
            failed.append({"path": str(path), "error": str(exc)})
    return {"removed": removed, "failed": failed}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest().lower()


def _verify_model_file(path: Path, expected_sha256: str) -> None:
    if not expected_sha256:
        return
    actual = _sha256_file(path)
    if actual != expected_sha256.lower().strip():
        raise RuntimeError(
            f"SHA256 mismatch for {path.name}: expected {expected_sha256}, got {actual}"
        )


def is_upscale_model_installed(model_id: str) -> bool:
    normalized = str(model_id or "").strip()
    if normalized not in UPSCALE_DOWNLOADABLE_MODEL_IDS:
        return False

    if _is_seedvr_model(normalized):
        if not _seedvr_repo_installed():
            return False
        spec = get_seedvr_model_spec(normalized)
        target = SEEDVR_MODEL_DIR / spec.dit_model_name
        return target.exists() and target.stat().st_size > 0

    if _is_realesrgan_model(normalized):
        spec = get_realesrgan_model_spec(normalized)
        for weight in spec.weights:
            target = REALESRGAN_MODEL_DIR / weight.filename
            if not target.exists() or target.stat().st_size <= 0:
                return False
        return True

    return False


def list_upscale_model_download_entries() -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for model_id in UPSCALE_DOWNLOADABLE_MODEL_IDS:
        entries.append(
            {
                "model_id": model_id,
                "display_name": _model_display_name(model_id),
                "installed": is_upscale_model_installed(model_id),
                "can_redownload": True,
                "install_hint": _model_install_hint(model_id),
            }
        )
    return entries


def _deploy_seedvr_repo(stage_file: Path, staging_root: Path) -> None:
    _deploy_archive_directory(
        stage_file,
        staging_root,
        SEEDVR_REPO_DIR,
        archive_kind="tar.gz",
        required_file="inference_cli.py",
    )
    _write_repo_ref_marker()


def _deploy_seedvr_weight_factory(model_id: str) -> Callable[[Path, Path], None]:
    spec = get_seedvr_model_spec(model_id)
    target = SEEDVR_MODEL_DIR / spec.dit_model_name

    def _deploy(stage_file: Path, _staging_root: Path) -> None:
        _verify_model_file(stage_file, spec.sha256)
        _replace_file_atomically(stage_file, target)

    return _deploy


def _deploy_realesrgan_weight_factory(
    *,
    model_id: str,
    filename: str,
    expected_sha256: str,
) -> Callable[[Path, Path], None]:
    target = REALESRGAN_MODEL_DIR / filename

    def _deploy(stage_file: Path, _staging_root: Path) -> None:
        _verify_model_file(stage_file, expected_sha256)
        _replace_file_atomically(stage_file, target)

    return _deploy


def _build_seedvr_download_plan(model_id: str, force: bool) -> List[DownloadItem]:
    spec = get_seedvr_model_spec(model_id)
    items: List[DownloadItem] = []

    if force or not _seedvr_repo_installed() or not _seedvr_repo_ref_matches():
        items.append(
            DownloadItem(
                key="seedvr_repo",
                display_name="SeedVR2 source",
                source_type="http",
                source=SEEDVR_REPO_ARCHIVE_URL,
                stage_filename="seedvr2_repo.tar.gz",
                deploy=_deploy_seedvr_repo,
            )
        )

    model_target = SEEDVR_MODEL_DIR / spec.dit_model_name
    if force or not model_target.exists():
        items.append(
            DownloadItem(
                key=f"seedvr_weight_{model_id}",
                display_name=f"SeedVR2 weight: {spec.dit_model_name}",
                source_type="http",
                source=spec.source_url,
                stage_filename=spec.dit_model_name,
                deploy=_deploy_seedvr_weight_factory(model_id),
            )
        )

    items.append(
        DownloadItem(
            key="seedvr_runtime_prepare",
            display_name="SeedVR runtime (Python 3.12 + scene split deps)",
            source_type="runtime_prepare_seedvr",
            source="",
            stage_filename="",
            deploy=lambda _stage, _root: None,
        )
    )
    return items


def _build_realesrgan_download_plan(model_id: str, force: bool) -> List[DownloadItem]:
    spec = get_realesrgan_model_spec(model_id)
    items: List[DownloadItem] = []

    for weight in spec.weights:
        target = REALESRGAN_MODEL_DIR / weight.filename
        if force or not target.exists():
            items.append(
                DownloadItem(
                    key=f"realesrgan_weight_{model_id}_{weight.filename}",
                    display_name=f"Real-ESRGAN weight: {weight.filename}",
                    source_type="http",
                    source=weight.source_url,
                    stage_filename=weight.filename,
                    deploy=_deploy_realesrgan_weight_factory(
                        model_id=model_id,
                        filename=weight.filename,
                        expected_sha256=weight.sha256,
                    ),
                )
            )

    items.append(
        DownloadItem(
            key="realesrgan_runtime_prepare",
            display_name="Real-ESRGAN runtime (Python 3.12)",
            source_type="runtime_prepare_realesrgan",
            source="",
            stage_filename="",
            deploy=lambda _stage, _root: None,
        )
    )
    return items


def _build_download_plan(model_id: str, force: bool) -> List[DownloadItem]:
    normalized = str(model_id or "").strip()
    if normalized in REMOVED_UPSCALE_MODEL_IDS:
        raise ValueError(
            f"Model removed: {normalized}. "
            f"Supported values: {', '.join(UPSCALE_DOWNLOADABLE_MODEL_IDS)}"
        )
    if normalized not in UPSCALE_DOWNLOADABLE_MODEL_IDS:
        raise ValueError(f"Unsupported model id: {normalized}")

    if _is_seedvr_model(normalized):
        return _build_seedvr_download_plan(normalized, force=force)
    if _is_realesrgan_model(normalized):
        return _build_realesrgan_download_plan(normalized, force=force)
    raise ValueError(f"Unsupported model id: {normalized}")


def download_upscale_model(
    model_id: str,
    *,
    force: bool = False,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    cancel_event: Optional[Any] = None,
) -> Dict[str, Any]:
    normalized = str(model_id or "").strip()
    if normalized in REMOVED_UPSCALE_MODEL_IDS:
        raise ValueError(
            f"Model removed: {normalized}. "
            f"Supported values: {', '.join(UPSCALE_DOWNLOADABLE_MODEL_IDS)}"
        )
    if normalized not in UPSCALE_DOWNLOADABLE_MODEL_IDS:
        raise ValueError(
            f"Unsupported model id: {normalized}. "
            f"Supported values: {', '.join(UPSCALE_DOWNLOADABLE_MODEL_IDS)}"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SEEDVR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REALESRGAN_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    items = _build_download_plan(normalized, force=bool(force))
    emitter = _ProgressEmitter(items, progress_callback)

    if not items:
        emitter.mark_done(f"{_model_display_name(normalized)} already installed")
        return {"model_id": normalized, "installed": True, "skipped": True}

    task_root = MODELS_DIR / ".download_staging" / f"upscale-{normalized}-{uuid.uuid4().hex}"
    downloads_dir = task_root / "downloads"
    task_root.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    try:
        for item in items:
            _assert_not_cancelled(cancel_event)

            if item.source_type == "runtime_prepare_seedvr":
                emitter.start_item(item.key, item.display_name, known_total=1)
                runtime = SeedVRRuntime()

                def runtime_progress(payload: Dict[str, Any]) -> None:
                    message = str(payload.get("message") or "Preparing SeedVR runtime...")
                    emitter._emit(message, current_file=item.display_name)  # type: ignore[attr-defined]

                runtime.ensure_runtime(progress_callback=runtime_progress, cancel_event=cancel_event)
                emitter.add_chunk(1)
                emitter.finish_item("SeedVR runtime ready")
                continue

            if item.source_type == "runtime_prepare_realesrgan":
                emitter.start_item(item.key, item.display_name, known_total=1)
                runtime = RealESRGANRuntime()

                def runtime_progress(payload: Dict[str, Any]) -> None:
                    message = str(payload.get("message") or "Preparing Real-ESRGAN runtime...")
                    emitter._emit(message, current_file=item.display_name)  # type: ignore[attr-defined]

                runtime.ensure_runtime(progress_callback=runtime_progress, cancel_event=cancel_event)
                emitter.add_chunk(1)
                emitter.finish_item("Real-ESRGAN runtime ready")
                continue

            if item.source_type != "http":
                raise RuntimeError(f"Unsupported source type: {item.source_type}")

            staged_file = downloads_dir / item.stage_filename
            _download_http(
                item.source,
                staged_file,
                cancel_event,
                emitter,
                item.key,
                item.display_name,
            )
            _assert_not_cancelled(cancel_event)
            item.deploy(staged_file, task_root)
            emitter.finish_item(f"Deployed {item.display_name}")

        emitter.mark_done(f"{_model_display_name(normalized)} download complete")
        return {
            "model_id": normalized,
            "installed": is_upscale_model_installed(normalized),
            "skipped": False,
        }
    except DownloadCancelled:
        emitter.mark_cancelled(f"{_model_display_name(normalized)} download cancelled")
        raise
    except Exception as exc:
        emitter.mark_failed(f"{_model_display_name(normalized)} download failed", str(exc))
        raise
    finally:
        shutil.rmtree(task_root, ignore_errors=True)


__all__ = [
    "UPSCALE_DOWNLOADABLE_MODEL_IDS",
    "download_upscale_model",
    "is_upscale_model_installed",
    "list_upscale_model_download_entries",
    "remove_legacy_upscale_model_files",
]

