#!/usr/bin/env python3
"""
手动打标流程的模型下载脚本。

支持模型：
- LaMa（单帧 ROI 去水印）
- ProPainter（时序去水印）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.model_downloader import (  # noqa: E402
    DOWNLOADABLE_MODEL_IDS,
    DownloadCancelled,
    download_model,
    list_model_download_entries,
)


def _print_header(title: str) -> None:
    """打印带分隔线的标题块，便于 CLI 输出分段阅读。"""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _format_bytes(num: int) -> str:
    """把字节数转成易读字符串（B/KB/MB/GB/TB）。"""
    value = float(max(0, int(num)))
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while value >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    return f"{value:.1f}{units[idx]}"


def _format_speed(speed_bps: float) -> str:
    """把速度（B/s）格式化为 MB/s 文本。"""
    speed = float(max(0.0, speed_bps))
    if speed <= 0:
        return "0.00 MB/s"
    return f"{speed / (1024 * 1024):.2f} MB/s"


def _print_progress(payload: Dict[str, object]) -> None:
    """下载进度回调：在同一行刷新进度条信息。"""
    progress = float(payload.get("progress") or 0.0)
    downloaded = int(payload.get("downloaded_bytes") or 0)
    total = int(payload.get("total_bytes") or 0)
    speed = float(payload.get("speed_bps") or 0.0)
    current_file = str(payload.get("current_file") or "")

    percent = max(0, min(100, int(round(progress * 100))))
    if total > 0:
        total_text = _format_bytes(total)
    else:
        total_text = "--"

    text = (
        f"\r  {percent:3d}% | {_format_bytes(downloaded)} / {total_text} | "
        f"{_format_speed(speed)} | {current_file}"
    )
    print(text, end="", flush=True)


def parse_args() -> argparse.Namespace:
    """解析命令行参数，决定需要部署哪些模型。"""
    parser = argparse.ArgumentParser(
        description="Deploy local model assets for Mac Watermark Remover"
    )
    parser.add_argument("--lama", action="store_true", help="Deploy LaMa model")
    parser.add_argument(
        "--propainter", action="store_true", help="Deploy ProPainter source + weights"
    )
    parser.add_argument("--all", action="store_true", help="Deploy all supported models")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even when model appears already installed",
    )
    return parser.parse_args()


def _resolve_targets(args: argparse.Namespace) -> List[str]:
    """把参数开关映射成标准模型 ID 列表。"""
    install_lama = args.all or args.lama
    install_propainter = args.all or args.propainter

    # 向后兼容：未指定参数时默认只安装 LaMa。
    if not (install_lama or install_propainter):
        install_lama = True

    targets: List[str] = []
    if install_lama:
        targets.append("lama_roi")
    if install_propainter:
        targets.append("propainter_roi")
    return targets


def main() -> bool:
    """
    脚本主入口。

    流程：
    1) 解析参数并校验目标模型；
    2) 逐个下载/部署并打印进度；
    3) 输出安装汇总并返回成功标记。
    """
    args = parse_args()
    targets = _resolve_targets(args)

    invalid = [model_id for model_id in targets if model_id not in DOWNLOADABLE_MODEL_IDS]
    if invalid:
        print(f"Unsupported model ids: {', '.join(invalid)}")
        return False

    print(f"Project root: {PROJECT_ROOT}")
    print("Deploy targets:", ", ".join(targets))
    if args.force:
        print("Mode: force re-download")

    ok = True
    for model_id in targets:
        _print_header(f"Deploying {model_id}")

        try:
            download_model(
                model_id=model_id,
                force=bool(args.force),
                progress_callback=_print_progress,
            )
            print("\n✅ done")
        except DownloadCancelled:
            ok = False
            print("\n❌ cancelled")
        except Exception as exc:
            ok = False
            print(f"\n❌ failed: {exc}")

    _print_header("Deployment Summary")
    status_map = {entry["model_id"]: bool(entry["installed"]) for entry in list_model_download_entries()}
    print(f"LaMa:       {'installed' if status_map.get('lama_roi') else 'missing'}")
    print(f"ProPainter: {'installed' if status_map.get('propainter_roi') else 'missing'}")
    print("Result:", "✅ success" if ok else "❌ partial/failed")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
