"""
Real-ESRGAN 本地推理 worker。

职责：
- 在独立进程内执行视频逐帧超分；
- 显式选择 mps/cuda/cpu 设备，避免默认逻辑漂移；
- 输出可解析进度日志，供上层 runtime 映射 UI 进度。
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


def _safe_print(message: str) -> None:
    print(str(message), flush=True)


def _round_even(value: float) -> int:
    rounded = int(round(float(value)))
    if rounded < 2:
        rounded = 2
    if rounded % 2 != 0:
        rounded += 1
    return rounded


def _resolve_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_model(
    *,
    model_id: str,
    weights_dir: Path,
    denoise_strength: float,
) -> Tuple[object, int, str | Sequence[str], Optional[Sequence[float]]]:
    normalized = str(model_id or "").strip()
    if normalized == "realesrgan_general_x4v3":
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        )
        netscale = 4
        base = str(weights_dir / "realesr-general-x4v3.pth")
        wdn = str(weights_dir / "realesr-general-wdn-x4v3.pth")
        if denoise_strength < 0.999:
            return model, netscale, [base, wdn], [denoise_strength, 1.0 - denoise_strength]
        return model, netscale, base, None

    if normalized == "realesrgan_x2plus":
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        netscale = 2
        return model, netscale, str(weights_dir / "RealESRGAN_x2plus.pth"), None

    raise ValueError(f"Unsupported Real-ESRGAN model_id: {normalized}")


def _open_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    candidates = ("mp4v", "avc1", "H264")
    for code in candidates:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*code), fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError("Failed to initialize video writer for Real-ESRGAN output")


def run(args: argparse.Namespace) -> int:
    input_path = str(args.input or "").strip()
    output_path = Path(str(args.output or "").strip())
    model_id = str(args.model_id or "").strip()
    weights_dir = Path(str(args.weights_dir or "").strip())
    outscale = float(args.outscale or 1.0)
    denoise_strength = max(0.0, min(1.0, float(args.denoise_strength or 0.5)))
    tile = max(0, int(args.tile or 0))
    tile_pad = max(0, int(args.tile_pad or 10))
    pre_pad = max(0, int(args.pre_pad or 0))

    if not input_path:
        raise ValueError("Missing --input")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not weights_dir.exists():
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")
    if outscale <= 0:
        raise ValueError("outscale must be > 0")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = _resolve_device()
    _safe_print(f"PROGRESS stage=warmup step=device_ready device={device.type}")

    model, netscale, model_path, dni_weight = _build_model(
        model_id=model_id,
        weights_dir=weights_dir,
        denoise_strength=denoise_strength,
    )
    _safe_print("PROGRESS stage=warmup step=model_config_ready")

    # 稳定优先：仅 CUDA 使用 half，MPS/CPU 保持 fp32。
    use_half = device.type == "cuda"
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=use_half,
        device=device,
    )
    _safe_print("PROGRESS stage=warmup step=load_model_done")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid input resolution")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        total_frames = 1
    out_w = _round_even(width * outscale)
    out_h = _round_even(height * outscale)

    writer = _open_writer(output_path, fps=fps, width=out_w, height=out_h)
    _safe_print("PROGRESS stage=warmup step=video_backend_ready")
    _safe_print("PROGRESS stage=warmup step=chunk_loop_start")

    processed = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            output_frame, _ = upsampler.enhance(frame, outscale=outscale)
            writer.write(output_frame)
            processed += 1
            _safe_print(f"PROGRESS frame={processed}/{max(1, total_frames)}")
    finally:
        cap.release()
        writer.release()

    if processed <= 0:
        raise RuntimeError("No frames were processed by Real-ESRGAN worker")

    _safe_print("PROGRESS stage=flush step=writer_closed")
    _safe_print(f"OUTPUT path={output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--model-id", required=True, type=str)
    parser.add_argument("--weights-dir", required=True, type=str)
    parser.add_argument("--outscale", type=float, default=1.0)
    parser.add_argument("--denoise-strength", type=float, default=0.5)
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--tile-pad", type=int, default=10)
    parser.add_argument("--pre-pad", type=int, default=0)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        _safe_print(f"ERROR {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

