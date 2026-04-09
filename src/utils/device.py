"""
设备检测工具模块。

这个文件负责：
1. 选择当前机器最合适的推理设备（MPS / CUDA / CPU）。
2. 返回设备名称、显存/内存、是否支持 FP16 等信息。
3. 估算在当前设备上更稳妥的批处理大小。
"""

import torch
import platform
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DeviceInfo:
    """统一描述当前计算设备的信息，便于 UI 和日志直接展示。"""

    name: str
    device_type: str
    memory_gb: float
    is_available: bool
    supports_fp16: bool
    
    def __str__(self) -> str:
        """把设备信息格式化成一行可读文本。"""
        status = "✅" if self.is_available else "❌"
        fp16 = " | FP16" if self.supports_fp16 else ""
        return f"{status} {self.name} ({self.device_type}) - {self.memory_gb:.1f}GB{fp16}"


def get_device() -> torch.device:
    """
    按优先级选择推理设备。

    返回：
    - `torch.device("mps")`：Apple Silicon 可用时优先。
    - `torch.device("cuda")`：其次使用 NVIDIA GPU。
    - `torch.device("cpu")`：都不可用时回退到 CPU。
    """
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def get_device_info() -> DeviceInfo:
    """
    根据当前选中的设备类型，分发到对应的采集函数。

    输出是统一的 `DeviceInfo`，上层不用关心平台差异。
    """
    device = get_device()
    
    if device.type == 'mps':
        return _get_mps_info()
    elif device.type == 'cuda':
        return _get_cuda_info()
    else:
        return _get_cpu_info()


def _get_mps_info() -> DeviceInfo:
    """
    读取 Apple Silicon 设备信息。

    主要通过 `system_profiler` 获取芯片名和内存；
    如果命令失败，返回一个可用的保底信息，避免程序中断。
    """
    try:
        result = subprocess.run(
            ['system_profiler', 'SPHardwareDataType'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        chip_name = "Apple Silicon"
        memory_gb = 8.0
        
        if result.returncode == 0:
            output = result.stdout
            for line in output.split('\n'):
                if 'Chip:' in line:
                    chip_name = line.split(':')[1].strip()
                elif 'Memory:' in line:
                    memory_str = line.split(':')[1].strip()
                    if 'GB' in memory_str:
                        memory_gb = float(memory_str.replace('GB', '').strip())
        
        return DeviceInfo(
            name=chip_name,
            device_type="MPS (Metal)",
            memory_gb=memory_gb,
            is_available=True,
            supports_fp16=True
        )
    except Exception:
        return DeviceInfo(
            name="Apple Silicon",
            device_type="MPS (Metal)",
            memory_gb=8.0,
            is_available=True,
            supports_fp16=True
        )


def _get_cuda_info() -> DeviceInfo:
    """
    读取 CUDA 设备信息。

    关键点：
    - 显存来自 `torch.cuda.get_device_properties`。
    - FP16 支持这里用计算能力主版本 >= 7 作为经验判断。
    - 若读取失败，返回“可继续运行”的默认值。
    """
    try:
        gpu_name = torch.cuda.get_device_name(0)
        memory_bytes = torch.cuda.get_device_properties(0).total_memory
        memory_gb = memory_bytes / (1024 ** 3)
        
        major, minor = torch.cuda.get_device_capability(0)
        supports_fp16 = major >= 7
        
        return DeviceInfo(
            name=gpu_name,
            device_type="CUDA",
            memory_gb=memory_gb,
            is_available=True,
            supports_fp16=supports_fp16
        )
    except Exception:
        return DeviceInfo(
            name="Unknown CUDA GPU",
            device_type="CUDA",
            memory_gb=8.0,
            is_available=True,
            supports_fp16=True
        )


def _get_cpu_info() -> DeviceInfo:
    """
    读取 CPU 信息和系统总内存。

    macOS 下优先用 `sysctl` 取更友好的 CPU 名称；
    其他系统走 `platform.processor()`。
    """
    try:
        if platform.system() == 'Darwin':
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True,
                text=True,
                timeout=2
            )
            cpu_name = result.stdout.strip() if result.returncode == 0 else "CPU"
        else:
            cpu_name = platform.processor() or "CPU"
        
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        cpu_name = "CPU"
        memory_gb = 8.0
    
    return DeviceInfo(
        name=cpu_name,
        device_type="CPU",
        memory_gb=memory_gb,
        is_available=True,
        supports_fp16=False
    )


def get_memory_usage() -> Tuple[float, float]:
    """
    获取当前进程内存占用和系统总内存（GB）。

    返回：`(used_gb, total_gb)`。
    失败时返回保底值，避免上层 UI 因异常崩溃。
    """
    try:
        import psutil
        process = psutil.Process()
        used_gb = process.memory_info().rss / (1024 ** 3)
        total_gb = psutil.virtual_memory().total / (1024 ** 3)
        return used_gb, total_gb
    except Exception:
        return 0.0, 8.0


def check_memory_available(required_gb: float) -> bool:
    """
    粗略判断系统可用内存是否满足需求。

    参数：
    - `required_gb`：预计至少需要的内存（GB）。
    """
    try:
        import psutil
        available = psutil.virtual_memory().available / (1024 ** 3)
        return available >= required_gb
    except Exception:
        return True


def get_optimal_batch_size(device: torch.device, image_size: Tuple[int, int] = (720, 1280)) -> int:
    """
    根据设备类型和显存规模，给出一个偏保守的批大小建议。

    说明：
    - 这里是经验值，不追求极限吞吐，目的是减少 OOM。
    - `image_size` 预留给后续更细粒度估算，目前只保留接口。
    """
    height, width = image_size
    pixels = height * width
    
    if device.type == 'mps':
        memory_gb = get_device_info().memory_gb
        if memory_gb >= 16:
            return 8
        elif memory_gb >= 8:
            return 4
        else:
            return 2
    elif device.type == 'cuda':
        memory_gb = get_device_info().memory_gb
        if memory_gb >= 24:
            return 16
        elif memory_gb >= 12:
            return 8
        elif memory_gb >= 8:
            return 4
        else:
            return 2
    else:
        return 1


if __name__ == "__main__":
    print("Device Detection Test")
    print("=" * 50)
    print(f"Selected Device: {get_device()}")
    print(f"Device Info: {get_device_info()}")
    used, total = get_memory_usage()
    print(f"Memory Usage: {used:.2f}GB / {total:.2f}GB")
    print(f"Optimal Batch Size: {get_optimal_batch_size(get_device())}")
