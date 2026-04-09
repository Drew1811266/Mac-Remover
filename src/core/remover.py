"""
LaMa 去水印核心封装。

新手可理解为：
- 这个类把“加载模型 + 单帧修复 + 批量修复”打包成统一接口。
- 上层只需要给图像和掩码，不需要关心底层是 TorchScript 还是 SimpleLama。
"""

import torch
import numpy as np
from typing import List, Optional, Union
from PIL import Image
from pathlib import Path

from ..utils.device import get_device, get_device_info
from ..utils.logger import logger
from ..utils.memory_cleanup import release_unified_memory

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
LOCAL_LAMA_PATH = MODELS_DIR / "big-lama"


class WatermarkRemover:
    """LaMa 修复引擎入口类。"""
    def __init__(
        self,
        device: Optional[torch.device] = None,
        use_fp16: bool = True
    ):
        # 设备优先级：外部传入 > 自动检测；FP16 只在设备支持时启用。
        self.device = device or get_device()
        self.use_fp16 = use_fp16 and get_device_info().supports_fp16
        self.model = None
        self._is_loaded = False
        self._torchscript_device = self.device
        
        logger.info(f"WatermarkRemover initialized on {self.device}")
    
    def load_model(self):
        """按需加载模型（重复调用是幂等的）。"""
        if self._is_loaded:
            return
        
        try:
            self._load_simple_lama()
            self._is_loaded = True
            logger.info("LaMA model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LaMA model: {e}")
            raise
    
    def _load_simple_lama(self):
        """
        加载 SimpleLama/TorchScript 模型。

        兼容逻辑：
        - 有本地 TorchScript 权重时优先加载；
        - 否则走 simple-lama-inpainting 的默认加载流程；
        - MPS + TorchScript 不稳定时强制回退 CPU。
        """
        try:
            import torch.jit
            from pathlib import Path
            
            model_path = Path.home() / '.cache' / 'torch' / 'hub' / 'checkpoints' / 'big-lama.pt'
            
            if not model_path.exists():
                # 无本地 TorchScript 文件时，走第三方封装的默认加载路径。
                from simple_lama_inpainting import SimpleLama
                self.model = SimpleLama(device=str(self.device))
                logger.info(f"SimpleLama model initialized on {self.device}")
                return
            
            self.model = torch.jit.load(model_path, map_location='cpu')

            # The bundled big-lama TorchScript model is unstable on MPS for
            # irregular ROI sizes and may fail with tensor size mismatch.
            if str(self.device) == 'mps':
                self._torchscript_device = torch.device('cpu')
                logger.warning("LaMA TorchScript on MPS is unstable; falling back to CPU inference")
            else:
                self._torchscript_device = self.device
                if str(self._torchscript_device) != 'cpu':
                    self.model = self.model.to(self._torchscript_device)
            
            self.model.eval()
            logger.info(f"LaMA model loaded on {self._torchscript_device}")
            
        except ImportError:
            # 开发环境缺依赖时尝试自动安装，提升可用性。
            logger.warning("simple-lama-inpainting not installed, installing now...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'simple-lama-inpainting'])
            from simple_lama_inpainting import SimpleLama
            self.model = SimpleLama(device=str(self.device))
    
    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        单帧修复入口。

        输入：
        - image: BGR/RGB numpy 或 PIL 图像
        - mask: 0/1 或 0/255 掩码
        输出：
        - 修复后的 numpy 图像（失败时返回原图，避免整条流水线中断）。
        """
        if not self._is_loaded:
            self.load_model()
        
        if isinstance(image, np.ndarray):
            pil_image = self._numpy_to_pil(image)
        else:
            pil_image = image
        
        if isinstance(mask, np.ndarray):
            if mask.max() > 1:
                pil_mask = Image.fromarray(mask)
            else:
                pil_mask = Image.fromarray((mask * 255).astype(np.uint8))
        else:
            pil_mask = mask
        
        try:
            # TorchScript 与 SimpleLama 走不同推理路径，但对外接口保持一致。
            if self._is_torchscript_model():
                result = self._inpaint_torchscript(pil_image, pil_mask)
            else:
                result = self.model(pil_image, pil_mask)
            
            if isinstance(result, Image.Image):
                return self._pil_to_numpy(result)
            else:
                return result
        except Exception as e:
            logger.error(f"Inpainting failed: {e}")
            return image
    
    def _is_torchscript_model(self) -> bool:
        """判断当前模型是否为 TorchScript ScriptModule。"""
        return isinstance(self.model, torch.jit.ScriptModule)
    
    def _inpaint_torchscript(self, pil_image: Image.Image, pil_mask: Image.Image) -> Image.Image:
        """
        TorchScript 推理实现。

        关键点：
        - 输入尺寸按 32 对齐补边，推理后再裁回原尺寸；
        - 掩码统一成单通道二值；
        - 输出值夹紧到 [0, 1] 后再转 uint8。
        """
        import cv2
        import torch.nn.functional as F
        from torchvision import transforms

        infer_device = getattr(self, '_torchscript_device', self.device)
        img_tensor = transforms.ToTensor()(pil_image).unsqueeze(0).float().to(infer_device)

        mask_gray = np.array(pil_mask.convert('L'), dtype=np.uint8)
        target_h = int(pil_image.height)
        target_w = int(pil_image.width)
        if mask_gray.shape[:2] != (target_h, target_w):
            mask_gray = cv2.resize(mask_gray, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        mask_tensor = torch.from_numpy((mask_gray > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(infer_device)

        h = int(img_tensor.shape[-2])
        w = int(img_tensor.shape[-1])
        pad_mod = 32
        pad_h = (pad_mod - (h % pad_mod)) % pad_mod
        pad_w = (pad_mod - (w % pad_mod)) % pad_mod

        if pad_h or pad_w:
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
            mask_tensor = F.pad(mask_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0.0)

        with torch.no_grad():
            result = self.model(img_tensor, mask_tensor)

        result = result[:, :, :h, :w]
        result = torch.clamp(result, 0.0, 1.0)
        result = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
        result = (result * 255).astype(np.uint8)

        return Image.fromarray(result)
    
    def inpaint_batch(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray],
        progress_callback: Optional[callable] = None
    ) -> List[np.ndarray]:
        """批量逐帧修复（内部仍按单帧循环，便于复用与容错）。"""
        if not self._is_loaded:
            self.load_model()
        
        results = []
        total = len(images)
        
        for i, (image, mask) in enumerate(zip(images, masks)):
            result = self.inpaint(image, mask)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results
    
    def inpaint_with_blend(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        feather_edges: bool = True
    ) -> np.ndarray:
        """
        单帧修复后做边缘融合，降低“修复块边界感”。
        """
        import cv2
        
        result = self.inpaint(image, mask)
        
        if feather_edges:
            kernel = np.ones((5, 5), np.uint8)
            dilated_mask = cv2.dilate(mask, kernel, iterations=2)
            blurred_mask = cv2.GaussianBlur(dilated_mask, (15, 15), 0)
            blend_mask = blurred_mask.astype(np.float32) / 255.0
            
            if len(blend_mask.shape) == 2:
                blend_mask = blend_mask[:, :, np.newaxis]
            
            result = (image * (1 - blend_mask) + result * blend_mask).astype(np.uint8)
        
        return result
    
    def _numpy_to_pil(self, image: np.ndarray) -> Image.Image:
        """numpy 图像转 PIL（自动处理 BGR/BGRA 通道顺序）。"""
        import cv2
        
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
            else:
                return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return Image.fromarray(image)
    
    def _pil_to_numpy(self, image: Image.Image) -> np.ndarray:
        """PIL 图像转 numpy（输出 BGR/BGRA 以兼容 OpenCV 流程）。"""
        import cv2
        
        if image.mode == 'RGBA':
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
        else:
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    def is_loaded(self) -> bool:
        """返回模型是否已加载。"""
        return self._is_loaded
    
    def unload_model(self):
        """释放模型并尝试清理 CUDA 缓存。"""
        try:
            if self.model is not None:
                del self.model
        except Exception as exc:  # pragma: no cover - 运行时兜底
            logger.warning(f"Failed to delete LaMA model reference: {exc}")
        finally:
            self.model = None
            self._is_loaded = False
            self._torchscript_device = self.device

        cleanup_result = release_unified_memory("watermark_remover_unload")
        if cleanup_result.get("success"):
            logger.info("LaMA model unloaded")
        else:
            logger.warning(
                "LaMA model unloaded with cleanup warnings: %s",
                "; ".join(cleanup_result.get("errors") or []),
            )
