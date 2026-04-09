"""
图像处理通用工具。

职责：
1. 生成/缩放掩码。
2. 图像尺寸调整与补边。
3. PIL 与 NumPy(OpenCV) 之间互转。
4. 按掩码融合修复结果。
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List, Optional


def create_mask(
    shape: Tuple[int, int, int],
    bboxes: List[List[int]],
    expand_pixels: int = 5
) -> np.ndarray:
    """
    根据多个矩形框生成二值掩码。

    参数：
    - `shape`：原图尺寸（只取前两个维度）。
    - `bboxes`：`[x1, y1, x2, y2]` 列表。
    - `expand_pixels`：每个框额外向外扩展的像素，减少边缘漏处理。
    """
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - expand_pixels)
        y1 = max(0, y1 - expand_pixels)
        x2 = min(width, x2 + expand_pixels)
        y2 = min(height, y2 + expand_pixels)
        mask[y1:y2, x1:x2] = 255
    
    return mask


def create_mask_from_points(
    shape: Tuple[int, int, int],
    points: List[Tuple[int, int, int, int]]
) -> np.ndarray:
    """
    根据 `(x, y, w, h)` 形式的标注点生成二值掩码。
    """
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for x, y, w, h in points:
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width, x + w)
        y2 = min(height, y + h)
        mask[y1:y2, x1:x2] = 255
    
    return mask


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    keep_aspect_ratio: bool = True
) -> Tuple[np.ndarray, float]:
    """
    缩放图像到目标大小。

    返回：
    - 缩放后图像
    - 缩放比例（保持比例模式下用于后续坐标还原）
    """
    if keep_aspect_ratio:
        h, w = image.shape[:2]
        target_h, target_w = target_size
        
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale
    else:
        resized = cv2.resize(image, target_size[::-1], interpolation=cv2.INTER_AREA)
        return resized, 1.0


def resize_mask(
    mask: np.ndarray,
    target_size: Tuple[int, int]
) -> np.ndarray:
    """
    缩放掩码并保持硬边界（最近邻插值）。
    """
    return cv2.resize(mask, target_size[::-1], interpolation=cv2.INTER_NEAREST)


def pad_image_to_size(
    image: np.ndarray,
    target_size: Tuple[int, int],
    pad_value: int = 0
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    把图像居中补边到指定尺寸。

    返回：
    - 补边后的图像
    - `(pad_top, pad_left)`，方便把坐标映射回原图。
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    if len(image.shape) == 3:
        padded = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value)
        )
    else:
        padded = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=pad_value
        )
    
    return padded, (pad_top, pad_left)


def numpy_to_pil(image: np.ndarray) -> Image.Image:
    """OpenCV(BGR/BGRA) -> PIL(RGB/RGBA)。"""
    if image.shape[2] == 4:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
    else:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """PIL(RGB/RGBA) -> OpenCV(BGR/BGRA)。"""
    if image.mode == 'RGBA':
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
    else:
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def blend_images(
    original: np.ndarray,
    inpainted: np.ndarray,
    mask: np.ndarray,
    feather_edges: bool = True
) -> np.ndarray:
    """
    用掩码把修复图和原图融合成最终图。

    - `feather_edges=True`：对掩码做膨胀+高斯模糊，过渡更自然。
    - `False`：硬切换，速度更快但边缘可能更明显。
    """
    if feather_edges:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        mask = mask.astype(np.float32) / 255.0
        
        if len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]
        
        result = (original * (1 - mask) + inpainted * mask).astype(np.uint8)
    else:
        mask = mask.astype(bool)
        if len(mask.shape) == 2:
            mask = np.stack([mask] * 3, axis=-1)
        result = original.copy()
        result[mask] = inpainted[mask]
    
    return result


def expand_bbox(
    bbox: List[int],
    image_shape: Tuple[int, int],
    expand_ratio: float = 0.1
) -> List[int]:
    """
    按比例扩展单个矩形框，并限制在图像边界内。
    """
    x1, y1, x2, y2 = bbox
    height, width = image_shape[:2]
    
    w = x2 - x1
    h = y2 - y1
    
    expand_w = int(w * expand_ratio)
    expand_h = int(h * expand_ratio)
    
    x1 = max(0, x1 - expand_w)
    y1 = max(0, y1 - expand_h)
    x2 = min(width, x2 + expand_w)
    y2 = min(height, y2 + expand_h)
    
    return [x1, y1, x2, y2]


def merge_bboxes(
    bboxes: List[List[int]],
    merge_threshold: int = 10
) -> List[List[int]]:
    """
    合并非常接近的矩形框，减少重复处理区域。

    这里采用简单阈值规则：四个边坐标都足够接近时就合并。
    """
    if not bboxes:
        return []
    
    merged = []
    used = set()
    
    for i, bbox1 in enumerate(bboxes):
        if i in used:
            continue
        
        x1, y1, x2, y2 = bbox1
        
        for j, bbox2 in enumerate(bboxes[i+1:], i+1):
            if j in used:
                continue
            
            bx1, by1, bx2, by2 = bbox2
            
            if (abs(x1 - bx1) < merge_threshold and 
                abs(y1 - by1) < merge_threshold and
                abs(x2 - bx2) < merge_threshold and
                abs(y2 - by2) < merge_threshold):
                x1 = min(x1, bx1)
                y1 = min(y1, by1)
                x2 = max(x2, bx2)
                y2 = max(y2, by2)
                used.add(j)
        
        merged.append([x1, y1, x2, y2])
    
    return merged
