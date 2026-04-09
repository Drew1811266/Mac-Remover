// 单个矩形标注框（基于视频原始分辨率坐标）。
export interface AnnotationRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

// 一个“标记段”：
// 在指定帧区间内，对某个矩形区域做去水印处理。
export interface AnnotationSegment {
  id: string;
  start_frame: number;
  end_frame: number;
  rect: AnnotationRect;
  expand_px: number;
  feather_px: number;
  enabled: boolean;
  created_at: string;
  updated_at: string;
}

// 视频元信息，用于校验 sidecar 是否匹配当前视频。
export interface VideoMeta {
  path: string;
  basename: string;
  sha1: string;
  size: number;
  mtime_ns: number;
  width: number;
  height: number;
  fps: number;
  frame_count: number;
}

// sidecar 文件整体结构。
export interface AnnotationSidecar {
  version: string;
  video_meta: VideoMeta;
  segments: AnnotationSegment[];
  updated_at: string;
}
