export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface AnnotationSegment {
  id?: string;
  start_frame: number;
  end_frame: number;
  rect: Rect;
  expand_px?: number;
  feather_px?: number;
  enabled?: boolean;
  created_at?: string;
  updated_at?: string;
}

export interface NormalizedAnnotationSegment extends Required<AnnotationSegment> {}

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

export interface LoadAnnotationsResult {
  success: boolean;
  exists?: boolean;
  warning?: string;
  error?: string;
  sidecar_path?: string;
  video_meta?: VideoMeta;
  segments?: NormalizedAnnotationSegment[];
}

export interface SaveAnnotationsResult extends LoadAnnotationsResult {}

export interface MediaInfoResult {
  success: boolean;
  type?: 'video' | 'image';
  fps?: number;
  frame_count?: number;
  duration?: number;
  width?: number;
  height?: number;
  error?: string;
}

export interface PreviewSessionResult {
  success: boolean;
  session_id?: string;
  preview_fps?: number;
  total_preview_frames?: number;
  width?: number;
  height?: number;
  error?: string;
}

export interface PreviewFrameResult {
  success: boolean;
  frame_index?: number;
  frame_url?: string;
  decode_ms?: number;
  error?: string;
}

export interface AppSettings {
  language: 'zh' | 'en';
  theme: 'light' | 'dark';
  output: {
    path: string;
    model_id: 'lama_roi';
  };
}

export interface ProgressEventPayload {
  progress?: number;
  message?: string;
  status?: string;
  processed_frames?: number;
  total_frames?: number;
  estimated_time?: string;
  phase?: string;
  eta_seconds?: number;
  throughput_fps?: number;
  opaque_infer?: boolean;
}
