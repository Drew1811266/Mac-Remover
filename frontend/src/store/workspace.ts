// 打标工作区状态仓库（zustand）。
// 专门管理视频帧、标记段、选中项等打标页状态。
import { create } from 'zustand';
import type { AnnotationRect, AnnotationSegment, VideoMeta } from '../types/annotation';

// 打标页模式：主流程页 or 工作台页。
export type ViewMode = 'main' | 'workspace';

export interface WorkspaceState {
  viewMode: ViewMode;
  videoPath: string;
  videoMeta: VideoMeta | null;
  currentFrame: number;
  selectedId: string | null;
  showAll: boolean;
  segments: AnnotationSegment[];
  setViewMode: (mode: ViewMode) => void;
  setVideoPath: (path: string) => void;
  setVideoMeta: (meta: VideoMeta | null) => void;
  setCurrentFrame: (frame: number) => void;
  selectSegment: (id: string | null) => void;
  setShowAll: (show: boolean) => void;
  // 直接替换整批标记段（例如从磁盘加载）。
  replaceSegments: (segments: AnnotationSegment[]) => void;
  // 新增或覆盖某个标记段。
  upsertSegment: (segment: AnnotationSegment) => void;
  updateSegment: (id: string, patch: Partial<AnnotationSegment>) => void;
  removeSegment: (id: string) => void;
  clearSegments: () => void;
  createSegmentFromRect: (rect: AnnotationRect, frame: number, options?: CreateSegmentOptions) => void;
}

export interface CreateSegmentOptions {
  fps?: number;
  frameMax?: number;
}

function nowIso(): string {
  // 统一时间戳格式，便于前后端对齐。
  return new Date().toISOString();
}

function clampFrame(value: number, frameMax?: number): number {
  const numeric = Number(value);
  const safeFrame = Number.isFinite(numeric) ? Math.max(0, Math.round(numeric)) : 0;
  if (typeof frameMax !== 'number' || !Number.isFinite(frameMax)) return safeFrame;
  return Math.min(Math.max(0, Math.round(frameMax)), safeFrame);
}

function createSegment(rect: AnnotationRect, frame: number, options?: CreateSegmentOptions): AnnotationSegment {
  // 旧版行为：新增标记段默认从当前帧开始，覆盖约 2 秒。
  const now = nowIso();
  const start = clampFrame(frame, options?.frameMax);
  const fps = Math.max(1, Number(options?.fps) || 24);
  const defaultSpan = Math.max(1, Math.round(fps * 2));
  const end = clampFrame(start + defaultSpan, options?.frameMax);
  return {
    id: `seg_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    start_frame: start,
    end_frame: Math.max(start, end),
    rect,
    expand_px: 5,
    feather_px: 3,
    enabled: true,
    created_at: now,
    updated_at: now,
  };
}

export function resolveVisibleStageSegments(
  segments: AnnotationSegment[],
  currentFrame: number,
): AnnotationSegment[] {
  return segments.filter((segment) =>
    segment.enabled !== false && segment.start_frame <= currentFrame && currentFrame <= segment.end_frame,
  );
}

// 打标工作区状态仓库：
// 管理当前视频、当前帧、选中项、标记段列表等核心数据。
export const useWorkspaceStore = create<WorkspaceState>((set) => ({
  viewMode: 'main',
  videoPath: '',
  videoMeta: null,
  currentFrame: 0,
  selectedId: null,
  showAll: true,
  segments: [],

  setViewMode: (mode) => set({ viewMode: mode }),
  setVideoPath: (path) => set({ videoPath: path }),
  setVideoMeta: (meta) => set({ videoMeta: meta }),
  setCurrentFrame: (frame) =>
    set(() => {
      // 防御式转换，保证 currentFrame 一定是 >=0 的整数。
      const numeric = Number(frame);
      const safeFrame = Number.isFinite(numeric) ? Math.max(0, Math.round(numeric)) : 0;
      return { currentFrame: safeFrame };
    }),
  selectSegment: (id) => set({ selectedId: id }),
  setShowAll: (show) => set({ showAll: show }),
  replaceSegments: (segments) =>
    set({
      segments,
      // 默认选中新列表最后一项，方便用户继续编辑。
      selectedId: segments.length > 0 ? segments[segments.length - 1].id : null,
    }),

  upsertSegment: (segment) =>
    set((state) => {
      const idx = state.segments.findIndex((seg) => seg.id === segment.id);
      if (idx < 0) {
        return { segments: [...state.segments, segment], selectedId: segment.id };
      }
      const next = [...state.segments];
      next[idx] = segment;
      return { segments: next, selectedId: segment.id };
    }),

  updateSegment: (id, patch) =>
    set((state) => ({
      segments: state.segments.map((seg) =>
        seg.id === id ? { ...seg, ...patch, updated_at: nowIso() } : seg,
      ),
    })),

  removeSegment: (id) =>
    set((state) => ({
      segments: state.segments.filter((seg) => seg.id !== id),
      selectedId: state.selectedId === id ? null : state.selectedId,
    })),

  clearSegments: () => set({ segments: [], selectedId: null }),

  createSegmentFromRect: (rect, frame, options) =>
    set((state) => {
      const segment = createSegment(rect, frame, options);
      return {
        segments: [...state.segments, segment],
        selectedId: segment.id,
      };
    }),
}));
