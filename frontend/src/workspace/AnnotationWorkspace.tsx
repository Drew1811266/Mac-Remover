// 打标工作台：
// 1) 在当前帧上绘制/编辑矩形标记段
// 2) 维护时间轴帧区间
// 3) 提供加载/保存/清空标注入口
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Button, Card, Space, Switch, Table, Tooltip, Typography } from '@douyinfe/semi-ui';
import { IconPlay, IconPause, IconArrowLeft, IconArrowRight } from '@douyinfe/semi-icons';

import type { AnnotationRect, AnnotationSegment } from '../types/annotation';
import { useWorkspaceStore } from '../store/workspace';
import { useI18n } from '../i18n/useI18n';

const { Text } = Typography;

interface StageMetrics {
  // 舞台容器尺寸 + 实际视频在容器内的位置与大小（含 letterbox 偏移）。
  width: number;
  height: number;
  videoLeft: number;
  videoTop: number;
  videoWidth: number;
  videoHeight: number;
}

interface DraftRect {
  // 鼠标拖拽中间态（舞台坐标）。
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

type ResizeHandle = 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w' | 'nw';

interface RectDragState {
  // 当前被拖拽的标记段及坐标换算参数。
  id: string;
  mode: 'move' | 'resize';
  handle?: ResizeHandle;
  startClientX: number;
  startClientY: number;
  baseRect: AnnotationRect;
  sourcePerStageX: number;
  sourcePerStageY: number;
  videoWidth: number;
  videoHeight: number;
}

const MIN_RECT_SIZE = 4;
const KEY_REPEAT_DELAY_MS = 180;
const KEY_REPEAT_INTERVAL_MS = 40;
const EMPTY_STAGE_WIDTH = 1280;
const EMPTY_STAGE_HEIGHT = 720;

function clamp(value: number, min: number, max: number): number {
  // 数值钳制工具。
  return Math.min(max, Math.max(min, value));
}

function normalizeDraftRect(draft: DraftRect): AnnotationRect {
  // 把任意方向拖拽统一成 x/y/width/height 形式。
  const left = Math.min(draft.x1, draft.x2);
  const top = Math.min(draft.y1, draft.y2);
  const right = Math.max(draft.x1, draft.x2);
  const bottom = Math.max(draft.y1, draft.y2);
  return {
    x: left,
    y: top,
    width: Math.max(1, right - left),
    height: Math.max(1, bottom - top),
  };
}

function computeVideoBox(containerWidth: number, containerHeight: number, mediaWidth: number, mediaHeight: number): StageMetrics {
  // 计算“视频实际显示区域”在容器中的位置，处理左右/上下留白。
  if (containerWidth <= 0 || containerHeight <= 0) {
    return {
      width: 1,
      height: 1,
      videoLeft: 0,
      videoTop: 0,
      videoWidth: 1,
      videoHeight: 1,
    };
  }

  const safeMediaWidth = Math.max(1, mediaWidth);
  const safeMediaHeight = Math.max(1, mediaHeight);
  const stageRatio = containerWidth / containerHeight;
  const mediaRatio = safeMediaWidth / safeMediaHeight;

  let videoWidth = containerWidth;
  let videoHeight = containerHeight;
  let videoLeft = 0;
  let videoTop = 0;

  if (stageRatio > mediaRatio) {
    videoHeight = containerHeight;
    videoWidth = videoHeight * mediaRatio;
    videoLeft = (containerWidth - videoWidth) / 2;
  } else {
    videoWidth = containerWidth;
    videoHeight = videoWidth / mediaRatio;
    videoTop = (containerHeight - videoHeight) / 2;
  }

  return {
    width: containerWidth,
    height: containerHeight,
    videoLeft,
    videoTop,
    videoWidth,
    videoHeight,
  };
}

function clampRectToVideoBounds(
  rect: AnnotationRect,
  videoWidth: number,
  videoHeight: number,
  minSize: number,
): AnnotationRect {
  // 把矩形限制在视频范围内，同时保证最小尺寸。
  const maxW = Math.max(1, Math.round(videoWidth));
  const maxH = Math.max(1, Math.round(videoHeight));
  const safeMin = Math.max(1, Math.min(minSize, Math.min(maxW, maxH)));

  let width = clamp(Math.round(rect.width), safeMin, maxW);
  let height = clamp(Math.round(rect.height), safeMin, maxH);
  let x = Math.round(rect.x);
  let y = Math.round(rect.y);

  x = clamp(x, 0, maxW - width);
  y = clamp(y, 0, maxH - height);
  width = clamp(width, safeMin, maxW - x);
  height = clamp(height, safeMin, maxH - y);

  return { x, y, width, height };
}

function applyMoveDeltaToRect(
  baseRect: AnnotationRect,
  deltaX: number,
  deltaY: number,
  videoWidth: number,
  videoHeight: number,
): AnnotationRect {
  // 移动模式：在原矩形上叠加位移，再做边界约束。
  return clampRectToVideoBounds(
    {
      ...baseRect,
      x: baseRect.x + deltaX,
      y: baseRect.y + deltaY,
    },
    videoWidth,
    videoHeight,
    MIN_RECT_SIZE,
  );
}

function applyResizeDeltaToRect(
  baseRect: AnnotationRect,
  deltaX: number,
  deltaY: number,
  handle: ResizeHandle,
  videoWidth: number,
  videoHeight: number,
): AnnotationRect {
  // 缩放模式：按 handle 方向改变边界，再做边界和最小尺寸约束。
  const maxW = Math.max(1, Math.round(videoWidth));
  const maxH = Math.max(1, Math.round(videoHeight));
  const safeMin = Math.max(1, Math.min(MIN_RECT_SIZE, Math.min(maxW, maxH)));

  let left = baseRect.x;
  let top = baseRect.y;
  let right = baseRect.x + baseRect.width;
  let bottom = baseRect.y + baseRect.height;

  if (handle.includes('w')) {
    left = Math.min(left + deltaX, right - safeMin);
  }
  if (handle.includes('e')) {
    right = Math.max(right + deltaX, left + safeMin);
  }
  if (handle.includes('n')) {
    top = Math.min(top + deltaY, bottom - safeMin);
  }
  if (handle.includes('s')) {
    bottom = Math.max(bottom + deltaY, top + safeMin);
  }

  left = clamp(left, 0, maxW - safeMin);
  top = clamp(top, 0, maxH - safeMin);
  right = clamp(right, left + safeMin, maxW);
  bottom = clamp(bottom, top + safeMin, maxH);

  return clampRectToVideoBounds(
    {
      x: left,
      y: top,
      width: right - left,
      height: bottom - top,
    },
    maxW,
    maxH,
    safeMin,
  );
}

interface AnnotationWorkspaceProps {
  frameImageUrl?: string;
  previewFrameWidth?: number;
  previewFrameHeight?: number;
  onSaveAnnotations: () => Promise<void>;
  onClearAnnotations: () => Promise<void>;
}

export function AnnotationWorkspace({
  frameImageUrl,
  previewFrameWidth,
  previewFrameHeight,
  onSaveAnnotations,
  onClearAnnotations,
}: AnnotationWorkspaceProps) {
  const { t } = useI18n();
  const stageRef = useRef<HTMLDivElement | null>(null);
  const workspaceStageWrapRef = useRef<HTMLDivElement | null>(null);
  const timelineRef = useRef<HTMLDivElement | null>(null);
  const trackRef = useRef<HTMLDivElement | null>(null);
  const heldKeyRef = useRef<string | null>(null);
  const repeatTimerRef = useRef<number | null>(null);
  const repeatIntervalRef = useRef<number | null>(null);
  const currentFrameRef = useRef(0);
  const selectedIdRef = useRef<string | null>(null);
  const segmentsRef = useRef<AnnotationSegment[]>([]);
  const activeBoundaryRef = useRef<{ segmentId: string; edge: 'start' | 'end' } | null>(null);
  const isPlayingRef = useRef(false);
  const frameMaxRef = useRef(0);
  const [metrics, setMetrics] = useState<StageMetrics>(() => computeVideoBox(1, 1, 16, 9));
  const [stageContainerWidth, setStageContainerWidth] = useState(0);
  const [stageContainerHeight, setStageContainerHeight] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [draftRect, setDraftRect] = useState<DraftRect | null>(null);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
  const [rectDrag, setRectDrag] = useState<RectDragState | null>(null);
  const [segmentDrag, setSegmentDrag] = useState<{
    id: string;
    edge: 'start' | 'end';
    startX: number;
    baseStart: number;
    baseEnd: number;
    trackWidth: number;
  } | null>(null);
  const [activeBoundary, setActiveBoundary] = useState<{
    segmentId: string;
    edge: 'start' | 'end';
  } | null>(null);

  // 从 workspace store 读取核心状态与操作。
  const {
    videoMeta,
    currentFrame,
    selectedId,
    segments,
    showAll,
    setCurrentFrame,
    selectSegment,
    setShowAll,
    updateSegment,
    removeSegment,
    createSegmentFromRect,
  } = useWorkspaceStore();

  const fps = Math.max(1, videoMeta?.fps ?? 24);
  const frameMax = Math.max(0, (videoMeta?.frame_count ?? 1) - 1);
  const sourceWidth = Number(
    previewFrameWidth && previewFrameWidth > 0
      ? previewFrameWidth
      : (videoMeta?.width ?? EMPTY_STAGE_WIDTH),
  );
  const sourceHeight = Number(
    previewFrameHeight && previewFrameHeight > 0
      ? previewFrameHeight
      : (videoMeta?.height ?? EMPTY_STAGE_HEIGHT),
  );
  const annotationWidth = Math.max(1, Number((videoMeta?.width ?? sourceWidth) || 1));
  const annotationHeight = Math.max(1, Number((videoMeta?.height ?? sourceHeight) || 1));
  const stageAspectRatio = sourceWidth > 0 && sourceHeight > 0 ? `${sourceWidth} / ${sourceHeight}` : undefined;
  const dpr = typeof window !== 'undefined' ? Math.max(1, window.devicePixelRatio || 1) : 1;
  const selectedSegment = useMemo(
    () => (selectedId ? segments.find((segment) => segment.id === selectedId) ?? null : null),
    [segments, selectedId],
  );

  useEffect(() => {
    currentFrameRef.current = currentFrame;
  }, [currentFrame]);

  useEffect(() => {
    selectedIdRef.current = selectedId;
  }, [selectedId]);

  useEffect(() => {
    segmentsRef.current = segments;
  }, [segments]);

  useEffect(() => {
    activeBoundaryRef.current = activeBoundary;
  }, [activeBoundary]);

  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

  useEffect(() => {
    frameMaxRef.current = frameMax;
  }, [frameMax]);

  const focusTimeline = useCallback(() => {
    window.requestAnimationFrame(() => {
      timelineRef.current?.focus();
    });
  }, []);

  const stopPlaybackIfNeeded = useCallback(() => {
    if (isPlayingRef.current) {
      isPlayingRef.current = false;
      setIsPlaying(false);
    }
  }, []);

  const applyBoundaryFrame = useCallback((
    segment: AnnotationSegment,
    edge: 'start' | 'end',
    nextFrame: number,
  ): number => {
    // 统一处理边界 frame 更新：clamp + start/end 关系约束。
    const clampedFrame = Math.round(clamp(nextFrame, 0, frameMax));
    if (edge === 'start') {
      const appliedStart = Math.min(clampedFrame, segment.end_frame);
      updateSegment(segment.id, { start_frame: appliedStart });
      return appliedStart;
    }
    const appliedEnd = Math.max(clampedFrame, segment.start_frame);
    updateSegment(segment.id, { end_frame: appliedEnd });
    return appliedEnd;
  }, [frameMax, updateSegment]);

  const resolveNearestBoundaryEdge = useCallback((segment: AnnotationSegment, frame: number): 'start' | 'end' => (
    Math.abs(frame - segment.start_frame) <= Math.abs(frame - segment.end_frame) ? 'start' : 'end'
  ), []);

  const selectSegmentWithExplicitBoundary = useCallback((
    segment: AnnotationSegment,
    edge: 'start' | 'end',
    options?: { focus?: boolean; syncFrame?: boolean },
  ) => {
    const nextBoundary = { segmentId: segment.id, edge };
    selectedIdRef.current = segment.id;
    activeBoundaryRef.current = nextBoundary;
    selectSegment(segment.id);
    setActiveBoundary(nextBoundary);
    if (options?.syncFrame) {
      const boundaryFrame = edge === 'start' ? segment.start_frame : segment.end_frame;
      currentFrameRef.current = boundaryFrame;
      setCurrentFrame(boundaryFrame);
    }
    if (options?.focus !== false) {
      focusTimeline();
    }
  }, [focusTimeline, selectSegment, setCurrentFrame]);

  const selectSegmentWithNearestBoundary = useCallback((
    segment: AnnotationSegment,
    options?: { focus?: boolean; syncFrame?: boolean },
  ) => {
    const edge = resolveNearestBoundaryEdge(segment, currentFrameRef.current);
    selectSegmentWithExplicitBoundary(segment, edge, options);
  }, [resolveNearestBoundaryEdge, selectSegmentWithExplicitBoundary]);

  const selectNewSegmentForRangeEditing = useCallback((segment: AnnotationSegment) => {
    selectSegmentWithExplicitBoundary(segment, 'end', { focus: true, syncFrame: false });
  }, [selectSegmentWithExplicitBoundary]);

  const removeSegmentWithBoundaryCleanup = useCallback((segmentId: string) => {
    // 删除片段时同步清理激活边界，避免悬空引用。
    if (selectedIdRef.current === segmentId) {
      selectedIdRef.current = null;
    }
    if (activeBoundaryRef.current?.segmentId === segmentId) {
      activeBoundaryRef.current = null;
    }
    removeSegment(segmentId);
    setActiveBoundary((prev) => (prev && prev.segmentId === segmentId ? null : prev));
  }, [removeSegment]);

  const stepPreviewFrame = useCallback((delta: number) => {
    stopPlaybackIfNeeded();
    const nextFrame = Math.round(clamp(currentFrameRef.current + delta, 0, frameMaxRef.current));
    if (nextFrame !== currentFrameRef.current) {
      currentFrameRef.current = nextFrame;
      setCurrentFrame(nextFrame);
    }
    focusTimeline();
    return nextFrame;
  }, [focusTimeline, setCurrentFrame, stopPlaybackIfNeeded]);

  const stepActiveBoundary = useCallback((delta: number) => {
    const selectedSegmentId = selectedIdRef.current;
    const currentBoundary = activeBoundaryRef.current;
    const segment = selectedSegmentId
      ? segmentsRef.current.find((item) => item.id === selectedSegmentId) ?? null
      : null;

    if (!segment || !currentBoundary || currentBoundary.segmentId !== segment.id) {
      return stepPreviewFrame(delta);
    }

    stopPlaybackIfNeeded();
    const baseFrame = currentBoundary.edge === 'start' ? segment.start_frame : segment.end_frame;
    const appliedFrame = applyBoundaryFrame(segment, currentBoundary.edge, baseFrame + delta);
    segmentsRef.current = useWorkspaceStore.getState().segments;
    if (appliedFrame !== currentFrameRef.current) {
      currentFrameRef.current = appliedFrame;
      setCurrentFrame(appliedFrame);
    }
    focusTimeline();
    return appliedFrame;
  }, [applyBoundaryFrame, focusTimeline, setCurrentFrame, stepPreviewFrame, stopPlaybackIfNeeded]);

  const applyCurrentFrameToActiveBoundary = useCallback(() => {
    const selectedSegmentId = selectedIdRef.current;
    const currentBoundary = activeBoundaryRef.current;
    const segment = selectedSegmentId
      ? segmentsRef.current.find((item) => item.id === selectedSegmentId) ?? null
      : null;

    if (!segment || !currentBoundary || currentBoundary.segmentId !== segment.id) {
      return;
    }

    const appliedFrame = applyBoundaryFrame(segment, currentBoundary.edge, currentFrameRef.current);
    segmentsRef.current = useWorkspaceStore.getState().segments;
    if (appliedFrame !== currentFrameRef.current) {
      currentFrameRef.current = appliedFrame;
      setCurrentFrame(appliedFrame);
    }
    focusTimeline();
  }, [applyBoundaryFrame, focusTimeline, setCurrentFrame]);

  const switchActiveBoundary = useCallback((direction: number) => {
    const selectedSegmentId = selectedIdRef.current;
    const segment = selectedSegmentId
      ? segmentsRef.current.find((item) => item.id === selectedSegmentId) ?? null
      : null;
    if (!segment) return;

    const currentBoundary = activeBoundaryRef.current;
    const currentEdge = currentBoundary && currentBoundary.segmentId === segment.id
      ? currentBoundary.edge
      : resolveNearestBoundaryEdge(segment, currentFrameRef.current);
    const nextEdge = direction >= 0
      ? (currentEdge === 'start' ? 'end' : 'start')
      : (currentEdge === 'end' ? 'start' : 'end');
    selectSegmentWithExplicitBoundary(segment, nextEdge, { focus: true, syncFrame: true });
  }, [resolveNearestBoundaryEdge, selectSegmentWithExplicitBoundary]);

  const stopArrowRepeat = useCallback(() => {
    if (repeatTimerRef.current !== null) {
      window.clearTimeout(repeatTimerRef.current);
      repeatTimerRef.current = null;
    }
    if (repeatIntervalRef.current !== null) {
      window.clearInterval(repeatIntervalRef.current);
      repeatIntervalRef.current = null;
    }
    heldKeyRef.current = null;
  }, []);

  const startArrowRepeat = useCallback((direction: number, step: number) => {
    stopArrowRepeat();
    heldKeyRef.current = `${direction}:${step}`;
    repeatTimerRef.current = window.setTimeout(() => {
      repeatIntervalRef.current = window.setInterval(() => {
        stepActiveBoundary(direction * step);
      }, KEY_REPEAT_INTERVAL_MS);
    }, KEY_REPEAT_DELAY_MS);
  }, [stepActiveBoundary, stopArrowRepeat]);

  useEffect(() => {
    window.addEventListener('blur', stopArrowRepeat);
    return () => {
      window.removeEventListener('blur', stopArrowRepeat);
      stopArrowRepeat();
    };
  }, [stopArrowRepeat]);

  let stageWidth = 0;
  let stageHeight = 0;
  if (sourceWidth > 0 && sourceHeight > 0) {
    const cssNativeWidth = sourceWidth / dpr;
    const cssNativeHeight = sourceHeight / dpr;
    const widthCap = stageContainerWidth > 0 ? stageContainerWidth : cssNativeWidth;
    const stageHeightCap = stageContainerHeight > 0
      ? Math.max(240, Math.min(cssNativeHeight, stageContainerHeight))
      : cssNativeHeight;
    const heightBasedWidthCap = stageHeightCap * (sourceWidth / sourceHeight);
    stageWidth = Math.min(cssNativeWidth, widthCap, heightBasedWidthCap);
    stageHeight = stageWidth * (sourceHeight / sourceWidth);
  }

  const hasComputedStageSize = Number.isFinite(stageWidth)
    && Number.isFinite(stageHeight)
    && stageWidth > 0
    && stageHeight > 0;

  const stageStyle = hasComputedStageSize
    ? {
        width: `${Math.round(stageWidth)}px`,
        height: `${Math.round(stageHeight)}px`,
        maxWidth: '100%',
      }
    : {
        width: '100%',
        aspectRatio: stageAspectRatio || '16 / 9',
      };

  const syncMetrics = useCallback(() => {
    // 根据当前舞台 DOM 尺寸同步视频显示区域参数。
    const stage = stageRef.current;
    if (!stage) return;
    const rect = stage.getBoundingClientRect();
    const mediaWidth = Math.max(1, sourceWidth || annotationWidth || 16);
    const mediaHeight = Math.max(1, sourceHeight || annotationHeight || 9);
    setMetrics(computeVideoBox(rect.width, rect.height, mediaWidth, mediaHeight));
  }, [annotationHeight, annotationWidth, sourceHeight, sourceWidth]);

  useEffect(() => {
    // 首次和窗口变化时都要重算 metrics。
    syncMetrics();
    window.addEventListener('resize', syncMetrics);
    return () => window.removeEventListener('resize', syncMetrics);
  }, [syncMetrics]);

  useEffect(() => {
    // 观测外层容器宽度，驱动舞台自适应尺寸。
    const node = workspaceStageWrapRef.current;
    if (!node) return;

    const measure = () => {
      const rect = node.getBoundingClientRect();
      setStageContainerWidth(Math.max(0, Math.floor(rect.width)));
      setStageContainerHeight(Math.max(0, Math.floor(rect.height)));
    };

    measure();

    let resizeObserver: ResizeObserver | null = null;
    if (typeof ResizeObserver !== 'undefined') {
      resizeObserver = new ResizeObserver(() => measure());
      resizeObserver.observe(node);
    }

    window.addEventListener('resize', measure);
    return () => {
      window.removeEventListener('resize', measure);
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
    };
  }, []);

  useEffect(() => {
    // 帧图、源尺寸或容器尺寸变化后重新同步坐标系。
    syncMetrics();
  }, [frameImageUrl, sourceHeight, sourceWidth, stageContainerWidth, syncMetrics]);

  useEffect(() => {
    // 播放状态下按 fps 推进帧索引。
    if (!isPlaying) return;
    const timer = window.setInterval(() => {
      setCurrentFrame((currentFrame + 1) > frameMax ? frameMax : currentFrame + 1);
    }, Math.max(30, Math.round(1000 / fps)));
    return () => window.clearInterval(timer);
  }, [currentFrame, fps, frameMax, isPlaying, setCurrentFrame]);

  useEffect(() => {
    // 播放到最后一帧自动停下。
    if (currentFrame >= frameMax && isPlaying) {
      setIsPlaying(false);
    }
  }, [currentFrame, frameMax, isPlaying]);

  useEffect(() => {
    // 时间轴区间拖拽：实时更新 start_frame / end_frame。
    if (!segmentDrag) return;
    const onMove = (event: MouseEvent) => {
      const deltaPx = event.clientX - segmentDrag.startX;
      const deltaFrame = Math.round((deltaPx / Math.max(1, segmentDrag.trackWidth)) * Math.max(1, frameMax));
      const segment = segments.find((item) => item.id === segmentDrag.id);
      if (!segment) return;
      const baseFrame = segmentDrag.edge === 'start'
        ? segmentDrag.baseStart
        : segmentDrag.baseEnd;
      const appliedFrame = applyBoundaryFrame(
        segment,
        segmentDrag.edge,
        baseFrame + deltaFrame,
      );
      segmentsRef.current = useWorkspaceStore.getState().segments;
      if (appliedFrame !== currentFrame) {
        currentFrameRef.current = appliedFrame;
        setCurrentFrame(appliedFrame);
      }
    };
    const onUp = () => setSegmentDrag(null);

    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [applyBoundaryFrame, currentFrame, frameMax, segmentDrag, segments, setCurrentFrame]);

  useEffect(() => {
    // 画布矩形拖拽/缩放：实时更新标记段矩形。
    if (!rectDrag) return;

    const onMove = (event: MouseEvent) => {
      const deltaX = Math.round((event.clientX - rectDrag.startClientX) * rectDrag.sourcePerStageX);
      const deltaY = Math.round((event.clientY - rectDrag.startClientY) * rectDrag.sourcePerStageY);

      let nextRect: AnnotationRect;
      if (rectDrag.mode === 'move') {
        nextRect = applyMoveDeltaToRect(
          rectDrag.baseRect,
          deltaX,
          deltaY,
          rectDrag.videoWidth,
          rectDrag.videoHeight,
        );
      } else {
        nextRect = applyResizeDeltaToRect(
          rectDrag.baseRect,
          deltaX,
          deltaY,
          rectDrag.handle ?? 'se',
          rectDrag.videoWidth,
          rectDrag.videoHeight,
        );
      }

      updateSegment(rectDrag.id, { rect: nextRect });
    };

    const onUp = () => {
      setRectDrag(null);
    };

    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    window.addEventListener('blur', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      window.removeEventListener('blur', onUp);
    };
  }, [rectDrag, updateSegment]);

  useEffect(() => {
    // 选中变化后保证 activeBoundary 有效：默认选中距离当前帧更近的一端。
    if (!selectedId) {
      activeBoundaryRef.current = null;
      setActiveBoundary(null);
      return;
    }
    const nextSelectedSegment = segments.find((segment) => segment.id === selectedId) ?? null;
    if (!nextSelectedSegment) {
      activeBoundaryRef.current = null;
      setActiveBoundary(null);
      return;
    }
    setActiveBoundary((prev) => {
      if (prev && prev.segmentId === selectedId) {
        activeBoundaryRef.current = prev;
        return prev;
      }
      const nextBoundary = {
        segmentId: selectedId,
        edge: resolveNearestBoundaryEdge(nextSelectedSegment, currentFrameRef.current),
      };
      activeBoundaryRef.current = nextBoundary;
      return nextBoundary;
    });
  }, [resolveNearestBoundaryEdge, segments, selectedId]);

  const stageSegments = useMemo(() => (
    // 画布只显示当前帧命中的已启用标记段。
    segments.filter((seg) => seg.enabled !== false && seg.start_frame <= currentFrame && currentFrame <= seg.end_frame)
  ), [currentFrame, segments]);

  const tableSegments = useMemo(() => {
    // 列表可切换“显示全部”或“仅显示当前帧命中”。
    if (showAll) return segments;
    return segments.filter((seg) => seg.start_frame <= currentFrame && currentFrame <= seg.end_frame);
  }, [currentFrame, segments, showAll]);

  const stageToSourceRect = useCallback((draft: DraftRect): AnnotationRect => {
    // 舞台坐标 -> 原视频坐标（保存时使用原视频坐标）。
    const normalized = normalizeDraftRect(draft);
    const left = clamp(normalized.x, metrics.videoLeft, metrics.videoLeft + metrics.videoWidth);
    const top = clamp(normalized.y, metrics.videoTop, metrics.videoTop + metrics.videoHeight);
    const width = clamp(
      normalized.width,
      1,
      metrics.videoLeft + metrics.videoWidth - left,
    );
    const height = clamp(
      normalized.height,
      1,
      metrics.videoTop + metrics.videoHeight - top,
    );

    const scaleX = annotationWidth / metrics.videoWidth;
    const scaleY = annotationHeight / metrics.videoHeight;

    return {
      x: Math.round((left - metrics.videoLeft) * scaleX),
      y: Math.round((top - metrics.videoTop) * scaleY),
      width: Math.max(1, Math.round(width * scaleX)),
      height: Math.max(1, Math.round(height * scaleY)),
    };
  }, [annotationHeight, annotationWidth, metrics]);

  const sourceToStageRect = useCallback((rect: AnnotationRect): AnnotationRect => {
    // 原视频坐标 -> 舞台坐标（渲染时换算到当前显示尺寸）。
    const scaleX = metrics.videoWidth / annotationWidth;
    const scaleY = metrics.videoHeight / annotationHeight;
    return {
      x: metrics.videoLeft + rect.x * scaleX,
      y: metrics.videoTop + rect.y * scaleY,
      width: Math.max(1, rect.width * scaleX),
      height: Math.max(1, rect.height * scaleY),
    };
  }, [annotationHeight, annotationWidth, metrics]);

  const startRectDrag = useCallback((
    segment: AnnotationSegment,
    mode: 'move' | 'resize',
    event: React.MouseEvent<HTMLElement>,
    handle?: ResizeHandle,
  ) => {
    // 启动矩形拖拽或缩放，并记录起始状态。
    event.preventDefault();
    event.stopPropagation();

    const sourcePerStageX = annotationWidth / Math.max(1, metrics.videoWidth);
    const sourcePerStageY = annotationHeight / Math.max(1, metrics.videoHeight);

    setDraftRect(null);
    setDragStart(null);
    selectSegmentWithNearestBoundary(segment, { focus: true, syncFrame: false });
    setRectDrag({
      id: segment.id,
      mode,
      handle,
      startClientX: event.clientX,
      startClientY: event.clientY,
      baseRect: segment.rect,
      sourcePerStageX,
      sourcePerStageY,
      videoWidth: annotationWidth,
      videoHeight: annotationHeight,
    });
  }, [annotationHeight, annotationWidth, metrics.videoHeight, metrics.videoWidth, selectSegmentWithNearestBoundary]);

  const resizeHandles: ResizeHandle[] = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'];

  const onStageMouseDown = (event: React.MouseEvent<HTMLDivElement>) => {
    // 左键在舞台空白处按下：开始绘制新矩形草稿。
    if (rectDrag) return;
    if (event.button !== 0) return;
    const stage = stageRef.current;
    if (!stage) return;
    const bounds = stage.getBoundingClientRect();
    const localX = clamp(event.clientX - bounds.left, metrics.videoLeft, metrics.videoLeft + metrics.videoWidth);
    const localY = clamp(event.clientY - bounds.top, metrics.videoTop, metrics.videoTop + metrics.videoHeight);
    setDragStart({ x: localX, y: localY });
    setDraftRect({ x1: localX, y1: localY, x2: localX, y2: localY });
  };

  const onStageMouseMove = (event: React.MouseEvent<HTMLDivElement>) => {
    // 更新绘制中的草稿矩形。
    if (rectDrag) return;
    if (!dragStart) return;
    const stage = stageRef.current;
    if (!stage) return;
    const bounds = stage.getBoundingClientRect();
    const localX = clamp(event.clientX - bounds.left, metrics.videoLeft, metrics.videoLeft + metrics.videoWidth);
    const localY = clamp(event.clientY - bounds.top, metrics.videoTop, metrics.videoTop + metrics.videoHeight);
    setDraftRect({ x1: dragStart.x, y1: dragStart.y, x2: localX, y2: localY });
  };

  const commitDraft = () => {
    // 鼠标抬起时提交草稿：尺寸太小则丢弃，否则生成新标记段。
    if (rectDrag) return;
    if (!draftRect) return;
    const rect = stageToSourceRect(draftRect);
    if (rect.width < 2 || rect.height < 2) {
      setDraftRect(null);
      setDragStart(null);
      return;
    }
    const existingIds = new Set(segmentsRef.current.map((segment) => segment.id));
    createSegmentFromRect(rect, currentFrameRef.current);
    const nextSegments = useWorkspaceStore.getState().segments;
    segmentsRef.current = nextSegments;
    const newSegment = nextSegments.find((segment) => !existingIds.has(segment.id))
      ?? nextSegments[nextSegments.length - 1]
      ?? null;
    if (newSegment) {
      selectNewSegmentForRangeEditing(newSegment);
    }
    setDraftRect(null);
    setDragStart(null);
    focusTimeline();
  };

  const columns = [
    // 右侧列表列定义：序号、帧区间、跳转、删除。
    {
      title: t('annotation.id'),
      dataIndex: 'id',
      width: 72,
      align: 'center' as const,
      render: (_: string, __: AnnotationSegment, idx: number) => (
        <span className="workspace-table-cell-id">{idx + 1}</span>
      ),
    },
    {
      title: t('annotation.range'),
      dataIndex: 'start_frame',
      width: 128,
      align: 'center' as const,
      render: (_: number, record: AnnotationSegment) => (
        <span className="workspace-table-cell-range">{`${record.start_frame} - ${record.end_frame}`}</span>
      ),
    },
    {
      title: t('annotation.actions'),
      width: 96,
      align: 'center' as const,
      render: (_: unknown, record: AnnotationSegment) => (
        <Button
          size="small"
          theme="borderless"
          className="workspace-table-jump-btn"
          onClick={(event) => {
            event.preventDefault();
            event.stopPropagation();
            selectSegmentWithExplicitBoundary(record, 'start', { focus: true, syncFrame: true });
          }}
        >
          {t('annotation.jump')}
        </Button>
      ),
    },
    {
      title: t('annotation.delete'),
      dataIndex: 'id',
      width: 72,
      align: 'center' as const,
      render: (_: string, record: AnnotationSegment) => (
        <button
          type="button"
          className="workspace-table-delete-btn"
          aria-label={t('annotation.deleteSelected')}
          onClick={(event) => {
            event.preventDefault();
            event.stopPropagation();
            removeSegmentWithBoundaryCleanup(record.id);
          }}
        >
          ×
        </button>
      ),
    },
  ];

  const getTrackStyle = useCallback((segment: AnnotationSegment) => {
    // 时间轴上单个标记段条块的位置和宽度。
    const totalFrames = Math.max(1, frameMax + 1);
    const left = (Math.max(0, segment.start_frame) / totalFrames) * 100;
    const width = ((Math.max(segment.start_frame, segment.end_frame) - Math.max(0, segment.start_frame) + 1) / totalFrames) * 100;
    return {
      left: `${left}%`,
      width: `${Math.max(width, 0.15)}%`,
      minWidth: '14px',
    };
  }, [frameMax]);

  const getTrackCursorStyle = useCallback(() => {
    // 时间轴当前帧游标位置。
    const safeMax = Math.max(1, frameMax);
    const left = (Math.max(0, currentFrame) / safeMax) * 100;
    return { left: `${left}%` };
  }, [currentFrame, frameMax]);

  const getTrackFrameFromClientX = useCallback((clientX: number) => {
    const track = trackRef.current;
    if (!track) return currentFrameRef.current;
    const rect = track.getBoundingClientRect();
    if (rect.width <= 0) return currentFrameRef.current;
    const ratio = clamp((clientX - rect.left) / rect.width, 0, 1);
    return Math.round(ratio * frameMaxRef.current);
  }, []);

  const onTrackBackgroundMouseDown = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    if (event.target !== event.currentTarget) return;
    event.preventDefault();
    stopPlaybackIfNeeded();
    const nextFrame = getTrackFrameFromClientX(event.clientX);
    if (nextFrame !== currentFrameRef.current) {
      currentFrameRef.current = nextFrame;
      setCurrentFrame(nextFrame);
    }
    focusTimeline();
  }, [focusTimeline, getTrackFrameFromClientX, setCurrentFrame, stopPlaybackIfNeeded]);

  const startSegmentDrag = (id: string, edge: 'start' | 'end', event: React.MouseEvent<HTMLElement>) => {
    // 从时间轴句柄开始拖拽标记段起点或终点。
    const track = trackRef.current;
    const seg = segments.find((item) => item.id === id);
    if (!track || !seg) return;
    event.preventDefault();
    event.stopPropagation();
    const width = track.getBoundingClientRect().width;
    stopPlaybackIfNeeded();
    selectSegmentWithExplicitBoundary(seg, edge, { focus: true, syncFrame: true });
    setSegmentDrag({
      id,
      edge,
      startX: event.clientX,
      baseStart: seg.start_frame,
      baseEnd: seg.end_frame,
      trackWidth: Math.max(1, width),
    });
  };

  const selectShortcutSegment = useCallback((slot: number) => {
    const visibleSegments = showAll
      ? segmentsRef.current
      : segmentsRef.current.filter(
          (seg) => seg.start_frame <= currentFrameRef.current && currentFrameRef.current <= seg.end_frame,
        );
    const segment = visibleSegments[slot];
    if (!segment) return false;
    selectSegmentWithNearestBoundary(segment, { focus: true, syncFrame: false });
    return true;
  }, [selectSegmentWithNearestBoundary, showAll]);

  const handleTimelineKeyDown = useCallback((event: React.KeyboardEvent<HTMLDivElement>) => {
    const target = event.target as HTMLElement | null;
    const tagName = target?.tagName?.toLowerCase();
    if (target?.isContentEditable || tagName === 'input' || tagName === 'textarea' || tagName === 'select') return;
    if (event.metaKey || event.ctrlKey || event.altKey) return;

    const digitKey = /^[1-9]$/.test(event.key)
      ? Number(event.key)
      : (/^Numpad[1-9]$/.test(event.code) ? Number(event.code.replace('Numpad', '')) : 0);
    if (digitKey > 0) {
      const selected = selectShortcutSegment(digitKey - 1);
      if (selected) {
        event.preventDefault();
      }
      return;
    }

    if (event.code === 'Space') {
      event.preventDefault();
      if (isPlayingRef.current) {
        stopPlaybackIfNeeded();
      } else {
        if (currentFrameRef.current >= frameMaxRef.current) {
          currentFrameRef.current = 0;
          setCurrentFrame(0);
        }
        isPlayingRef.current = true;
        setIsPlaying(true);
      }
      focusTimeline();
      return;
    }

    if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
      const selectedSegmentId = selectedIdRef.current;
      const segment = selectedSegmentId
        ? segmentsRef.current.find((item) => item.id === selectedSegmentId) ?? null
        : null;
      if (!segment) return;
      event.preventDefault();
      selectSegmentWithExplicitBoundary(
        segment,
        event.key === 'ArrowUp' ? 'start' : 'end',
        { focus: true, syncFrame: true },
      );
      return;
    }

    if (event.key === 'Tab') {
      const selectedSegmentId = selectedIdRef.current;
      const segment = selectedSegmentId
        ? segmentsRef.current.find((item) => item.id === selectedSegmentId) ?? null
        : null;
      if (!segment) return;
      event.preventDefault();
      switchActiveBoundary(event.shiftKey ? -1 : 1);
      return;
    }

    if (event.key === 'Enter') {
      const currentBoundary = activeBoundaryRef.current;
      if (!currentBoundary) return;
      event.preventDefault();
      applyCurrentFrameToActiveBoundary();
      return;
    }

    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault();
      if (event.repeat) return;
      const direction = event.key === 'ArrowLeft' ? -1 : 1;
      const step = event.shiftKey ? 10 : 1;
      stepActiveBoundary(direction * step);
      startArrowRepeat(direction, step);
      return;
    }

    if ((event.key === 'Delete' || event.key === 'Backspace') && selectedIdRef.current) {
      event.preventDefault();
      removeSegmentWithBoundaryCleanup(selectedIdRef.current);
    }
  }, [
    applyCurrentFrameToActiveBoundary,
    focusTimeline,
    removeSegmentWithBoundaryCleanup,
    selectSegmentWithExplicitBoundary,
    selectShortcutSegment,
    setCurrentFrame,
    startArrowRepeat,
    stepActiveBoundary,
    stopPlaybackIfNeeded,
    switchActiveBoundary,
  ]);

  const handleTimelineKeyUp = useCallback((event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      stopArrowRepeat();
    }
  }, [stopArrowRepeat]);

  const activeBoundaryLabel = activeBoundary?.edge === 'end'
    ? t('annotation.boundaryMode.end')
    : activeBoundary?.edge === 'start'
      ? t('annotation.boundaryMode.start')
      : t('annotation.boundaryMode.none');
  const selectedSegmentStart = selectedSegment ? selectedSegment.start_frame : '--';
  const selectedSegmentEnd = selectedSegment ? selectedSegment.end_frame : '--';

  return (
    // 主体布局：左侧画布+时间轴，右侧标记段管理表。
    <div className="workspace-shell">
      <Card
        title={t('annotate.title')}
        bodyStyle={{ padding: 12 }}
        headerExtraContent={(
          <Space wrap>
            <Button size="small" type="primary" onClick={() => void onSaveAnnotations()}>{t('annotation.save')}</Button>
            <Button size="small" type="danger" theme="light" onClick={() => void onClearAnnotations()}>
              {t('annotation.clear')}
            </Button>
          </Space>
        )}
      >
        <div className="workspace-main-grid">
          <div className="workspace-player-pane">
            <div className="workspace-stage-wrap" ref={workspaceStageWrapRef}>
              <div
                ref={stageRef}
                className="workspace-stage"
                style={stageStyle}
                onMouseDown={onStageMouseDown}
                onMouseMove={onStageMouseMove}
                onMouseUp={commitDraft}
                onMouseLeave={commitDraft}
              >
                {frameImageUrl ? (
                  <img src={frameImageUrl} alt="frame" className="workspace-frame" onLoad={syncMetrics} />
                ) : (
                  <div className="workspace-placeholder">{t('annotation.framePlaceholder')}</div>
                )}

                {stageSegments.map((segment) => {
                  const rect = sourceToStageRect(segment.rect);
                  const selected = selectedId === segment.id;
                  return (
                    <div
                      key={segment.id}
                      className={`workspace-segment active ${selected ? 'selected' : ''}`}
                      style={{
                        left: rect.x,
                        top: rect.y,
                        width: rect.width,
                        height: rect.height,
                      }}
                      onMouseDown={(event) => {
                        if (selected) {
                          startRectDrag(segment, 'move', event);
                          return;
                        }
                        event.stopPropagation();
                      }}
                      onClick={() => selectSegmentWithNearestBoundary(segment, { focus: true, syncFrame: false })}
                    />
                  );
                })}

                {stageSegments.map((segment) => {
                  const rect = sourceToStageRect(segment.rect);
                  const selected = selectedId === segment.id;
                  if (!selected) return null;

                  return (
                    <div
                      key={`${segment.id}-overlay`}
                      className="workspace-segment-overlay"
                      style={{
                        left: rect.x,
                        top: rect.y,
                        width: rect.width,
                        height: rect.height,
                      }}
                      onMouseDown={(event) => event.stopPropagation()}
                    >
                      <Tooltip content={t('annotation.deleteSelected')}>
                        <button
                          type="button"
                          className="workspace-segment-delete-btn"
                          onMouseDown={(event) => {
                            event.preventDefault();
                            event.stopPropagation();
                          }}
                          onClick={(event) => {
                            event.preventDefault();
                            event.stopPropagation();
                            removeSegmentWithBoundaryCleanup(segment.id);
                          }}
                        >
                          ×
                        </button>
                      </Tooltip>

                      {resizeHandles.map((handle) => (
                        <span
                          key={`${segment.id}-${handle}`}
                          className={`workspace-segment-handle workspace-segment-handle-${handle}`}
                          onMouseDown={(event) => startRectDrag(segment, 'resize', event, handle)}
                        />
                      ))}
                    </div>
                  );
                })}

                {draftRect && (() => {
                  const rect = normalizeDraftRect(draftRect);
                  return (
                    <div
                      className="workspace-segment draft"
                      style={{ left: rect.x, top: rect.y, width: rect.width, height: rect.height }}
                    />
                  );
                })()}
              </div>
            </div>

            <div
              ref={timelineRef}
              className="workspace-timeline"
              tabIndex={0}
              onKeyDown={handleTimelineKeyDown}
              onKeyUp={handleTimelineKeyUp}
            >
              <Space className="workspace-timeline-controls">
                <Button
                  icon={<IconPlay />}
                  size="small"
                  onClick={() => {
                    if (currentFrameRef.current >= frameMaxRef.current) {
                      currentFrameRef.current = 0;
                      setCurrentFrame(0);
                    }
                    isPlayingRef.current = true;
                    setIsPlaying(true);
                    focusTimeline();
                  }}
                  disabled={isPlaying}
                >
                  {t('common.play')}
                </Button>
                <Button
                  icon={<IconPause />}
                  size="small"
                  onClick={() => {
                    stopPlaybackIfNeeded();
                    focusTimeline();
                  }}
                  disabled={!isPlaying}
                >
                  {t('common.pause')}
                </Button>
                <Button icon={<IconArrowLeft />} size="small" onClick={() => stepPreviewFrame(-1)}>
                  -1
                </Button>
                <Button icon={<IconArrowRight />} size="small" onClick={() => stepPreviewFrame(1)}>
                  +1
                </Button>
              </Space>

              <div className="workspace-timeline-status">
                <div className="workspace-timeline-stat">
                  <Text type="tertiary">{t('annotation.previewFrameLabel')}</Text>
                  <Text>{`${currentFrame} / ${frameMax}`}</Text>
                </div>
                <div className={`workspace-timeline-stat ${activeBoundary ? 'is-active' : ''}`}>
                  <Text type="tertiary">{t('annotation.editingLabel')}</Text>
                  <Text>{activeBoundaryLabel}</Text>
                </div>
                <div className="workspace-timeline-stat">
                  <Text type="tertiary">{t('annotation.inPointLabel')}</Text>
                  <Text>{selectedSegmentStart}</Text>
                </div>
                <div className="workspace-timeline-stat">
                  <Text type="tertiary">{t('annotation.outPointLabel')}</Text>
                  <Text>{selectedSegmentEnd}</Text>
                </div>
              </div>

              <div ref={trackRef} className="workspace-track" onMouseDown={onTrackBackgroundMouseDown}>
                {segments.map((segment) => (
                  <div
                    key={`track-${segment.id}`}
                    className={`workspace-track-segment ${selectedId === segment.id ? 'selected' : ''}`}
                    style={getTrackStyle(segment)}
                    onMouseDown={() => {
                      focusTimeline();
                    }}
                    onClick={(event) => {
                      event.stopPropagation();
                      selectSegmentWithNearestBoundary(segment, { focus: true, syncFrame: false });
                    }}
                  >
                    <span
                      className={`workspace-track-handle left ${
                        activeBoundary?.segmentId === segment.id && activeBoundary.edge === 'start' ? 'active' : ''
                      }`}
                      onMouseDown={(event) => {
                        event.stopPropagation();
                        startSegmentDrag(segment.id, 'start', event);
                      }}
                      onClick={(event) => {
                        event.preventDefault();
                        event.stopPropagation();
                      }}
                    />
                    <span
                      className={`workspace-track-handle right ${
                        activeBoundary?.segmentId === segment.id && activeBoundary.edge === 'end' ? 'active' : ''
                      }`}
                      onMouseDown={(event) => {
                        event.stopPropagation();
                        startSegmentDrag(segment.id, 'end', event);
                      }}
                      onClick={(event) => {
                        event.preventDefault();
                        event.stopPropagation();
                      }}
                    />
                  </div>
                ))}
                <div className="workspace-track-cursor" style={getTrackCursorStyle()} />
              </div>

              <div className="workspace-timeline-tips">
                <span className="workspace-timeline-tip">{t('annotation.timelineTip.preview')}</span>
                <span className="workspace-timeline-tip">{t('annotation.timelineTip.switch')}</span>
                <span className="workspace-timeline-tip">{t('annotation.timelineTip.apply')}</span>
                <span className="workspace-timeline-tip">{t('annotation.timelineTip.fineTune')}</span>
                <span className="workspace-timeline-tip">{t('annotation.timelineTip.selectSegment')}</span>
              </div>
            </div>
          </div>

          <div className="workspace-side-pane">
            <Card
              className="workspace-manager-card"
              title={t('annotation.manager')}
              bodyStyle={{ padding: 10 }}
              headerExtraContent={(
                <Tooltip content={t('annotation.showAll')}>
                  <Space>
                    <Text type="tertiary">{t('annotation.showAll')}</Text>
                    <Switch size="small" checked={showAll} onChange={setShowAll} />
                  </Space>
                </Tooltip>
              )}
            >
              <Table
                className="workspace-manager-table"
                dataSource={tableSegments}
                columns={columns}
                pagination={{ pageSize: 8 }}
                rowKey="id"
                size="small"
                onRow={(record?: AnnotationSegment) => ({
                  onClick: () => {
                    if (record) {
                      selectSegmentWithNearestBoundary(record, { focus: true, syncFrame: false });
                    }
                  },
                })}
              />
            </Card>
          </div>
        </div>
      </Card>
    </div>
  );
}
