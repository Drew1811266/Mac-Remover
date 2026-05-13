// 打标工作台：
// 交互语义严格对齐旧 pywebview 前端：拖拽只生成草稿，点击“新增标记段”后才创建标记。
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import type { AnnotationRect, AnnotationSegment } from '../types/annotation';
import { resolveVisibleStageSegments, useWorkspaceStore } from '../store/workspace';
import { useI18n } from '../i18n/useI18n';
import {
  MaterialIcon,
  MdButton,
  MdEmptyState,
  MdIconButton,
  MdInspectorList,
  MdInspectorRow,
  MdSlider,
  MdSurface,
  MdSwitch,
} from '../material';

interface StageMetrics {
  width: number;
  height: number;
  videoLeft: number;
  videoTop: number;
  videoWidth: number;
  videoHeight: number;
}

interface DraftRect {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface SegmentDragState {
  id: string;
  edge: 'start' | 'end';
  startX: number;
  baseStart: number;
  baseEnd: number;
  trackWidth: number;
}

interface AnnotationWorkspaceProps {
  frameImageUrl?: string;
  previewFrameWidth?: number;
  previewFrameHeight?: number;
  onSaveAnnotations: () => Promise<void>;
  onClearAnnotations: () => Promise<void>;
}

const MIN_RECT_SIZE = 4;
const EMPTY_STAGE_WIDTH = 1280;
const EMPTY_STAGE_HEIGHT = 720;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function clampFrame(frame: number, frameMax: number): number {
  const safeMax = Math.max(0, Math.round(frameMax));
  const numeric = Number(frame);
  return clamp(Number.isFinite(numeric) ? Math.round(numeric) : 0, 0, safeMax);
}

function normalizeDraftRect(draft: DraftRect): AnnotationRect {
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

function durationText(segment: AnnotationSegment, fps: number): string {
  const safeFps = Math.max(1, Number(fps) || 24);
  const frameSpan = Math.max(1, Number(segment.end_frame) - Number(segment.start_frame) + 1);
  return `${frameSpan} 帧 / ${(frameSpan / safeFps).toFixed(2)}s`;
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
  const trackRef = useRef<HTMLDivElement | null>(null);
  const currentFrameRef = useRef(0);
  const selectedIdRef = useRef<string | null>(null);
  const segmentsRef = useRef<AnnotationSegment[]>([]);
  const isPlayingRef = useRef(false);
  const frameMaxRef = useRef(0);
  const [metrics, setMetrics] = useState<StageMetrics>(() => computeVideoBox(1, 1, 16, 9));
  const [stageContainerWidth, setStageContainerWidth] = useState(0);
  const [stageContainerHeight, setStageContainerHeight] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [draftRect, setDraftRect] = useState<DraftRect | null>(null);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
  const [segmentDrag, setSegmentDrag] = useState<SegmentDragState | null>(null);

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
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

  useEffect(() => {
    frameMaxRef.current = frameMax;
  }, [frameMax]);

  const stopPlaybackIfNeeded = useCallback(() => {
    if (isPlayingRef.current) {
      isPlayingRef.current = false;
      setIsPlaying(false);
    }
  }, []);

  const seekAnnotationFrame = useCallback((frame: number) => {
    stopPlaybackIfNeeded();
    const nextFrame = clampFrame(frame, frameMaxRef.current);
    currentFrameRef.current = nextFrame;
    setCurrentFrame(nextFrame);
  }, [setCurrentFrame, stopPlaybackIfNeeded]);

  const stepAnnotation = useCallback((delta: number) => {
    seekAnnotationFrame(currentFrameRef.current + delta);
  }, [seekAnnotationFrame]);

  const togglePlayback = useCallback(() => {
    if (isPlayingRef.current) {
      stopPlaybackIfNeeded();
      return;
    }
    if (currentFrameRef.current >= frameMaxRef.current) {
      currentFrameRef.current = 0;
      setCurrentFrame(0);
    }
    isPlayingRef.current = true;
    setIsPlaying(true);
  }, [setCurrentFrame, stopPlaybackIfNeeded]);

  const removeAnnotation = useCallback((id: string) => {
    removeSegment(id);
    if (selectedIdRef.current === id) {
      selectedIdRef.current = null;
    }
  }, [removeSegment]);

  useEffect(() => {
    if (!isPlaying) return;
    const intervalMs = Math.max(30, Math.round(1000 / Math.max(1, Math.round(fps))));
    const timer = window.setInterval(() => {
      if (currentFrameRef.current >= frameMaxRef.current) {
        stopPlaybackIfNeeded();
        return;
      }
      const nextFrame = clampFrame(currentFrameRef.current + 1, frameMaxRef.current);
      currentFrameRef.current = nextFrame;
      setCurrentFrame(nextFrame);
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [fps, isPlaying, setCurrentFrame, stopPlaybackIfNeeded]);

  const syncMetrics = useCallback(() => {
    const stage = stageRef.current;
    if (!stage) return;
    const rect = stage.getBoundingClientRect();
    const mediaWidth = Math.max(1, sourceWidth || annotationWidth || 16);
    const mediaHeight = Math.max(1, sourceHeight || annotationHeight || 9);
    setMetrics(computeVideoBox(rect.width, rect.height, mediaWidth, mediaHeight));
  }, [annotationHeight, annotationWidth, sourceHeight, sourceWidth]);

  useEffect(() => {
    syncMetrics();
    window.addEventListener('resize', syncMetrics);
    return () => window.removeEventListener('resize', syncMetrics);
  }, [syncMetrics]);

  useEffect(() => {
    const node = workspaceStageWrapRef.current;
    if (!node) return;

    const measure = () => {
      const rect = node.getBoundingClientRect();
      const style = window.getComputedStyle(node);
      const horizontalPadding = Number.parseFloat(style.paddingLeft || '0') + Number.parseFloat(style.paddingRight || '0');
      const verticalPadding = Number.parseFloat(style.paddingTop || '0') + Number.parseFloat(style.paddingBottom || '0');
      setStageContainerWidth(Math.max(0, Math.floor(rect.width - horizontalPadding)));
      setStageContainerHeight(Math.max(0, Math.floor(rect.height - verticalPadding)));
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
      resizeObserver?.disconnect();
    };
  }, []);

  useEffect(() => {
    syncMetrics();
  }, [frameImageUrl, sourceHeight, sourceWidth, stageContainerWidth, stageContainerHeight, syncMetrics]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const tagName = target?.tagName?.toLowerCase();
      if (target?.isContentEditable || tagName === 'input' || tagName === 'textarea' || tagName === 'select') return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;

      if (event.code === 'Space') {
        event.preventDefault();
        togglePlayback();
        return;
      }
      if (event.key === 'ArrowLeft') {
        event.preventDefault();
        stepAnnotation(event.shiftKey ? -10 : -1);
        return;
      }
      if (event.key === 'ArrowRight') {
        event.preventDefault();
        stepAnnotation(event.shiftKey ? 10 : 1);
        return;
      }
      if (event.key === 'Delete' && selectedIdRef.current) {
        event.preventDefault();
        removeAnnotation(selectedIdRef.current);
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [removeAnnotation, stepAnnotation, togglePlayback]);

  useEffect(() => {
    if (!segmentDrag) return;
    const onMove = (event: MouseEvent) => {
      const deltaPx = event.clientX - segmentDrag.startX;
      const deltaFrame = Math.round((deltaPx / Math.max(1, segmentDrag.trackWidth)) * Math.max(1, frameMaxRef.current));
      const segment = segmentsRef.current.find((item) => item.id === segmentDrag.id);
      if (!segment) return;

      if (segmentDrag.edge === 'start') {
        const nextStart = clampFrame(segmentDrag.baseStart + deltaFrame, frameMaxRef.current);
        updateSegment(segmentDrag.id, { start_frame: Math.min(nextStart, segment.end_frame) });
      } else {
        const nextEnd = clampFrame(segmentDrag.baseEnd + deltaFrame, frameMaxRef.current);
        updateSegment(segmentDrag.id, { end_frame: Math.max(nextEnd, segment.start_frame) });
      }
    };
    const onUp = () => setSegmentDrag(null);

    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [segmentDrag, updateSegment]);

  let stageWidth = 0;
  let stageHeight = 0;
  if (sourceWidth > 0 && sourceHeight > 0) {
    const mediaRatio = sourceWidth / sourceHeight;
    const availableWidth = stageContainerWidth > 0 ? stageContainerWidth : sourceWidth / dpr;
    const availableHeight = stageContainerHeight > 0 ? stageContainerHeight : sourceHeight / dpr;
    const widthFromHeight = availableHeight * mediaRatio;
    stageWidth = Math.min(availableWidth, widthFromHeight);
    stageHeight = stageWidth / mediaRatio;
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

  const clampStagePointToVideo = useCallback((clientX: number, clientY: number) => {
    const stage = stageRef.current;
    if (!stage) return null;
    const bounds = stage.getBoundingClientRect();
    return {
      x: clamp(clientX - bounds.left, metrics.videoLeft, metrics.videoLeft + metrics.videoWidth),
      y: clamp(clientY - bounds.top, metrics.videoTop, metrics.videoTop + metrics.videoHeight),
    };
  }, [metrics]);

  const stageToSourceRect = useCallback((draft: DraftRect): AnnotationRect => {
    const normalized = normalizeDraftRect(draft);
    const left = clamp(normalized.x, metrics.videoLeft, metrics.videoLeft + metrics.videoWidth);
    const top = clamp(normalized.y, metrics.videoTop, metrics.videoTop + metrics.videoHeight);
    const width = clamp(normalized.width, 1, metrics.videoLeft + metrics.videoWidth - left);
    const height = clamp(normalized.height, 1, metrics.videoTop + metrics.videoHeight - top);

    const scaleX = annotationWidth / metrics.videoWidth;
    const scaleY = annotationHeight / metrics.videoHeight;

    return {
      x: Math.max(0, Math.round((left - metrics.videoLeft) * scaleX)),
      y: Math.max(0, Math.round((top - metrics.videoTop) * scaleY)),
      width: Math.max(1, Math.round(width * scaleX)),
      height: Math.max(1, Math.round(height * scaleY)),
    };
  }, [annotationHeight, annotationWidth, metrics]);

  const sourceToStageRect = useCallback((rect: AnnotationRect): AnnotationRect => {
    const scaleX = metrics.videoWidth / annotationWidth;
    const scaleY = metrics.videoHeight / annotationHeight;
    const x = metrics.videoLeft + rect.x * scaleX;
    const y = metrics.videoTop + rect.y * scaleY;
    const maxX = metrics.videoLeft + metrics.videoWidth;
    const maxY = metrics.videoTop + metrics.videoHeight;
    return {
      x: clamp(x, metrics.videoLeft, maxX),
      y: clamp(y, metrics.videoTop, maxY),
      width: Math.max(1, Math.min(maxX - x, rect.width * scaleX)),
      height: Math.max(1, Math.min(maxY - y, rect.height * scaleY)),
    };
  }, [annotationHeight, annotationWidth, metrics]);

  const onStageMouseDown = (event: React.MouseEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    syncMetrics();
    const point = clampStagePointToVideo(event.clientX, event.clientY);
    if (!point) return;
    stopPlaybackIfNeeded();
    setDragStart(point);
    setDraftRect({ x1: point.x, y1: point.y, x2: point.x, y2: point.y });
  };

  const onStageMouseMove = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!dragStart) return;
    const point = clampStagePointToVideo(event.clientX, event.clientY);
    if (!point) return;
    setDraftRect({ x1: dragStart.x, y1: dragStart.y, x2: point.x, y2: point.y });
  };

  const finishDraft = (event?: React.MouseEvent<HTMLDivElement>) => {
    if (!dragStart) return;
    if (event) {
      const point = clampStagePointToVideo(event.clientX, event.clientY);
      if (point) {
        const nextDraft = { x1: dragStart.x, y1: dragStart.y, x2: point.x, y2: point.y };
        const rect = normalizeDraftRect(nextDraft);
        setDraftRect(rect.width < MIN_RECT_SIZE || rect.height < MIN_RECT_SIZE ? null : nextDraft);
      }
    } else if (draftRect) {
      const rect = normalizeDraftRect(draftRect);
      if (rect.width < MIN_RECT_SIZE || rect.height < MIN_RECT_SIZE) {
        setDraftRect(null);
      }
    }
    setDragStart(null);
  };

  const addAnnotationFromDraft = useCallback(() => {
    if (!draftRect) return;
    const rect = normalizeDraftRect(draftRect);
    if (rect.width < MIN_RECT_SIZE || rect.height < MIN_RECT_SIZE) {
      setDraftRect(null);
      return;
    }

    createSegmentFromRect(stageToSourceRect(draftRect), currentFrameRef.current, {
      fps,
      frameMax,
    });
    setDraftRect(null);
  }, [createSegmentFromRect, draftRect, fps, frameMax, stageToSourceRect]);

  const setAnnotationStart = useCallback((id: string, frame: number) => {
    const segment = segmentsRef.current.find((item) => item.id === id);
    if (!segment) return;
    const nextStart = clampFrame(frame, frameMaxRef.current);
    updateSegment(id, { start_frame: Math.min(nextStart, segment.end_frame) });
  }, [updateSegment]);

  const setAnnotationEnd = useCallback((id: string, frame: number) => {
    const segment = segmentsRef.current.find((item) => item.id === id);
    if (!segment) return;
    const nextEnd = clampFrame(frame, frameMaxRef.current);
    updateSegment(id, { end_frame: Math.max(nextEnd, segment.start_frame) });
  }, [updateSegment]);

  const jumpToAnnotation = useCallback((segment: AnnotationSegment) => {
    selectSegment(segment.id);
    selectedIdRef.current = segment.id;
    seekAnnotationFrame(segment.start_frame);
  }, [seekAnnotationFrame, selectSegment]);

  const getTrackStyle = useCallback((segment: AnnotationSegment) => {
    const safeMax = Math.max(1, frameMax);
    const left = (Math.max(0, segment.start_frame) / safeMax) * 100;
    const right = (Math.max(0, segment.end_frame) / safeMax) * 100;
    return {
      left: `${left}%`,
      width: `${Math.max(1, right - left)}%`,
      minWidth: '14px',
    };
  }, [frameMax]);

  const getTrackCursorStyle = useCallback(() => {
    const safeMax = Math.max(1, frameMax);
    const left = (Math.max(0, currentFrame) / safeMax) * 100;
    return { left: `${left}%` };
  }, [currentFrame, frameMax]);

  const startSegmentDrag = (segment: AnnotationSegment, edge: 'start' | 'end', event: React.MouseEvent<HTMLElement>) => {
    const track = trackRef.current;
    if (!track) return;
    event.preventDefault();
    event.stopPropagation();
    stopPlaybackIfNeeded();
    selectSegment(segment.id);
    selectedIdRef.current = segment.id;
    setSegmentDrag({
      id: segment.id,
      edge,
      startX: event.clientX,
      baseStart: segment.start_frame,
      baseEnd: segment.end_frame,
      trackWidth: Math.max(1, track.getBoundingClientRect().width),
    });
  };

  const stageSegments = useMemo(
    () => resolveVisibleStageSegments(segments, currentFrame),
    [currentFrame, segments],
  );
  const draftReady = !!draftRect && normalizeDraftRect(draftRect).width >= MIN_RECT_SIZE && normalizeDraftRect(draftRect).height >= MIN_RECT_SIZE;
  const selectedSegmentStart = selectedSegment ? selectedSegment.start_frame : '--';
  const selectedSegmentEnd = selectedSegment ? selectedSegment.end_frame : '--';

  return (
    <div className="workspace-shell">
      <MdSurface className="workspace-surface">
        <div className="surface-header workspace-header">
          <div>
            <h2>{t('annotate.title')}</h2>
            <p>{`${segments.length} ${t('annotation.manager')} · ${currentFrame}/${frameMax}`}</p>
          </div>
          <div className="button-row wrap">
            <MdButton variant="filled" icon="save" onClick={() => void onSaveAnnotations()}>
              {t('annotation.save')}
            </MdButton>
            <MdButton variant="outlined" tone="danger" icon="delete" onClick={() => void onClearAnnotations()}>
              {t('annotation.clear')}
            </MdButton>
          </div>
        </div>
        <div className="workspace-main-grid">
          <div className="workspace-player-pane">
            <div className="workspace-stage-wrap" ref={workspaceStageWrapRef}>
              <div
                ref={stageRef}
                className="workspace-stage"
                style={stageStyle}
                onMouseDown={onStageMouseDown}
                onMouseMove={onStageMouseMove}
                onMouseUp={finishDraft}
                onMouseLeave={() => finishDraft()}
              >
                {frameImageUrl ? (
                  <img
                    src={frameImageUrl}
                    alt="frame"
                    className="workspace-frame"
                    draggable={false}
                    onDragStart={(event) => event.preventDefault()}
                    onLoad={syncMetrics}
                  />
                ) : (
                  <MdEmptyState
                    className="workspace-placeholder"
                    icon="frame_inspect"
                    title={t('annotation.framePlaceholder')}
                    description={t('process.noVideoHint')}
                  />
                )}

                {stageSegments.map((segment) => {
                  const rect = sourceToStageRect(segment.rect);
                  const selected = selectedId === segment.id;
                  return (
                    <div
                      key={segment.id}
                      className={`workspace-segment ${segment.enabled === false ? 'disabled' : 'active'} ${selected ? 'selected' : ''}`}
                      style={{
                        left: rect.x,
                        top: rect.y,
                        width: rect.width,
                        height: rect.height,
                      }}
                      onMouseDown={(event) => event.stopPropagation()}
                      onClick={(event) => {
                        event.stopPropagation();
                        selectSegment(segment.id);
                      }}
                    />
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

            <div className="workspace-timeline">
              <div className="workspace-timeline-controls">
                <MdButton icon={isPlaying ? 'pause' : 'play_arrow'} onClick={togglePlayback}>
                  {isPlaying ? t('common.pause') : t('common.play')}
                </MdButton>
                <MdButton icon="chevron_left" onClick={() => stepAnnotation(-1)}>
                  -1
                </MdButton>
                <MdButton icon="chevron_right" onClick={() => stepAnnotation(1)}>
                  +1
                </MdButton>
                <MdButton
                  variant="filled"
                  icon="add_box"
                  disabled={!draftReady}
                  onClick={addAnnotationFromDraft}
                >
                  {t('annotation.addSegment')}
                </MdButton>
              </div>

              <div className="workspace-timeline-status">
                <div className="workspace-timeline-stat">
                  <span>{t('annotation.previewFrameLabel')}</span>
                  <strong>{`${currentFrame} / ${frameMax}`}</strong>
                </div>
                <div className="workspace-timeline-stat">
                  <span>{t('annotation.editingLabel')}</span>
                  <strong>{selectedSegment ? `#${segments.findIndex((item) => item.id === selectedSegment.id) + 1}` : '--'}</strong>
                </div>
                <div className="workspace-timeline-stat">
                  <span>{t('annotation.inPointLabel')}</span>
                  <strong>{selectedSegmentStart}</strong>
                </div>
                <div className="workspace-timeline-stat">
                  <span>{t('annotation.outPointLabel')}</span>
                  <strong>{selectedSegmentEnd}</strong>
                </div>
              </div>

              <MdSlider
                min={0}
                max={frameMax}
                value={clampFrame(currentFrame, frameMax)}
                ariaLabel={t('annotation.previewFrameLabel')}
                onChange={seekAnnotationFrame}
              />

              <div ref={trackRef} className="workspace-track">
                {segments.map((segment) => (
                  <div
                    key={`track-${segment.id}`}
                    className={`workspace-track-segment ${selectedId === segment.id ? 'selected' : ''}`}
                    style={getTrackStyle(segment)}
                    onClick={(event) => {
                      event.stopPropagation();
                      selectSegment(segment.id);
                    }}
                  >
                    <span
                      className="workspace-track-handle left"
                      onMouseDown={(event) => startSegmentDrag(segment, 'start', event)}
                      onClick={(event) => {
                        event.preventDefault();
                        event.stopPropagation();
                      }}
                    />
                    <span
                      className="workspace-track-handle right"
                      onMouseDown={(event) => startSegmentDrag(segment, 'end', event)}
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
                <span className="workspace-timeline-tip">{t('annotation.timelineTip.draw')}</span>
                <span className="workspace-timeline-tip">{t('annotation.timelineTip.add')}</span>
                <span className="workspace-timeline-tip">{t('annotation.timelineTip.legacyKeys')}</span>
              </div>
            </div>
          </div>

          <div className="workspace-side-pane">
            <section className="workspace-manager-card">
              <div className="workspace-manager-head">
                <div>
                  <h3>{t('annotation.manager')}</h3>
                  <p>{`${segments.length}/${segments.length}`}</p>
                </div>
                <label className="workspace-show-all">
                  <span>{t('annotation.showAll')}</span>
                  <MdSwitch checked={showAll} label={t('annotation.showAll')} onChange={setShowAll} />
                </label>
              </div>
              <MdInspectorList className="workspace-manager-table legacy">
                {segments.length === 0 ? (
                  <div className="workspace-manager-empty">
                    <MaterialIcon name="ink_highlighter" />
                    <span>{t('annotation.emptyLegacy')}</span>
                  </div>
                ) : null}
                {segments.map((segment, index) => (
                  <MdInspectorRow
                    key={segment.id}
                    label={`#${index + 1}`}
                    value={(
                      <span className="workspace-segment-summary">
                        <strong>{`${segment.start_frame} - ${segment.end_frame}`}</strong>
                        <span>{durationText(segment, fps)}</span>
                        <span>{`x:${segment.rect.x} y:${segment.rect.y} w:${segment.rect.width} h:${segment.rect.height}`}</span>
                      </span>
                    )}
                    selected={selectedId === segment.id}
                    onClick={() => selectSegment(segment.id)}
                    action={(
                      <span className="workspace-table-actions legacy">
                        <label className="workspace-segment-enabled">
                          <MdSwitch
                            checked={segment.enabled !== false}
                            label={t('annotation.enabled')}
                            onChange={(enabled) => updateSegment(segment.id, { enabled })}
                          />
                          <span>{segment.enabled !== false ? t('annotation.enabled') : t('annotation.disabled')}</span>
                        </label>
                        <MdButton
                          variant="text"
                          icon="my_location"
                          onClick={(event) => {
                            event.preventDefault();
                            event.stopPropagation();
                            jumpToAnnotation(segment);
                          }}
                        >
                          {t('annotation.jump')}
                        </MdButton>
                        <MdButton
                          variant="text"
                          onClick={(event) => {
                            event.preventDefault();
                            event.stopPropagation();
                            setAnnotationStart(segment.id, currentFrameRef.current);
                          }}
                        >
                          {t('annotation.setStart')}
                        </MdButton>
                        <MdButton
                          variant="text"
                          onClick={(event) => {
                            event.preventDefault();
                            event.stopPropagation();
                            setAnnotationEnd(segment.id, currentFrameRef.current);
                          }}
                        >
                          {t('annotation.setEnd')}
                        </MdButton>
                        <MdIconButton
                          icon="delete"
                          label={t('annotation.deleteSelected')}
                          onClick={(event) => {
                            event.preventDefault();
                            event.stopPropagation();
                            removeAnnotation(segment.id);
                          }}
                        />
                      </span>
                    )}
                  />
                ))}
              </MdInspectorList>
            </section>
          </div>
        </div>
      </MdSurface>
    </div>
  );
}
