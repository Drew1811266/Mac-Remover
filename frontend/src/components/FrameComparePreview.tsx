import { useCallback, useEffect, useRef, useState } from 'react';
import { desktopClient } from '../services/desktop';
import { useI18n } from '../i18n/useI18n';
import { MdButton, MdSlider } from '../material';

export interface FrameComparePreviewProps {
  outputPath: string;
  outputUrlFallback: string;
  sourcePath?: string;
  isVideo: boolean;
  fpsHint: number;
  frameCountHint: number;
  widthHint: number;
  heightHint: number;
  beforeLabel: string;
  afterLabel: string;
  compareUnavailableText: string;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

export function FrameComparePreview({
  outputPath,
  outputUrlFallback,
  sourcePath,
  isVideo,
  fpsHint,
  frameCountHint,
  widthHint,
  heightHint,
  beforeLabel,
  afterLabel,
  compareUnavailableText,
}: FrameComparePreviewProps) {
  const { t } = useI18n();
  const [sessionId, setSessionId] = useState('');
  const [previewFrameUrl, setPreviewFrameUrl] = useState('');
  const [previewFrameIndex, setPreviewFrameIndex] = useState(0);
  const [previewTotalFrames, setPreviewTotalFrames] = useState(0);
  const [previewFps, setPreviewFps] = useState(15);
  const [sourceSessionId, setSourceSessionId] = useState('');
  const [sourcePreviewFrameUrl, setSourcePreviewFrameUrl] = useState('');
  const [sourcePreviewTotalFrames, setSourcePreviewTotalFrames] = useState(0);
  const [previewWidth, setPreviewWidth] = useState(0);
  const [previewHeight, setPreviewHeight] = useState(0);
  const [previewContainerWidth, setPreviewContainerWidth] = useState(0);
  const [previewAvailableHeight, setPreviewAvailableHeight] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [previewError, setPreviewError] = useState('');
  const [compareWarning, setCompareWarning] = useState('');
  const [compareRatio, setCompareRatio] = useState(0.5);
  const [isCompareDragging, setIsCompareDragging] = useState(false);

  const frameFetchInFlightRef = useRef(false);
  const pendingFrameRef = useRef<number | null>(null);
  const lastRenderedOutputFrameRef = useRef<number | null>(null);
  const lastRenderedSourceFrameRef = useRef<number | null>(null);
  const sessionIdRef = useRef('');
  const sourceSessionIdRef = useRef('');
  const previewWrapRef = useRef<HTMLDivElement | null>(null);
  const compareStageRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);

  useEffect(() => {
    sourceSessionIdRef.current = sourceSessionId;
  }, [sourceSessionId]);

  const closeSession = useCallback(async (sid?: string) => {
    const target = sid || sessionIdRef.current;
    if (!target) return;
    try {
      await desktopClient.closeVideoPreviewSession(target);
    } catch {
      // Best effort cleanup.
    }
    if (target === sessionIdRef.current) {
      sessionIdRef.current = '';
      pendingFrameRef.current = null;
      lastRenderedOutputFrameRef.current = null;
      setSessionId('');
    }
  }, []);

  const closeSourceSession = useCallback(async (sid?: string) => {
    const target = sid || sourceSessionIdRef.current;
    if (!target) return;
    try {
      await desktopClient.closeVideoPreviewSession(target);
    } catch {
      // Best effort cleanup.
    }
    if (target === sourceSessionIdRef.current) {
      sourceSessionIdRef.current = '';
      lastRenderedSourceFrameRef.current = null;
      setSourceSessionId('');
    }
  }, []);

  const mapSourceFrameIndex = useCallback((outputFrameIndex: number) => {
    const outputMax = Math.max(0, previewTotalFrames - 1);
    const sourceMax = Math.max(0, sourcePreviewTotalFrames - 1);
    if (outputMax <= 0 || sourceMax <= 0) return 0;
    const safeOutputIndex = Math.max(0, Math.min(outputMax, outputFrameIndex));
    const mapped = Math.round((safeOutputIndex / outputMax) * sourceMax);
    return Math.max(0, Math.min(sourceMax, mapped));
  }, [previewTotalFrames, sourcePreviewTotalFrames]);

  useEffect(() => {
    let cancelled = false;

    const open = async () => {
      const safeFpsHint = Math.max(1, Math.round(Number(fpsHint) || 15));

      setIsPlaying(false);
      setIsCompareDragging(false);
      setCompareRatio(0.5);
      setPreviewError('');
      setCompareWarning('');
      setPreviewFrameUrl('');
      setSourcePreviewFrameUrl('');
      setPreviewFrameIndex(0);
      setPreviewTotalFrames(0);
      setSourcePreviewTotalFrames(0);
      setPreviewFps(safeFpsHint);
      setPreviewWidth(0);
      setPreviewHeight(0);

      if (!outputPath || !isVideo) {
        await closeSession();
        await closeSourceSession();
        return;
      }

      const decodeMaxWidth = Math.max(
        960,
        Math.min(
          1920,
          Math.round((window.innerWidth || 1280) * Math.max(1, window.devicePixelRatio || 1)),
        ),
      );

      await closeSession();
      await closeSourceSession();
      const opened = await desktopClient.openVideoPreviewSession(
        outputPath,
        Math.max(15, safeFpsHint),
        decodeMaxWidth,
      );

      if (cancelled) return;
      if (!opened.success || !opened.session_id) {
        setPreviewError(opened.error || 'Failed to open video preview session');
        return;
      }

      setSessionId(opened.session_id);
      setPreviewTotalFrames(Math.max(1, Number(opened.total_preview_frames || frameCountHint || 1)));
      setPreviewFps(Math.max(1, Number(opened.preview_fps || safeFpsHint)));
      setPreviewWidth(Math.max(0, Number(opened.width || 0)));
      setPreviewHeight(Math.max(0, Number(opened.height || 0)));

      if (!sourcePath) return;

      const sourceOpened = await desktopClient.openVideoPreviewSession(
        sourcePath,
        Math.max(15, safeFpsHint),
        decodeMaxWidth,
      );

      if (cancelled) return;
      if (!sourceOpened.success || !sourceOpened.session_id) {
        setCompareWarning(compareUnavailableText);
        return;
      }

      setSourceSessionId(sourceOpened.session_id);
      setSourcePreviewTotalFrames(Math.max(1, Number(sourceOpened.total_preview_frames || 1)));
    };

    void open();
    return () => {
      cancelled = true;
    };
  }, [
    closeSession,
    closeSourceSession,
    compareUnavailableText,
    fpsHint,
    frameCountHint,
    isVideo,
    outputPath,
    sourcePath,
  ]);

  const pumpPreviewFrames = useCallback(async () => {
    if (frameFetchInFlightRef.current) return;
    const outputSessionId = sessionIdRef.current;
    if (!outputSessionId) return;

    frameFetchInFlightRef.current = true;
    try {
      while (true) {
        const outputTargetFrame = pendingFrameRef.current;
        pendingFrameRef.current = null;
        if (outputTargetFrame === null) break;

        const activeOutputSessionId = sessionIdRef.current;
        if (!activeOutputSessionId) break;
        const previousOutputFrame = lastRenderedOutputFrameRef.current;
        const useSequentialOutputRead = previousOutputFrame !== null && outputTargetFrame === previousOutputFrame + 1;

        const outputFramePromise = useSequentialOutputRead
          ? desktopClient.readVideoPreviewFrame(activeOutputSessionId)
          : desktopClient.readVideoPreviewFrame(activeOutputSessionId, outputTargetFrame);

        const activeSourceSessionId = sourceSessionIdRef.current;
        const shouldLoadSource = !!activeSourceSessionId && sourcePreviewTotalFrames > 0;
        let sourceTargetFrame = 0;
        let sourceFramePromise: Promise<Awaited<ReturnType<typeof desktopClient.readVideoPreviewFrame>> | null> = Promise.resolve(null);

        if (shouldLoadSource) {
          sourceTargetFrame = mapSourceFrameIndex(outputTargetFrame);
          const previousSourceFrame = lastRenderedSourceFrameRef.current;
          const useSequentialSourceRead = previousSourceFrame !== null && sourceTargetFrame === previousSourceFrame + 1;
          sourceFramePromise = useSequentialSourceRead
            ? desktopClient.readVideoPreviewFrame(activeSourceSessionId)
            : desktopClient.readVideoPreviewFrame(activeSourceSessionId, sourceTargetFrame);
        }

        const [outputFrame, sourceFrame] = await Promise.all([outputFramePromise, sourceFramePromise]);
        if (activeOutputSessionId !== sessionIdRef.current) break;
        if (!outputFrame.success || !outputFrame.frame_url) {
          setPreviewError(outputFrame.error || 'Failed to read video preview frame');
          setIsPlaying(false);
          break;
        }

        setPreviewError('');
        setPreviewFrameUrl(outputFrame.frame_url);
        lastRenderedOutputFrameRef.current = (
          typeof outputFrame.frame_index === 'number'
            ? Number(outputFrame.frame_index)
            : outputTargetFrame
        );

        if (!shouldLoadSource) continue;
        if (activeSourceSessionId !== sourceSessionIdRef.current) continue;

        if (sourceFrame && sourceFrame.success && sourceFrame.frame_url) {
          setSourcePreviewFrameUrl(sourceFrame.frame_url);
          setCompareWarning('');
          lastRenderedSourceFrameRef.current = (
            typeof sourceFrame.frame_index === 'number'
              ? Number(sourceFrame.frame_index)
              : sourceTargetFrame
          );
        } else {
          setSourcePreviewFrameUrl('');
          setCompareWarning(compareUnavailableText);
        }
      }
    } finally {
      frameFetchInFlightRef.current = false;
      if (pendingFrameRef.current !== null) {
        void pumpPreviewFrames();
      }
    }
  }, [compareUnavailableText, mapSourceFrameIndex, sourcePreviewTotalFrames]);

  useEffect(() => {
    if (!sessionId) return;
    pendingFrameRef.current = Math.max(0, Number(previewFrameIndex) || 0);
    void pumpPreviewFrames();
  }, [previewFrameIndex, pumpPreviewFrames, sessionId]);

  useEffect(() => {
    if (!sessionId || !isPlaying) return;
    const frameTotal = Math.max(1, previewTotalFrames);
    const timer = window.setInterval(() => {
      setPreviewFrameIndex((value) => {
        const next = value + 1;
        if (next >= frameTotal) {
          setIsPlaying(false);
          return frameTotal - 1;
        }
        return next;
      });
    }, Math.max(40, Math.round(1000 / Math.max(1, previewFps))));
    return () => window.clearInterval(timer);
  }, [isPlaying, previewFps, previewTotalFrames, sessionId]);

  useEffect(() => {
    return () => {
      void closeSession();
      void closeSourceSession();
    };
  }, [closeSession, closeSourceSession]);

  useEffect(() => {
    const node = previewWrapRef.current;
    if (!node) return;

    const bottomSafePadding = 24;
    const minPreviewHeight = 140;
    const measure = () => {
      const rect = node.getBoundingClientRect();
      setPreviewContainerWidth(Math.max(0, Math.floor(rect.width)));
      const available = Math.floor(window.innerHeight - rect.top - bottomSafePadding);
      setPreviewAvailableHeight(Math.max(minPreviewHeight, available));
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
  }, [isVideo, outputPath]);

  const updateCompareRatio = useCallback((clientX: number) => {
    const stage = compareStageRef.current;
    if (!stage) return;
    const rect = stage.getBoundingClientRect();
    if (rect.width <= 0) return;
    const ratio = (clientX - rect.left) / rect.width;
    setCompareRatio(clamp01(ratio));
  }, []);

  useEffect(() => {
    if (!isCompareDragging) return;
    const onMove = (event: MouseEvent) => {
      updateCompareRatio(event.clientX);
    };
    const onUp = () => setIsCompareDragging(false);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    window.addEventListener('blur', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      window.removeEventListener('blur', onUp);
    };
  }, [isCompareDragging, updateCompareRatio]);

  const maxFrame = Math.max(0, previewTotalFrames - 1);
  const compareReady = isVideo && !!previewFrameUrl && !!sourcePreviewFrameUrl && !!sourceSessionId;
  const compareLeftInset = `${(clamp01(compareRatio) * 100).toFixed(3)}%`;
  const compareHandleLeft = `${(clamp01(compareRatio) * 100).toFixed(3)}%`;
  const videoWidth = widthHint > 0 ? widthHint : (previewWidth > 0 ? previewWidth : 0);
  const videoHeight = heightHint > 0 ? heightHint : (previewHeight > 0 ? previewHeight : 0);
  const dpr = Math.max(1, window.devicePixelRatio || 1);

  let stageWidth = 0;
  let stageHeight = 0;
  if (videoWidth > 0 && videoHeight > 0) {
    const cssNativeWidth = videoWidth / dpr;
    const cssNativeHeight = videoHeight / dpr;

    const widthCap = previewContainerWidth > 0 ? previewContainerWidth : cssNativeWidth;
    const heightCap = previewAvailableHeight > 0 ? previewAvailableHeight : cssNativeHeight;

    stageWidth = Math.min(cssNativeWidth, widthCap);
    stageHeight = stageWidth * (videoHeight / videoWidth);

    if (stageHeight > heightCap) {
      stageHeight = heightCap;
      stageWidth = stageHeight * (videoWidth / videoHeight);
    }
  }

  const hasComputedStageSize = Number.isFinite(stageWidth)
    && Number.isFinite(stageHeight)
    && stageWidth > 0
    && stageHeight > 0;

  const previewStageStyle = hasComputedStageSize
    ? {
        width: `${Math.round(stageWidth)}px`,
        height: `${Math.round(stageHeight)}px`,
        maxWidth: '100%',
      }
    : {
        width: '100%',
        aspectRatio: videoWidth > 0 && videoHeight > 0 ? `${videoWidth} / ${videoHeight}` : '16 / 9',
      };

  return (
    <>
      <div className="result-preview-wrap" ref={previewWrapRef}>
        <div className="result-preview-stage" style={previewStageStyle}>
          {isVideo ? (
            compareReady ? (
              <div className="result-compare-root" ref={compareStageRef}>
                <img src={sourcePreviewFrameUrl} alt="result-compare-before" className="result-image result-compare-image" />
                <div className="result-compare-after-layer" style={{ clipPath: `inset(0 0 0 ${compareLeftInset})` }}>
                  <img src={previewFrameUrl} alt="result-preview-frame" className="result-image result-compare-image" />
                </div>
                <div className="result-compare-divider" style={{ left: compareHandleLeft }}>
                  <button
                    type="button"
                    className="result-compare-handle"
                    aria-label={t('result.compare.handle')}
                    onMouseDown={(event) => {
                      event.preventDefault();
                      event.stopPropagation();
                      updateCompareRatio(event.clientX);
                      setIsCompareDragging(true);
                    }}
                  />
                </div>
                <div className="result-compare-label result-compare-label-before">{beforeLabel}</div>
                <div className="result-compare-label result-compare-label-after">{afterLabel}</div>
              </div>
            ) : previewFrameUrl ? (
              <img src={previewFrameUrl} alt="result-preview-frame" className="result-image" />
            ) : (
              <video src={outputUrlFallback} controls className="result-video" />
            )
          ) : (
            <img src={outputUrlFallback} alt="result" className="result-image" />
          )}
        </div>
      </div>

      {isVideo && sessionId && (
        <div className="result-playback-controls">
          <MdButton
            icon={isPlaying ? 'pause' : 'play_arrow'}
            disabled={previewTotalFrames <= 1}
            onClick={() => {
              if (isPlaying) {
                setIsPlaying(false);
                return;
              }
              if (previewFrameIndex >= maxFrame) {
                setPreviewFrameIndex(0);
              }
              setIsPlaying(true);
            }}
          >
            {isPlaying ? t('common.pause') : t('common.play')}
          </MdButton>
          <div className="result-slider-wrap">
            <MdSlider
              min={0}
              max={maxFrame}
              value={Math.min(maxFrame, previewFrameIndex)}
              onChange={(value) => {
                setIsPlaying(false);
                setPreviewFrameIndex(Number(value));
              }}
              disabled={previewTotalFrames <= 1}
            />
          </div>
          <span className="metadata-text">{`${Math.min(maxFrame, previewFrameIndex) + 1}/${Math.max(1, previewTotalFrames)}`}</span>
        </div>
      )}
      {isVideo && previewError && <p className="form-error">{previewError}</p>}
      {isVideo && compareWarning && <p className="metadata-text">{compareWarning}</p>}
    </>
  );
}
