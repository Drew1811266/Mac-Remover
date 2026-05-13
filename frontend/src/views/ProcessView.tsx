// 处理页视图：
// 左侧预览与播放控制，右侧处理参数与进度展示。
import { useCallback, useEffect, useRef, useState } from 'react';
import type { AnnotationSegment, VideoMeta } from '../types/annotation';
import type { AppSettings, DeviceInfoState, ProcessState } from '../types/app';
import { useI18n } from '../i18n/useI18n';
import {
  MdButton,
  MdChip,
  MdEmptyState,
  MdLinearProgress,
  MdSelect,
  MdSlider,
  MdStatusMetric,
  MdSurface,
  MdTaskPanel,
} from '../material';

interface ProcessViewProps {
  // 当前视频与预览帧信息。
  videoPath: string;
  videoUrl: string;
  videoMeta: VideoMeta | null;
  frameImageUrl: string;
  previewFrameWidth?: number;
  previewFrameHeight?: number;
  currentFrame: number;
  isSelectingVideo: boolean;
  isPlaying: boolean;
  settings: AppSettings;
  process: ProcessState;
  deviceInfo: DeviceInfoState;
  segments: AnnotationSegment[];
  // 来自应用层的状态与操作回调。
  onSelectVideo: () => void;
  onSetCurrentFrame: (frame: number) => void;
  onTogglePlay: () => void;
  onSelectOutputFolder: () => void;
  onChangeModelId: (modelId: AppSettings['modelId']) => void;
  onStartProcessing: () => void;
  onStopProcessing: () => void;
  onGoAnnotate: () => void;
}

export function ProcessView({
  videoPath,
  videoUrl,
  videoMeta,
  frameImageUrl,
  previewFrameWidth,
  previewFrameHeight,
  currentFrame,
  isSelectingVideo,
  isPlaying,
  settings,
  process,
  deviceInfo,
  segments,
  onSelectVideo,
  onSetCurrentFrame,
  onTogglePlay,
  onSelectOutputFolder,
  onChangeModelId,
  onStartProcessing,
  onStopProcessing,
  onGoAnnotate,
}: ProcessViewProps) {
  const { t } = useI18n();
  const previewWrapRef = useRef<HTMLDivElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const currentFrameRef = useRef(0);
  const lastSyncedVideoFrameRef = useRef(-1);
  const [previewContainerWidth, setPreviewContainerWidth] = useState(0);
  const [previewContainerHeight, setPreviewContainerHeight] = useState(0);
  // 预览区域和处理状态的派生数据，统一在视图层计算。
  const frameMax = Math.max(0, (videoMeta?.frame_count ?? 1) - 1);
  const enabledCount = segments.filter((item) => item.enabled !== false).length;
  const hasEnabledSegments = enabledCount > 0;
  const sliderValue = Math.min(frameMax, Math.max(0, Number(currentFrame) || 0));
  const sliderKey = `${videoPath || 'no-video'}:${frameMax}`;
  const etaText = process.estimatedTime || '--:--';
  const speedText = process.throughputFps && process.throughputFps > 0
    ? `${process.throughputFps.toFixed(2)} ${t('process.speedUnit')}`
    : `-- ${t('process.speedUnit')}`;
  const phaseKey = process.phase ? `process.phase.${process.phase}` : 'status.idle';
  const phaseText = t(phaseKey);
  const playbackFps = Math.max(1, Number(videoMeta?.fps || 24));

  useEffect(() => {
    currentFrameRef.current = currentFrame;
  }, [currentFrame]);

  useEffect(() => {
    // 监听容器尺寸变化，计算预览可用宽度以维持正确比例。
    const node = previewWrapRef.current;
    if (!node) return;

    const measure = () => {
      const rect = node.getBoundingClientRect();
      setPreviewContainerWidth(Math.max(0, Math.floor(rect.width)));
      setPreviewContainerHeight(Math.max(0, Math.floor(rect.height)));
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

  const seekVideoToFrame = useCallback((frame: number) => {
    const video = videoRef.current;
    if (!video || !Number.isFinite(playbackFps) || playbackFps <= 0) return;
    const nextTime = Math.max(0, Math.min(frameMax / playbackFps, frame / playbackFps));
    if (Number.isFinite(nextTime) && Math.abs(video.currentTime - nextTime) > 0.015) {
      video.currentTime = nextTime;
    }
  }, [frameMax, playbackFps]);

  const syncFrameFromVideo = useCallback((time: number) => {
    if (!videoPath || !Number.isFinite(time)) return;
    const nextFrame = Math.max(0, Math.min(frameMax, Math.round(time * playbackFps)));
    if (nextFrame === lastSyncedVideoFrameRef.current) return;
    lastSyncedVideoFrameRef.current = nextFrame;
    onSetCurrentFrame(nextFrame);
  }, [frameMax, onSetCurrentFrame, playbackFps, videoPath]);

  useEffect(() => {
    const video = videoRef.current;
    if (!videoUrl || !video) return;

    if (isPlaying) {
      if (video.paused || video.ended) {
        seekVideoToFrame(currentFrameRef.current);
      }
      void video.play().catch(() => {
        // The user can press Play again if the platform blocks playback.
      });
      return;
    }

    video.pause();
    seekVideoToFrame(currentFrameRef.current);
  }, [isPlaying, seekVideoToFrame, videoUrl]);

  useEffect(() => {
    if (isPlaying || !videoUrl) return;
    seekVideoToFrame(currentFrame);
  }, [currentFrame, isPlaying, seekVideoToFrame, videoUrl]);

  useEffect(() => {
    const video = videoRef.current;
    if (!videoUrl || !video || !isPlaying) return;

    type VideoFrameCallbackMetadata = { mediaTime?: number };
    type VideoWithFrameCallback = HTMLVideoElement & {
      requestVideoFrameCallback?: (
        callback: (now: number, metadata: VideoFrameCallbackMetadata) => void,
      ) => number;
      cancelVideoFrameCallback?: (handle: number) => void;
    };

    const callbackVideo = video as VideoWithFrameCallback;
    if (!callbackVideo.requestVideoFrameCallback) return;

    let cancelled = false;
    let handle = 0;
    const onVideoFrame = (_now: number, metadata: VideoFrameCallbackMetadata) => {
      if (cancelled) return;
      syncFrameFromVideo(typeof metadata.mediaTime === 'number' ? metadata.mediaTime : video.currentTime);
      handle = callbackVideo.requestVideoFrameCallback?.(onVideoFrame) ?? 0;
    };

    handle = callbackVideo.requestVideoFrameCallback(onVideoFrame);
    return () => {
      cancelled = true;
      if (handle && callbackVideo.cancelVideoFrameCallback) {
        callbackVideo.cancelVideoFrameCallback(handle);
      }
    };
  }, [isPlaying, syncFrameFromVideo, videoUrl]);

  const sourceWidth = Number(previewFrameWidth && previewFrameWidth > 0 ? previewFrameWidth : (videoMeta?.width ?? 0));
  const sourceHeight = Number(previewFrameHeight && previewFrameHeight > 0 ? previewFrameHeight : (videoMeta?.height ?? 0));
  const dpr = Math.max(1, window.devicePixelRatio || 1);

  let stageWidth = 0;
  let stageHeight = 0;
  if (sourceWidth > 0 && sourceHeight > 0) {
    const cssNativeWidth = sourceWidth / dpr;
    const cssNativeHeight = sourceHeight / dpr;
    const widthCap = previewContainerWidth > 0 ? previewContainerWidth : cssNativeWidth;
    const heightCap = previewContainerHeight > 0
      ? Math.max(160, Math.min(cssNativeHeight, previewContainerHeight))
      : cssNativeHeight;
    const heightBasedWidthCap = heightCap * (sourceWidth / sourceHeight);
    stageWidth = Math.min(cssNativeWidth, widthCap, heightBasedWidthCap);
    stageHeight = stageWidth * (sourceHeight / sourceWidth);
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
        aspectRatio: sourceWidth > 0 && sourceHeight > 0 ? `${sourceWidth} / ${sourceHeight}` : '16 / 9',
      };

  return (
    // 布局：左侧预览，右侧配置与进度。
    <div className="process-layout">
      <MdSurface className="process-left-card">
        <div className="surface-header">
          <div>
            <h2>{t('process.leftTitle')}</h2>
            <p>{videoMeta?.basename || t('process.noVideoHint')}</p>
          </div>
          <MdButton loading={isSelectingVideo} icon="video_file" variant="filled" onClick={onSelectVideo}>
            {t('common.selectVideo')}
          </MdButton>
        </div>
        <div className="process-toolbar">
          <MdChip>{`${t('common.fileName')}: ${videoMeta?.basename || '-'}`}</MdChip>
          <MdChip>{`${t('common.resolution')}: ${videoMeta ? `${videoMeta.width}×${videoMeta.height}` : '-'}`}</MdChip>
          <MdChip>{`${t('common.fps')}: ${videoMeta?.fps ? videoMeta.fps.toFixed(2) : '-'}`}</MdChip>
          <MdChip>{`${t('common.frames')}: ${videoMeta?.frame_count ?? 0}`}</MdChip>
        </div>

        <div className="process-preview-wrap" ref={previewWrapRef}>
          <div className={`process-preview-stage ${videoUrl || frameImageUrl ? '' : 'is-empty'}`.trim()} style={previewStageStyle}>
            {videoUrl ? (
              <video
                ref={videoRef}
                src={videoUrl}
                className="process-preview-video"
                preload="metadata"
                playsInline
                onTimeUpdate={(event) => {
                  if (!('requestVideoFrameCallback' in HTMLVideoElement.prototype)) {
                    syncFrameFromVideo(event.currentTarget.currentTime);
                  }
                }}
                onLoadedMetadata={() => seekVideoToFrame(currentFrame)}
                onEnded={() => {
                  syncFrameFromVideo(videoRef.current?.duration || frameMax / playbackFps);
                  if (isPlaying) onTogglePlay();
                }}
              />
            ) : frameImageUrl ? (
              <img src={frameImageUrl} alt="preview" className="process-preview-image" />
            ) : (
              <MdEmptyState
                className="process-preview-empty"
                icon="movie"
                title={t('process.noVideo')}
                description={t('process.noVideoHint')}
                action={(
                  <MdButton variant="outlined" icon="video_file" loading={isSelectingVideo} onClick={onSelectVideo}>
                    {t('common.selectVideo')}
                  </MdButton>
                )}
              />
            )}
          </div>
        </div>

        <div className="process-playback-controls">
          <MdButton
            icon={isPlaying ? 'pause' : 'play_arrow'}
            disabled={!videoPath}
            onClick={onTogglePlay}
          >
            {isPlaying ? t('common.pause') : t('common.play')}
          </MdButton>
          <div className="process-slider-wrap">
            <MdSlider
              min={0}
              max={frameMax}
              ariaLabel={sliderKey}
              value={sliderValue}
              onChange={(value) => {
                const nextFrame = Number(value);
                seekVideoToFrame(nextFrame);
                onSetCurrentFrame(nextFrame);
              }}
              disabled={!videoPath}
            />
          </div>
          <span className="metadata-text">{`${sliderValue}/${frameMax}`}</span>
        </div>
      </MdSurface>

      <MdSurface className="process-right-card supporting-pane">
        <div className="surface-header">
          <div>
            <h2>{t('process.rightTitle')}</h2>
            <p>{`${enabledCount} ${t('annotation.manager')}`}</p>
          </div>
        </div>
        <div className="process-task-stack">
          {!hasEnabledSegments && (
            <div className="process-warning-box">
              <span>{t('process.noSegmentsTip')}</span>
              <MdButton variant="text" icon="ink_highlighter" onClick={onGoAnnotate}>
                {t('process.goAnnotate')}
              </MdButton>
            </div>
          )}

          <MdTaskPanel icon="folder_open" title={t('process.outputPath')}>
            <div className="process-output-row">
              <span className="process-output-path-text" title={settings.outputPath || '-'}>
                {settings.outputPath || '-'}
              </span>
              <MdButton variant="outlined" icon="folder_open" onClick={onSelectOutputFolder}>
                {t('common.browse')}
              </MdButton>
            </div>
          </MdTaskPanel>

          <MdTaskPanel icon="model_training" title={t('process.model')} subtitle={t('process.model.hintShort')}>
            <MdSelect
              value={settings.modelId}
              onChange={(value) => onChangeModelId(value)}
              options={[
                { label: t('process.model.lama'), value: 'lama_roi' },
              ]}
            />
            <span className="process-field-help" title={t('process.model.hint')}>
              {t('process.model.hint')}
            </span>
          </MdTaskPanel>

          <MdTaskPanel
            icon="speed"
            title={t('process.progress')}
            footer={(
              <div className="button-row">
                <MdButton
                  variant="filled"
                  icon="play_arrow"
                  disabled={!videoPath || process.isProcessing}
                  onClick={onStartProcessing}
                >
                  {t('common.start')}
                </MdButton>
                <MdButton
                  variant="outlined"
                  tone="danger"
                  icon="stop"
                  disabled={!process.isProcessing}
                  onClick={onStopProcessing}
                >
                  {t('common.stop')}
                </MdButton>
              </div>
            )}
          >
            <div className="progress-with-value">
              <MdLinearProgress value={process.progress} />
              <span>{Math.round(process.progress * 100)}%</span>
            </div>
            <div className="process-progress-details">
              <span>{`${t('process.etaLabel')}: ${etaText}`}</span>
              <span>{`${t('process.speedLabel')}: ${speedText}`}</span>
              <span>{`${t('process.phaseLabel')}: ${phaseText}`}</span>
            </div>
          </MdTaskPanel>

          <div className="process-meta-box">
            <MdStatusMetric label={t('process.status')} value={process.statusMessage || t('status.idle')} />
            <MdStatusMetric label={t('common.frames')} value={`${process.processedFrames}/${process.totalFrames}`} />
            <MdStatusMetric label={t('process.device')} value={deviceInfo.device} />
            <MdStatusMetric label={t('process.memory')} value={deviceInfo.memory} />
          </div>
        </div>
      </MdSurface>
    </div>
  );
}
