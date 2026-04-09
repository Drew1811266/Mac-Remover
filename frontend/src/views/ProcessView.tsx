// 处理页视图：
// 左侧预览与播放控制，右侧处理参数与进度展示。
import { useEffect, useRef, useState } from 'react';
import { Button, Card, Progress, Select, Slider, Space, Tag, Tooltip, Typography } from '@douyinfe/semi-ui';
import { IconPause, IconPlay } from '@douyinfe/semi-icons';
import type { AnnotationSegment, VideoMeta } from '../types/annotation';
import type { AppSettings, DeviceInfoState, ProcessState } from '../types/app';
import { useI18n } from '../i18n/useI18n';

const { Text, Title } = Typography;

interface ProcessViewProps {
  // 当前视频与预览帧信息。
  videoPath: string;
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
  const [previewContainerWidth, setPreviewContainerWidth] = useState(0);
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

  useEffect(() => {
    // 监听容器尺寸变化，计算预览可用宽度以维持正确比例。
    const node = previewWrapRef.current;
    if (!node) return;

    const measure = () => {
      const rect = node.getBoundingClientRect();
      setPreviewContainerWidth(Math.max(0, Math.floor(rect.width)));
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

  const sourceWidth = Number(previewFrameWidth && previewFrameWidth > 0 ? previewFrameWidth : (videoMeta?.width ?? 0));
  const sourceHeight = Number(previewFrameHeight && previewFrameHeight > 0 ? previewFrameHeight : (videoMeta?.height ?? 0));
  const dpr = Math.max(1, window.devicePixelRatio || 1);

  let stageWidth = 0;
  let stageHeight = 0;
  if (sourceWidth > 0 && sourceHeight > 0) {
    const cssNativeWidth = sourceWidth / dpr;
    const widthCap = previewContainerWidth > 0 ? previewContainerWidth : cssNativeWidth;
    stageWidth = Math.min(cssNativeWidth, widthCap);
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
      <Card className="process-left-card" title={t('process.leftTitle')}>
        <Space className="process-toolbar" wrap>
          <Button loading={isSelectingVideo} onClick={onSelectVideo}>
            {t('common.selectVideo')}
          </Button>
          <Tag>{`${t('common.fileName')}: ${videoMeta?.basename || '-'}`}</Tag>
          <Tag>{`${t('common.resolution')}: ${videoMeta ? `${videoMeta.width}×${videoMeta.height}` : '-'}`}</Tag>
          <Tag>{`${t('common.fps')}: ${videoMeta?.fps ? videoMeta.fps.toFixed(2) : '-'}`}</Tag>
          <Tag>{`${t('common.frames')}: ${videoMeta?.frame_count ?? 0}`}</Tag>
        </Space>

        <div className="process-preview-wrap" ref={previewWrapRef}>
          <div className="process-preview-stage" style={previewStageStyle}>
            {frameImageUrl ? (
              <img src={frameImageUrl} alt="preview" className="process-preview-image" />
            ) : (
              <div className="process-preview-empty">
                <Title heading={5}>{t('process.noVideo')}</Title>
                <Text type="tertiary">{t('process.noVideoHint')}</Text>
              </div>
            )}
          </div>
        </div>

        <Space className="process-playback-controls">
          <Button
            icon={isPlaying ? <IconPause /> : <IconPlay />}
            disabled={!videoPath}
            onClick={onTogglePlay}
          >
            {isPlaying ? t('common.pause') : t('common.play')}
          </Button>
          <div className="process-slider-wrap">
            <Slider
              key={sliderKey}
              min={0}
              max={frameMax}
              value={sliderValue}
              onChange={(value) => onSetCurrentFrame(Number(value))}
              disabled={!videoPath}
            />
          </div>
          <Text type="tertiary">{`${sliderValue}/${frameMax}`}</Text>
        </Space>
      </Card>

      <Card className="process-right-card" title={t('process.rightTitle')}>
        {!hasEnabledSegments && (
          <div className="process-warning-box">
            <Text>{t('process.noSegmentsTip')}</Text>
            <Button size="small" onClick={onGoAnnotate}>
              {t('process.goAnnotate')}
            </Button>
          </div>
        )}

        <div className="process-field">
          <Text className="process-field-label">{t('process.outputPath')}</Text>
          <Space align="center" className="process-output-row">
            <Text ellipsis={{ showTooltip: true }} className="process-output-path-text">
              {settings.outputPath || '-'}
            </Text>
            <Button size="small" onClick={onSelectOutputFolder}>
              {t('common.browse')}
            </Button>
          </Space>
        </div>

        <div className="process-field">
          <Text className="process-field-label">{t('process.model')}</Text>
          <Select
            value={settings.modelId}
            onChange={(value) => onChangeModelId(String(value) as AppSettings['modelId'])}
            optionList={[
              { label: t('process.model.lama'), value: 'lama_roi' },
              { label: t('process.model.propainter'), value: 'propainter_roi' },
            ]}
          />
          <Text className="process-field-help" type="tertiary">
            <Tooltip content={t('process.model.hint')}>
              <span>{t('process.model.hintShort')}</span>
            </Tooltip>
          </Text>
        </div>

        <div className="process-field">
          <Space>
            <Button
              type="primary"
              disabled={!videoPath || process.isProcessing}
              onClick={onStartProcessing}
            >
              {t('common.start')}
            </Button>
            <Button
              type="danger"
              theme="light"
              disabled={!process.isProcessing}
              onClick={onStopProcessing}
            >
              {t('common.stop')}
            </Button>
          </Space>
        </div>

        <div className="process-field">
          <Text className="process-field-label">{t('process.progress')}</Text>
          <Progress percent={Math.round(process.progress * 100)} showInfo />
          <Space className="process-progress-details" wrap>
            <Text type="tertiary">{`${t('process.etaLabel')}: ${etaText}`}</Text>
            <Text type="tertiary">{`${t('process.speedLabel')}: ${speedText}`}</Text>
            <Text type="tertiary">{`${t('process.phaseLabel')}: ${phaseText}`}</Text>
          </Space>
        </div>

        <div className="process-meta-box">
          <Text>{`${t('process.status')}: ${process.statusMessage || t('status.idle')}`}</Text>
          <Text>{`${t('common.frames')}: ${process.processedFrames}/${process.totalFrames}`}</Text>
          <Text>{`${t('process.device')}: ${deviceInfo.device}`}</Text>
          <Text>{`${t('process.memory')}: ${deviceInfo.memory}`}</Text>
        </div>
      </Card>
    </div>
  );
}
