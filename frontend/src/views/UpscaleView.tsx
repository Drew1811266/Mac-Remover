import { Button, Card, Space, Tag, Typography } from '@douyinfe/semi-ui';
import type { ResultState, UpscaleResultState, UpscaleTaskState } from '../types/app';
import { useI18n } from '../i18n/useI18n';
import { FrameComparePreview } from '../components/FrameComparePreview';

const { Text, Title } = Typography;

interface UpscaleViewProps {
  sourceResult: ResultState;
  upscaleResult: UpscaleResultState;
  upscaleTask: UpscaleTaskState;
  onOpenOutputDir: () => void;
  onBackToResult: () => void;
}

function isVideoPath(path: string): boolean {
  return /\.(mp4|mov|mkv|avi|webm|m4v)(\?|#|$)/i.test(path || '');
}

export function UpscaleView({
  sourceResult,
  upscaleResult,
  upscaleTask,
  onOpenOutputDir,
  onBackToResult,
}: UpscaleViewProps) {
  const { t } = useI18n();
  const hasUpscale = !!upscaleResult.outputPath && !!upscaleResult.outputUrl;
  const sourcePath = sourceResult.mediaType === 'video' ? sourceResult.outputPath : undefined;
  const upscaleIsVideo = isVideoPath(upscaleResult.outputPath || upscaleResult.outputUrl || '');

  return (
    <Card
      className="upscale-view-card"
      title={t('upscale.view.title')}
      headerExtraContent={(
        <Space>
          <Button onClick={onBackToResult}>{t('upscale.view.back')}</Button>
          <Button onClick={onOpenOutputDir}>{t('common.openOutput')}</Button>
        </Space>
      )}
    >
      <Text type="tertiary">{t('upscale.view.subtitle')}</Text>

      {!hasUpscale && (
        <div className="result-empty">
          <Title heading={5}>{t('upscale.view.empty')}</Title>
          <Text type="tertiary">{upscaleTask.message || t('upscale.status.idle')}</Text>
        </div>
      )}

      {hasUpscale && (
        <>
          <Space className="result-meta-tags" wrap>
            <Tag>{`${t('common.fileName')}: ${upscaleResult.outputPath.split('/').pop()}`}</Tag>
            <Tag>{`${t('common.resolution')}: ${upscaleResult.width}×${upscaleResult.height}`}</Tag>
            <Tag>{`${t('common.fps')}: ${upscaleResult.fps ? upscaleResult.fps.toFixed(2) : '-'}`}</Tag>
            <Tag>{`${t('common.frames')}: ${upscaleResult.frameCount || '-'}`}</Tag>
            <Tag>{`${t('upscale.mode')}: ${upscaleResult.mode || '-'}`}</Tag>
            <Tag>{`${t('upscale.engine')}: ${upscaleResult.engine || '-'}`}</Tag>
            <Tag>{`${t('upscale.model')}: ${upscaleResult.modelId || '-'}`}</Tag>
          </Space>

          <FrameComparePreview
            outputPath={upscaleResult.outputPath}
            outputUrlFallback={upscaleResult.previewUrl || upscaleResult.outputUrl}
            sourcePath={sourcePath}
            isVideo={upscaleIsVideo}
            fpsHint={upscaleResult.fps}
            frameCountHint={upscaleResult.frameCount}
            widthHint={upscaleResult.width}
            heightHint={upscaleResult.height}
            beforeLabel={t('upscale.compare.before')}
            afterLabel={t('upscale.compare.after')}
            compareUnavailableText={t('upscale.compare.unavailable')}
          />

          {upscaleResult.warning && <Text type="tertiary">{upscaleResult.warning}</Text>}
        </>
      )}
    </Card>
  );
}
