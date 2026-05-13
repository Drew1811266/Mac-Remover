import { useState } from 'react';
import type { ResultState, UpscaleResultState, UpscaleTaskState } from '../types/app';
import { useI18n } from '../i18n/useI18n';
import { FrameComparePreview } from '../components/FrameComparePreview';
import { MaterialIcon, MdButton, MdChip, MdEmptyState, MdSurface } from '../material';

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
  const [showAllMeta, setShowAllMeta] = useState(false);
  const hasUpscale = !!upscaleResult.outputPath && !!upscaleResult.outputUrl;
  const sourcePath = sourceResult.mediaType === 'video' ? sourceResult.outputPath : undefined;
  const upscaleIsVideo = isVideoPath(upscaleResult.outputPath || upscaleResult.outputUrl || '');
  const metaItems = hasUpscale
    ? [
        `${t('common.fileName')}: ${upscaleResult.outputPath.split('/').pop()}`,
        `${t('common.resolution')}: ${upscaleResult.width}×${upscaleResult.height}`,
        `${t('common.fps')}: ${upscaleResult.fps ? upscaleResult.fps.toFixed(2) : '-'}`,
        `${t('common.frames')}: ${upscaleResult.frameCount || '-'}`,
        `${t('upscale.mode')}: ${upscaleResult.mode || '-'}`,
        `${t('upscale.engine')}: ${upscaleResult.engine || '-'}`,
        `${t('upscale.model')}: ${upscaleResult.modelId || '-'}`,
      ]
    : [];
  const visibleMetaItems = showAllMeta ? metaItems : metaItems.slice(0, 4);

  return (
    <MdSurface className="upscale-view-card">
      <div className="surface-header">
        <div>
          <h2>{t('upscale.view.title')}</h2>
          <p>{t('upscale.view.subtitle')}</p>
        </div>
        <div className="button-row">
          <MdButton variant="outlined" icon="arrow_back" onClick={onBackToResult}>
            {t('upscale.view.back')}
          </MdButton>
          <MdButton variant="tonal" icon="folder_open" onClick={onOpenOutputDir}>
            {t('common.openOutput')}
          </MdButton>
        </div>
      </div>

      {!hasUpscale && (
        <div className="result-empty">
          <MdEmptyState
            icon="auto_awesome_motion"
            title={t('upscale.view.empty')}
            description={upscaleTask.message || t('upscale.status.idle')}
          />
        </div>
      )}

      {hasUpscale && (
        <>
          <div className="result-meta-tags">
            {visibleMetaItems.map((item) => (
              <MdChip key={item}>{item}</MdChip>
            ))}
            {metaItems.length > 4 ? (
              <button
                type="button"
                className="meta-more-button"
                aria-expanded={showAllMeta}
                onClick={() => setShowAllMeta((value) => !value)}
              >
                <span>{showAllMeta ? t('common.less') : t('common.more')}</span>
                <MaterialIcon name={showAllMeta ? 'expand_less' : 'expand_more'} />
              </button>
            ) : null}
          </div>

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

          {upscaleResult.warning && <p className="metadata-text">{upscaleResult.warning}</p>}
        </>
      )}
    </MdSurface>
  );
}
