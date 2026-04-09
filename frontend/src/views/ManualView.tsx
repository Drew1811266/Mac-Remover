// 软件说明书侧栏视图：
// 展示功能简介、快速开始、页面功能、使用建议与常见问题。
import { Typography } from '@douyinfe/semi-ui';
import { useI18n } from '../i18n/useI18n';

const { Text } = Typography;

const QUICK_START_KEYS = [
  'manual.quickStart.step1',
  'manual.quickStart.step2',
  'manual.quickStart.step3',
  'manual.quickStart.step4',
];

const FEATURE_ITEMS = [
  {
    titleKey: 'manual.features.process.title',
    descKey: 'manual.features.process.desc',
  },
  {
    titleKey: 'manual.features.annotate.title',
    descKey: 'manual.features.annotate.desc',
  },
  {
    titleKey: 'manual.features.result.title',
    descKey: 'manual.features.result.desc',
  },
  {
    titleKey: 'manual.features.upscale.title',
    descKey: 'manual.features.upscale.desc',
  },
  {
    titleKey: 'manual.features.settings.title',
    descKey: 'manual.features.settings.desc',
  },
];

const TIPS_KEYS = [
  'manual.tips.item1',
  'manual.tips.item2',
  'manual.tips.item3',
];

const FAQ_ITEMS = [
  {
    questionKey: 'manual.faq.preview.question',
    answerKey: 'manual.faq.preview.answer',
  },
  {
    questionKey: 'manual.faq.model.question',
    answerKey: 'manual.faq.model.answer',
  },
  {
    questionKey: 'manual.faq.speed.question',
    answerKey: 'manual.faq.speed.answer',
  },
];

export function ManualView() {
  const { t } = useI18n();

  return (
    <div className="manual-panel">
      <div className="manual-panel-intro">
        <Text type="tertiary" className="manual-subtitle">
          {t('manual.subtitle')}
        </Text>
      </div>

      <section className="manual-section">
        <Text className="manual-section-title">{t('manual.section.overview')}</Text>
        <Text className="manual-body-text">{t('manual.overview.body')}</Text>
      </section>

      <section className="manual-section">
        <Text className="manual-section-title">{t('manual.section.quickStart')}</Text>
        <ol className="manual-list manual-list-ordered">
          {QUICK_START_KEYS.map((key) => (
            <li key={key}>
              <Text>{t(key)}</Text>
            </li>
          ))}
        </ol>
      </section>

      <section className="manual-section">
        <Text className="manual-section-title">{t('manual.section.features')}</Text>
        <ul className="manual-list">
          {FEATURE_ITEMS.map((item) => (
            <li key={item.titleKey} className="manual-feature-item">
              <Text className="manual-feature-title">{t(item.titleKey)}</Text>
              <Text type="tertiary">{t(item.descKey)}</Text>
            </li>
          ))}
        </ul>
      </section>

      <section className="manual-section">
        <Text className="manual-section-title">{t('manual.section.tips')}</Text>
        <ul className="manual-list">
          {TIPS_KEYS.map((key) => (
            <li key={key}>
              <Text>{t(key)}</Text>
            </li>
          ))}
        </ul>
      </section>

      <section className="manual-section">
        <Text className="manual-section-title">{t('manual.section.faq')}</Text>
        <ul className="manual-list">
          {FAQ_ITEMS.map((item) => (
            <li key={item.questionKey} className="manual-faq-item">
              <Text className="manual-feature-title">{t(item.questionKey)}</Text>
              <Text type="tertiary">{t(item.answerKey)}</Text>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
