// 软件说明书侧栏视图：
// 展示功能简介、快速开始、页面功能、使用建议与常见问题。
import { useState } from 'react';
import { useI18n } from '../i18n/useI18n';

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

type ManualSection = 'overview' | 'quickStart' | 'features' | 'tips' | 'faq';

export function ManualView() {
  const { t } = useI18n();
  const [activeSection, setActiveSection] = useState<ManualSection>('overview');
  const sectionTabs: Array<{ key: ManualSection; labelKey: string }> = [
    { key: 'overview', labelKey: 'manual.section.overview' },
    { key: 'quickStart', labelKey: 'manual.section.quickStart' },
    { key: 'features', labelKey: 'manual.section.features' },
    { key: 'tips', labelKey: 'manual.section.tips' },
    { key: 'faq', labelKey: 'manual.section.faq' },
  ];

  return (
    <div className="manual-panel">
      <div className="manual-panel-intro">
        <p className="manual-subtitle">{t('manual.subtitle')}</p>
      </div>

      <div className="manual-tabs" role="tablist" aria-label={t('manual.title')}>
        {sectionTabs.map((item) => (
          <button
            key={item.key}
            type="button"
            role="tab"
            aria-selected={activeSection === item.key}
            className={activeSection === item.key ? 'is-selected' : ''}
            onClick={() => setActiveSection(item.key)}
          >
            {t(item.labelKey)}
          </button>
        ))}
      </div>

      <div className="manual-section-frame">
        {activeSection === 'overview' ? (
          <section className="manual-section" role="tabpanel">
            <h3 className="manual-section-title">{t('manual.section.overview')}</h3>
            <p className="manual-body-text">{t('manual.overview.body')}</p>
          </section>
        ) : null}

        {activeSection === 'quickStart' ? (
          <section className="manual-section" role="tabpanel">
            <h3 className="manual-section-title">{t('manual.section.quickStart')}</h3>
            <ol className="manual-list manual-list-ordered">
              {QUICK_START_KEYS.map((key) => (
                <li key={key}>{t(key)}</li>
              ))}
            </ol>
          </section>
        ) : null}

        {activeSection === 'features' ? (
          <section className="manual-section" role="tabpanel">
            <h3 className="manual-section-title">{t('manual.section.features')}</h3>
            <ul className="manual-list">
              {FEATURE_ITEMS.map((item) => (
                <li key={item.titleKey} className="manual-feature-item">
                  <strong className="manual-feature-title">{t(item.titleKey)}</strong>
                  <span>{t(item.descKey)}</span>
                </li>
              ))}
            </ul>
          </section>
        ) : null}

        {activeSection === 'tips' ? (
          <section className="manual-section" role="tabpanel">
            <h3 className="manual-section-title">{t('manual.section.tips')}</h3>
            <ul className="manual-list">
              {TIPS_KEYS.map((key) => (
                <li key={key}>{t(key)}</li>
              ))}
            </ul>
          </section>
        ) : null}

        {activeSection === 'faq' ? (
          <section className="manual-section" role="tabpanel">
            <h3 className="manual-section-title">{t('manual.section.faq')}</h3>
            <ul className="manual-list">
              {FAQ_ITEMS.map((item) => (
                <li key={item.questionKey} className="manual-faq-item">
                  <strong className="manual-feature-title">{t(item.questionKey)}</strong>
                  <span>{t(item.answerKey)}</span>
                </li>
              ))}
            </ul>
          </section>
        ) : null}
      </div>
    </div>
  );
}
