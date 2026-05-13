import type { Language, ThemeMode } from '../types/app';

export function applyDocumentTheme(theme: ThemeMode, language: Language): void {
  document.body.setAttribute('theme-mode', theme);
  document.body.dataset.materialDensity = 'compact';
  document.documentElement.lang = language === 'en' ? 'en' : 'zh-CN';
}
