// i18n Hook：
// 从全局设置读取当前语言，并返回翻译函数 t(key)。
import { useCallback, useMemo } from 'react';
import { messages } from './messages';
import { useAppStore } from '../store/app';

export function useI18n() {
  // 语言开关来自全局设置。
  const language = useAppStore((state) => state.settings.language);

  // 当前语言找不到时回退中文词典。
  const dictionary = useMemo(() => messages[language] ?? messages.zh, [language]);

  // 翻译函数：缺失 key 时原样返回 key，便于排查漏翻译。
  const t = useCallback((key: string): string => dictionary[key] ?? key, [dictionary]);

  return { t, language };
}
