import { mkdir, readFile, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import type { AppSettings } from './types.js';
import { isInpaintModelId } from './modelCatalog.js';

const SETTINGS_FILE = 'settings.json';

export function defaultSettings(): AppSettings {
  return {
    language: 'zh',
    theme: 'light',
    output: {
      path: path.join(os.homedir(), 'Downloads', 'WatermarkRemover'),
      model_id: 'lama_roi',
    },
  };
}

export async function loadSettings(userDataDir: string): Promise<AppSettings> {
  const defaults = defaultSettings();
  try {
    const raw = JSON.parse(await readFile(path.join(userDataDir, SETTINGS_FILE), 'utf8')) as Partial<AppSettings>;
    return sanitizeSettings(raw, defaults);
  } catch {
    return defaults;
  }
}

export async function saveSettings(
  userDataDir: string,
  settings: Partial<AppSettings>,
): Promise<{ success: boolean; error?: string }> {
  try {
    await mkdir(userDataDir, { recursive: true });
    const current = await loadSettings(userDataDir);
    const next = sanitizeSettings(settings, current);
    await writeFile(path.join(userDataDir, SETTINGS_FILE), `${JSON.stringify(next, null, 2)}\n`, 'utf8');
    return { success: true };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

function sanitizeSettings(raw: Partial<AppSettings>, fallback: AppSettings): AppSettings {
  return {
    language: raw.language === 'en' ? 'en' : raw.language === 'zh' ? 'zh' : fallback.language,
    theme: raw.theme === 'dark' ? 'dark' : raw.theme === 'light' ? 'light' : fallback.theme,
    output: {
      path:
        typeof raw.output?.path === 'string' && raw.output.path.trim()
          ? raw.output.path
          : fallback.output.path,
      model_id: isInpaintModelId(raw.output?.model_id) ? raw.output.model_id : fallback.output.model_id,
    },
  };
}
