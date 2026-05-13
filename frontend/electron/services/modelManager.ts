import { createHash, randomUUID } from 'node:crypto';
import { access, mkdir, rename, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';

import {
  ALL_MODEL_SPECS,
  INPAINT_MODEL_SPECS,
  isDownloadableModelId,
  isInpaintModelId,
  modelAssetPath,
  modelInstallDir,
  type DownloadableModelId,
  type InpaintModelId,
  type ModelSpec,
} from './modelCatalog.js';
import { validateManifest, type ManifestAsset, type ModelManifest } from './modelManifest.js';

export interface ModelDownloadTask {
  state: 'idle' | 'running' | 'success' | 'failed' | 'cancelled';
  model_id: DownloadableModelId | '';
  progress: number;
  downloaded_bytes: number;
  total_bytes: number;
  speed_bps: number;
  current_file: string;
  message: string;
  error: string;
}

export interface ModelDownloadStatus {
  success: boolean;
  models: Array<{
    model_id: InpaintModelId;
    display_name: string;
    installed: boolean;
    can_redownload: boolean;
    install_hint: string;
  }>;
  task: ModelDownloadTask;
  error?: string;
}

export interface ModelManagerOptions {
  userDataDir: string;
  manifestUrl?: string;
}

export class ModelManager {
  private task: ModelDownloadTask = createIdleTask();
  private abortController: AbortController | null = null;

  constructor(private readonly options: ModelManagerOptions) {}

  modelPath(modelId: InpaintModelId = 'lama_roi'): string {
    const spec = INPAINT_MODEL_SPECS[modelId];
    return modelAssetPath(this.options.userDataDir, spec, spec.assets[0]);
  }

  async isModelInstalled(modelId: DownloadableModelId): Promise<boolean> {
    const spec = ALL_MODEL_SPECS[modelId];
    if (!spec) return false;
    try {
      for (const asset of spec.assets) {
        const modelStat = await stat(modelAssetPath(this.options.userDataDir, spec, asset));
        if (!modelStat.isFile() || modelStat.size <= 0) return false;
      }
      return true;
    } catch {
      return false;
    }
  }

  async getStatus(): Promise<ModelDownloadStatus> {
    const modelEntries = await Promise.all(
      Object.values(INPAINT_MODEL_SPECS).map(async (spec) => {
        const installed = await this.isModelInstalled(spec.modelId);
        return {
          model_id: spec.modelId as InpaintModelId,
          display_name: spec.displayName,
          installed,
          can_redownload: true,
          install_hint: installed ? '' : spec.installHint,
        };
      }),
    );
    return {
      success: true,
      models: modelEntries,
      task: { ...this.task },
    };
  }

  async startDownload(modelId: unknown = 'lama_roi', force = false): Promise<{ success: boolean; error?: string }> {
    if (this.task.state === 'running') {
      return { success: true };
    }
    if (!isInpaintModelId(modelId)) {
      return {
        success: false,
        error: `Invalid model_id: ${String(modelId || '')}. Supported values: ${Object.keys(INPAINT_MODEL_SPECS).join(', ')}`,
      };
    }
    if (!force && (await this.isModelInstalled(modelId))) {
      this.task = { ...createIdleTask(), state: 'success', progress: 1, message: 'Model already installed' };
      return { success: true };
    }
    if (!this.options.manifestUrl) {
      return {
        success: false,
        error: `WMR_MODEL_MANIFEST_URL is not configured; cannot download non-Python assets for ${modelId}`,
      };
    }

    this.abortController = new AbortController();
    this.task = {
      ...createIdleTask(),
      state: 'running',
      model_id: modelId,
      message: 'Downloading model manifest',
    };

    void this.downloadFromManifest(modelId, this.abortController.signal).catch((error) => {
      this.task = {
        ...this.task,
        state: this.task.state === 'cancelled' ? 'cancelled' : 'failed',
        error: error instanceof Error ? error.message : String(error),
        message: 'Model download failed',
      };
    });

    return { success: true };
  }

  cancelDownload(): { success: boolean; error?: string } {
    if (this.task.state === 'running') {
      this.task = { ...this.task, state: 'cancelled', message: 'Model download cancelled' };
      this.abortController?.abort();
    }
    return { success: true };
  }

  private async downloadFromManifest(modelId: InpaintModelId, signal: AbortSignal): Promise<void> {
    const manifest = await this.fetchManifest(signal);
    const spec = INPAINT_MODEL_SPECS[modelId];
    const assets = resolveManifestAssets(manifest, spec);
    if (!spec.implemented) {
      throw new Error(spec.blockedReason || `${spec.displayName} is blocked by the non-Python equivalence gate.`);
    }

    await mkdir(modelInstallDir(this.options.userDataDir, spec), { recursive: true });

    for (const { manifestAsset, fileName } of assets) {
      const destination = modelAssetPath(this.options.userDataDir, spec, { assetId: manifestAsset.asset_id, fileName });
      const tmpPath = path.join(path.dirname(destination), `${fileName}.${randomUUID()}.download`);
      await this.downloadAsset(manifestAsset, tmpPath, signal);

      const sha256 = await sha256File(tmpPath);
      if (sha256 !== manifestAsset.sha256) {
        await rm(tmpPath, { force: true });
        throw new Error(`Checksum mismatch for ${manifestAsset.fileName}`);
      }

      await rename(tmpPath, destination);
    }
    this.task = {
      ...this.task,
      state: 'success',
      progress: 1,
      downloaded_bytes: assets.reduce((sum, item) => sum + item.manifestAsset.size, 0),
      total_bytes: assets.reduce((sum, item) => sum + item.manifestAsset.size, 0),
      speed_bps: 0,
      current_file: spec.displayName,
      message: 'Model download complete',
      error: '',
    };
  }

  private async fetchManifest(signal: AbortSignal): Promise<ModelManifest> {
    const response = await fetch(this.options.manifestUrl!, { signal });
    if (!response.ok) {
      throw new Error(`Failed to fetch model manifest: HTTP ${response.status}`);
    }
    return validateManifest(await response.json());
  }

  private async downloadAsset(asset: ManifestAsset, destination: string, signal: AbortSignal): Promise<void> {
    this.task = {
      ...this.task,
      current_file: asset.fileName,
      total_bytes: asset.size,
      message: `Downloading ${asset.fileName}`,
    };

    const response = await fetch(asset.url, { signal });
    if (!response.ok || !response.body) {
      throw new Error(`Failed to download ${asset.fileName}: HTTP ${response.status}`);
    }

    const started = Date.now();
    const buffer = Buffer.from(await response.arrayBuffer());
    await writeFile(destination, buffer);
    const elapsedSec = Math.max(0.001, (Date.now() - started) / 1000);
    this.task = {
      ...this.task,
      downloaded_bytes: buffer.byteLength,
      total_bytes: asset.size,
      progress: asset.size > 0 ? Math.min(1, buffer.byteLength / asset.size) : 1,
      speed_bps: buffer.byteLength / elapsedSec,
    };
  }
}

export async function localModelPath(userDataDir: string, modelId: InpaintModelId = 'lama_roi'): Promise<string | null> {
  const spec = INPAINT_MODEL_SPECS[modelId];
  const modelPath = modelAssetPath(userDataDir, spec, spec.assets[0]);
  try {
    await access(modelPath);
    return modelPath;
  } catch {
    return null;
  }
}

export async function isInstalledModel(userDataDir: string, modelId: DownloadableModelId): Promise<boolean> {
  if (!isDownloadableModelId(modelId)) return false;
  const spec = ALL_MODEL_SPECS[modelId];
  try {
    for (const asset of spec.assets) {
      await access(modelAssetPath(userDataDir, spec, asset));
    }
    return true;
  } catch {
    return false;
  }
}

function createIdleTask(): ModelDownloadTask {
  return {
    state: 'idle',
    model_id: '',
    progress: 0,
    downloaded_bytes: 0,
    total_bytes: 0,
    speed_bps: 0,
    current_file: '',
    message: '',
    error: '',
  };
}

async function sha256File(filePath: string): Promise<string> {
  const data = await import('node:fs/promises').then((fs) => fs.readFile(filePath));
  return createHash('sha256').update(data).digest('hex');
}

function resolveManifestAssets(
  manifest: ModelManifest,
  spec: ModelSpec,
): Array<{ manifestAsset: ManifestAsset; fileName: string }> {
  return spec.assets.map((asset) => {
    const manifestAsset = manifest.assets.find(
      (item) => item.kind === 'model' && item.asset_id === asset.assetId && item.model_id === spec.modelId,
    );
    if (!manifestAsset) {
      throw new Error(`Manifest does not contain required asset ${asset.assetId} for ${spec.modelId}`);
    }
    return { manifestAsset, fileName: asset.fileName };
  });
}
