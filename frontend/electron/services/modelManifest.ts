export type AssetKind = 'model' | 'runtime';
export type RuntimeKind =
  | 'onnx'
  | 'ncnn'
  | 'gguf-runner'
  | 'native-addon'
  | 'ffmpeg'
  | 'archive';

export type ManifestModelId =
  | 'lama_roi'
  | 'realesrgan_general_x4v3'
  | 'realesrgan_x2plus'
  | 'seedvr2_3b_q8_0_gguf'
  | 'seedvr2_3b_q4_k_m_gguf';

export type ManifestEngine = 'lama' | 'realesrgan' | 'seedvr2' | 'ffmpeg' | 'native';

export interface ManifestAsset {
  asset_id: string;
  id: string;
  kind: AssetKind;
  model_id?: ManifestModelId;
  engine: ManifestEngine;
  runtime_kind: RuntimeKind;
  version: string;
  url: string;
  sha256: string;
  size: number;
  file_name: string;
  fileName: string;
  license: string;
  platform?: string;
  arch?: string;
}

export interface ModelManifest {
  version: number;
  assets: ManifestAsset[];
}

const SHA256_RE = /^[a-f0-9]{64}$/i;

export function validateManifest(input: unknown): ModelManifest {
  if (!input || typeof input !== 'object') {
    throw new Error('Invalid model manifest: expected object');
  }

  const manifest = input as Partial<ModelManifest>;
  if (manifest.version !== 1 || !Array.isArray(manifest.assets)) {
    throw new Error('Invalid model manifest: expected version 1 with assets');
  }

  const assets = manifest.assets.map((asset, index) => validateAsset(asset, index));
  return { version: 1, assets };
}

function validateAsset(input: unknown, index: number): ManifestAsset {
  if (!input || typeof input !== 'object') {
    throw new Error(`Invalid manifest asset at ${index}: expected object`);
  }
  const asset = input as Partial<ManifestAsset> & {
    id?: string;
    asset_id?: string;
    fileName?: string;
    file_name?: string;
  };
  const assetId = firstNonEmpty(asset.asset_id, asset.id);
  const fileName = firstNonEmpty(asset.file_name, asset.fileName);
  const kindOk = asset.kind === 'model' || asset.kind === 'runtime';
  const engineOk =
    asset.engine === 'lama' ||
    asset.engine === 'realesrgan' ||
    asset.engine === 'seedvr2' ||
    asset.engine === 'ffmpeg' ||
    asset.engine === 'native';
  const runtimeKindOk =
    asset.runtime_kind === 'onnx' ||
    asset.runtime_kind === 'ncnn' ||
    asset.runtime_kind === 'gguf-runner' ||
    asset.runtime_kind === 'native-addon' ||
    asset.runtime_kind === 'ffmpeg' ||
    asset.runtime_kind === 'archive';
  const commonOk =
    nonEmpty(assetId) &&
    kindOk &&
    engineOk &&
    runtimeKindOk &&
    nonEmpty(asset.version) &&
    nonEmpty(asset.url) &&
    nonEmpty(fileName) &&
    nonEmpty(asset.license) &&
    Number.isFinite(asset.size) &&
    Number(asset.size) > 0 &&
    nonEmpty(asset.sha256) &&
    SHA256_RE.test(asset.sha256);

  if (!commonOk) {
    throw new Error(`Invalid manifest asset at ${index}`);
  }

  if (asset.kind === 'model' && !isManifestModelId(asset.model_id)) {
    throw new Error(`Invalid manifest asset at ${index}: model asset requires supported model_id`);
  }

  if (asset.kind === 'runtime' && (!nonEmpty(asset.platform) || !nonEmpty(asset.arch))) {
    throw new Error(`Invalid manifest asset at ${index}: runtime asset requires platform and arch`);
  }

  const kind = asset.kind as AssetKind;
  const version = asset.version as string;
  const url = asset.url as string;
  const sha256 = asset.sha256 as string;
  const normalizedAssetId = assetId as string;
  const normalizedFileName = fileName as string;

  return {
    asset_id: normalizedAssetId,
    id: normalizedAssetId,
    kind,
    model_id: asset.model_id,
    engine: asset.engine as ManifestEngine,
    runtime_kind: asset.runtime_kind as RuntimeKind,
    version,
    url,
    sha256: sha256.toLowerCase(),
    size: Number(asset.size),
    file_name: normalizedFileName,
    fileName: normalizedFileName,
    license: asset.license as string,
    platform: asset.platform,
    arch: asset.arch,
  };
}

function nonEmpty(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0;
}

function firstNonEmpty(...values: unknown[]): string | undefined {
  for (const value of values) {
    if (nonEmpty(value)) return value;
  }
  return undefined;
}

function isManifestModelId(value: unknown): value is ManifestModelId {
  return (
    value === 'lama_roi' ||
    value === 'realesrgan_general_x4v3' ||
    value === 'realesrgan_x2plus' ||
    value === 'seedvr2_3b_q8_0_gguf' ||
    value === 'seedvr2_3b_q4_k_m_gguf'
  );
}
