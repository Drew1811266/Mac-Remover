import sharp from 'sharp';

import type { NormalizedAnnotationSegment } from './types.js';

import type { InferenceSession, Tensor } from 'onnxruntime-node';

type Ort = typeof import('onnxruntime-node');

interface LoadedOnnxModel {
  ort: Ort;
  session: InferenceSession;
  inputs: {
    image: string;
    mask: string;
  };
  fixedInputSize: { width: number; height: number } | null;
}

interface CropRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

const modelCache = new Map<string, Promise<LoadedOnnxModel>>();

export async function inpaintImageWithOnnx(
  modelPath: string,
  imagePath: string,
  segments: NormalizedAnnotationSegment[],
): Promise<void> {
  if (segments.length === 0) return;

  const source = sharp(imagePath).removeAlpha();
  const metadata = await source.metadata();
  const width = metadata.width || 0;
  const height = metadata.height || 0;
  if (width <= 0 || height <= 0) {
    throw new Error(`Cannot read frame dimensions: ${imagePath}`);
  }

  const model = await loadOnnxModel(modelPath);
  if (model.fixedInputSize) {
    await inpaintFixedInputCrops(model, imagePath, width, height, segments);
    return;
  }

  const padWidth = alignTo(width, 32);
  const padHeight = alignTo(height, 32);
  const raw = await source
    .extend({
      right: padWidth - width,
      bottom: padHeight - height,
      background: { r: 0, g: 0, b: 0 },
    })
    .raw()
    .toBuffer();
  const mask = createMask(width, height, padWidth, padHeight, segments);
  const imageTensor = rgbToNchwFloat(raw, padWidth, padHeight);

  const output = await runInpaintModel(model, imageTensor, mask, padWidth, padHeight);

  const out = nchwFloatToRgb(output, padWidth, padHeight, width, height);
  await sharp(out, { raw: { width, height, channels: 3 } }).png().toFile(imagePath);
}

async function loadOnnxModel(modelPath: string): Promise<LoadedOnnxModel> {
  let cached = modelCache.get(modelPath);
  if (!cached) {
    cached = loadOnnxModelUncached(modelPath);
    modelCache.set(modelPath, cached);
  }
  return cached;
}

async function loadOnnxModelUncached(modelPath: string): Promise<LoadedOnnxModel> {
  const ort = await import('onnxruntime-node');
  const session = await ort.InferenceSession.create(modelPath, { executionProviders: ['cpu'] });
  if (session.inputNames.length < 2) {
    throw new Error('LaMa ONNX model must expose image and mask inputs');
  }

  const inputs = resolveModelInputs(session);
  return {
    ort,
    session,
    inputs,
    fixedInputSize: resolveFixedInputSize(session, inputs.image),
  };
}

function resolveModelInputs(session: InferenceSession): LoadedOnnxModel['inputs'] {
  const tensors = session.inputMetadata.filter((meta) => meta.isTensor);
  const image = tensors.find((meta) => meta.shape[1] === 3)?.name || session.inputNames[0];
  const mask = tensors.find((meta) => meta.shape[1] === 1)?.name || session.inputNames.find((name) => name !== image) || session.inputNames[1];
  return { image, mask };
}

function resolveFixedInputSize(session: InferenceSession, imageInputName: string): LoadedOnnxModel['fixedInputSize'] {
  const imageInput = session.inputMetadata.find((meta) => meta.name === imageInputName);
  if (!imageInput?.isTensor) return null;

  const height = imageInput.shape[2];
  const width = imageInput.shape[3];
  if (typeof width === 'number' && typeof height === 'number' && width > 0 && height > 0) {
    return { width, height };
  }
  return null;
}

async function runInpaintModel(
  model: LoadedOnnxModel,
  imageTensor: Float32Array,
  mask: Float32Array,
  width: number,
  height: number,
): Promise<Float32Array> {
  const feeds: Record<string, Tensor> = {
    [model.inputs.image]: new model.ort.Tensor('float32', imageTensor, [1, 3, height, width]),
    [model.inputs.mask]: new model.ort.Tensor('float32', mask, [1, 1, height, width]),
  };
  const outputs = await model.session.run(feeds);
  const outputName = model.session.outputNames[0];
  const output = outputs[outputName];
  if (!output || output.type !== 'float32') {
    throw new Error('LaMa ONNX model returned unsupported output tensor');
  }
  return output.data as Float32Array;
}

async function inpaintFixedInputCrops(
  model: LoadedOnnxModel,
  imagePath: string,
  width: number,
  height: number,
  segments: NormalizedAnnotationSegment[],
): Promise<void> {
  const inputSize = model.fixedInputSize;
  if (!inputSize) return;

  for (const segment of segments) {
    const crop = computeCropRect(segment, width, height);
    if (crop.width <= 0 || crop.height <= 0) continue;

    const raw = await sharp(imagePath)
      .extract({ left: crop.x, top: crop.y, width: crop.width, height: crop.height })
      .removeAlpha()
      .resize({ width: inputSize.width, height: inputSize.height, fit: 'fill' })
      .raw()
      .toBuffer();
    const mask = createMaskForCrop(crop, inputSize.width, inputSize.height, [segment]);
    const imageTensor = rgbToNchwFloat(raw, inputSize.width, inputSize.height);
    const output = await runInpaintModel(model, imageTensor, mask, inputSize.width, inputSize.height);
    const outputRgb = nchwFloatToRgb(output, inputSize.width, inputSize.height, inputSize.width, inputSize.height);
    const resizedRgb = await sharp(outputRgb, { raw: { width: inputSize.width, height: inputSize.height, channels: 3 } })
      .resize({ width: crop.width, height: crop.height, fit: 'fill' })
      .raw()
      .toBuffer();
    const alpha = createAlphaMaskForCrop(crop, crop.width, crop.height, [segment]);
    const overlay = await sharp(rgbaWithAlpha(resizedRgb, alpha), {
      raw: { width: crop.width, height: crop.height, channels: 4 },
    })
      .png()
      .toBuffer();
    const composited = await sharp(imagePath).composite([{ input: overlay, left: crop.x, top: crop.y }]).png().toBuffer();
    await sharp(composited).png().toFile(imagePath);
  }
}

function createMask(
  width: number,
  height: number,
  padWidth: number,
  padHeight: number,
  segments: NormalizedAnnotationSegment[],
): Float32Array {
  const mask = new Float32Array(padWidth * padHeight);
  for (const segment of segments) {
    const expand = Math.max(0, segment.expand_px || 0);
    const x0 = clamp(segment.rect.x - expand, 0, width - 1);
    const y0 = clamp(segment.rect.y - expand, 0, height - 1);
    const x1 = clamp(segment.rect.x + segment.rect.width + expand, 0, width);
    const y1 = clamp(segment.rect.y + segment.rect.height + expand, 0, height);
    for (let y = y0; y < y1; y += 1) {
      const row = y * padWidth;
      for (let x = x0; x < x1; x += 1) {
        mask[row + x] = 1;
      }
    }
  }
  return mask;
}

function createMaskForCrop(
  crop: CropRect,
  targetWidth: number,
  targetHeight: number,
  segments: NormalizedAnnotationSegment[],
): Float32Array {
  const mask = new Float32Array(targetWidth * targetHeight);
  for (const segment of segments) {
    const expand = Math.max(0, segment.expand_px || 0);
    const x0 = clamp(Math.floor((segment.rect.x - expand - crop.x) * (targetWidth / crop.width)), 0, targetWidth - 1);
    const y0 = clamp(Math.floor((segment.rect.y - expand - crop.y) * (targetHeight / crop.height)), 0, targetHeight - 1);
    const x1 = clamp(Math.ceil((segment.rect.x + segment.rect.width + expand - crop.x) * (targetWidth / crop.width)), 0, targetWidth);
    const y1 = clamp(Math.ceil((segment.rect.y + segment.rect.height + expand - crop.y) * (targetHeight / crop.height)), 0, targetHeight);
    for (let y = y0; y < y1; y += 1) {
      const row = y * targetWidth;
      for (let x = x0; x < x1; x += 1) {
        mask[row + x] = 1;
      }
    }
  }
  return mask;
}

function createAlphaMaskForCrop(
  crop: CropRect,
  targetWidth: number,
  targetHeight: number,
  segments: NormalizedAnnotationSegment[],
): Buffer {
  const alpha = Buffer.alloc(targetWidth * targetHeight);
  for (const segment of segments) {
    const expand = Math.max(0, segment.expand_px || 0);
    const x0 = clamp(Math.floor((segment.rect.x - expand - crop.x) * (targetWidth / crop.width)), 0, targetWidth - 1);
    const y0 = clamp(Math.floor((segment.rect.y - expand - crop.y) * (targetHeight / crop.height)), 0, targetHeight - 1);
    const x1 = clamp(Math.ceil((segment.rect.x + segment.rect.width + expand - crop.x) * (targetWidth / crop.width)), 0, targetWidth);
    const y1 = clamp(Math.ceil((segment.rect.y + segment.rect.height + expand - crop.y) * (targetHeight / crop.height)), 0, targetHeight);
    for (let y = y0; y < y1; y += 1) {
      const row = y * targetWidth;
      for (let x = x0; x < x1; x += 1) {
        alpha[row + x] = 255;
      }
    }
  }
  return alpha;
}

function computeCropRect(segment: NormalizedAnnotationSegment, width: number, height: number): CropRect {
  const expand = Math.max(0, segment.expand_px || 0);
  const x0 = clamp(Math.floor(segment.rect.x - expand), 0, width);
  const y0 = clamp(Math.floor(segment.rect.y - expand), 0, height);
  const x1 = clamp(Math.ceil(segment.rect.x + segment.rect.width + expand), 0, width);
  const y1 = clamp(Math.ceil(segment.rect.y + segment.rect.height + expand), 0, height);
  const segmentWidth = Math.max(1, x1 - x0);
  const segmentHeight = Math.max(1, y1 - y0);
  const side = Math.min(Math.max(width, height), Math.max(160, Math.ceil(Math.max(segmentWidth, segmentHeight) * 2.5)));
  const cropWidth = Math.min(width, side);
  const cropHeight = Math.min(height, side);
  const centerX = (x0 + x1) / 2;
  const centerY = (y0 + y1) / 2;
  return {
    x: Math.round(clamp(centerX - cropWidth / 2, 0, width - cropWidth)),
    y: Math.round(clamp(centerY - cropHeight / 2, 0, height - cropHeight)),
    width: cropWidth,
    height: cropHeight,
  };
}

function rgbToNchwFloat(raw: Buffer, width: number, height: number): Float32Array {
  const out = new Float32Array(3 * width * height);
  const plane = width * height;
  for (let i = 0; i < plane; i += 1) {
    out[i] = raw[i * 3] / 255;
    out[plane + i] = raw[i * 3 + 1] / 255;
    out[plane * 2 + i] = raw[i * 3 + 2] / 255;
  }
  return out;
}

function nchwFloatToRgb(data: Float32Array, sourceWidth: number, sourceHeight: number, width: number, height: number): Buffer {
  const out = Buffer.alloc(width * height * 3);
  const plane = sourceWidth * sourceHeight;
  const scale = detectOutputScale(data);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const src = y * sourceWidth + x;
      const dst = (y * width + x) * 3;
      out[dst] = toByte(data[src], scale);
      out[dst + 1] = toByte(data[plane + src], scale);
      out[dst + 2] = toByte(data[plane * 2 + src], scale);
    }
  }
  void sourceHeight;
  return out;
}

function rgbaWithAlpha(rgb: Buffer, alpha: Buffer): Buffer {
  const out = Buffer.alloc(alpha.length * 4);
  for (let i = 0; i < alpha.length; i += 1) {
    out[i * 4] = rgb[i * 3];
    out[i * 4 + 1] = rgb[i * 3 + 1];
    out[i * 4 + 2] = rgb[i * 3 + 2];
    out[i * 4 + 3] = alpha[i];
  }
  return out;
}

function detectOutputScale(data: Float32Array): number {
  let max = 0;
  const step = Math.max(1, Math.floor(data.length / 8192));
  for (let i = 0; i < data.length; i += step) {
    max = Math.max(max, data[i]);
  }
  return max > 2 ? 1 : 255;
}

function alignTo(value: number, mod: number): number {
  return value % mod === 0 ? value : value + mod - (value % mod);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function toByte(value: number, scale: number): number {
  return Math.max(0, Math.min(255, Math.round(value * scale)));
}
