#!/usr/bin/env node
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, rm, stat, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import {
  DEFAULT_FUNCTIONAL_ROOT,
  FRONTEND_ROOT,
  PROJECT_ROOT,
  ensureFunctionalFixtures,
} from './functional-fixtures.mjs';

const args = new Set(process.argv.slice(2));
const requireProcessing = args.has('--require-processing');
const keepWorkdir = args.has('--keep-workdir');
const workRoot = path.resolve(process.env.WMR_FUNCTIONAL_ROOT || DEFAULT_FUNCTIONAL_ROOT);
const reportPath = path.join(workRoot, 'functional-smoke-report.json');
const userDataDir = await mkdtemp(path.join(os.tmpdir(), 'wmr-functional-userdata-'));
const outputDir = path.join(workRoot, 'outputs');

const results = [];
const progressEvents = [];

try {
  const fixtures = await ensureFunctionalFixtures({ root: workRoot });
  await mkdir(outputDir, { recursive: true });

  const services = await loadCompiledServices();
  const media = new services.MediaService({ userDataDir, appRoot: PROJECT_ROOT });
  const preview = new services.PreviewService({ userDataDir, appRoot: PROJECT_ROOT });
  const modelManager = new services.ModelManager({ userDataDir });
  const upscale = new services.UpscaleService({ userDataDir, appRoot: PROJECT_ROOT });

  await check('runtime tools resolve to executable FFmpeg and FFprobe', async () => {
    for (const tool of ['ffmpeg', 'ffprobe']) {
      const resolved = await services.resolveRuntimeTool({ tool, userDataDir, appRoot: PROJECT_ROOT });
      await services.runProcess(resolved.path, ['-version'], { timeoutMs: 10_000 });
      assert.match(resolved.path, new RegExp(tool));
    }
  });

  await check('media info accepts valid videos and rejects invalid files', async () => {
    const mp4 = await media.getMediaInfo(fixtures.fixtures.h264Aac.path);
    assert.equal(mp4.success, true);
    assert.equal(mp4.type, 'video');
    assert.equal(mp4.width, 1280);
    assert.equal(mp4.height, 720);
    assert.ok(Number(mp4.fps) > 0);
    assert.ok(Number(mp4.frame_count) > 0);

    const noAudio = await media.getMediaInfo(fixtures.fixtures.h264NoAudio.path);
    assert.equal(noAudio.success, true);
    assert.equal(noAudio.type, 'video');

    const webm = await media.getMediaInfo(fixtures.fixtures.webm.path);
    assert.equal(webm.success, true);
    assert.equal(webm.type, 'video');
    assert.equal(webm.width, 640);
    assert.equal(webm.height, 360);

    const invalid = await media.getMediaInfo(fixtures.fixtures.invalid.path);
    assert.equal(invalid.success, false);
    assert.ok(invalid.error);
  });

  await check('preview session opens, reads JPEG data URLs, and closes cleanly', async () => {
    const opened = await preview.openVideoPreviewSession(fixtures.fixtures.h264Aac.path, 12, 640);
    assert.equal(opened.success, true);
    assert.ok(opened.session_id);
    assert.equal(opened.width, 640);
    assert.ok(Number(opened.height) > 0);

    const first = await preview.readVideoPreviewFrame(opened.session_id, 0);
    assert.equal(first.success, true);
    assert.match(String(first.frame_url), /^data:image\/jpeg;base64,/);

    const wrapped = await preview.readVideoPreviewFrame(opened.session_id, 99999);
    assert.equal(wrapped.success, true);
    assert.ok(Number(wrapped.frame_index) >= 0);

    const closed = preview.closeVideoPreviewSession(opened.session_id);
    assert.equal(closed.success, true);
    const afterClose = await preview.readVideoPreviewFrame(opened.session_id, 0);
    assert.equal(afterClose.success, false);
  });

  await check('sidecar annotations save, normalize, reload, and delete', async () => {
    const videoPath = fixtures.fixtures.h264Aac.path;
    const meta = await media.getVideoMeta(videoPath);
    const saved = await services.saveAnnotations({
      videoPath,
      videoMeta: meta,
      segments: fixtures.segments.multi,
    });
    assert.equal(saved.success, true);
    assert.equal(saved.exists, true);
    assert.equal(saved.segments.length, fixtures.segments.multi.length);

    const sidecarRaw = JSON.parse(await services.readFile(services.buildSidecarPath(videoPath), 'utf8'));
    assert.equal(sidecarRaw.version, '1.0');
    assert.ok(sidecarRaw.video_meta);
    assert.ok(Array.isArray(sidecarRaw.segments));
    assert.ok(sidecarRaw.updated_at);

    const loaded = await services.loadAnnotations(videoPath, meta);
    assert.equal(loaded.success, true);
    assert.equal(loaded.exists, true);
    assert.equal(loaded.segments.length, fixtures.segments.multi.length);

    const edgeSaved = await services.saveAnnotations({
      videoPath,
      videoMeta: meta,
      segments: fixtures.segments.edge,
    });
    assert.equal(edgeSaved.success, true);
    const edge = edgeSaved.segments[0];
    assert.ok(edge.rect.x + edge.rect.width <= meta.width);
    assert.ok(edge.rect.y + edge.rect.height <= meta.height);

    const deleted = await services.deleteAnnotations(videoPath);
    assert.equal(deleted.success, true);
    const missing = await services.loadAnnotations(videoPath, meta);
    assert.equal(missing.success, true);
    assert.equal(missing.exists, false);
  });

  await check('settings persist language, theme, output directory, and sanitize removed model choices', async () => {
    const defaults = await services.loadSettings(userDataDir);
    assert.equal(defaults.language, 'zh');
    assert.equal(defaults.theme, 'light');
    assert.equal(defaults.output.model_id, 'lama_roi');

    const saved = await services.saveSettings(userDataDir, {
      language: 'en',
      theme: 'dark',
      output: { path: outputDir, model_id: 'propainter_roi' },
    });
    assert.equal(saved.success, true);

    const reloaded = await services.loadSettings(userDataDir);
    assert.equal(reloaded.language, 'en');
    assert.equal(reloaded.theme, 'dark');
    assert.equal(reloaded.output.path, outputDir);
    assert.equal(reloaded.output.model_id, 'lama_roi');
  });

  await check('model download status and invalid download errors are structured', async () => {
    const status = await modelManager.getStatus();
    assert.equal(status.success, true);
    assert.ok(status.models.some((entry) => entry.model_id === 'lama_roi'));
    assert.equal(status.models.some((entry) => entry.model_id === 'propainter_roi'), false);

    const invalid = await modelManager.startDownload('not_a_model');
    assert.equal(invalid.success, false);
    assert.match(String(invalid.error), /Invalid model_id/);

    const blocked = await modelManager.startDownload('lama_roi');
    assert.equal(blocked.success, false);
    assert.match(String(blocked.error), /WMR_MODEL_MANIFEST_URL/);

    const cancel = modelManager.cancelDownload();
    assert.equal(cancel.success, true);
  });

  await check('upscale capabilities and blocked native engines are explicit', async () => {
    const capabilities = await upscale.getCapabilities();
    assert.equal(capabilities.success, true);
    assert.ok(capabilities.engines.length >= 2);
    assert.ok(capabilities.engines.every((entry) => entry.available === false));
    assert.ok(capabilities.models.some((entry) => entry.model_id === 'realesrgan_x2plus'));
    assert.ok(capabilities.models.some((entry) => entry.model_id === 'seedvr2_3b_q4_k_m_gguf'));

    const download = await upscale.startModelDownload({ model_id: 'realesrgan_x2plus' });
    assert.equal(download.success, false);
    assert.ok(download.error);

    const start = await upscale.startUpscale({
      input_path: fixtures.fixtures.h264Aac.path,
      output_dir: outputDir,
      mode: 'enhance_same_resolution',
      engine: 'realesrgan',
      model_id: 'realesrgan_x2plus',
      keep_audio: true,
    });
    assert.equal(start.success, false);
    const task = await upscale.getTaskStatus();
    assert.equal(task.success, true);
    assert.equal(task.task.state, 'failed');
    assert.ok(task.task.error);
  });

  await check('processor rejects invalid and removed inpaint models without fallback', async () => {
    const processor = services.createVideoProcessor({
      userDataDir,
      appRoot: PROJECT_ROOT,
      emitProgress: (payload) => progressEvents.push(payload),
    });

    const invalid = await processor.processVideo({
      input_path: fixtures.fixtures.h264Aac.path,
      output_path: outputDir,
      annotation_segments: fixtures.segments.single,
      settings: { model_id: 'not_a_model' },
    });
    assert.equal(invalid.success, false);
    assert.match(String(invalid.error), /Invalid model_id/);

    const removed = await processor.processVideo({
      input_path: fixtures.fixtures.h264Aac.path,
      output_path: outputDir,
      annotation_segments: fixtures.segments.single,
      settings: { model_id: 'propainter_roi' },
    });
    assert.equal(removed.success, false);
    assert.match(String(removed.error), /Invalid model_id/);
  });

  await maybeRunLamaProcessingSmoke(fixtures, services, modelManager);

  await writeReport();
  printSummary(reportPath);
  if (results.some((result) => result.status === 'failed')) {
    process.exitCode = 1;
  }
} finally {
  if (!keepWorkdir) {
    await rm(userDataDir, { recursive: true, force: true });
  }
}

async function maybeRunLamaProcessingSmoke(fixtures, services, modelManager) {
  const lamaInstalled = await modelManager.isModelInstalled('lama_roi');
  if (!lamaInstalled) {
    const reason = `lamaInstalled=${lamaInstalled}`;
    if (requireProcessing) {
      await check('LaMa processing completes when required', async () => {
        assert.fail(`LaMa processing required but unavailable: ${reason}`);
      });
      return;
    }
    skip('LaMa processing completes when installed', reason);
    return;
  }

  await check('LaMa processing completes, emits monotonic progress, and preserves audio', async () => {
    const events = [];
    const processor = services.createVideoProcessor({
      userDataDir,
      appRoot: PROJECT_ROOT,
      emitProgress: (payload) => events.push(payload),
    });
    const result = await processor.processVideo({
      input_path: fixtures.fixtures.h264Aac.path,
      output_path: outputDir,
      annotation_segments: fixtures.segments.single,
      settings: { model_id: 'lama_roi' },
    });
    assert.equal(result.success, true);
    assert.ok(result.output_path);
    const outputStat = await stat(result.output_path);
    assert.ok(outputStat.size > 0);
    const media = new services.MediaService({ userDataDir, appRoot: PROJECT_ROOT });
    const info = await media.getMediaInfo(result.output_path);
    assert.equal(info.success, true);
    assert.equal(info.type, 'video');
    assert.equal(info.width, 1280);
    assert.equal(info.height, 720);
    assertProgressMonotonic(events);
    assert.equal(await hasAudioStream(result.output_path, services), true);
  });
}

async function hasAudioStream(videoPath, services) {
  const ffprobe = await services.resolveRuntimeTool({ tool: 'ffprobe', userDataDir, appRoot: PROJECT_ROOT });
  const result = await services.runProcess(ffprobe.path, [
    '-v',
    'error',
    '-select_streams',
    'a:0',
    '-show_entries',
    'stream=codec_type',
    '-of',
    'json',
    videoPath,
  ]);
  const parsed = JSON.parse(result.stdout || '{}');
  return Array.isArray(parsed.streams) && parsed.streams.some((stream) => stream.codec_type === 'audio');
}

function assertProgressMonotonic(events) {
  assert.ok(events.length > 0);
  let previous = -Infinity;
  for (const event of events) {
    if (typeof event.progress !== 'number') continue;
    assert.ok(event.progress >= previous, `progress regressed from ${previous} to ${event.progress}`);
    previous = event.progress;
  }
}

async function loadCompiledServices() {
  const serviceRoot = path.join(FRONTEND_ROOT, 'dist-electron', 'electron', 'services');
  try {
    await stat(path.join(serviceRoot, 'media.js'));
  } catch {
    throw new Error('Compiled Electron services are missing. Run `npm run electron:build` first.');
  }
  const [
    fs,
    media,
    preview,
    annotations,
    settings,
    modelManager,
    processor,
    nativeCore,
    ffmpeg,
    processRunner,
    upscale,
  ] = await Promise.all([
    import('node:fs/promises'),
    import('../dist-electron/electron/services/media.js'),
    import('../dist-electron/electron/services/preview.js'),
    import('../dist-electron/electron/services/annotations.js'),
    import('../dist-electron/electron/services/settings.js'),
    import('../dist-electron/electron/services/modelManager.js'),
    import('../dist-electron/electron/services/processor.js'),
    import('../dist-electron/electron/services/nativeCore.js'),
    import('../dist-electron/electron/services/ffmpeg.js'),
    import('../dist-electron/electron/services/processRunner.js'),
    import('../dist-electron/electron/services/upscale.js'),
  ]);
  return {
    ...fs,
    ...media,
    ...preview,
    ...annotations,
    ...settings,
    ...modelManager,
    ...processor,
    ...nativeCore,
    ...ffmpeg,
    ...processRunner,
    ...upscale,
  };
}

async function check(name, fn) {
  try {
    const detail = await fn();
    results.push({ name, status: 'passed', detail: detail || '' });
    console.log(`PASS ${name}`);
  } catch (error) {
    const message = error instanceof Error ? error.stack || error.message : String(error);
    results.push({ name, status: 'failed', detail: message });
    console.error(`FAIL ${name}`);
    console.error(message);
  }
}

function skip(name, detail) {
  results.push({ name, status: 'skipped', detail });
  console.log(`SKIP ${name}: ${detail}`);
}

async function writeReport() {
  await mkdir(path.dirname(reportPath), { recursive: true });
  await writeFile(
    reportPath,
    `${JSON.stringify(
      {
        generated_at: new Date().toISOString(),
        platform: process.platform,
        arch: process.arch,
        node: process.version,
        project_root: PROJECT_ROOT,
        work_root: workRoot,
        output_dir: outputDir,
        results,
      },
      null,
      2,
    )}\n`,
    'utf8',
  );
}

function printSummary(pathToReport) {
  const counts = results.reduce(
    (acc, result) => {
      acc[result.status] += 1;
      return acc;
    },
    { passed: 0, failed: 0, skipped: 0 },
  );
  console.log(`Functional smoke summary: ${counts.passed} passed, ${counts.failed} failed, ${counts.skipped} skipped`);
  console.log(`Report: ${pathToReport}`);
}
