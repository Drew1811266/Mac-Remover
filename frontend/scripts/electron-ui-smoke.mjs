#!/usr/bin/env node
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdir, mkdtemp, rm, stat, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import { DEFAULT_FUNCTIONAL_ROOT, FRONTEND_ROOT } from './functional-fixtures.mjs';

class CdpSession {
  constructor(socket) {
    this.socket = socket;
    this.nextId = 1;
    this.pending = new Map();
    socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (!message.id) return;
      const pending = this.pending.get(message.id);
      if (!pending) return;
      this.pending.delete(message.id);
      if (message.error) pending.reject(new Error(message.error.message || JSON.stringify(message.error)));
      else pending.resolve(message.result || {});
    };
  }

  static connect(url) {
    return new Promise((resolve, reject) => {
      const socket = new WebSocket(url);
      socket.onopen = () => resolve(new CdpSession(socket));
      socket.onerror = () => reject(new Error(`Failed to connect to CDP websocket: ${url}`));
    });
  }

  send(method, params = {}) {
    const id = this.nextId;
    this.nextId += 1;
    const payload = JSON.stringify({ id, method, params });
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.socket.send(payload);
    });
  }

  close() {
    this.socket.close();
  }
}

const port = Number(process.env.WMR_ELECTRON_SMOKE_PORT || 9339);
const screenshotDir = path.join(DEFAULT_FUNCTIONAL_ROOT, 'screenshots');
const mainPath = path.join(FRONTEND_ROOT, 'dist-electron', 'electron', 'main.js');
const electronPath = resolveElectronBinary();
const userDataDir = await mkdtemp(path.join(os.tmpdir(), 'wmr-electron-ui-userdata-'));
const results = [];

let child;
let cdp;

try {
  await stat(mainPath);
  await mkdir(screenshotDir, { recursive: true });
  child = spawn(electronPath, [`--remote-debugging-port=${port}`, mainPath], {
    cwd: FRONTEND_ROOT,
    env: {
      ...process.env,
      WMR_MODEL_MANIFEST_URL: '',
      WMR_USER_DATA_DIR: userDataDir,
    },
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
  });

  child.stdout.on('data', (chunk) => process.stdout.write(chunk));
  child.stderr.on('data', (chunk) => process.stderr.write(chunk));

  const target = await waitForElectronTarget();
  cdp = await CdpSession.connect(target.webSocketDebuggerUrl);
  await cdp.send('Runtime.enable');
  await cdp.send('Page.enable');
  await waitForRendererReady();

  await check('renderer exposes window.wmr and loads the app shell', async () => {
    const state = await evaluateJson(`(() => ({
      title: document.title,
      hasShell: !!document.querySelector('.app-shell'),
      hasDesktopApi: !!window.wmr,
      apiMethods: Object.keys(window.wmr || {}).sort(),
      topTitle: document.querySelector('.app-window-title')?.textContent?.trim() || '',
      navText: Array.from(document.querySelectorAll('.app-nav-destination')).map((item) => item.textContent.trim()),
      bodyText: document.body.innerText.slice(0, 800)
    }))()`);
    assert.equal(state.hasShell, true);
    assert.equal(state.hasDesktopApi, true);
    assert.ok(state.apiMethods.includes('process_video'));
    assert.equal(state.topTitle, 'Mac Watermark Remover');
    assert.ok(state.navText.some((text) => text.includes('视频导入')));
    assert.ok(state.bodyText.includes('源视频预览'));
  });

  await check('top chrome has no right rail divider and main viewport is not page-scrolled', async () => {
    const layout = await evaluateJson(`(() => {
      const rail = document.querySelector('.app-nav-rail');
      const main = document.querySelector('.app-main-content');
      const top = document.querySelector('.app-top-bar');
      const railStyle = rail ? getComputedStyle(rail) : null;
      const topStyle = top ? getComputedStyle(top) : null;
      return {
        railBorderRightWidth: railStyle?.borderRightWidth || '',
        topBorderBottomWidth: topStyle?.borderBottomWidth || '',
        bodyOverflowY: document.documentElement.scrollHeight - document.documentElement.clientHeight,
        mainOverflowY: main ? main.scrollHeight - main.clientHeight : 0
      };
    })()`);
    assert.equal(layout.railBorderRightWidth, '0px');
    assert.equal(layout.topBorderBottomWidth, '0px');
    assert.ok(layout.bodyOverflowY <= 1, `body overflow ${layout.bodyOverflowY}px`);
    assert.ok(layout.mainOverflowY <= 1, `main overflow ${layout.mainOverflowY}px`);
  });

  await screenshot('process');

  await check('visible interactive controls have usable center hit targets on every main page', async () => {
    await clickByText('.app-nav-destination', '视频导入');
    await sleep(150);
    await assertInteractiveCenterHitTargets('视频导入');

    for (const destination of ['水印打标', '效果预览', '视频放大']) {
      await clickByText('.app-nav-destination', destination);
      await sleep(150);
      await assertInteractiveCenterHitTargets(destination);
    }
  });

  for (const destination of ['水印打标', '效果预览', '视频放大']) {
    await check(`navigation opens ${destination}`, async () => {
      await clickByText('.app-nav-destination', destination);
      await sleep(250);
      const text = await evaluateJson('document.body.innerText');
      assert.ok(String(text).includes(destination));
    });
    await screenshot(destination);
  }

  await check('annotation workspace fills the Electron desktop content width', async () => {
    await clickByText('.app-nav-destination', '水印打标');
    await sleep(250);
    const layout = await evaluateJson(`(() => {
      const main = document.querySelector('.app-main-content');
      const shell = document.querySelector('.workspace-shell');
      const stageWrap = document.querySelector('.workspace-stage-wrap');
      const stage = document.querySelector('.workspace-stage');

      function box(el) {
        const rect = el.getBoundingClientRect();
        const style = getComputedStyle(el);
        return {
          left: rect.left,
          width: rect.width,
          height: rect.height,
          innerWidth: rect.width - Number.parseFloat(style.paddingLeft || '0') - Number.parseFloat(style.paddingRight || '0'),
          innerHeight: rect.height - Number.parseFloat(style.paddingTop || '0') - Number.parseFloat(style.paddingBottom || '0'),
          paddingLeft: Number.parseFloat(style.paddingLeft || '0')
        };
      }

      return {
        main: box(main),
        shell: box(shell),
        stageWrap: box(stageWrap),
        stage: box(stage)
      };
    })()`);
    const mainContentLeft = layout.main.left + layout.main.paddingLeft;
    const expectedStageWidth = Math.min(layout.stageWrap.innerWidth, layout.stageWrap.innerHeight * (16 / 9));
    assert.ok(
      Math.abs(layout.shell.left - mainContentLeft) <= 2,
      `workspace shell is horizontally offset: ${JSON.stringify(layout)}`,
    );
    assert.ok(
      layout.shell.width >= layout.main.innerWidth - 2,
      `workspace shell should use the full content width: ${JSON.stringify(layout)}`,
    );
    assert.ok(
      layout.stage.width >= expectedStageWidth - 2,
      `workspace stage should expand to the available 16:9 box: ${JSON.stringify(layout)}`,
    );
  });

  await check('annotation frame drag creates only an ROI draft and never starts native image dragging', async () => {
    await clickByText('.app-nav-destination', '水印打标');
    await sleep(250);
    await evaluateJson(`(() => {
      const stage = document.querySelector('.workspace-stage');
      if (!stage) return false;
      const previous = stage.querySelector('.workspace-frame');
      if (previous) previous.remove();
      const frame = document.createElement('img');
      frame.className = 'workspace-frame';
      frame.alt = 'synthetic frame';
      frame.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 720"%3E%3Crect width="1280" height="720" fill="%23eef3f8"/%3E%3Crect x="900" y="40" width="260" height="90" fill="%23ffffff" opacity=".9"/%3E%3C/svg%3E';
      stage.appendChild(frame);
      return true;
    })()`);

    const frameState = await evaluateJson(`(() => {
      const frame = document.querySelector('.workspace-frame');
      const style = getComputedStyle(frame);
      const rect = frame.getBoundingClientRect();
      const hit = document.elementFromPoint(Math.round(rect.left + rect.width / 2), Math.round(rect.top + rect.height / 2));
      return {
        userDrag: style.webkitUserDrag || style.userDrag || '',
        pointerEvents: style.pointerEvents,
        centerHitClass: hit ? String(hit.className || '') : ''
      };
    })()`);
    assert.equal(frameState.userDrag, 'none', `workspace frame should disable native drag CSS: ${JSON.stringify(frameState)}`);
    assert.equal(frameState.pointerEvents, 'none', `workspace frame should let stage own pointer events: ${JSON.stringify(frameState)}`);
    assert.ok(!frameState.centerHitClass.includes('workspace-frame'), `workspace frame should not own pointer hit testing: ${JSON.stringify(frameState)}`);

    let state = await getAnnotationState();
    const initialRows = state.rows;
    await dragCenter('.workspace-stage', 130, 80);
    await sleep(200);
    state = await getAnnotationState();
    assert.equal(state.rows, initialRows, 'dragging the rendered frame should not auto-create a segment');
    assert.equal(state.hasDraft, true, 'dragging the rendered frame should leave an ROI draft');
    assert.equal(state.addDisabled, false, 'the ROI draft should enable Add Segment');
  });

  await check('annotation workspace keeps legacy draft-add-delete interaction semantics', async () => {
    await clickByText('.app-nav-destination', '视频导入');
    await sleep(150);
    await clickByText('.app-nav-destination', '水印打标');
    await sleep(250);

    let state = await getAnnotationState();
    assert.equal(state.rows, 0);
    assert.equal(state.hasDraft, false);

    await dragCenter('.workspace-stage', 110, 70);
    await sleep(200);
    state = await getAnnotationState();
    assert.equal(state.rows, 0, 'dragging the frame should only create a draft, not a segment');
    assert.equal(state.hasDraft, true);
    assert.equal(state.addDisabled, false);

    await clickByText('md-filled-button, md-filled-tonal-button, md-outlined-button, md-text-button, button', '新增标记段');
    await sleep(200);
    state = await getAnnotationState();
    assert.equal(state.rows, 1, 'Add Segment should commit the draft to the manager list');
    assert.equal(state.hasDraft, false);
    assert.match(state.text, /0 - 0|0 - \d+/);

    await clickCenter('.workspace-segment-enabled .md-switch');
    await sleep(150);
    state = await getAnnotationState();
    assert.match(state.text, /停用|Disabled/);

    await clickByText('md-text-button, button', '设起点');
    await clickByText('md-text-button, button', '设终点');
    await clickByText('md-text-button, button', '跳转');
    await pressKey('Delete');
    await sleep(200);
    state = await getAnnotationState();
    assert.equal(state.rows, 0, 'Delete should remove the selected segment');
  });

  await check('settings and manual panels open and close from the icon center point', async () => {
    await clickTopAction(1);
    await sleep(250);
    let state = await evaluateJson(`(() => ({
      hasModal: !!document.querySelector('.md-modal-layer'),
      text: document.body.innerText
    }))()`);
    assert.equal(state.hasModal, true);
    assert.match(state.text, /设置|Settings/);
    await assertInteractiveCenterHitTargets('设置弹层');
    await screenshot('settings-panel');
    await clickCenter('.settings-sidesheet .md-side-dialog-header .md-icon-button');
    await sleep(150);
    state = await evaluateJson(`(() => ({ hasModal: !!document.querySelector('.md-modal-layer') }))()`);
    assert.equal(state.hasModal, false, 'settings close icon center click did not close the panel');

    await clickTopAction(0);
    await sleep(250);
    state = await evaluateJson(`(() => ({
      hasModal: !!document.querySelector('.md-modal-layer'),
      text: document.body.innerText
    }))()`);
    assert.equal(state.hasModal, true);
    assert.match(state.text, /说明书|Manual|Guide/);
    await assertInteractiveCenterHitTargets('说明书弹层');
    await screenshot('manual-panel');
    await clickCenter('.manual-sidesheet .md-side-dialog-header .md-icon-button');
    await sleep(150);
    state = await evaluateJson(`(() => ({ hasModal: !!document.querySelector('.md-modal-layer') }))()`);
    assert.equal(state.hasModal, false, 'manual close icon center click did not close the panel');
  });

  await writeReport();
  printSummary();
  if (results.some((result) => result.status === 'failed')) process.exitCode = 1;
} finally {
  if (cdp) cdp.close();
  if (child && !child.killed) {
    child.kill('SIGTERM');
    await sleep(500);
    if (!child.killed) child.kill('SIGKILL');
  }
  await rm(userDataDir, { recursive: true, force: true });
}

function resolveElectronBinary() {
  if (process.env.ELECTRON_BINARY) return process.env.ELECTRON_BINARY;
  if (process.platform === 'darwin') {
    return path.join(FRONTEND_ROOT, 'node_modules', 'electron', 'dist', 'Electron.app', 'Contents', 'MacOS', 'Electron');
  }
  if (process.platform === 'win32') {
    return path.join(FRONTEND_ROOT, 'node_modules', 'electron', 'dist', 'electron.exe');
  }
  return path.join(FRONTEND_ROOT, 'node_modules', 'electron', 'dist', 'electron');
}

async function waitForElectronTarget() {
  const started = Date.now();
  while (Date.now() - started < 20_000) {
    try {
      const response = await fetch(`http://127.0.0.1:${port}/json`);
      const targets = await response.json();
      const target = targets.find((entry) => entry.type === 'page' && entry.webSocketDebuggerUrl);
      if (target) return target;
    } catch {
      // Electron may still be booting.
    }
    await sleep(150);
  }
  throw new Error(`Timed out waiting for Electron DevTools target on port ${port}`);
}

async function waitForRendererReady() {
  const started = Date.now();
  while (Date.now() - started < 20_000) {
    const ready = await evaluateJson(`(() => ({
      readyState: document.readyState,
      hasShell: !!document.querySelector('.app-shell'),
      bodyText: document.body.innerText.slice(0, 200)
    }))()`);
    if (ready.readyState !== 'loading' && ready.hasShell && ready.bodyText.length > 20) return;
    await sleep(150);
  }
  throw new Error('Timed out waiting for renderer app shell');
}

async function clickByText(selector, text) {
  const clicked = await evaluateJson(`(() => {
    const el = Array.from(document.querySelectorAll(${JSON.stringify(selector)}))
      .find((item) => item.textContent.includes(${JSON.stringify(text)}));
    if (!el) return false;
    el.click();
    return true;
  })()`);
  assert.equal(clicked, true, `Could not click ${selector} containing ${text}`);
}

async function clickByAriaLabel(label) {
  const clicked = await evaluateJson(`(() => {
    const el = document.querySelector('[aria-label=${JSON.stringify(label)}]');
    if (!el) return false;
    el.click();
    return true;
  })()`);
  assert.equal(clicked, true, `Could not click aria-label=${label}`);
}

async function clickTopAction(index) {
  const clicked = await evaluateJson(`(() => {
    const actions = Array.from(document.querySelectorAll('.app-top-actions .md-icon-button, .app-top-actions md-icon-button, .app-top-actions button'));
    const el = actions[${Number(index)}];
    if (!el) return false;
    el.click();
    return true;
  })()`);
  assert.equal(clicked, true, `Could not click top action index ${index}`);
}

async function getAnnotationState() {
  return evaluateJson(`(() => {
    const addButton = Array.from(document.querySelectorAll('md-filled-button, md-filled-tonal-button, md-outlined-button, md-text-button, button'))
      .find((el) => el.textContent.includes('新增标记段') || el.textContent.includes('Add Segment'));
    return {
      rows: document.querySelectorAll('.workspace-manager-table .md-inspector-row').length,
      hasDraft: !!document.querySelector('.workspace-segment.draft'),
      addDisabled: !addButton || addButton.hasAttribute('disabled') || addButton.disabled === true,
      text: document.body.innerText
    };
  })()`);
}

async function clickCenter(selector) {
  const point = await evaluateJson(`(() => {
    const el = document.querySelector(${JSON.stringify(selector)});
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    return {
      x: Math.round(rect.left + rect.width / 2),
      y: Math.round(rect.top + rect.height / 2),
      width: Math.round(rect.width),
      height: Math.round(rect.height)
    };
  })()`);
  assert.ok(point, `Could not find ${selector}`);
  assert.ok(point.width > 0 && point.height > 0, `${selector} has empty hit box`);
  await cdp.send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: point.x, y: point.y, button: 'none' });
  await cdp.send('Input.dispatchMouseEvent', {
    type: 'mousePressed',
    x: point.x,
    y: point.y,
    button: 'left',
    clickCount: 1,
  });
  await cdp.send('Input.dispatchMouseEvent', {
    type: 'mouseReleased',
    x: point.x,
    y: point.y,
    button: 'left',
    clickCount: 1,
  });
}

async function dragCenter(selector, deltaX, deltaY) {
  const point = await evaluateJson(`(() => {
    const el = document.querySelector(${JSON.stringify(selector)});
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    return {
      x: Math.round(rect.left + rect.width / 2),
      y: Math.round(rect.top + rect.height / 2),
      width: Math.round(rect.width),
      height: Math.round(rect.height)
    };
  })()`);
  assert.ok(point, `Could not find ${selector}`);
  assert.ok(point.width > 0 && point.height > 0, `${selector} has empty hit box`);
  const endX = point.x + Number(deltaX);
  const endY = point.y + Number(deltaY);
  await cdp.send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: point.x, y: point.y, button: 'none' });
  await cdp.send('Input.dispatchMouseEvent', {
    type: 'mousePressed',
    x: point.x,
    y: point.y,
    button: 'left',
    clickCount: 1,
  });
  await cdp.send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: endX, y: endY, button: 'left' });
  await cdp.send('Input.dispatchMouseEvent', {
    type: 'mouseReleased',
    x: endX,
    y: endY,
    button: 'left',
    clickCount: 1,
  });
}

async function pressKey(key) {
  const keyCodes = {
    Delete: 46,
    Space: 32,
    ArrowLeft: 37,
    ArrowRight: 39,
  };
  const windowsVirtualKeyCode = keyCodes[key] || 0;
  await cdp.send('Input.dispatchKeyEvent', { type: 'keyDown', windowsVirtualKeyCode, key, code: key });
  await cdp.send('Input.dispatchKeyEvent', { type: 'keyUp', windowsVirtualKeyCode, key, code: key });
}

async function assertInteractiveCenterHitTargets(scopeName) {
  const issues = await evaluateJson(`(() => {
    const modal = document.querySelector('.md-modal-layer');
    const scope = modal || document;
    const selector = [
      'button',
      'md-filled-button',
      'md-filled-tonal-button',
      'md-outlined-button',
      'md-text-button',
      'md-elevated-button',
      'md-icon-button',
      'md-switch',
      'md-checkbox',
      'md-slider',
      'md-outlined-select',
      '[role="button"]'
    ].join(',');

    function isDisabled(el) {
      return el.disabled === true || el.hasAttribute('disabled') || el.getAttribute('aria-disabled') === 'true';
    }

    function isVisible(el) {
      const style = getComputedStyle(el);
      const rect = el.getBoundingClientRect();
      return (
        style.display !== 'none' &&
        style.visibility !== 'hidden' &&
        Number(style.opacity || '1') > 0.01 &&
        rect.width >= 4 &&
        rect.height >= 4 &&
        rect.bottom > 0 &&
        rect.right > 0 &&
        rect.top < window.innerHeight &&
        rect.left < window.innerWidth
      );
    }

    function closestInteractive(el) {
      let current = el;
      while (current) {
        if (current.matches?.(selector)) return current;
        const root = current.getRootNode?.();
        current = current.parentElement || root?.host || null;
      }
      return null;
    }

    function nameFor(el) {
      return (
        el.getAttribute('aria-label') ||
        el.getAttribute('title') ||
        el.textContent?.trim().replace(/\\s+/g, ' ').slice(0, 80) ||
        el.tagName.toLowerCase()
      );
    }

    const candidates = Array.from(scope.querySelectorAll(selector));
    return candidates.flatMap((el, index) => {
      if (isDisabled(el) || !isVisible(el)) return [];
      const rect = el.getBoundingClientRect();
      const x = Math.round(rect.left + rect.width / 2);
      const y = Math.round(rect.top + rect.height / 2);
      const hit = document.elementFromPoint(x, y);
      const interactiveHit = hit ? closestInteractive(hit) : null;
      if (interactiveHit === el || (interactiveHit && el.contains(interactiveHit))) return [];
      return [
        {
          index,
          name: nameFor(el),
          tag: el.tagName.toLowerCase(),
          className: String(el.className || ''),
          rect: {
            left: Math.round(rect.left),
            top: Math.round(rect.top),
            width: Math.round(rect.width),
            height: Math.round(rect.height)
          },
          center: { x, y },
          hitTag: hit?.tagName?.toLowerCase() || null,
          hitClassName: hit ? String(hit.className || '') : '',
          hitText: hit?.textContent?.trim().replace(/\\s+/g, ' ').slice(0, 80) || ''
        }
      ];
    });
  })()`);

  assert.deepEqual(issues, [], `${scopeName} has blocked or misaligned interactive center targets: ${JSON.stringify(issues, null, 2)}`);
}

async function pressEscape() {
  await cdp.send('Input.dispatchKeyEvent', { type: 'keyDown', windowsVirtualKeyCode: 27, key: 'Escape', code: 'Escape' });
  await cdp.send('Input.dispatchKeyEvent', { type: 'keyUp', windowsVirtualKeyCode: 27, key: 'Escape', code: 'Escape' });
}

async function screenshot(name) {
  const safeName = encodeURIComponent(name).replace(/%/g, '').replace(/[^\w.-]+/g, '-');
  const result = await cdp.send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false });
  const outPath = path.join(screenshotDir, `electron-${safeName}.png`);
  await writeFile(outPath, Buffer.from(result.data, 'base64'));
  console.log(`Screenshot: ${outPath}`);
}

async function evaluateJson(expression) {
  const result = await cdp.send('Runtime.evaluate', {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  if (result.exceptionDetails) {
    throw new Error(result.exceptionDetails.text || 'Runtime.evaluate failed');
  }
  return result.result.value;
}

async function waitForCondition(expression, description, timeoutMs = 12_000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    if (await evaluateJson(expression)) return;
    await sleep(150);
  }
  throw new Error(`Timed out waiting for ${description}`);
}

async function check(name, fn) {
  try {
    await fn();
    results.push({ name, status: 'passed' });
    console.log(`PASS ${name}`);
  } catch (error) {
    const detail = error instanceof Error ? error.stack || error.message : String(error);
    results.push({ name, status: 'failed', detail });
    console.error(`FAIL ${name}`);
    console.error(detail);
  }
}

async function writeReport() {
  const reportPath = path.join(DEFAULT_FUNCTIONAL_ROOT, 'electron-ui-smoke-report.json');
  await mkdir(path.dirname(reportPath), { recursive: true });
  await writeFile(
    reportPath,
    `${JSON.stringify(
      {
        generated_at: new Date().toISOString(),
        platform: process.platform,
        arch: process.arch,
        screenshots: screenshotDir,
        results,
      },
      null,
      2,
    )}\n`,
    'utf8',
  );
  console.log(`Report: ${reportPath}`);
}

function printSummary() {
  const counts = results.reduce(
    (acc, result) => {
      acc[result.status] += 1;
      return acc;
    },
    { passed: 0, failed: 0 },
  );
  console.log(`Electron UI smoke summary: ${counts.passed} passed, ${counts.failed} failed`);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
