#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const mainPath = path.join(frontendRoot, 'dist-electron', 'electron', 'main.js');
const electronPath = resolveElectronBinary();

if (process.platform === 'darwin') {
  const child = spawn(electronPath, [mainPath, ...process.argv.slice(2)], {
    cwd: frontendRoot,
    env: process.env,
    detached: true,
    stdio: 'ignore',
  });
  child.unref();
  console.log(`Started development Electron (pid ${child.pid})`);
  process.exit(0);
}

const child = spawn(electronPath, [mainPath, ...process.argv.slice(2)], {
  cwd: frontendRoot,
  env: process.env,
  stdio: 'inherit',
});

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    if (!child.killed) child.kill(signal);
  });
}

child.on('exit', (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 0);
});

function resolveElectronBinary() {
  if (process.env.ELECTRON_BINARY) return process.env.ELECTRON_BINARY;
  const platformBinary =
    process.platform === 'darwin'
      ? path.join(frontendRoot, 'node_modules', 'electron', 'dist', 'Electron.app', 'Contents', 'MacOS', 'Electron')
      : process.platform === 'win32'
        ? path.join(frontendRoot, 'node_modules', 'electron', 'dist', 'electron.exe')
        : path.join(frontendRoot, 'node_modules', 'electron', 'dist', 'electron');

  if (!existsSync(platformBinary)) {
    throw new Error(`Electron binary not found: ${platformBinary}. Run npm install in frontend/.`);
  }
  return platformBinary;
}
