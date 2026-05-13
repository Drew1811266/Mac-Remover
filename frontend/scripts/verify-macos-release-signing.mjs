#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';

const releaseDir = path.resolve(process.argv[2] || 'release');
const appPath = path.join(releaseDir, 'mac-arm64', 'Mac Watermark Remover.app');

if (!existsSync(appPath)) {
  fail(`Packaged macOS app not found: ${appPath}`);
}

const codesign = run('codesign', ['-dv', '--verbose=4', appPath]);
const signature = `${codesign.stdout}\n${codesign.stderr}`;

if (!signature.includes('flags=') || !signature.includes('runtime')) {
  fail('macOS app is not signed with hardened runtime enabled.');
}
if (signature.includes('Signature=adhoc')) {
  fail('macOS app is ad-hoc signed. Build with a Developer ID Application certificate.');
}
if (!/Authority=Developer ID Application:/u.test(signature)) {
  fail('macOS app is not signed by a Developer ID Application identity.');
}
if (/TeamIdentifier=not set/u.test(signature) || !/TeamIdentifier=/u.test(signature)) {
  fail('macOS app signature does not contain a TeamIdentifier.');
}

const stapler = run('xcrun', ['stapler', 'validate', appPath], { allowFailure: true });
const staplerOutput = `${stapler.stdout}\n${stapler.stderr}`;
if (stapler.status !== 0 || /does not have a ticket|could not validate|rejected/i.test(staplerOutput)) {
  fail('macOS app does not have a valid notarization ticket stapled.');
}

console.log(`macOS signing and notarization checks passed for ${appPath}`);

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: process.cwd(),
    encoding: 'utf8',
  });
  if (!options.allowFailure && result.status !== 0) {
    fail(`${command} ${args.join(' ')} failed:\n${result.stderr || result.stdout}`);
  }
  return result;
}

function fail(message) {
  console.error(message);
  process.exit(1);
}
