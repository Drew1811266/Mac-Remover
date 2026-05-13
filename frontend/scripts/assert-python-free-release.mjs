#!/usr/bin/env node
import { readdir, stat } from 'node:fs/promises';
import path from 'node:path';

const roots = process.argv.slice(2);
const targets = roots.length > 0 ? roots : ['release'];
const forbiddenSegments = [
  'python',
  'pywebview',
  'site-packages',
  'venv',
  '__pycache__',
  'torch',
  'torchvision',
];
const forbiddenExtensions = new Set(['.py', '.pyc', '.pyo']);
const findings = [];

for (const target of targets) {
  await scan(path.resolve(target), true);
}

if (findings.length > 0) {
  console.error('Python artifacts found in release output:');
  for (const finding of findings) {
    console.error(`- ${finding}`);
  }
  process.exit(1);
}

console.log(`Python-free release scan passed for ${targets.join(', ')}`);

async function scan(target, required = false) {
  let targetStat;
  try {
    targetStat = await stat(target);
  } catch {
    if (required) {
      findings.push(`${target} (missing release output)`);
    }
    return;
  }

  const normalized = target.split(path.sep).map((segment) => segment.toLowerCase());
  const base = path.basename(target).toLowerCase();
  if (normalized.some((segment) => forbiddenSegments.includes(segment)) || forbiddenExtensions.has(path.extname(base))) {
    findings.push(target);
  }

  if (!targetStat.isDirectory()) return;
  const entries = await readdir(target, { withFileTypes: true });
  await Promise.all(entries.map((entry) => scan(path.join(target, entry.name))));
}
