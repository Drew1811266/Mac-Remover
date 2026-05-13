#!/usr/bin/env node
import { ensureFunctionalFixtures } from './functional-fixtures.mjs';

const args = new Set(process.argv.slice(2));
const outputDirArg = process.argv.find((arg) => arg.startsWith('--output-dir='));

const result = await ensureFunctionalFixtures({
  force: args.has('--force'),
  root: outputDirArg ? outputDirArg.slice('--output-dir='.length) : undefined,
});

console.log(`Functional fixtures ready: ${result.manifestPath}`);
for (const fixture of Object.values(result.fixtures)) {
  console.log(`- ${fixture.name}: ${fixture.path}`);
}
