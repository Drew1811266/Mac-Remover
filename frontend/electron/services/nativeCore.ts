import { access } from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);

export interface NativeCoreStatus {
  available: boolean;
  opencvAlgorithms: boolean;
  path: string;
  reason: string;
}

export function nativeCorePath(appRoot: string, platform = process.platform, arch = process.arch): string {
  return path.resolve(appRoot, 'native', 'prebuilds', `${platform}-${arch}`, nativeModuleName(platform));
}

export async function getNativeCoreStatus(appRoot: string): Promise<NativeCoreStatus> {
  const modulePath = nativeCorePath(appRoot);
  try {
    await access(modulePath);
  } catch {
    return {
      available: false,
      opencvAlgorithms: false,
      path: modulePath,
      reason: 'Prebuilt C++ Node-API core is missing for this platform.',
    };
  }

  try {
    const loaded = require(modulePath) as { getCapabilities?: () => { opencv_algorithms?: unknown } };
    const capabilities = typeof loaded.getCapabilities === 'function' ? loaded.getCapabilities() : {};
    const opencvAlgorithms = capabilities.opencv_algorithms === true;
    return {
      available: true,
      opencvAlgorithms,
      path: modulePath,
      reason: opencvAlgorithms ? '' : 'Prebuilt C++ core does not expose the required OpenCV-equivalent algorithms.',
    };
  } catch (error) {
    return {
      available: false,
      opencvAlgorithms: false,
      path: modulePath,
      reason: error instanceof Error ? error.message : String(error),
    };
  }
}

function nativeModuleName(platform: string): string {
  return platform === 'win32' ? 'wmr_native.node' : 'wmr_native.node';
}
