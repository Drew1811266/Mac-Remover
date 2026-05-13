import { spawn } from 'node:child_process';

export interface RunProcessResult {
  stdout: string;
  stderr: string;
  stdoutBuffer: Buffer;
  stderrBuffer: Buffer;
}

export function runProcess(
  command: string,
  args: string[],
  options: { signal?: AbortSignal; timeoutMs?: number } = {},
): Promise<RunProcessResult> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true,
    });
    const stdout: Buffer[] = [];
    const stderr: Buffer[] = [];
    let timeout: NodeJS.Timeout | undefined;

    const cleanup = () => {
      if (timeout) clearTimeout(timeout);
      options.signal?.removeEventListener('abort', onAbort);
    };
    const onAbort = () => {
      child.kill('SIGTERM');
      cleanup();
      reject(new Error('Operation cancelled'));
    };

    if (options.signal) {
      if (options.signal.aborted) {
        onAbort();
        return;
      }
      options.signal.addEventListener('abort', onAbort, { once: true });
    }

    if (options.timeoutMs && options.timeoutMs > 0) {
      timeout = setTimeout(() => {
        child.kill('SIGTERM');
        cleanup();
        reject(new Error(`Process timed out: ${command}`));
      }, options.timeoutMs);
    }

    child.stdout.on('data', (chunk: Buffer) => stdout.push(chunk));
    child.stderr.on('data', (chunk: Buffer) => stderr.push(chunk));
    child.on('error', (error) => {
      cleanup();
      reject(error);
    });
    child.on('close', (code) => {
      cleanup();
      const stdoutBuffer = Buffer.concat(stdout);
      const stderrBuffer = Buffer.concat(stderr);
      const result = {
        stdout: stdoutBuffer.toString('utf8'),
        stderr: stderrBuffer.toString('utf8'),
        stdoutBuffer,
        stderrBuffer,
      };
      if (code === 0) {
        resolve(result);
      } else {
        reject(new Error(`${command} exited with ${code}: ${result.stderr.slice(-1200)}`));
      }
    });
  });
}
