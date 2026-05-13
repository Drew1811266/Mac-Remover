import { createVideoProcessor, type ProcessVideoPayload, type VideoProcessorOptions } from './processor.js';

interface ProcessorChildRequest {
  payload: ProcessVideoPayload;
  options: Pick<VideoProcessorOptions, 'userDataDir' | 'appRoot'>;
}

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => {
  input += chunk;
});

process.stdin.on('end', async () => {
  try {
    const request = JSON.parse(input) as ProcessorChildRequest;
    const processor = createVideoProcessor({
      userDataDir: request.options.userDataDir,
      appRoot: request.options.appRoot,
      emitProgress: (payload) => {
        process.stdout.write(`${JSON.stringify({ type: 'progress', payload })}\n`);
      },
    });
    const result = await processor.processVideo(request.payload);
    process.stdout.write(`${JSON.stringify({ type: 'result', payload: result })}\n`);
  } catch (error) {
    process.stdout.write(
      `${JSON.stringify({
        type: 'result',
        payload: { success: false, error: error instanceof Error ? error.message : String(error) },
      })}\n`,
    );
  }
});
