import { describe, expect, it, vi, beforeEach } from 'vitest';

const runProcessMock = vi.hoisted(() => vi.fn());

vi.mock('./ffmpeg.js', () => ({
  resolveRuntimeTool: vi.fn().mockResolvedValue({ path: '/usr/local/bin/ffmpeg' }),
}));

vi.mock('./media.js', () => ({
  MediaService: class {
    async getVideoMeta(input: string) {
      return {
        path: input,
        fps: 30,
        frame_count: 105,
        width: 1280,
        height: 704,
      };
    }
  },
}));

vi.mock('./processRunner.js', () => ({
  runProcess: runProcessMock,
}));

import { PreviewService } from './preview.js';

describe('preview service', () => {
  beforeEach(() => {
    runProcessMock.mockReset();
  });

  it('falls back to exact frame selection when fast seek cannot encode the requested frame', async () => {
    runProcessMock
      .mockRejectedValueOnce(new Error('ffmpeg exited with 234: Non full-range YUV is non-standard'))
      .mockResolvedValueOnce({
        stdout: '',
        stderr: '',
        stdoutBuffer: Buffer.from([1, 2, 3]),
        stderrBuffer: Buffer.alloc(0),
      });

    const service = new PreviewService({ userDataDir: '/tmp/wmr-user-data', appRoot: '/tmp/wmr-app' });
    const opened = await service.openVideoPreviewSession('/tmp/source.mp4', 30, 1280);
    expect(opened.success).toBe(true);

    const frame = await service.readVideoPreviewFrame(String(opened.session_id), 104);

    expect(frame.success).toBe(true);
    expect(frame.frame_index).toBe(104);
    expect(frame.frame_url).toBe('data:image/jpeg;base64,AQID');
    expect(runProcessMock).toHaveBeenCalledTimes(2);
    expect(runProcessMock.mock.calls[1][1]).toContain('select=eq(n\\,104),scale=1280:704,format=yuvj420p');
  });
});
