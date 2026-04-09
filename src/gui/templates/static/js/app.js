function app() {
  return {
    theme: initTheme(),
    mode: 'single',
    showAdvancedSettings: false,
    
    videoExtensions: ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'],
    currentFile: null,
    previewUrl: null,
    processedPath: null,
    processedUrl: null,
    dragOver: false,
    nativePlayerAvailable: false,
    sourcePlayer: null,
    processedPlayer: null,
    sourcePreviewMode: 'video',
    sourceSessionId: null,
    sourceFrameUrl: null,
    sourceFrameIndex: 0,
    sourceFrameTotal: 0,
    sourcePreviewFps: 12,
    sourceFrameTimer: null,
    sourceFrameBusy: false,
    processedPreviewMode: 'video',
    processedSessionId: null,
    processedFrameUrl: null,
    processedFrameIndex: 0,
    processedFrameTotal: 0,
    processedPreviewFps: 12,
    processedFrameTimer: null,
    processedFrameBusy: false,
    sourceMediaInfo: { type: '', fps: 0, frame_count: 0, duration: 0, width: 0, height: 0 },
    processedMediaInfo: { type: '', fps: 0, frame_count: 0, duration: 0, width: 0, height: 0 },
    sourceNativePosition: 0,
    sourceNativeDuration: 0,
    processedNativePosition: 0,
    processedNativeDuration: 0,
    
    isProcessing: false,
    isDetecting: false,
    isProcessed: false,
    progress: 0,
    statusMessage: '就绪',
    processedFrames: 0,
    totalFrames: 0,
    estimatedTime: '--:--',
    
    detectedMasks: [],
    
    settings: {
      videoType: 'general',
      soraOptimized: true,
      detectionMode: 'auto',
      detectionSkip: 3,
      sensitivity: 50,
      outputFormat: 'MP4',
      outputPath: '',
      outputQuality: 'high',
      useGpu: true,
      useFp16: true,
      batchSize: 4,
      handleFade: true,
      fadeIn: 0.5,
      fadeOut: 0.5
    },
    
    deviceInfo: 'M1 Pro (MPS)',
    memoryUsage: '4.2GB / 16GB',
    showAnnotationWorkspace: false,
    annotationSegments: [],
    selectedAnnotationId: null,
    annotationShowAll: true,
    annotationDraftRect: null,
    annotationCurrentFrame: 0,
    annotationPlaybackTimer: null,
    annotationStageMetrics: {
      left: 0,
      top: 0,
      width: 1,
      height: 1,
      videoLeft: 0,
      videoTop: 0,
      videoWidth: 1,
      videoHeight: 1
    },
    annotationDrawing: false,
    annotationDrawStart: null,
    annotationSegmentDrag: null,
    annotationDirty: false,
    
    get statusClass() {
      if (this.isProcessing) return 'processing';
      if (this.isProcessed) return 'done';
      return 'idle';
    },
    
    get statusText() {
      if (this.isProcessing) return '处理中...';
      if (this.isProcessed) return '完成';
      return '就绪';
    },
    
    get currentIsVideo() {
      return this.isVideoPath(this.currentFile);
    },
    
    get processedIsVideo() {
      return this.isVideoPath(this.processedPath);
    },

    get annotationFrameMax() {
      const frameCount = Number(this.sourceMediaInfo.frame_count) || Number(this.sourceFrameTotal) || 0;
      return Math.max(0, frameCount - 1);
    },
    
    init() {
      if (typeof window !== 'undefined') {
        window.__wmrApp = this;
      }
      this.loadSettings();
      this.detectNativePlayer();
      this.getDeviceInfo();
      setInterval(() => this.getDeviceInfo(), 5000);
      setInterval(() => this.pollNativePlayerState(), 500);
      window.addEventListener('wmr-progress', (event) => {
        if (event && event.detail) {
          this.updateProgress(event.detail);
        }
      });
      window.addEventListener('resize', () => this.syncAnnotationStageMetrics());
      window.addEventListener('mousemove', (event) => this.onAnnotationGlobalMouseMove(event));
      window.addEventListener('mouseup', () => this.onAnnotationGlobalMouseUp());
      window.addEventListener('keydown', (event) => this.onAnnotationKeydown(event));
      window.addEventListener('beforeunload', () => {
        this.pauseAnnotation();
        this.teardownPreviewSessions();
      });
    },
    
    toggleTheme() {
      this.theme = window.toggleTheme();
    },
    
    toFileUrl(path) {
      if (!path) return null;
      const normalized = String(path).replace(/\\/g, '/');
      const isWindowsPath = /^[a-zA-Z]:\//.test(normalized);
      const segments = normalized.split('/').map((segment, index) => {
        if (segment === '' && (index === 0 || (isWindowsPath && index === 1))) {
          return segment;
        }
        return encodeURIComponent(segment);
      });
      if (isWindowsPath) {
        return `file:///${segments.join('/')}`;
      }
      return `file://${segments.join('/')}`;
    },
    
    isVideoPath(path) {
      if (!path) return false;
      const lower = path.toLowerCase();
      return this.videoExtensions.some(ext => lower.endsWith(ext));
    },

    videoMimeType(path) {
      if (!path) return 'video/mp4';
      const lower = String(path).toLowerCase();
      if (lower.endsWith('.webm')) return 'video/webm';
      if (lower.endsWith('.mkv')) return 'video/x-matroska';
      if (lower.endsWith('.mov')) return 'video/quicktime';
      if (lower.endsWith('.avi')) return 'video/x-msvideo';
      if (lower.endsWith('.wmv')) return 'video/x-ms-wmv';
      if (lower.endsWith('.flv')) return 'video/x-flv';
      return 'video/mp4';
    },

    async detectNativePlayer() {
      this.nativePlayerAvailable = false;
      try {
        if (!window.pywebview || !pywebview.api || !pywebview.api.native_player_status) {
          return;
        }
        const status = await pywebview.api.native_player_status();
        if (status && status.success) {
          this.nativePlayerAvailable = !!status.available;
        }
      } catch (e) {
        console.warn('Detect native player error:', e);
      }
    },

    resetSourceMediaInfo() {
      this.sourceMediaInfo = { type: '', fps: 0, frame_count: 0, duration: 0, width: 0, height: 0 };
      this.sourceNativePosition = 0;
      this.sourceNativeDuration = 0;
    },

    resetProcessedMediaInfo() {
      this.processedMediaInfo = { type: '', fps: 0, frame_count: 0, duration: 0, width: 0, height: 0 };
      this.processedNativePosition = 0;
      this.processedNativeDuration = 0;
    },

    formatFps(value) {
      const fps = Number(value) || 0;
      if (fps <= 0) return '--';
      return fps >= 10 ? fps.toFixed(2).replace(/\.00$/, '') : fps.toFixed(3);
    },

    formatDuration(value) {
      const total = Math.max(0, Number(value) || 0);
      const h = Math.floor(total / 3600);
      const m = Math.floor((total % 3600) / 60);
      const s = Math.floor(total % 60);
      if (h > 0) {
        return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
      }
      return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    },

    newAnnotationId() {
      return `seg_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    },

    resetAnnotations() {
      this.pauseAnnotation();
      this.annotationSegments = [];
      this.selectedAnnotationId = null;
      this.annotationDraftRect = null;
      this.annotationCurrentFrame = 0;
      this.annotationDirty = false;
    },

    async ensureAnnotationFrameSession() {
      if (!this.currentFile || !this.currentIsVideo) return false;
      if (this.sourcePreviewMode !== 'frames' || !this.sourceSessionId) {
        await this.switchSourceToFramePreview();
      }
      return !!this.sourceSessionId;
    },

    async enterAnnotationWorkspace() {
      if (!this.currentFile || !this.currentIsVideo || this.mode === 'batch') return;
      const ok = await this.ensureAnnotationFrameSession();
      if (!ok) {
        this.statusMessage = '无法进入打标模式：预览会话初始化失败';
        return;
      }
      this.showAnnotationWorkspace = true;
      this.annotationCurrentFrame = Number(this.sourceFrameIndex) || 0;
      if (this.annotationSegments.length === 0) {
        await this.loadAnnotations();
      }
      if (typeof this.$nextTick === 'function') {
        await this.$nextTick();
      }
      this.syncAnnotationStageMetrics();
    },

    exitAnnotationWorkspace() {
      this.pauseAnnotation();
      this.showAnnotationWorkspace = false;
      this.annotationDrawing = false;
      this.annotationSegmentDrag = null;
    },

    syncAnnotationStageMetrics() {
      const stage = this.$refs ? this.$refs.annotationStage : null;
      if (!stage) return;
      const rect = stage.getBoundingClientRect();
      const stageWidth = Math.max(1, rect.width);
      const stageHeight = Math.max(1, rect.height);
      const image = this.$refs ? this.$refs.annotationImage : null;
      let mediaWidth = Number(this.sourceMediaInfo.width) || 0;
      let mediaHeight = Number(this.sourceMediaInfo.height) || 0;
      if ((mediaWidth <= 0 || mediaHeight <= 0) && image) {
        mediaWidth = Number(image.naturalWidth) || 0;
        mediaHeight = Number(image.naturalHeight) || 0;
      }

      let videoWidth = stageWidth;
      let videoHeight = stageHeight;
      let videoLeft = 0;
      let videoTop = 0;
      if (mediaWidth > 0 && mediaHeight > 0) {
        const stageRatio = stageWidth / stageHeight;
        const mediaRatio = mediaWidth / mediaHeight;
        if (stageRatio > mediaRatio) {
          videoHeight = stageHeight;
          videoWidth = Math.max(1, videoHeight * mediaRatio);
          videoLeft = (stageWidth - videoWidth) / 2;
          videoTop = 0;
        } else {
          videoWidth = stageWidth;
          videoHeight = Math.max(1, videoWidth / mediaRatio);
          videoLeft = 0;
          videoTop = (stageHeight - videoHeight) / 2;
        }
      }

      this.annotationStageMetrics = {
        left: rect.left,
        top: rect.top,
        width: stageWidth,
        height: stageHeight,
        videoLeft,
        videoTop,
        videoWidth: Math.max(1, videoWidth),
        videoHeight: Math.max(1, videoHeight)
      };
    },

    _annotationClampToVideoBox(x, y) {
      const m = this.annotationStageMetrics;
      const maxX = m.videoLeft + m.videoWidth;
      const maxY = m.videoTop + m.videoHeight;
      return {
        x: Math.max(m.videoLeft, Math.min(maxX, x)),
        y: Math.max(m.videoTop, Math.min(maxY, y))
      };
    },

    _clampFrameIndex(frame) {
      const n = Number(frame) || 0;
      return Math.max(0, Math.min(this.annotationFrameMax, Math.round(n)));
    },

    getVisibleAnnotationSegments() {
      if (this.annotationShowAll) return this.annotationSegments;
      return this.annotationSegments.filter((seg) => this.isAnnotationSegmentActive(seg));
    },

    isAnnotationSegmentActive(seg) {
      if (!seg || seg.enabled === false) return false;
      return this.annotationCurrentFrame >= seg.start_frame && this.annotationCurrentFrame <= seg.end_frame;
    },

    selectAnnotation(id) {
      this.selectedAnnotationId = id;
    },

    getSelectedAnnotation() {
      return this.annotationSegments.find((seg) => seg.id === this.selectedAnnotationId) || null;
    },

    _segmentDisplayRect(seg) {
      const metrics = this.annotationStageMetrics;
      const srcWidth = Math.max(1, Number(this.sourceMediaInfo.width) || 1);
      const srcHeight = Math.max(1, Number(this.sourceMediaInfo.height) || 1);
      const rect = seg && seg.rect ? seg.rect : { x: 0, y: 0, width: 0, height: 0 };
      const scaleX = metrics.videoWidth / srcWidth;
      const scaleY = metrics.videoHeight / srcHeight;

      const rawX = metrics.videoLeft + Number(rect.x) * scaleX;
      const rawY = metrics.videoTop + Number(rect.y) * scaleY;
      const maxX = metrics.videoLeft + metrics.videoWidth;
      const maxY = metrics.videoTop + metrics.videoHeight;
      const x = Math.max(metrics.videoLeft, Math.min(maxX, rawX));
      const y = Math.max(metrics.videoTop, Math.min(maxY, rawY));
      const width = Math.max(1, Math.min(maxX - x, Number(rect.width) * scaleX));
      const height = Math.max(1, Math.min(maxY - y, Number(rect.height) * scaleY));
      return { x, y, width, height };
    },

    annotationRectStyle(seg) {
      const r = this._segmentDisplayRect(seg);
      return {
        left: `${r.x}px`,
        top: `${r.y}px`,
        width: `${r.width}px`,
        height: `${r.height}px`
      };
    },

    annotationDraftRectStyle() {
      if (!this.annotationDraftRect) return {};
      const d = this.annotationDraftRect;
      const left = Math.min(d.x1, d.x2);
      const top = Math.min(d.y1, d.y2);
      const width = Math.max(1, Math.abs(d.x2 - d.x1));
      const height = Math.max(1, Math.abs(d.y2 - d.y1));
      return {
        left: `${left}px`,
        top: `${top}px`,
        width: `${width}px`,
        height: `${height}px`
      };
    },

    startAnnotationDraw(event) {
      if (!this.showAnnotationWorkspace) return;
      if (event && typeof event.button === 'number' && event.button !== 0) return;
      this.syncAnnotationStageMetrics();
      const m = this.annotationStageMetrics;
      const clamped = this._annotationClampToVideoBox(
        event.clientX - m.left,
        event.clientY - m.top
      );
      const x = clamped.x;
      const y = clamped.y;
      this.annotationDrawing = true;
      this.annotationDrawStart = { x, y };
      this.annotationDraftRect = { x1: x, y1: y, x2: x, y2: y };
    },

    moveAnnotationDraw(event) {
      if (!this.annotationDrawing || !this.annotationDrawStart) return;
      const m = this.annotationStageMetrics;
      const clamped = this._annotationClampToVideoBox(
        event.clientX - m.left,
        event.clientY - m.top
      );
      const x = clamped.x;
      const y = clamped.y;
      this.annotationDraftRect = {
        x1: this.annotationDrawStart.x,
        y1: this.annotationDrawStart.y,
        x2: x,
        y2: y
      };
    },

    endAnnotationDraw(event) {
      if (!this.annotationDrawing) return;
      if (event) {
        this.moveAnnotationDraw(event);
      }
      this.annotationDrawing = false;
      this.annotationDrawStart = null;

      if (!this.annotationDraftRect) return;
      const width = Math.abs(this.annotationDraftRect.x2 - this.annotationDraftRect.x1);
      const height = Math.abs(this.annotationDraftRect.y2 - this.annotationDraftRect.y1);
      if (width < 4 || height < 4) {
        this.annotationDraftRect = null;
      }
    },

    addAnnotationFromDraft() {
      if (!this.annotationDraftRect) return;
      const m = this.annotationStageMetrics;
      const srcWidth = Math.max(1, Number(this.sourceMediaInfo.width) || 1);
      const srcHeight = Math.max(1, Number(this.sourceMediaInfo.height) || 1);
      const scaleX = srcWidth / m.videoWidth;
      const scaleY = srcHeight / m.videoHeight;

      const left = Math.max(
        m.videoLeft,
        Math.min(this.annotationDraftRect.x1, this.annotationDraftRect.x2)
      );
      const top = Math.max(
        m.videoTop,
        Math.min(this.annotationDraftRect.y1, this.annotationDraftRect.y2)
      );
      const right = Math.min(
        m.videoLeft + m.videoWidth,
        Math.max(this.annotationDraftRect.x1, this.annotationDraftRect.x2)
      );
      const bottom = Math.min(
        m.videoTop + m.videoHeight,
        Math.max(this.annotationDraftRect.y1, this.annotationDraftRect.y2)
      );
      const width = Math.max(1, right - left);
      const height = Math.max(1, bottom - top);

      const x = Math.max(0, Math.round((left - m.videoLeft) * scaleX));
      const y = Math.max(0, Math.round((top - m.videoTop) * scaleY));
      const w = Math.max(1, Math.round(width * scaleX));
      const h = Math.max(1, Math.round(height * scaleY));

      const startFrame = this._clampFrameIndex(this.annotationCurrentFrame);
      const defaultSpan = Math.max(1, Math.round((Number(this.sourceMediaInfo.fps) || 24) * 2));
      const endFrame = this._clampFrameIndex(startFrame + defaultSpan);
      const now = new Date().toISOString();

      const seg = {
        id: this.newAnnotationId(),
        start_frame: Math.min(startFrame, endFrame),
        end_frame: Math.max(startFrame, endFrame),
        rect: { x, y, width: w, height: h },
        expand_px: 5,
        feather_px: 3,
        enabled: true,
        created_at: now,
        updated_at: now
      };
      this.annotationSegments = [...this.annotationSegments, seg];
      this.selectedAnnotationId = seg.id;
      this.annotationDraftRect = null;
      this.annotationDirty = true;
    },

    updateAnnotationSegment(id, updater) {
      let changed = false;
      this.annotationSegments = this.annotationSegments.map((seg) => {
        if (seg.id !== id) return seg;
        const next = typeof updater === 'function' ? updater(seg) : { ...seg, ...updater };
        changed = true;
        return { ...next, updated_at: new Date().toISOString() };
      });
      if (changed) {
        this.annotationDirty = true;
      }
    },

    removeAnnotation(id) {
      const before = this.annotationSegments.length;
      this.annotationSegments = this.annotationSegments.filter((seg) => seg.id !== id);
      if (this.selectedAnnotationId === id) {
        this.selectedAnnotationId = null;
      }
      if (this.annotationSegments.length !== before) {
        this.annotationDirty = true;
      }
    },

    async clearAnnotations() {
      this.annotationSegments = [];
      this.selectedAnnotationId = null;
      this.annotationDraftRect = null;
      this.annotationDrawing = false;
      this.annotationDirty = false;

      if (
        this.currentFile &&
        window.pywebview &&
        pywebview.api &&
        pywebview.api.delete_annotations
      ) {
        try {
          await pywebview.api.delete_annotations({ video_path: this.currentFile });
        } catch (e) {
          console.error('Delete annotations error:', e);
        }
      }
      this.statusMessage = '标注已清空';
    },

    annotationDurationText(seg) {
      if (!seg) return '--';
      const fps = Math.max(1, Number(this.sourceMediaInfo.fps) || Number(this.sourcePreviewFps) || 24);
      const frameSpan = Math.max(1, Number(seg.end_frame) - Number(seg.start_frame) + 1);
      const sec = frameSpan / fps;
      return `${frameSpan} 帧 / ${sec.toFixed(2)}s`;
    },

    toggleAnnotationEnabled(id, enabled) {
      this.updateAnnotationSegment(id, { enabled: !!enabled });
    },

    setAnnotationStart(id, frame) {
      const f = this._clampFrameIndex(frame);
      this.updateAnnotationSegment(id, (seg) => {
        const start = Math.min(f, seg.end_frame);
        return { ...seg, start_frame: start };
      });
    },

    setAnnotationEnd(id, frame) {
      const f = this._clampFrameIndex(frame);
      this.updateAnnotationSegment(id, (seg) => {
        const end = Math.max(f, seg.start_frame);
        return { ...seg, end_frame: end };
      });
    },

    async jumpToAnnotation(id) {
      const seg = this.annotationSegments.find((item) => item.id === id);
      if (!seg) return;
      this.selectedAnnotationId = id;
      await this.seekAnnotationFrame(seg.start_frame);
    },

    annotationTrackStyle(seg) {
      const max = Math.max(1, this.annotationFrameMax);
      const left = (Math.max(0, seg.start_frame) / max) * 100;
      const right = (Math.max(0, seg.end_frame) / max) * 100;
      const width = Math.max(1, right - left);
      return {
        left: `${left}%`,
        width: `${width}%`
      };
    },

    annotationTrackCursorStyle() {
      const max = Math.max(1, this.annotationFrameMax);
      const left = (Math.max(0, this.annotationCurrentFrame) / max) * 100;
      return { left: `${left}%` };
    },

    startAnnotationSegmentDrag(id, edge, event) {
      const track = this.$refs ? this.$refs.annotationTrack : null;
      if (!track) return;
      const seg = this.annotationSegments.find((item) => item.id === id);
      if (!seg) return;
      this.annotationSegmentDrag = {
        id,
        edge,
        startX: event.clientX,
        baseStart: seg.start_frame,
        baseEnd: seg.end_frame,
        trackWidth: Math.max(1, track.getBoundingClientRect().width)
      };
    },

    onAnnotationGlobalMouseMove(event) {
      if (!this.annotationSegmentDrag) return;
      const drag = this.annotationSegmentDrag;
      const frameMax = this.annotationFrameMax;
      const deltaPx = event.clientX - drag.startX;
      const deltaFrame = Math.round((deltaPx / drag.trackWidth) * Math.max(1, frameMax));

      if (drag.edge === 'start') {
        const nextStart = this._clampFrameIndex(drag.baseStart + deltaFrame);
        this.updateAnnotationSegment(drag.id, (seg) => ({
          ...seg,
          start_frame: Math.min(nextStart, seg.end_frame)
        }));
      } else if (drag.edge === 'end') {
        const nextEnd = this._clampFrameIndex(drag.baseEnd + deltaFrame);
        this.updateAnnotationSegment(drag.id, (seg) => ({
          ...seg,
          end_frame: Math.max(nextEnd, seg.start_frame)
        }));
      }
    },

    onAnnotationGlobalMouseUp() {
      this.annotationSegmentDrag = null;
    },

    async seekAnnotationFrame(value) {
      const frame = this._clampFrameIndex(value);
      const ok = await this.ensureAnnotationFrameSession();
      if (!ok) return;
      await this.seekSourceFrame(frame);
      this.annotationCurrentFrame = Number(this.sourceFrameIndex) || frame;
    },

    async stepAnnotation(delta) {
      const next = this._clampFrameIndex(this.annotationCurrentFrame + Number(delta || 0));
      await this.seekAnnotationFrame(next);
    },

    playAnnotation() {
      if (this.annotationPlaybackTimer) return;
      const fps = Math.max(1, Math.round(Number(this.sourceMediaInfo.fps) || Number(this.sourcePreviewFps) || 12));
      const intervalMs = Math.max(30, Math.round(1000 / fps));
      this.annotationPlaybackTimer = setInterval(async () => {
        if (this.annotationCurrentFrame >= this.annotationFrameMax) {
          this.pauseAnnotation();
          return;
        }
        await this.stepAnnotation(1);
      }, intervalMs);
    },

    pauseAnnotation() {
      if (this.annotationPlaybackTimer) {
        clearInterval(this.annotationPlaybackTimer);
        this.annotationPlaybackTimer = null;
      }
    },

    onAnnotationKeydown(event) {
      if (!this.showAnnotationWorkspace) return;
      const tag = (event.target && event.target.tagName) ? event.target.tagName.toLowerCase() : '';
      if (tag === 'input' || tag === 'textarea' || tag === 'select') return;

      if (event.code === 'Space') {
        event.preventDefault();
        if (this.annotationPlaybackTimer) this.pauseAnnotation();
        else this.playAnnotation();
        return;
      }
      if (event.key === 'ArrowLeft') {
        event.preventDefault();
        const step = event.shiftKey ? -10 : -1;
        this.stepAnnotation(step);
        return;
      }
      if (event.key === 'ArrowRight') {
        event.preventDefault();
        const step = event.shiftKey ? 10 : 1;
        this.stepAnnotation(step);
        return;
      }
      if (event.key === 'Delete' && this.selectedAnnotationId) {
        event.preventDefault();
        this.removeAnnotation(this.selectedAnnotationId);
      }
    },

    async saveAnnotations() {
      if (!this.currentFile || !window.pywebview || !pywebview.api || !pywebview.api.save_annotations) return;
      try {
        const result = await pywebview.api.save_annotations({
          video_path: this.currentFile,
          segments: this.annotationSegments,
          video_meta: {
            path: this.currentFile,
            width: Number(this.sourceMediaInfo.width) || 0,
            height: Number(this.sourceMediaInfo.height) || 0,
            fps: Number(this.sourceMediaInfo.fps) || 0,
            frame_count: Number(this.sourceMediaInfo.frame_count) || 0
          }
        });
        if (result && result.success) {
          if (Array.isArray(result.segments)) {
            this.annotationSegments = result.segments;
          }
          this.annotationDirty = false;
          this.statusMessage = `标注已保存 (${this.annotationSegments.length} 条)`;
        } else if (result && result.error) {
          this.statusMessage = `保存标注失败: ${result.error}`;
        }
      } catch (e) {
        console.error('Save annotations error:', e);
        this.statusMessage = '保存标注失败';
      }
    },

    async loadAnnotations() {
      if (!this.currentFile || !window.pywebview || !pywebview.api || !pywebview.api.load_annotations) return;
      try {
        const result = await pywebview.api.load_annotations({ video_path: this.currentFile });
        if (result && result.success) {
          if (Array.isArray(result.segments)) {
            this.annotationSegments = result.segments;
            this.selectedAnnotationId = this.annotationSegments.length > 0
              ? this.annotationSegments[this.annotationSegments.length - 1].id
              : null;
            this.annotationDirty = false;
          }
          if (result.warning) {
            this.statusMessage = result.warning;
          } else {
            this.statusMessage = `已加载标注 (${this.annotationSegments.length} 条)`;
          }
        } else if (result && result.error) {
          this.statusMessage = `加载标注失败: ${result.error}`;
        }
      } catch (e) {
        console.error('Load annotations error:', e);
        this.statusMessage = '加载标注失败';
      }
    },
    
    async loadSourceMediaInfo(path) {
      this.resetSourceMediaInfo();
      if (!path || !window.pywebview || !pywebview.api || !pywebview.api.get_media_info) return;
      try {
        const info = await pywebview.api.get_media_info({ path });
        if (info && info.success) {
          this.sourceMediaInfo = info;
          if (Number(info.fps) > 0) {
            this.sourcePreviewFps = Number(info.fps);
          }
          if (info.duration) {
            this.sourceNativeDuration = Number(info.duration) || 0;
          }
        }
      } catch (e) {
        console.warn('Load source media info error:', e);
      }
    },

    async loadProcessedMediaInfo(path) {
      this.resetProcessedMediaInfo();
      if (!path || !window.pywebview || !pywebview.api || !pywebview.api.get_media_info) return;
      try {
        const info = await pywebview.api.get_media_info({ path });
        if (info && info.success) {
          this.processedMediaInfo = info;
          if (Number(info.fps) > 0) {
            this.processedPreviewFps = Number(info.fps);
          }
          if (info.duration) {
            this.processedNativeDuration = Number(info.duration) || 0;
          }
        }
      } catch (e) {
        console.warn('Load processed media info error:', e);
      }
    },

    async openSourceNativePlayer(path = null, autoplay = false) {
      if (!this.nativePlayerAvailable) return false;
      try {
        const targetPath = path || this.currentFile;
        if (!targetPath || !this.isVideoPath(targetPath)) return false;
        const result = await pywebview.api.open_native_player({
          role: 'source',
          path: targetPath,
          title: '原始视频预览',
          autoplay: !!autoplay
        });
        if (result && result.success) {
          this.sourcePreviewMode = 'native';
          this.sourceNativeDuration = Number(result.duration || this.sourceMediaInfo.duration) || this.sourceNativeDuration;
          if (result.fps && Number(result.fps) > 0) {
            this.sourceMediaInfo.fps = Number(result.fps);
          }
          return true;
        }
      } catch (e) {
        console.warn('Open source native player error:', e);
      }
      return false;
    },

    async openProcessedNativePlayer(path = null, autoplay = false) {
      if (!this.nativePlayerAvailable) return false;
      try {
        const targetPath = path || this.processedPath;
        if (!targetPath || !this.isVideoPath(targetPath)) return false;
        const result = await pywebview.api.open_native_player({
          role: 'processed',
          path: targetPath,
          title: '处理后视频预览',
          autoplay: !!autoplay
        });
        if (result && result.success) {
          this.processedPreviewMode = 'native';
          this.processedNativeDuration = Number(result.duration || this.processedMediaInfo.duration) || this.processedNativeDuration;
          if (result.fps && Number(result.fps) > 0) {
            this.processedMediaInfo.fps = Number(result.fps);
          }
          return true;
        }
      } catch (e) {
        console.warn('Open processed native player error:', e);
      }
      return false;
    },
    
    async preparePreviewPath(path) {
      if (!path || !this.isVideoPath(path)) {
        return path;
      }
      
      try {
        if (!window.pywebview || !pywebview.api || !pywebview.api.prepare_video_preview) {
          return path;
        }
        
        const result = await pywebview.api.prepare_video_preview({ path });
        if (result && result.success && result.path) {
          if (result.warning) {
            console.warn('Prepare preview warning:', result.warning);
          }
          return result.path;
        }
        
        if (result && result.error) {
          console.warn('Prepare preview error:', result.error);
        }
      } catch (e) {
        console.error('Prepare preview path error:', e);
      }
      
      return path;
    },

    clearSourceFrameTimer() {
      if (this.sourceFrameTimer) {
        clearInterval(this.sourceFrameTimer);
        this.sourceFrameTimer = null;
      }
    },

    clearProcessedFrameTimer() {
      if (this.processedFrameTimer) {
        clearInterval(this.processedFrameTimer);
        this.processedFrameTimer = null;
      }
    },

    disposeSourcePlayer() {
      if (this.sourcePlayer && typeof this.sourcePlayer.dispose === 'function') {
        try {
          this.sourcePlayer.dispose();
        } catch (e) {
          console.warn('Dispose source player error:', e);
        }
      }
      this.sourcePlayer = null;
    },

    disposeProcessedPlayer() {
      if (this.processedPlayer && typeof this.processedPlayer.dispose === 'function') {
        try {
          this.processedPlayer.dispose();
        } catch (e) {
          console.warn('Dispose processed player error:', e);
        }
      }
      this.processedPlayer = null;
    },

    async mountSourcePlayer() {
      if (this.sourcePreviewMode !== 'video' || !this.currentIsVideo || !this.previewUrl) {
        this.disposeSourcePlayer();
        return;
      }
      if (typeof videojs === 'undefined') {
        console.warn('videojs is unavailable');
        return;
      }

      if (typeof this.$nextTick === 'function') {
        await this.$nextTick();
      }
      const el = this.$refs ? this.$refs.sourceVideo : null;
      if (!el) return;

      this.disposeSourcePlayer();
      const player = videojs(el, {
        controls: true,
        preload: 'auto',
        autoplay: false,
        fluid: false,
        responsive: false
      });
      player.on('error', () => {
        this.onSourceVideoError();
      });
      player.src({ src: this.previewUrl, type: this.videoMimeType(this.currentFile) });
      player.load();
      this.sourcePlayer = player;
    },

    async mountProcessedPlayer() {
      if (this.processedPreviewMode !== 'video' || !this.processedIsVideo || !this.processedUrl) {
        this.disposeProcessedPlayer();
        return;
      }
      if (typeof videojs === 'undefined') {
        console.warn('videojs is unavailable');
        return;
      }

      if (typeof this.$nextTick === 'function') {
        await this.$nextTick();
      }
      const el = this.$refs ? this.$refs.processedVideo : null;
      if (!el) return;

      this.disposeProcessedPlayer();
      const player = videojs(el, {
        controls: true,
        preload: 'auto',
        autoplay: false,
        fluid: false,
        responsive: false
      });
      player.on('error', () => {
        this.onProcessedVideoError();
      });
      player.src({ src: this.processedUrl, type: this.videoMimeType(this.processedPath) });
      player.load();
      this.processedPlayer = player;
    },

    async closeSourcePreviewSession() {
      this.clearSourceFrameTimer();
      this.sourceFrameBusy = false;
      this.disposeSourcePlayer();
      if (window.pywebview && pywebview.api && pywebview.api.close_native_player) {
        try {
          await pywebview.api.close_native_player({ role: 'source' });
        } catch (e) {
          console.warn('Close source native player error:', e);
        }
      }
      if (this.sourceSessionId && window.pywebview && pywebview.api && pywebview.api.close_video_preview_session) {
        try {
          await pywebview.api.close_video_preview_session({ session_id: this.sourceSessionId });
        } catch (e) {
          console.warn('Close source preview session error:', e);
        }
      }
      this.sourceSessionId = null;
      this.sourceFrameUrl = null;
      this.sourceFrameIndex = 0;
      this.sourceFrameTotal = 0;
      this.sourcePreviewFps = 12;
      this.sourceNativePosition = 0;
      this.sourceNativeDuration = 0;
      this.sourcePreviewMode = 'video';
    },

    async closeProcessedPreviewSession() {
      this.clearProcessedFrameTimer();
      this.processedFrameBusy = false;
      this.disposeProcessedPlayer();
      if (window.pywebview && pywebview.api && pywebview.api.close_native_player) {
        try {
          await pywebview.api.close_native_player({ role: 'processed' });
        } catch (e) {
          console.warn('Close processed native player error:', e);
        }
      }
      if (this.processedSessionId && window.pywebview && pywebview.api && pywebview.api.close_video_preview_session) {
        try {
          await pywebview.api.close_video_preview_session({ session_id: this.processedSessionId });
        } catch (e) {
          console.warn('Close processed preview session error:', e);
        }
      }
      this.processedSessionId = null;
      this.processedFrameUrl = null;
      this.processedFrameIndex = 0;
      this.processedFrameTotal = 0;
      this.processedPreviewFps = 12;
      this.processedNativePosition = 0;
      this.processedNativeDuration = 0;
      this.processedPreviewMode = 'video';
    },

    teardownPreviewSessions() {
      this.closeSourcePreviewSession();
      this.closeProcessedPreviewSession();
    },

    async switchSourceToFramePreview(reason) {
      if (!this.currentFile || !this.currentIsVideo) return;
      if (!window.pywebview || !pywebview.api || !pywebview.api.open_video_preview_session || !pywebview.api.read_video_preview_frame) {
        return;
      }

      await this.closeSourcePreviewSession();
      try {
        const targetFps = Math.max(1, Math.round(Number(this.sourceMediaInfo.fps) || 15));
        const opened = await pywebview.api.open_video_preview_session({
          path: this.currentFile,
          target_fps: targetFps,
          max_width: 640
        });
        if (!opened || !opened.success || !opened.session_id) {
          return;
        }

        this.sourceSessionId = opened.session_id;
        this.sourceFrameTotal = opened.total_preview_frames || 0;
        this.sourcePreviewFps = opened.preview_fps || 12;
        this.sourcePreviewMode = 'frames';
        this.sourceFrameIndex = 0;
        await this.fetchSourceFrame();
        if (reason) {
          this.statusMessage = reason;
        }
      } catch (e) {
        console.error('Switch source to frame preview error:', e);
      }
    },

    async switchProcessedToFramePreview(reason) {
      if (!this.processedPath || !this.processedIsVideo) return;
      if (!window.pywebview || !pywebview.api || !pywebview.api.open_video_preview_session || !pywebview.api.read_video_preview_frame) {
        return;
      }

      await this.closeProcessedPreviewSession();
      try {
        const targetFps = Math.max(1, Math.round(Number(this.processedMediaInfo.fps) || 15));
        const opened = await pywebview.api.open_video_preview_session({
          path: this.processedPath,
          target_fps: targetFps,
          max_width: 640
        });
        if (!opened || !opened.success || !opened.session_id) {
          return;
        }

        this.processedSessionId = opened.session_id;
        this.processedFrameTotal = opened.total_preview_frames || 0;
        this.processedPreviewFps = opened.preview_fps || 12;
        this.processedPreviewMode = 'frames';
        this.processedFrameIndex = 0;
        await this.fetchProcessedFrame();
        if (reason) {
          this.statusMessage = reason;
        }
      } catch (e) {
        console.error('Switch processed to frame preview error:', e);
      }
    },

    async fetchSourceFrame(targetIndex = null) {
      if (!this.sourceSessionId || this.sourceFrameBusy) return;
      this.sourceFrameBusy = true;
      try {
        const payload = { session_id: this.sourceSessionId };
        if (targetIndex !== null && targetIndex !== undefined) {
          payload.frame_index = Number(targetIndex);
        }
        const frame = await pywebview.api.read_video_preview_frame(payload);
        if (frame && frame.success && frame.frame_url) {
          this.sourceFrameUrl = frame.frame_url;
          if (typeof frame.frame_index === 'number') {
            this.sourceFrameIndex = frame.frame_index;
            if (this.showAnnotationWorkspace) {
              this.annotationCurrentFrame = frame.frame_index;
            }
          }
        }
      } catch (e) {
        console.error('Fetch source frame error:', e);
      } finally {
        this.sourceFrameBusy = false;
      }
    },

    async fetchProcessedFrame(targetIndex = null) {
      if (!this.processedSessionId || this.processedFrameBusy) return;
      this.processedFrameBusy = true;
      try {
        const payload = { session_id: this.processedSessionId };
        if (targetIndex !== null && targetIndex !== undefined) {
          payload.frame_index = Number(targetIndex);
        }
        const frame = await pywebview.api.read_video_preview_frame(payload);
        if (frame && frame.success && frame.frame_url) {
          this.processedFrameUrl = frame.frame_url;
          if (typeof frame.frame_index === 'number') {
            this.processedFrameIndex = frame.frame_index;
          }
        }
      } catch (e) {
        console.error('Fetch processed frame error:', e);
      } finally {
        this.processedFrameBusy = false;
      }
    },

    async seekSourceFrame(value) {
      if (!this.sourceSessionId) return;
      this.clearSourceFrameTimer();
      const index = Math.max(0, Number(value) || 0);
      this.sourceFrameIndex = index;
      await this.fetchSourceFrame(index);
    },

    async seekProcessedFrame(value) {
      if (!this.processedSessionId) return;
      this.clearProcessedFrameTimer();
      const index = Math.max(0, Number(value) || 0);
      this.processedFrameIndex = index;
      await this.fetchProcessedFrame(index);
    },

    async focusSourceNativePlayer() {
      if (!this.currentFile || !this.currentIsVideo) return;
      await this.openSourceNativePlayer(this.currentFile, false);
    },

    async focusProcessedNativePlayer() {
      if (!this.processedPath || !this.processedIsVideo) return;
      await this.openProcessedNativePlayer(this.processedPath, false);
    },

    async seekSourceNative(seconds) {
      if (this.sourcePreviewMode !== 'native') return;
      const target = Math.max(0, Number(seconds) || 0);
      this.sourceNativePosition = target;
      if (!window.pywebview || !pywebview.api || !pywebview.api.native_player_seek) return;
      try {
        await pywebview.api.native_player_seek({ role: 'source', seconds: target });
      } catch (e) {
        console.warn('Seek source native player error:', e);
      }
    },

    async seekProcessedNative(seconds) {
      if (this.processedPreviewMode !== 'native') return;
      const target = Math.max(0, Number(seconds) || 0);
      this.processedNativePosition = target;
      if (!window.pywebview || !pywebview.api || !pywebview.api.native_player_seek) return;
      try {
        await pywebview.api.native_player_seek({ role: 'processed', seconds: target });
      } catch (e) {
        console.warn('Seek processed native player error:', e);
      }
    },

    async pollNativePlayerState() {
      if (!this.nativePlayerAvailable || !window.pywebview || !pywebview.api || !pywebview.api.native_player_state) {
        return;
      }

      try {
        if (this.sourcePreviewMode === 'native') {
          const sourceState = await pywebview.api.native_player_state({ role: 'source' });
          if (sourceState && sourceState.success) {
            this.sourceNativePosition = Number(sourceState.position) || 0;
            this.sourceNativeDuration = Number(sourceState.duration) || this.sourceNativeDuration;
          }
        }
      } catch (e) {
        console.warn('Poll source native player state error:', e);
      }

      try {
        if (this.processedPreviewMode === 'native') {
          const processedState = await pywebview.api.native_player_state({ role: 'processed' });
          if (processedState && processedState.success) {
            this.processedNativePosition = Number(processedState.position) || 0;
            this.processedNativeDuration = Number(processedState.duration) || this.processedNativeDuration;
          }
        }
      } catch (e) {
        console.warn('Poll processed native player state error:', e);
      }
    },
    
    async loadSourcePreview(path) {
      await this.closeSourcePreviewSession();
      this.previewUrl = this.toFileUrl(path);
      
      if (!this.isVideoPath(path)) {
        return;
      }
      
      const selectedPath = path;
      if (this.nativePlayerAvailable) {
        const opened = await this.openSourceNativePlayer(path, false);
        if (opened && this.currentFile === selectedPath) {
          const fpsText = this.formatFps(this.sourceMediaInfo.fps);
          this.statusMessage = `已启用高性能原生播放器预览（原始帧率 ${fpsText} FPS）`;
          return;
        }
      }
      const previewPath = await this.preparePreviewPath(path);
      if (this.currentFile === selectedPath) {
        this.sourcePreviewMode = 'video';
        this.previewUrl = this.toFileUrl(previewPath);
        await this.mountSourcePlayer();
      }
    },
    
    async selectFile() {
      try {
        const result = await pywebview.api.select_file();
        if (result) {
          this.exitAnnotationWorkspace();
          this.resetAnnotations();
          await this.closeProcessedPreviewSession();
          this.resetProcessedMediaInfo();
          this.mode = 'single';
          this.currentFile = result.path;
          this.isProcessed = false;
          this.processedPath = null;
          this.processedUrl = null;
          this.detectedMasks = [];
          await this.loadSourceMediaInfo(result.path);
          await this.loadSourcePreview(result.path);
        }
      } catch (e) {
        console.error('Select file error:', e);
      }
    },
    
    async selectFolder() {
      try {
        const result = await pywebview.api.select_folder();
        if (result) {
          this.exitAnnotationWorkspace();
          this.resetAnnotations();
          await this.closeSourcePreviewSession();
          await this.closeProcessedPreviewSession();
          this.resetSourceMediaInfo();
          this.resetProcessedMediaInfo();
          this.mode = 'batch';
          this.currentFile = result.path;
          this.previewUrl = null;
          this.isProcessed = false;
          this.processedPath = null;
          this.processedUrl = null;
          this.detectedMasks = [];
        }
      } catch (e) {
        console.error('Select folder error:', e);
      }
    },
    
    async handleDrop(event) {
      this.dragOver = false;
      let droppedPath = null;
      const files = event.dataTransfer.files;
      if (files.length > 0) {
        const file = files[0];
        droppedPath = file.path || null;
      }
      
      if (!droppedPath) {
        const uriList = event.dataTransfer.getData('text/uri-list') || event.dataTransfer.getData('text/plain');
        if (uriList && uriList.startsWith('file://')) {
          droppedPath = decodeURIComponent(uriList.replace(/^file:\/\//, '').split('\\n')[0].trim());
        }
      }
      
      if (!droppedPath) {
        console.error('Drop file error: unable to resolve local file path');
        return;
      }
      
      this.currentFile = droppedPath;
      this.mode = 'single';
      this.exitAnnotationWorkspace();
      this.resetAnnotations();
      await this.closeProcessedPreviewSession();
      this.resetProcessedMediaInfo();
      await this.loadSourceMediaInfo(droppedPath);
      await this.loadSourcePreview(droppedPath);
      this.isProcessed = false;
      this.processedPath = null;
      this.processedUrl = null;
      this.detectedMasks = [];
    },
    
    async selectOutputPath() {
      try {
        const result = await pywebview.api.select_folder();
        if (result) {
          this.settings.outputPath = result.path;
        }
      } catch (e) {
        console.error('Select output path error:', e);
      }
    },
    
    async detectWatermark() {
      if (!this.currentFile || this.isDetecting) return;
      
      this.isDetecting = true;
      this.statusMessage = '正在检测水印...';
      
      try {
        const result = await pywebview.api.detect_watermark({
          path: this.currentFile
        });
        
        if (result && result.success && result.masks) {
          this.detectedMasks = result.masks;
          this.statusMessage = `检测到 ${result.masks.length} 个水印区域`;
        } else if (result && result.error) {
          this.statusMessage = `检测失败: ${result.error}`;
        }
      } catch (e) {
        console.error('Detect watermark error:', e);
        this.statusMessage = '检测失败';
      } finally {
        this.isDetecting = false;
      }
    },
    
    addManualMask() {
      this.detectedMasks.push({
        x: 100,
        y: 100,
        width: 200,
        height: 50
      });
    },
    
    clearMasks() {
      this.detectedMasks = [];
    },
    
    async previewDetection() {
      await this.detectWatermark();
    },
    
    async startProcessing() {
      if (!this.currentFile || this.isProcessing) return;
      
      this.isProcessing = true;
      this.isProcessed = false;
      this.progress = 0;
      this.statusMessage = '正在处理...';
      this.processedFrames = 0;
      this.totalFrames = 0;
      this.estimatedTime = '--:--';
      
      try {
        if (this.mode === 'batch') {
          const batchResult = await pywebview.api.process_batch({
            input_path: this.currentFile,
            output_path: this.settings.outputPath,
            settings: this.settings,
            sora_mode: this.settings.videoType === 'sora'
          });
          
          if (batchResult && batchResult.success) {
            this.isProcessed = true;
            this.processedUrl = null;
            this.statusMessage = `批量处理完成 (${batchResult.success_count}/${batchResult.total})`;
          } else if (batchResult && batchResult.error) {
            this.statusMessage = '处理失败: ' + batchResult.error;
          }
          return;
        }
        
        const lowerPath = this.currentFile.toLowerCase();
        const videoExts = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'];
        const isVideo = videoExts.some(ext => lowerPath.endsWith(ext));
        const enabledAnnotationSegments = this.annotationSegments.filter((seg) => seg && seg.enabled !== false);
        let result;
        
        if (isVideo) {
          if (this.showAnnotationWorkspace) {
            this.pauseAnnotation();
            if (enabledAnnotationSegments.length === 0) {
              this.statusMessage = '请先在打标工作台添加至少 1 条启用标记';
              return;
            }
          }
          if (enabledAnnotationSegments.length === 0) {
            this.statusMessage = '未设置人工标注，当前将使用已有检测区域处理';
          } else {
            this.statusMessage = `按人工标注处理（${enabledAnnotationSegments.length} 段）...`;
          }
          result = await pywebview.api.process_video({
            input_path: this.currentFile,
            output_path: this.settings.outputPath,
            masks: enabledAnnotationSegments.length > 0 ? [] : this.detectedMasks,
            annotation_segments: enabledAnnotationSegments,
            settings: this.settings,
            sora_mode: this.settings.videoType === 'sora'
          });
        } else {
          result = await pywebview.api.process_image({
            input_path: this.currentFile,
            output_path: this.settings.outputPath,
            masks: this.detectedMasks
          });
        }
        
        if (result && result.success && result.output_path) {
          this.processedPath = result.output_path;
          await this.loadProcessedMediaInfo(result.output_path);
          await this.closeProcessedPreviewSession();
          if (this.nativePlayerAvailable && this.isVideoPath(result.output_path)) {
            const opened = await this.openProcessedNativePlayer(result.output_path, false);
            if (!opened) {
              const processedPreviewPath = await this.preparePreviewPath(result.output_path);
              this.processedUrl = this.toFileUrl(processedPreviewPath);
              this.processedPreviewMode = 'video';
              await this.mountProcessedPlayer();
            } else {
              this.processedUrl = this.toFileUrl(result.output_path);
            }
          } else {
            const processedPreviewPath = await this.preparePreviewPath(result.output_path);
            this.processedUrl = this.toFileUrl(processedPreviewPath);
            this.processedPreviewMode = 'video';
            await this.mountProcessedPlayer();
          }
          this.isProcessed = true;
          if (this.isVideoPath(result.output_path) && Number(this.processedMediaInfo.fps) > 0) {
            this.statusMessage = `处理完成（输出帧率 ${this.formatFps(this.processedMediaInfo.fps)} FPS）`;
          } else {
            this.statusMessage = '处理完成';
          }
        } else if (result && result.error) {
          this.statusMessage = '处理失败: ' + result.error;
        }
      } catch (e) {
        console.error('Process error:', e);
        this.statusMessage = '处理失败: ' + e.message;
      } finally {
        this.isProcessing = false;
      }
    },
    
    async stopProcessing() {
      try {
        await pywebview.api.stop_processing();
        this.isProcessing = false;
        this.statusMessage = '已停止';
      } catch (e) {
        console.error('Stop processing error:', e);
      }
    },
    
    async openOutputDir() {
      try {
        await pywebview.api.open_output_dir();
      } catch (e) {
        console.error('Open output dir error:', e);
      }
    },
    
    async playSource() {
      if (this.sourcePreviewMode === 'native') {
        if (window.pywebview && pywebview.api && pywebview.api.native_player_play) {
          const result = await pywebview.api.native_player_play({ role: 'source' });
          if (result && result.success) return;
        }
        await this.openSourceNativePlayer(this.currentFile, true);
        return;
      }

      if (this.sourcePreviewMode === 'frames') {
        if (!this.sourceSessionId) {
          await this.switchSourceToFramePreview('已切换到兼容逐帧预览模式');
        }
        if (!this.sourceSessionId || this.sourceFrameTimer) return;
        const intervalMs = Math.max(40, Math.floor(1000 / Math.max(1, this.sourcePreviewFps)));
        this.sourceFrameTimer = setInterval(() => {
          this.fetchSourceFrame();
        }, intervalMs);
        return;
      }

      if (this.sourcePlayer) {
        try {
          await this.sourcePlayer.play();
        } catch (e) {
          console.error('Play source video error:', e);
          await this.switchSourceToFramePreview('HTML5播放器启动失败，已切换到兼容逐帧预览');
        }
        return;
      }

      const nativeVideo = this.$refs ? this.$refs.sourceVideo : null;
      if (nativeVideo) {
        try {
          await nativeVideo.play();
        } catch (e) {
          console.error('Play native source video error:', e);
          await this.switchSourceToFramePreview('HTML5播放器启动失败，已切换到兼容逐帧预览');
        }
      }
    },
    
    pauseSource() {
      if (this.sourcePreviewMode === 'native') {
        if (window.pywebview && pywebview.api && pywebview.api.native_player_pause) {
          pywebview.api.native_player_pause({ role: 'source' }).catch((e) => {
            console.warn('Pause source native player error:', e);
          });
        }
        return;
      }

      if (this.sourcePreviewMode === 'frames') {
        this.clearSourceFrameTimer();
        return;
      }
      if (this.sourcePlayer) {
        this.sourcePlayer.pause();
        return;
      }
      const nativeVideo = this.$refs ? this.$refs.sourceVideo : null;
      if (nativeVideo) {
        nativeVideo.pause();
      }
    },
    
    async playProcessed() {
      if (this.processedPreviewMode === 'native') {
        if (window.pywebview && pywebview.api && pywebview.api.native_player_play) {
          const result = await pywebview.api.native_player_play({ role: 'processed' });
          if (result && result.success) return;
        }
        await this.openProcessedNativePlayer(this.processedPath, true);
        return;
      }

      if (this.processedPreviewMode === 'frames') {
        if (!this.processedSessionId) {
          await this.switchProcessedToFramePreview('已切换到处理后兼容逐帧预览模式');
        }
        if (!this.processedSessionId || this.processedFrameTimer) return;
        const intervalMs = Math.max(40, Math.floor(1000 / Math.max(1, this.processedPreviewFps)));
        this.processedFrameTimer = setInterval(() => {
          this.fetchProcessedFrame();
        }, intervalMs);
        return;
      }

      if (this.processedPlayer) {
        try {
          await this.processedPlayer.play();
        } catch (e) {
          console.error('Play processed video error:', e);
          await this.switchProcessedToFramePreview('处理后HTML5播放器启动失败，已切换到兼容逐帧预览');
        }
        return;
      }

      const nativeVideo = this.$refs ? this.$refs.processedVideo : null;
      if (nativeVideo) {
        try {
          await nativeVideo.play();
        } catch (e) {
          console.error('Play native processed video error:', e);
          await this.switchProcessedToFramePreview('处理后HTML5播放器启动失败，已切换到兼容逐帧预览');
        }
      }
    },
    
    pauseProcessed() {
      if (this.processedPreviewMode === 'native') {
        if (window.pywebview && pywebview.api && pywebview.api.native_player_pause) {
          pywebview.api.native_player_pause({ role: 'processed' }).catch((e) => {
            console.warn('Pause processed native player error:', e);
          });
        }
        return;
      }

      if (this.processedPreviewMode === 'frames') {
        this.clearProcessedFrameTimer();
        return;
      }
      if (this.processedPlayer) {
        this.processedPlayer.pause();
        return;
      }
      const nativeVideo = this.$refs ? this.$refs.processedVideo : null;
      if (nativeVideo) {
        nativeVideo.pause();
      }
    },
    
    async onSourceVideoError() {
      const fallbackUrl = this.toFileUrl(this.currentFile);
      if (this.previewUrl && fallbackUrl && this.previewUrl !== fallbackUrl) {
        this.previewUrl = fallbackUrl;
        this.statusMessage = '预览转码文件加载失败，已回退原视频';
        await this.mountSourcePlayer();
        return;
      }
      await this.switchSourceToFramePreview('原视频预览加载失败，已切换到兼容逐帧预览');
    },
    
    async onProcessedVideoError() {
      const fallbackUrl = this.toFileUrl(this.processedPath);
      if (this.processedUrl && fallbackUrl && this.processedUrl !== fallbackUrl) {
        this.processedUrl = fallbackUrl;
        this.statusMessage = '处理后预览转码文件加载失败，已回退原视频';
        await this.mountProcessedPlayer();
        return;
      }
      await this.switchProcessedToFramePreview('处理后视频预览加载失败，已切换到兼容逐帧预览');
    },
    
    async getDeviceInfo() {
      try {
        const result = await pywebview.api.get_device_info();
        if (result) {
          this.deviceInfo = result.device;
          this.memoryUsage = result.memory;
        }
      } catch (e) {
        console.error('Get device info error:', e);
      }
    },
    
    loadSettings() {
      const saved = localStorage.getItem('settings');
      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          this.settings = { ...this.settings, ...parsed };
        } catch (e) {
          console.error('Load settings error:', e);
        }
      }
    },
    
    saveSettings() {
      localStorage.setItem('settings', JSON.stringify(this.settings));
    },
    
    updateProgress(data) {
      if (data.progress !== undefined) {
        const raw = Number(data.progress);
        if (Number.isFinite(raw)) {
          const clamped = Math.max(0, Math.min(1, raw));
          if (this.isProcessing) {
            this.progress = Math.max(this.progress || 0, clamped);
          } else {
            this.progress = clamped;
          }
        }
      }
      if (data.processed_frames !== undefined) {
        this.processedFrames = data.processed_frames;
      }
      if (data.total_frames !== undefined) {
        this.totalFrames = data.total_frames;
      }
      if (data.estimated_time !== undefined) {
        this.estimatedTime = data.estimated_time;
      }
      if (data.status) {
        this.statusMessage = data.status;
      } else if (data.message) {
        this.statusMessage = data.message;
      }

      const statusText = String(data.status || '').toLowerCase();
      const messageText = String(data.message || '').toLowerCase();
      if (
        statusText.includes('complete') ||
        statusText.includes('完成') ||
        messageText.includes('complete') ||
        messageText.includes('完成')
      ) {
        this.progress = 1;
      }
    }
  };
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = app;
}
