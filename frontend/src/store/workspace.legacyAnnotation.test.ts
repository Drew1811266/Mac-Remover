import { describe, expect, it, beforeEach } from 'vitest';

import { resolveVisibleStageSegments, useWorkspaceStore } from './workspace';
import type { AnnotationSegment } from '../types/annotation';

describe('legacy annotation segment behavior', () => {
  beforeEach(() => {
    useWorkspaceStore.setState({
      currentFrame: 0,
      selectedId: null,
      showAll: true,
      segments: [],
    });
  });

  it('creates a new segment with the legacy two-second default frame span', () => {
    useWorkspaceStore.getState().createSegmentFromRect(
      { x: 10, y: 20, width: 120, height: 60 },
      15,
      { fps: 30, frameMax: 100 },
    );

    const [segment] = useWorkspaceStore.getState().segments;
    expect(segment.start_frame).toBe(15);
    expect(segment.end_frame).toBe(75);
    expect(segment.enabled).toBe(true);
    expect(segment.expand_px).toBe(5);
    expect(segment.feather_px).toBe(3);
  });

  it('clamps the legacy default end frame to the video frame max', () => {
    useWorkspaceStore.getState().createSegmentFromRect(
      { x: 10, y: 20, width: 120, height: 60 },
      90,
      { fps: 30, frameMax: 100 },
    );

    const [segment] = useWorkspaceStore.getState().segments;
    expect(segment.start_frame).toBe(90);
    expect(segment.end_frame).toBe(100);
  });

  it('keeps canvas segments hidden outside their frame range even when show-all is enabled', () => {
    const segments: AnnotationSegment[] = [
      makeSegment('active', 10, 20, true),
      makeSegment('inactive-hit', 10, 20, false),
      makeSegment('future', 30, 40, true),
    ];

    expect(resolveVisibleStageSegments(segments, 15).map((segment) => segment.id)).toEqual(['active']);
  });
});

function makeSegment(id: string, start: number, end: number, enabled: boolean): AnnotationSegment {
  return {
    id,
    start_frame: start,
    end_frame: end,
    rect: { x: 0, y: 0, width: 10, height: 10 },
    expand_px: 5,
    feather_px: 3,
    enabled,
    created_at: '2026-01-01T00:00:00.000Z',
    updated_at: '2026-01-01T00:00:00.000Z',
  };
}
