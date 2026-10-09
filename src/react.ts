// React bindings (any React 18+ framework: Next.js, Remix, Vite…). Client-side only: use in components that
// render in the browser ('use client' in Next.js).
import { createElement, useEffect, useRef, useState, type VideoHTMLAttributes } from 'react';
import { createFaceSwap, type FaceSwap, type FaceSwapOptions, type IdentitySource, type Progress, type Stats } from './index.ts';

export type FaceSwapState =
  | { status: 'idle' | 'loading'; progress: Progress | null; swap: null; error: null }
  | { status: 'ready'; progress: null; swap: FaceSwap; error: null }
  | { status: 'error'; progress: null; swap: null; error: Error };

/**
 * Loads the swapper when `enabled` turns true (models download once, then come from the cache) and
 * disposes it on unmount. Pass a stable `options` object (useMemo) to avoid reloading.
 */
export function useFaceSwap(options: FaceSwapOptions, enabled = true): FaceSwapState & { stats: Stats | null } {
  const [state, setState] = useState<FaceSwapState>({ status: 'idle', progress: null, swap: null, error: null });
  const [stats, setStats] = useState<Stats | null>(null);
  useEffect(() => {
    if (!enabled) return;
    let alive = true;
    let made: FaceSwap | null = null;
    setState({ status: 'loading', progress: null, swap: null, error: null });
    createFaceSwap({
      ...options,
      onProgress: (p) => { options.onProgress?.(p); if (alive) setState({ status: 'loading', progress: p, swap: null, error: null }); },
      onStats: (s) => { options.onStats?.(s); if (alive) setStats(s); },
    }).then((s) => { made = s; if (alive) setState({ status: 'ready', progress: null, swap: s, error: null }); else s.dispose(); })
      .catch((error: Error) => alive && setState({ status: 'error', progress: null, swap: null, error }));
    return () => { alive = false; made?.dispose(); };
  }, [options, enabled]);
  return { ...state, stats };
}

/** The swapped version of a camera track; follows identity changes (morphing over `morphMs`). */
export function useSwappedTrack(swap: FaceSwap | null, track: MediaStreamTrack | null, identity: IdentitySource | null, morphMs = 1200) {
  const [out, setOut] = useState<MediaStreamTrack | null>(null);
  useEffect(() => {
    if (!swap || !track) return;
    const t = swap.process(track);
    setOut(t);
    return () => { t.stop(); setOut(null); };
  }, [swap, track]);
  const first = useRef(true);
  useEffect(() => {
    if (!swap) return;
    if (!identity) { swap.setStrength(0, 400); return; }
    swap.setIdentity(identity, { morphMs: first.current ? 0 : morphMs }).then(() => swap.setStrength(1, 400)).catch(() => {});
    first.current = false;
  }, [swap, identity, morphMs]);
  return out;
}

/** A <video> showing a track (e.g. the swapped one). */
export function TrackVideo({ track, ...props }: { track: MediaStreamTrack | null } & VideoHTMLAttributes<HTMLVideoElement>) {
  const ref = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.srcObject = track ? new MediaStream([track]) : null;
  }, [track]);
  return createElement('video', { ref, autoPlay: true, playsInline: true, muted: true, ...props });
}
