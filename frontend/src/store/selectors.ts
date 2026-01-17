/**
 * Selector hooks for derived state from the simulation store.
 *
 * These selectors compute derived values and provide type-safe access
 * to commonly used state combinations.
 */

import { useSimulationStore, T_CRITICAL, FrameMetadata } from './simulationStore';

/**
 * Get the current phase based on temperature.
 */
export function usePhase(): 'ordered' | 'critical' | 'disordered' {
  const temperature = useSimulationStore((state) => state.temperature);

  if (temperature < T_CRITICAL - 0.2) return 'ordered';
  if (temperature > T_CRITICAL + 0.2) return 'disordered';
  return 'critical';
}

/**
 * Check if temperature is near critical point.
 */
export function useIsNearCritical(): boolean {
  const temperature = useSimulationStore((state) => state.temperature);
  return Math.abs(temperature - T_CRITICAL) < 0.2;
}

/**
 * Get animation progress as a percentage (0-100).
 */
export function useAnimationProgress(): number {
  const { animationFrames, currentFrame } = useSimulationStore((state) => ({
    animationFrames: state.animationFrames,
    currentFrame: state.currentFrame,
  }));

  if (animationFrames.length === 0) return 0;
  return ((currentFrame + 1) / animationFrames.length) * 100;
}

/**
 * Check if animation has frames.
 */
export function useHasAnimation(): boolean {
  const animationFrames = useSimulationStore((state) => state.animationFrames);
  return animationFrames.length > 0;
}

/**
 * Get animation frame count.
 */
export function useFrameCount(): { current: number; total: number } {
  const { animationFrames, currentFrame } = useSimulationStore((state) => ({
    animationFrames: state.animationFrames,
    currentFrame: state.currentFrame,
  }));

  return {
    current: currentFrame + 1,
    total: animationFrames.length,
  };
}

/**
 * Check if simulation can be started.
 */
export function useCanStartSimulation(): boolean {
  const { isConnected, isRunning } = useSimulationStore((state) => ({
    isConnected: state.isConnected,
    isRunning: state.isRunning,
  }));

  return isConnected && !isRunning;
}

/**
 * Get current configuration parameters.
 */
export function useConfig() {
  return useSimulationStore((state) => ({
    temperature: state.temperature,
    latticeSize: state.latticeSize,
    samplerType: state.samplerType,
    numSteps: state.numSteps,
    playbackSpeed: state.playbackSpeed,
  }));
}

/**
 * Get current physical observables.
 */
export function useObservables() {
  return useSimulationStore((state) => ({
    energy: state.energy,
    magnetization: state.magnetization,
    spins: state.spins,
  }));
}

/**
 * Get connection and error state.
 */
export function useConnectionState() {
  return useSimulationStore((state) => ({
    isConnected: state.isConnected,
    error: state.error,
  }));
}

/**
 * Get animation state.
 */
export function useAnimationState() {
  return useSimulationStore((state) => ({
    isRunning: state.isRunning,
    isPlaying: state.isPlaying,
    currentFrame: state.currentFrame,
    frameCount: state.animationFrames.length,
    playbackSpeed: state.playbackSpeed,
  }));
}

/**
 * Get total spin count (lattice size squared).
 */
export function useTotalSpins(): number {
  const latticeSize = useSimulationStore((state) => state.latticeSize);
  return latticeSize * latticeSize;
}

/**
 * Get absolute magnetization (order parameter).
 */
export function useAbsoluteMagnetization(): number | null {
  const magnetization = useSimulationStore((state) => state.magnetization);
  return magnetization !== null ? Math.abs(magnetization) : null;
}

/**
 * Check if currently showing results from diffusion sampler.
 */
export function useIsDiffusionMode(): boolean {
  const samplerType = useSimulationStore((state) => state.samplerType);
  return samplerType === 'diffusion';
}

/**
 * Get frame delay in milliseconds based on playback speed.
 * Base delay is 50ms (20fps), speed 2x = 25ms, speed 0.5x = 100ms.
 */
export function useFrameDelay(): number {
  const playbackSpeed = useSimulationStore((state) => state.playbackSpeed);
  const baseDelay = 50; // 20fps
  return Math.round(baseDelay / playbackSpeed);
}

/**
 * Get metadata for the current animation frame.
 * Returns null if no animation data or current frame has no metadata.
 */
export function useCurrentFrameMetadata(): FrameMetadata | null {
  return useSimulationStore((state) => {
    const { frameMetadata, currentFrame } = state;
    return frameMetadata[currentFrame] || null;
  });
}

/**
 * Get diffusion-specific metadata for current frame.
 * Returns null if not using diffusion sampler or no data available.
 */
export function useDiffusionMetadata(): {
  t: number;
  sigma: number;
  progress: number;
} | null {
  return useSimulationStore((state) => {
    const { frameMetadata, currentFrame } = state;
    const metadata = frameMetadata[currentFrame];

    if (!metadata || metadata.sampler !== 'diffusion') {
      return null;
    }

    return {
      t: metadata.t ?? 1.0,
      sigma: metadata.sigma ?? 0,
      progress: metadata.progress ?? 0,
    };
  });
}

/**
 * Get the current diffusion phase based on time t.
 * Returns null if not using diffusion sampler.
 */
export function useDiffusionPhase():
  | 'pure_noise'
  | 'emerging'
  | 'refining'
  | 'crystallized'
  | null {
  const metadata = useDiffusionMetadata();

  if (!metadata) {
    return null;
  }

  const { t } = metadata;

  if (t >= 0.8) return 'pure_noise';
  if (t >= 0.5) return 'emerging';
  if (t >= 0.2) return 'refining';
  return 'crystallized';
}
