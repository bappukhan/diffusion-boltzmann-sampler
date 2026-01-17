/**
 * Store exports.
 */

export {
  useSimulationStore,
  T_CRITICAL,
  DEFAULT_CONFIG,
} from './simulationStore';

export type { SamplerType, FrameMetadata } from './simulationStore';

// Selector hooks
export {
  usePhase,
  useIsNearCritical,
  useAnimationProgress,
  useHasAnimation,
  useFrameCount,
  useCanStartSimulation,
  useConfig,
  useObservables,
  useConnectionState,
  useAnimationState,
  useTotalSpins,
  useAbsoluteMagnetization,
  useIsDiffusionMode,
  useFrameDelay,
  useCurrentFrameMetadata,
  useDiffusionMetadata,
  useDiffusionPhase,
} from './selectors';
