/**
 * Custom hooks module.
 */

export { useHealthCheck, default as useHealthCheckDefault } from './useHealthCheck';

export {
  useWebSocket,
  default as useWebSocketDefault,
} from './useWebSocket';

export type {
  WebSocketState,
  FrameData,
  WebSocketMessage,
  SamplingParams,
  UseWebSocketOptions,
  UseWebSocketReturn,
} from './useWebSocket';

export {
  useSynchronizedPlayback,
  default as useSynchronizedPlaybackDefault,
} from './useSynchronizedPlayback';

export type {
  SequenceState,
  SynchronizedPlaybackConfig,
  UseSynchronizedPlaybackReturn,
} from './useSynchronizedPlayback';
