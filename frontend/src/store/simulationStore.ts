import { create } from 'zustand';

export type SamplerType = 'mcmc' | 'diffusion';

/** Default configuration values */
export const DEFAULT_CONFIG = {
  temperature: 2.27, // Critical temperature
  latticeSize: 32,
  samplerType: 'mcmc' as SamplerType,
  numSteps: 100,
  playbackSpeed: 1.0,
} as const;

interface SimulationState {
  // Configuration
  temperature: number;
  latticeSize: number;
  samplerType: SamplerType;
  numSteps: number;
  playbackSpeed: number;

  // Current state
  spins: number[][] | null;
  energy: number | null;
  magnetization: number | null;

  // Animation
  isRunning: boolean;
  animationFrames: number[][][];
  currentFrame: number;
  isPlaying: boolean;

  // Connection
  isConnected: boolean;
  error: string | null;

  // Actions
  setTemperature: (temp: number) => void;
  setLatticeSize: (size: number) => void;
  setSamplerType: (type: SamplerType) => void;
  setNumSteps: (steps: number) => void;
  setPlaybackSpeed: (speed: number) => void;
  setSpins: (spins: number[][]) => void;
  setEnergy: (energy: number) => void;
  setMagnetization: (mag: number) => void;
  setIsRunning: (running: boolean) => void;
  addAnimationFrame: (frame: number[][]) => void;
  clearAnimationFrames: () => void;
  setCurrentFrame: (frame: number) => void;
  setIsPlaying: (playing: boolean) => void;
  setIsConnected: (connected: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
  resetConfig: () => void;
  clearState: () => void;
}

/** Initial state values */
const initialState = {
  // Configuration
  temperature: DEFAULT_CONFIG.temperature,
  latticeSize: DEFAULT_CONFIG.latticeSize,
  samplerType: DEFAULT_CONFIG.samplerType,
  numSteps: DEFAULT_CONFIG.numSteps,
  playbackSpeed: DEFAULT_CONFIG.playbackSpeed,

  // Current state
  spins: null as number[][] | null,
  energy: null as number | null,
  magnetization: null as number | null,

  // Animation
  isRunning: false,
  animationFrames: [] as number[][][],
  currentFrame: 0,
  isPlaying: false,

  // Connection
  isConnected: false,
  error: null as string | null,
};

export const useSimulationStore = create<SimulationState>((set) => ({
  ...initialState,

  // Actions
  setTemperature: (temperature) => set({ temperature }),
  setLatticeSize: (latticeSize) => set({ latticeSize }),
  setSamplerType: (samplerType) => set({ samplerType }),
  setNumSteps: (numSteps) => set({ numSteps }),
  setPlaybackSpeed: (playbackSpeed) => set({ playbackSpeed }),
  setSpins: (spins) => set({ spins }),
  setEnergy: (energy) => set({ energy }),
  setMagnetization: (magnetization) => set({ magnetization }),
  setIsRunning: (isRunning) => set({ isRunning }),
  addAnimationFrame: (frame) =>
    set((state) => ({
      animationFrames: [...state.animationFrames, frame],
      spins: frame,
    })),
  clearAnimationFrames: () =>
    set({ animationFrames: [], currentFrame: 0, isPlaying: false }),
  setCurrentFrame: (currentFrame) =>
    set((state) => ({
      currentFrame,
      spins: state.animationFrames[currentFrame] || state.spins,
    })),
  setIsPlaying: (isPlaying) => set({ isPlaying }),
  setIsConnected: (isConnected) => set({ isConnected }),
  setError: (error) => set({ error }),

  // Reset actions
  reset: () => set({ ...initialState }),
  resetConfig: () =>
    set({
      temperature: DEFAULT_CONFIG.temperature,
      latticeSize: DEFAULT_CONFIG.latticeSize,
      samplerType: DEFAULT_CONFIG.samplerType,
      numSteps: DEFAULT_CONFIG.numSteps,
      playbackSpeed: DEFAULT_CONFIG.playbackSpeed,
    }),
  clearState: () =>
    set({
      spins: null,
      energy: null,
      magnetization: null,
      animationFrames: [],
      currentFrame: 0,
      isPlaying: false,
      isRunning: false,
      error: null,
    }),
}));

// Critical temperature constant
export const T_CRITICAL = 2.269;
