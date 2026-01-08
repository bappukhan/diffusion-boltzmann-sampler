import { create } from 'zustand';

export type SamplerType = 'mcmc' | 'diffusion';

interface SimulationState {
  // Configuration
  temperature: number;
  latticeSize: number;
  samplerType: SamplerType;
  numSteps: number;

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
}

export const useSimulationStore = create<SimulationState>((set) => ({
  // Default configuration
  temperature: 2.27, // Critical temperature
  latticeSize: 32,
  samplerType: 'mcmc',
  numSteps: 100,

  // Initial state
  spins: null,
  energy: null,
  magnetization: null,

  // Animation
  isRunning: false,
  animationFrames: [],
  currentFrame: 0,
  isPlaying: false,

  // Connection
  isConnected: false,
  error: null,

  // Actions
  setTemperature: (temperature) => set({ temperature }),
  setLatticeSize: (latticeSize) => set({ latticeSize }),
  setSamplerType: (samplerType) => set({ samplerType }),
  setNumSteps: (numSteps) => set({ numSteps }),
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
    set({ animationFrames: [], currentFrame: 0 }),
  setCurrentFrame: (currentFrame) =>
    set((state) => ({
      currentFrame,
      spins: state.animationFrames[currentFrame] || state.spins,
    })),
  setIsPlaying: (isPlaying) => set({ isPlaying }),
  setIsConnected: (isConnected) => set({ isConnected }),
  setError: (error) => set({ error }),
}));

// Critical temperature constant
export const T_CRITICAL = 2.269;
