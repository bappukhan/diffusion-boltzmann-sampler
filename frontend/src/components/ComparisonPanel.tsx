import React, { useMemo } from 'react';
import { FrameMetadata, SamplerType } from '../store/simulationStore';

interface ComparisonPanelProps {
  /** Sampler type for display */
  samplerType: SamplerType;
  /** Current spin configuration */
  spins: number[][] | null;
  /** Metadata for current frame */
  metadata: FrameMetadata | null;
  /** Current frame index */
  currentFrame: number;
  /** Total number of frames */
  totalFrames: number;
  /** Whether data is loading */
  isLoading: boolean;
  /** Error message if any */
  error: string | null;
  /** Custom title override */
  title?: string;
  /** Custom accent color class */
  accentColor?: string;
}

/** Color mapping for spin values */
const SPIN_COLORS = {
  up: '#3b82f6', // blue-500
  down: '#f97316', // orange-500
  neutral: '#64748b', // slate-500
};

/**
 * Renders a spin configuration as a canvas-like grid.
 */
const SpinGrid: React.FC<{ spins: number[][]; size?: number }> = ({
  spins,
  size = 200,
}) => {
  const gridSize = spins.length;
  const cellSize = size / gridSize;

  const cells = useMemo(() => {
    const result: React.ReactNode[] = [];
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        const spin = spins[i][j];
        const color =
          spin > 0
            ? SPIN_COLORS.up
            : spin < 0
            ? SPIN_COLORS.down
            : SPIN_COLORS.neutral;
        result.push(
          <rect
            key={`${i}-${j}`}
            x={j * cellSize}
            y={i * cellSize}
            width={cellSize}
            height={cellSize}
            fill={color}
          />
        );
      }
    }
    return result;
  }, [spins, gridSize, cellSize]);

  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      className="rounded"
    >
      {cells}
    </svg>
  );
};

/**
 * Single panel in the comparison view showing one sampler's output.
 */
export const ComparisonPanel: React.FC<ComparisonPanelProps> = ({
  samplerType,
  spins,
  metadata,
  currentFrame,
  totalFrames,
  isLoading,
  error,
  title,
  accentColor,
}) => {
  const displayTitle =
    title ||
    (samplerType === 'mcmc' ? 'MCMC (Metropolis-Hastings)' : 'Diffusion (Score-Based)');

  const accent = accentColor || (samplerType === 'mcmc' ? 'text-orange-400' : 'text-purple-400');

  return (
    <div className="bg-slate-700/50 rounded-lg p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <h3 className={`text-sm font-medium ${accent}`}>{displayTitle}</h3>
        {isLoading && (
          <span className="text-xs text-slate-400 animate-pulse">
            Sampling...
          </span>
        )}
        {error && (
          <span className="text-xs text-red-400" title={error}>
            Error
          </span>
        )}
      </div>

      {/* Visualization */}
      <div className="aspect-square bg-slate-800 rounded flex items-center justify-center overflow-hidden">
        {spins ? (
          <SpinGrid spins={spins} size={200} />
        ) : isLoading ? (
          <div className="flex flex-col items-center gap-2">
            <div className="w-8 h-8 border-2 border-slate-600 border-t-blue-500 rounded-full animate-spin" />
            <span className="text-xs text-slate-500">Generating...</span>
          </div>
        ) : (
          <span className="text-sm text-slate-500">
            Run comparison to see results
          </span>
        )}
      </div>

      {/* Frame counter */}
      {totalFrames > 0 && (
        <div className="mt-2 flex items-center justify-between text-xs text-slate-400">
          <span>
            Frame {currentFrame + 1} / {totalFrames}
          </span>
          {metadata && (
            <span className="font-mono">
              {samplerType === 'mcmc'
                ? `E: ${metadata.energy.toFixed(3)}`
                : `t: ${(metadata.t ?? 0).toFixed(3)}`}
            </span>
          )}
        </div>
      )}

      {/* Metadata display */}
      {metadata && (
        <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
          <div className="bg-slate-800/50 rounded p-1.5">
            <span className="text-slate-500">Energy: </span>
            <span className="text-slate-300 font-mono">
              {metadata.energy.toFixed(4)}
            </span>
          </div>
          <div className="bg-slate-800/50 rounded p-1.5">
            <span className="text-slate-500">|M|: </span>
            <span className="text-slate-300 font-mono">
              {Math.abs(metadata.magnetization).toFixed(4)}
            </span>
          </div>
          {samplerType === 'mcmc' && metadata.temperature !== undefined && (
            <div className="bg-slate-800/50 rounded p-1.5 col-span-2">
              <span className="text-slate-500">Temperature: </span>
              <span className="text-orange-400 font-mono">
                {metadata.temperature.toFixed(3)}
              </span>
            </div>
          )}
          {samplerType === 'diffusion' && metadata.sigma !== undefined && (
            <div className="bg-slate-800/50 rounded p-1.5 col-span-2">
              <span className="text-slate-500">Sigma: </span>
              <span className="text-purple-400 font-mono">
                {metadata.sigma.toFixed(4)}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ComparisonPanel;
