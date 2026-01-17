import React, { useEffect, useRef } from 'react';
import { useSimulationStore, FrameMetadata } from '../store/simulationStore';
import { useFrameDelay } from '../store/selectors';

interface DiffusionAnimationProps {
  onFrameChange?: (frame: number) => void;
}

/** Component to display noise level indicator for diffusion sampling */
const NoiseLevelIndicator: React.FC<{ metadata: FrameMetadata | null }> = ({
  metadata,
}) => {
  if (!metadata || metadata.sampler !== 'diffusion') {
    return null;
  }

  const sigma = metadata.sigma ?? 0;
  const t = metadata.t ?? 0;
  const noisePercent = Math.min(100, sigma * 100);

  return (
    <div className="space-y-2 pt-2 border-t border-slate-700">
      <div className="flex items-center justify-between">
        <span className="text-xs text-slate-500">Noise Level</span>
        <span className="text-xs font-mono text-purple-400">
          sigma = {sigma.toFixed(3)}
        </span>
      </div>

      {/* Noise level bar */}
      <div className="relative h-3 bg-slate-700 rounded-full overflow-hidden">
        <div
          className="absolute h-full bg-gradient-to-r from-purple-600 to-pink-500 transition-all duration-150"
          style={{ width: `${noisePercent}%` }}
        />
        {/* Tick marks */}
        <div className="absolute inset-0 flex justify-between px-1">
          {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
            <div
              key={tick}
              className="w-px h-full bg-slate-600 opacity-50"
            />
          ))}
        </div>
      </div>

      {/* Diffusion time indicator */}
      <div className="flex items-center justify-between text-xs">
        <span className="text-slate-500">t = {t.toFixed(2)}</span>
        <span className="text-slate-500">
          {t > 0.7 ? 'Pure noise' : t > 0.3 ? 'Denoising' : 'Crystallizing'}
        </span>
      </div>
    </div>
  );
};

const SPEED_OPTIONS = [
  { value: 0.5, label: '0.5x' },
  { value: 1, label: '1x' },
  { value: 2, label: '2x' },
  { value: 4, label: '4x' },
];

export const DiffusionAnimation: React.FC<DiffusionAnimationProps> = ({
  onFrameChange,
}) => {
  const {
    animationFrames,
    frameMetadata,
    currentFrame,
    isPlaying,
    playbackSpeed,
    setCurrentFrame,
    setIsPlaying,
    setPlaybackSpeed,
  } = useSimulationStore();

  const currentMetadata = frameMetadata[currentFrame] || null;

  const frameDelay = useFrameDelay();

  const animationRef = useRef<number>();

  // Animation loop
  useEffect(() => {
    if (isPlaying && animationFrames.length > 0) {
      if (currentFrame < animationFrames.length - 1) {
        animationRef.current = window.setTimeout(() => {
          const nextFrame = currentFrame + 1;
          setCurrentFrame(nextFrame);
          onFrameChange?.(nextFrame);
        }, frameDelay);
      } else {
        setIsPlaying(false);
      }
    }

    return () => {
      if (animationRef.current) {
        clearTimeout(animationRef.current);
      }
    };
  }, [isPlaying, currentFrame, animationFrames.length, frameDelay, setCurrentFrame, setIsPlaying, onFrameChange]);

  if (animationFrames.length === 0) {
    return (
      <div className="bg-slate-800 rounded-lg p-4 shadow-lg">
        <h3 className="text-sm font-medium text-slate-400 mb-3">Animation</h3>
        <div className="text-center text-slate-500 py-8">
          Generate a sample to see the animation
        </div>
      </div>
    );
  }

  const progress = ((currentFrame + 1) / animationFrames.length) * 100;

  return (
    <div className="bg-slate-800 rounded-lg p-4 shadow-lg space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-slate-400">Animation</h3>
        <span className="text-xs text-slate-500 font-mono">
          Frame {currentFrame + 1} / {animationFrames.length}
        </span>
      </div>

      {/* Progress bar */}
      <div className="relative h-2 bg-slate-700 rounded-full overflow-hidden">
        <div
          className="absolute h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-100"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Scrubber */}
      <input
        type="range"
        min={0}
        max={animationFrames.length - 1}
        value={currentFrame}
        onChange={(e) => {
          const frame = parseInt(e.target.value);
          setCurrentFrame(frame);
          onFrameChange?.(frame);
        }}
        className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
      />

      {/* Controls */}
      <div className="flex items-center justify-center gap-4">
        <button
          onClick={() => {
            setCurrentFrame(0);
            onFrameChange?.(0);
          }}
          className="p-2 text-slate-400 hover:text-white transition-colors"
          title="Go to start"
        >
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
            <path d="M6 6h2v12H6zm3.5 6l8.5 6V6z" />
          </svg>
        </button>

        <button
          onClick={() => {
            if (currentFrame > 0) {
              setCurrentFrame(currentFrame - 1);
              onFrameChange?.(currentFrame - 1);
            }
          }}
          className="p-2 text-slate-400 hover:text-white transition-colors"
          title="Previous frame"
        >
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
            <path d="M6 6h2v12H6zm3.5 6l8.5 6V6z" />
          </svg>
        </button>

        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className={`p-3 rounded-full transition-colors ${
            isPlaying
              ? 'bg-purple-600 text-white'
              : 'bg-blue-600 text-white hover:bg-blue-700'
          }`}
          title={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? (
            <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
              <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
            </svg>
          ) : (
            <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
              <path d="M8 5v14l11-7z" />
            </svg>
          )}
        </button>

        <button
          onClick={() => {
            if (currentFrame < animationFrames.length - 1) {
              setCurrentFrame(currentFrame + 1);
              onFrameChange?.(currentFrame + 1);
            }
          }}
          className="p-2 text-slate-400 hover:text-white transition-colors"
          title="Next frame"
        >
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
            <path d="M6 18l8.5-6L6 6v12zM16 6v12h2V6h-2z" />
          </svg>
        </button>

        <button
          onClick={() => {
            setCurrentFrame(animationFrames.length - 1);
            onFrameChange?.(animationFrames.length - 1);
          }}
          className="p-2 text-slate-400 hover:text-white transition-colors"
          title="Go to end"
        >
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
            <path d="M6 18l8.5-6L6 6v12zM16 6v12h2V6h-2z" />
          </svg>
        </button>
      </div>

      {/* Playback Speed */}
      <div className="flex items-center justify-center gap-2 pt-2 border-t border-slate-700">
        <span className="text-xs text-slate-500">Speed:</span>
        <div className="flex gap-1">
          {SPEED_OPTIONS.map((option) => (
            <button
              key={option.value}
              onClick={() => setPlaybackSpeed(option.value)}
              className={`px-2 py-1 text-xs rounded transition-colors ${
                playbackSpeed === option.value
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-700 text-slate-400 hover:bg-slate-600 hover:text-slate-200'
              }`}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      {/* Noise Level Indicator (diffusion only) */}
      <NoiseLevelIndicator metadata={currentMetadata} />
    </div>
  );
};

export default DiffusionAnimation;
