import React, { useEffect, useRef } from 'react';
import { useSimulationStore } from '../store/simulationStore';

interface DiffusionAnimationProps {
  onFrameChange?: (frame: number) => void;
}

export const DiffusionAnimation: React.FC<DiffusionAnimationProps> = ({
  onFrameChange,
}) => {
  const {
    animationFrames,
    currentFrame,
    isPlaying,
    setCurrentFrame,
    setIsPlaying,
  } = useSimulationStore();

  const animationRef = useRef<number>();

  // Animation loop
  useEffect(() => {
    if (isPlaying && animationFrames.length > 0) {
      if (currentFrame < animationFrames.length - 1) {
        animationRef.current = window.setTimeout(() => {
          const nextFrame = currentFrame + 1;
          setCurrentFrame(nextFrame);
          onFrameChange?.(nextFrame);
        }, 50); // 20fps
      } else {
        setIsPlaying(false);
      }
    }

    return () => {
      if (animationRef.current) {
        clearTimeout(animationRef.current);
      }
    };
  }, [isPlaying, currentFrame, animationFrames.length, setCurrentFrame, setIsPlaying, onFrameChange]);

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
    </div>
  );
};

export default DiffusionAnimation;
