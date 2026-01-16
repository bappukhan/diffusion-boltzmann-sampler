import { useEffect, useState } from 'react';
import { useSimulationStore } from '../store/simulationStore';

interface Shortcut {
  key: string;
  description: string;
  modifier?: 'ctrl' | 'alt' | 'shift';
}

const SHORTCUTS: Shortcut[] = [
  { key: 'Space', description: 'Play/Pause animation' },
  { key: 'ArrowLeft', description: 'Previous frame' },
  { key: 'ArrowRight', description: 'Next frame' },
  { key: 'Home', description: 'Go to first frame' },
  { key: 'End', description: 'Go to last frame' },
  { key: 'r', description: 'Randomize configuration' },
  { key: 's', description: 'Generate sample' },
  { key: '?', description: 'Show keyboard shortcuts' },
];

interface KeyboardShortcutsProps {
  onSample: () => void;
  onRandomize: () => void;
}

/**
 * Keyboard shortcuts handler component.
 *
 * Listens for keyboard events and triggers corresponding actions.
 */
export function KeyboardShortcuts({
  onSample,
  onRandomize,
}: KeyboardShortcutsProps): JSX.Element {
  const [showHelp, setShowHelp] = useState(false);

  const {
    animationFrames,
    currentFrame,
    isPlaying,
    isRunning,
    setCurrentFrame,
    setIsPlaying,
  } = useSimulationStore();

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Ignore if user is typing in an input
      if (
        event.target instanceof HTMLInputElement ||
        event.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      switch (event.key) {
        case ' ':
          event.preventDefault();
          if (animationFrames.length > 0) {
            setIsPlaying(!isPlaying);
          }
          break;

        case 'ArrowLeft':
          event.preventDefault();
          if (currentFrame > 0) {
            setCurrentFrame(currentFrame - 1);
          }
          break;

        case 'ArrowRight':
          event.preventDefault();
          if (currentFrame < animationFrames.length - 1) {
            setCurrentFrame(currentFrame + 1);
          }
          break;

        case 'Home':
          event.preventDefault();
          if (animationFrames.length > 0) {
            setCurrentFrame(0);
          }
          break;

        case 'End':
          event.preventDefault();
          if (animationFrames.length > 0) {
            setCurrentFrame(animationFrames.length - 1);
          }
          break;

        case 'r':
        case 'R':
          if (!event.ctrlKey && !event.metaKey && !isRunning) {
            event.preventDefault();
            onRandomize();
          }
          break;

        case 's':
        case 'S':
          if (!event.ctrlKey && !event.metaKey && !isRunning) {
            event.preventDefault();
            onSample();
          }
          break;

        case '?':
          event.preventDefault();
          setShowHelp((prev) => !prev);
          break;

        case 'Escape':
          if (showHelp) {
            event.preventDefault();
            setShowHelp(false);
          }
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [
    animationFrames.length,
    currentFrame,
    isPlaying,
    isRunning,
    showHelp,
    setCurrentFrame,
    setIsPlaying,
    onSample,
    onRandomize,
  ]);

  if (!showHelp) {
    return <></>;
  }

  return (
    <div
      className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center"
      onClick={() => setShowHelp(false)}
    >
      <div
        className="bg-slate-800 rounded-lg shadow-xl border border-slate-700 p-6 max-w-md w-full mx-4"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-white">Keyboard Shortcuts</h2>
          <button
            onClick={() => setShowHelp(false)}
            className="p-1 text-slate-400 hover:text-white transition-colors"
          >
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                clipRule="evenodd"
              />
            </svg>
          </button>
        </div>

        <div className="space-y-2">
          {SHORTCUTS.map((shortcut) => (
            <div
              key={shortcut.key}
              className="flex items-center justify-between py-2 border-b border-slate-700 last:border-0"
            >
              <span className="text-slate-300">{shortcut.description}</span>
              <kbd className="px-2 py-1 bg-slate-900 text-slate-400 rounded text-sm font-mono">
                {shortcut.modifier && (
                  <span className="mr-1">
                    {shortcut.modifier === 'ctrl' ? 'Ctrl+' : ''}
                    {shortcut.modifier === 'alt' ? 'Alt+' : ''}
                    {shortcut.modifier === 'shift' ? 'Shift+' : ''}
                  </span>
                )}
                {shortcut.key}
              </kbd>
            </div>
          ))}
        </div>

        <div className="mt-4 text-xs text-slate-500 text-center">
          Press <kbd className="px-1 bg-slate-900 rounded">?</kbd> anytime to toggle
        </div>
      </div>
    </div>
  );
}

export default KeyboardShortcuts;
