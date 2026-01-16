import { useHealthCheck } from '../hooks/useHealthCheck';
import { formatDuration } from '../utils/format';

interface ConnectionStatusProps {
  /** Show full details instead of compact view */
  detailed?: boolean;
  /** Custom class for positioning */
  className?: string;
}

/**
 * Connection status indicator component.
 *
 * Shows the current connection state to the backend server
 * with optional detailed information.
 */
export function ConnectionStatus({
  detailed = false,
  className = '',
}: ConnectionStatusProps): JSX.Element {
  const {
    isConnected,
    isChecking,
    error,
    version,
    lastChecked,
    checkNow,
  } = useHealthCheck({
    interval: 10000,
    immediate: true,
  });

  const statusColor = isConnected ? 'green' : 'red';
  const statusText = isConnected ? 'Connected' : 'Disconnected';

  if (!detailed) {
    return (
      <div
        className={`
          flex items-center gap-2 px-3 py-1 rounded-full text-sm
          ${isConnected ? 'bg-green-900/50 text-green-400' : 'bg-red-900/50 text-red-400'}
          ${className}
        `}
      >
        <span
          className={`
            w-2 h-2 rounded-full
            ${isConnected ? 'bg-green-400' : 'bg-red-400'}
            ${isChecking ? 'animate-pulse' : ''}
          `}
        />
        {statusText}
      </div>
    );
  }

  return (
    <div
      className={`
        bg-slate-800 rounded-lg p-4 shadow-lg
        border border-slate-700
        ${className}
      `}
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-slate-300">Server Status</h3>
        <button
          onClick={checkNow}
          disabled={isChecking}
          className={`
            text-xs px-2 py-1 rounded
            ${isChecking
              ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
              : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }
            transition-colors
          `}
        >
          {isChecking ? 'Checking...' : 'Check Now'}
        </button>
      </div>

      <div className="space-y-2">
        {/* Connection status */}
        <div className="flex items-center gap-2">
          <span
            className={`
              w-3 h-3 rounded-full
              bg-${statusColor}-400
              ${isChecking ? 'animate-pulse' : ''}
            `}
            style={{
              backgroundColor: isConnected ? '#4ade80' : '#f87171',
            }}
          />
          <span className={`text-${statusColor}-400`} style={{
            color: isConnected ? '#4ade80' : '#f87171',
          }}>
            {statusText}
          </span>
        </div>

        {/* Version info */}
        {version && (
          <div className="text-xs text-slate-500">
            Version: <span className="text-slate-400">{version}</span>
          </div>
        )}

        {/* Last checked */}
        {lastChecked && (
          <div className="text-xs text-slate-500">
            Last checked:{' '}
            <span className="text-slate-400">
              {formatDuration(Date.now() - lastChecked.getTime())} ago
            </span>
          </div>
        )}

        {/* Error message */}
        {error && (
          <div className="mt-2 p-2 bg-red-900/30 rounded text-xs text-red-300">
            {error}
          </div>
        )}

        {/* Help text when disconnected */}
        {!isConnected && !error && (
          <div className="mt-2 text-xs text-slate-500">
            Start the server with:{' '}
            <code className="bg-slate-900 px-1 rounded">
              make backend
            </code>
          </div>
        )}
      </div>
    </div>
  );
}

export default ConnectionStatus;
