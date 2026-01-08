import React from 'react';
import { useSimulationStore, T_CRITICAL, SamplerType } from '../store/simulationStore';

interface ControlPanelProps {
  onSample: () => void;
  onRandomize: () => void;
}

export const ControlPanel: React.FC<ControlPanelProps> = ({
  onSample,
  onRandomize,
}) => {
  const {
    temperature,
    latticeSize,
    samplerType,
    numSteps,
    isRunning,
    energy,
    magnetization,
    setTemperature,
    setLatticeSize,
    setSamplerType,
    setNumSteps,
  } = useSimulationStore();

  const isNearCritical = Math.abs(temperature - T_CRITICAL) < 0.2;

  return (
    <div className="bg-slate-800 rounded-lg p-6 shadow-lg space-y-6">
      <h2 className="text-xl font-semibold text-white">Controls</h2>

      {/* Temperature Slider */}
      <div className="space-y-2">
        <label className="flex justify-between text-sm text-slate-300">
          <span>Temperature</span>
          <span className="font-mono">
            T = {temperature.toFixed(2)}
            {isNearCritical && (
              <span className="ml-2 text-yellow-400 text-xs">near T_c</span>
            )}
          </span>
        </label>
        <input
          type="range"
          min="0.5"
          max="5"
          step="0.01"
          value={temperature}
          onChange={(e) => setTemperature(parseFloat(e.target.value))}
          className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
        />
        <div className="flex justify-between text-xs text-slate-500">
          <span>Ordered</span>
          <span
            className="text-yellow-400 cursor-pointer hover:text-yellow-300"
            onClick={() => setTemperature(T_CRITICAL)}
          >
            T_c = {T_CRITICAL.toFixed(3)}
          </span>
          <span>Disordered</span>
        </div>
      </div>

      {/* Lattice Size */}
      <div className="space-y-2">
        <label className="flex justify-between text-sm text-slate-300">
          <span>Lattice Size</span>
          <span className="font-mono">{latticeSize} x {latticeSize}</span>
        </label>
        <select
          value={latticeSize}
          onChange={(e) => setLatticeSize(parseInt(e.target.value))}
          className="w-full bg-slate-700 text-white rounded-lg p-2 focus:ring-2 focus:ring-blue-500"
        >
          <option value={16}>16 x 16 (fast)</option>
          <option value={32}>32 x 32 (recommended)</option>
          <option value={48}>48 x 48</option>
          <option value={64}>64 x 64 (slow)</option>
        </select>
      </div>

      {/* Sampler Type */}
      <div className="space-y-2">
        <label className="text-sm text-slate-300">Sampler</label>
        <div className="flex gap-2">
          <button
            onClick={() => setSamplerType('mcmc')}
            className={`flex-1 py-2 px-4 rounded-lg font-medium transition-colors ${
              samplerType === 'mcmc'
                ? 'bg-blue-600 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            MCMC
          </button>
          <button
            onClick={() => setSamplerType('diffusion')}
            className={`flex-1 py-2 px-4 rounded-lg font-medium transition-colors ${
              samplerType === 'diffusion'
                ? 'bg-purple-600 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Diffusion
          </button>
        </div>
      </div>

      {/* Animation Steps */}
      <div className="space-y-2">
        <label className="flex justify-between text-sm text-slate-300">
          <span>Animation Steps</span>
          <span className="font-mono">{numSteps}</span>
        </label>
        <input
          type="range"
          min="20"
          max="200"
          step="10"
          value={numSteps}
          onChange={(e) => setNumSteps(parseInt(e.target.value))}
          className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
        />
      </div>

      {/* Action Buttons */}
      <div className="flex gap-3">
        <button
          onClick={onSample}
          disabled={isRunning}
          className={`flex-1 py-3 px-4 rounded-lg font-semibold transition-all ${
            isRunning
              ? 'bg-slate-600 text-slate-400 cursor-not-allowed'
              : 'bg-gradient-to-r from-blue-500 to-purple-500 text-white hover:from-blue-600 hover:to-purple-600 shadow-lg hover:shadow-xl'
          }`}
        >
          {isRunning ? 'Sampling...' : 'Generate Sample'}
        </button>
        <button
          onClick={onRandomize}
          disabled={isRunning}
          className="py-3 px-4 bg-slate-700 text-slate-300 rounded-lg hover:bg-slate-600 transition-colors"
        >
          Random
        </button>
      </div>

      {/* Current State */}
      {(energy !== null || magnetization !== null) && (
        <div className="bg-slate-900 rounded-lg p-4 space-y-2">
          <h3 className="text-sm font-medium text-slate-400">Current State</h3>
          <div className="grid grid-cols-2 gap-4">
            {energy !== null && (
              <div>
                <div className="text-xs text-slate-500">Energy/spin</div>
                <div className="text-lg font-mono text-white">
                  {energy.toFixed(3)}
                </div>
              </div>
            )}
            {magnetization !== null && (
              <div>
                <div className="text-xs text-slate-500">Magnetization</div>
                <div className="text-lg font-mono text-white">
                  {magnetization.toFixed(3)}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ControlPanel;
