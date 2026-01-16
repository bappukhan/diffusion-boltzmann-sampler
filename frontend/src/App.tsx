import { useState, useCallback } from 'react';
import { useSimulationStore, T_CRITICAL } from './store/simulationStore';
import { useHealthCheck } from './hooks/useHealthCheck';
import {
  getRandomConfiguration,
  createSamplingWebSocket,
  compareSamplers,
  AnalysisResponse,
} from './services/api';
import { IsingVisualizer } from './components/IsingVisualizer';
import { ControlPanel } from './components/ControlPanel';
import { DiffusionAnimation } from './components/DiffusionAnimation';
import { CorrelationPlot, DistributionPlot } from './components/CorrelationPlot';

function App() {
  const {
    temperature,
    latticeSize,
    samplerType,
    numSteps,
    spins,
    setSpins,
    setEnergy,
    setMagnetization,
    setIsRunning,
    addAnimationFrame,
    clearAnimationFrames,
    setIsConnected,
    setError,
  } = useSimulationStore();

  const [analysisData, setAnalysisData] = useState<AnalysisResponse | null>(null);
  const [isComparing, setIsComparing] = useState(false);
  const [activeTab, setActiveTab] = useState<'simulation' | 'analysis'>('simulation');

  // Use health check hook for backend connection monitoring
  const { isConnected } = useHealthCheck({
    interval: 10000,
    immediate: true,
    onStatusChange: (connected) => {
      setIsConnected(connected);
      if (!connected) {
        setError('Backend not available. Start the server with: uvicorn backend.api.main:app --reload');
      } else {
        setError(null);
      }
    },
  });

  // Generate sample via WebSocket
  const handleSample = useCallback(() => {
    if (!isConnected) {
      setError('Backend not connected');
      return;
    }

    setIsRunning(true);
    clearAnimationFrames();

    const ws = createSamplingWebSocket(
      // onFrame
      (data) => {
        addAnimationFrame(data.spins);
        setEnergy(data.energy);
        setMagnetization(data.magnetization);
      },
      // onDone
      () => {
        setIsRunning(false);
      },
      // onError
      (error) => {
        setError(error);
        setIsRunning(false);
      }
    );

    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          temperature,
          lattice_size: latticeSize,
          sampler: samplerType,
          num_steps: numSteps,
        })
      );
    };
  }, [
    isConnected,
    temperature,
    latticeSize,
    samplerType,
    numSteps,
    setIsRunning,
    clearAnimationFrames,
    addAnimationFrame,
    setEnergy,
    setMagnetization,
    setError,
  ]);

  // Randomize configuration
  const handleRandomize = useCallback(async () => {
    try {
      const result = await getRandomConfiguration(latticeSize);
      setSpins(result.spins);
      setEnergy(result.energy);
      setMagnetization(result.magnetization);
      clearAnimationFrames();
    } catch {
      setError('Failed to get random configuration');
    }
  }, [latticeSize, setSpins, setEnergy, setMagnetization, clearAnimationFrames, setError]);

  // Run comparison analysis
  const handleCompare = useCallback(async () => {
    if (!isConnected) {
      setError('Backend not connected');
      return;
    }

    setIsComparing(true);
    try {
      const data = await compareSamplers({
        temperature,
        lattice_size: latticeSize,
        n_samples: 100,
      });
      setAnalysisData(data);
    } catch {
      setError('Comparison failed');
    } finally {
      setIsComparing(false);
    }
  }, [isConnected, temperature, latticeSize, setError]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <header className="border-b border-slate-700">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-white">
                Diffusion Boltzmann Sampler
              </h1>
              <p className="text-sm text-slate-400">
                Neural sampling from statistical mechanics using score-based diffusion
              </p>
            </div>
            <div className="flex items-center gap-4">
              <div
                className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm ${
                  isConnected
                    ? 'bg-green-900/50 text-green-400'
                    : 'bg-red-900/50 text-red-400'
                }`}
              >
                <span
                  className={`w-2 h-2 rounded-full ${
                    isConnected ? 'bg-green-400' : 'bg-red-400'
                  }`}
                />
                {isConnected ? 'Connected' : 'Disconnected'}
              </div>
            </div>
          </div>

          {/* Tabs */}
          <div className="flex gap-4 mt-4">
            <button
              onClick={() => setActiveTab('simulation')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                activeTab === 'simulation'
                  ? 'bg-blue-600 text-white'
                  : 'text-slate-400 hover:text-white hover:bg-slate-700'
              }`}
            >
              Simulation
            </button>
            <button
              onClick={() => setActiveTab('analysis')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                activeTab === 'analysis'
                  ? 'bg-purple-600 text-white'
                  : 'text-slate-400 hover:text-white hover:bg-slate-700'
              }`}
            >
              Analysis
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {activeTab === 'simulation' ? (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left Column - Controls */}
            <div className="lg:col-span-1 space-y-6">
              <ControlPanel onSample={handleSample} onRandomize={handleRandomize} />
              <DiffusionAnimation />
            </div>

            {/* Right Column - Visualization */}
            <div className="lg:col-span-2 space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <IsingVisualizer
                  spins={spins}
                  size={latticeSize}
                  title={`${samplerType.toUpperCase()} Sample`}
                />
                <div className="bg-slate-800 rounded-lg p-6 shadow-lg">
                  <h3 className="text-lg font-semibold text-white mb-4">
                    Physics Background
                  </h3>
                  <div className="space-y-4 text-sm text-slate-300">
                    <div>
                      <h4 className="font-medium text-white">2D Ising Model</h4>
                      <p>
                        A lattice of spins s_i in {'{-1, +1}'} with Hamiltonian:
                      </p>
                      <code className="block bg-slate-900 p-2 rounded mt-1 text-blue-300">
                        H = -J sum s_i s_j - h sum s_i
                      </code>
                    </div>
                    <div>
                      <h4 className="font-medium text-white">Critical Temperature</h4>
                      <p>
                        Phase transition at T_c = {T_CRITICAL.toFixed(3)}
                      </p>
                      <ul className="list-disc list-inside mt-1 text-slate-400">
                        <li>T &lt; T_c: Ordered (ferromagnetic)</li>
                        <li>T &gt; T_c: Disordered (paramagnetic)</li>
                      </ul>
                    </div>
                    <div>
                      <h4 className="font-medium text-white">Score Function</h4>
                      <p>For Boltzmann distribution p(x) = exp(-E/kT):</p>
                      <code className="block bg-slate-900 p-2 rounded mt-1 text-purple-300">
                        score(x) = -grad E(x) / kT = Force / kT
                      </code>
                    </div>
                  </div>
                </div>
              </div>

              {/* Instructions */}
              <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                <h4 className="font-medium text-white mb-2">Quick Start</h4>
                <ol className="list-decimal list-inside text-sm text-slate-400 space-y-1">
                  <li>Adjust temperature using the slider (try near T_c = 2.27)</li>
                  <li>Select sampler type (MCMC for baseline, Diffusion for neural)</li>
                  <li>Click "Generate Sample" to see the sampling animation</li>
                  <li>Switch to Analysis tab to compare both methods</li>
                </ol>
              </div>
            </div>
          </div>
        ) : (
          /* Analysis Tab */
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-semibold text-white">
                  Sampler Comparison
                </h2>
                <p className="text-sm text-slate-400">
                  Compare MCMC baseline vs neural diffusion sampler
                </p>
              </div>
              <button
                onClick={handleCompare}
                disabled={isComparing || !isConnected}
                className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                  isComparing || !isConnected
                    ? 'bg-slate-600 text-slate-400 cursor-not-allowed'
                    : 'bg-gradient-to-r from-blue-500 to-purple-500 text-white hover:from-blue-600 hover:to-purple-600'
                }`}
              >
                {isComparing ? 'Analyzing...' : 'Run Comparison'}
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <CorrelationPlot
                mcmcData={analysisData?.mcmc.correlation}
                diffusionData={analysisData?.diffusion.correlation}
                title="Spin-Spin Correlation C(r)"
              />
              <DistributionPlot
                mcmcData={
                  analysisData?.mcmc.magnetization
                    ? {
                        values: analysisData.mcmc.magnetization.M,
                        probabilities: analysisData.mcmc.magnetization.P_M,
                      }
                    : undefined
                }
                diffusionData={
                  analysisData?.diffusion.magnetization
                    ? {
                        values: analysisData.diffusion.magnetization.M,
                        probabilities: analysisData.diffusion.magnetization.P_M,
                      }
                    : undefined
                }
                title="Magnetization Distribution P(M)"
                xlabel="Magnetization M"
              />
              <DistributionPlot
                mcmcData={
                  analysisData?.mcmc.energy
                    ? {
                        values: analysisData.mcmc.energy.E,
                        probabilities: analysisData.mcmc.energy.P_E,
                      }
                    : undefined
                }
                diffusionData={
                  analysisData?.diffusion.energy
                    ? {
                        values: analysisData.diffusion.energy.E,
                        probabilities: analysisData.diffusion.energy.P_E,
                      }
                    : undefined
                }
                title="Energy Distribution P(E/N)"
                xlabel="Energy per spin"
              />

              {/* Comparison Metrics */}
              {analysisData && (
                <div className="bg-slate-800 rounded-lg p-6 shadow-lg">
                  <h3 className="text-lg font-semibold text-white mb-4">
                    Comparison Metrics
                  </h3>
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-slate-900 rounded p-3">
                        <div className="text-xs text-slate-500">MCMC Mean |M|</div>
                        <div className="text-xl font-mono text-blue-400">
                          {Math.abs(analysisData.mcmc.mean_mag).toFixed(3)}
                        </div>
                      </div>
                      <div className="bg-slate-900 rounded p-3">
                        <div className="text-xs text-slate-500">Diffusion Mean |M|</div>
                        <div className="text-xl font-mono text-purple-400">
                          {Math.abs(analysisData.diffusion.mean_mag).toFixed(3)}
                        </div>
                      </div>
                      <div className="bg-slate-900 rounded p-3">
                        <div className="text-xs text-slate-500">MCMC tau_int</div>
                        <div className="text-xl font-mono text-blue-400">
                          {analysisData.mcmc.autocorrelation_time.toFixed(1)}
                        </div>
                      </div>
                      <div className="bg-slate-900 rounded p-3">
                        <div className="text-xs text-slate-500">Diffusion tau_int</div>
                        <div className="text-xl font-mono text-purple-400">1.0</div>
                        <div className="text-xs text-green-400">
                          {analysisData.mcmc.autocorrelation_time.toFixed(0)}x faster
                        </div>
                      </div>
                    </div>
                    <div className="text-xs text-slate-500">
                      Lower autocorrelation time = more independent samples per step.
                      Diffusion samples are independent by design.
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-700 mt-auto">
        <div className="container mx-auto px-4 py-4 text-center text-sm text-slate-500">
          Diffusion Boltzmann Sampler - Neural sampling from statistical mechanics
        </div>
      </footer>
    </div>
  );
}

export default App;
