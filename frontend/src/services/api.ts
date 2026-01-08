const API_BASE = 'http://localhost:8000';

export interface SampleResponse {
  samples: number[][][];
  energies: number[];
  magnetizations: number[];
  temperature: number;
  lattice_size: number;
}

export interface AnalysisResponse {
  mcmc: {
    magnetization: { M: number[]; P_M: number[] };
    energy: { E: number[]; P_E: number[] };
    correlation: { r: number[]; C_r: number[] };
    autocorrelation_time: number;
    mean_mag: number;
    var_mag: number;
  };
  diffusion: {
    magnetization: { M: number[]; P_M: number[] };
    energy: { E: number[]; P_E: number[] };
    correlation: { r: number[]; C_r: number[] };
    mean_mag: number;
    var_mag: number;
  };
  comparison_metrics: Record<string, number>;
  temperature: number;
  lattice_size: number;
}

export interface PhaseDiagramResponse {
  temperatures: number[];
  mean_magnetization: number[];
  std_magnetization: number[];
  T_critical: number;
  lattice_size: number;
}

export async function checkHealth(): Promise<{ status: string }> {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) throw new Error('Backend not available');
  return response.json();
}

export async function getConfig(): Promise<Record<string, any>> {
  const response = await fetch(`${API_BASE}/config`);
  return response.json();
}

export async function sampleMCMC(params: {
  temperature: number;
  lattice_size: number;
  n_samples: number;
  n_sweeps?: number;
  burn_in?: number;
}): Promise<SampleResponse> {
  const response = await fetch(`${API_BASE}/sample/mcmc`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!response.ok) throw new Error('Sampling failed');
  return response.json();
}

export async function sampleDiffusion(params: {
  temperature: number;
  lattice_size: number;
  n_samples: number;
  num_steps?: number;
}): Promise<SampleResponse> {
  const response = await fetch(`${API_BASE}/sample/diffusion`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!response.ok) throw new Error('Diffusion sampling failed');
  return response.json();
}

export async function getRandomConfiguration(
  latticeSize: number
): Promise<{
  spins: number[][];
  energy: number;
  magnetization: number;
}> {
  const response = await fetch(
    `${API_BASE}/sample/random?lattice_size=${latticeSize}`
  );
  return response.json();
}

export async function compareSamplers(params: {
  temperature: number;
  lattice_size: number;
  n_samples: number;
}): Promise<AnalysisResponse> {
  const response = await fetch(`${API_BASE}/analysis/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!response.ok) throw new Error('Comparison failed');
  return response.json();
}

export async function getPhaseDiagram(params: {
  lattice_size?: number;
  n_temps?: number;
  n_samples_per_temp?: number;
}): Promise<PhaseDiagramResponse> {
  const queryParams = new URLSearchParams();
  if (params.lattice_size) queryParams.set('lattice_size', params.lattice_size.toString());
  if (params.n_temps) queryParams.set('n_temps', params.n_temps.toString());
  if (params.n_samples_per_temp)
    queryParams.set('n_samples_per_temp', params.n_samples_per_temp.toString());

  const response = await fetch(
    `${API_BASE}/analysis/phase_diagram?${queryParams}`
  );
  return response.json();
}

export function createSamplingWebSocket(
  onFrame: (data: { spins: number[][]; energy: number; magnetization: number; t?: number }) => void,
  onDone: () => void,
  onError: (error: string) => void
): WebSocket {
  const ws = new WebSocket('ws://localhost:8000/ws/sample');

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'frame') {
      onFrame(data);
    } else if (data.type === 'done') {
      onDone();
    } else if (data.type === 'error') {
      onError(data.message);
    }
  };

  ws.onerror = () => {
    onError('WebSocket connection error');
  };

  return ws;
}
