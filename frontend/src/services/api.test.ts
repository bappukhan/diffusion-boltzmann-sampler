/**
 * Unit tests for API service functions.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  checkHealth,
  getConfig,
  sampleMCMC,
  sampleDiffusion,
  getRandomConfiguration,
  compareSamplers,
  getPhaseDiagram,
  createSamplingWebSocket,
} from './api';
import { APIError, TimeoutError, NetworkError } from '../utils/errors';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Mock WebSocket
class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  url: string;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onopen: ((event: Event) => void) | null = null;
  readyState = MockWebSocket.CONNECTING;

  constructor(url: string) {
    this.url = url;
  }

  send(data: string) {
    // Mock send
  }

  close() {
    this.readyState = MockWebSocket.CLOSED;
  }
}

global.WebSocket = MockWebSocket as unknown as typeof WebSocket;

describe('API service', () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('checkHealth', () => {
    it('should return health status on success', async () => {
      const healthResponse = { status: 'healthy', version: '1.0.0' };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(healthResponse),
      });

      const result = await checkHealth();

      expect(result).toEqual(healthResponse);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/health'),
        expect.objectContaining({ signal: expect.any(AbortSignal) })
      );
    });

    it('should throw APIError on non-ok response', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 503,
        statusText: 'Service Unavailable',
        json: () => Promise.resolve({ detail: 'Server is starting up' }),
      });

      await expect(checkHealth()).rejects.toThrow(APIError);
    });

    it('should throw TimeoutError on abort', async () => {
      mockFetch.mockImplementationOnce(() => {
        const error = new Error('Aborted');
        error.name = 'AbortError';
        return Promise.reject(error);
      });

      await expect(checkHealth()).rejects.toThrow(TimeoutError);
    });
  });

  describe('getConfig', () => {
    it('should return configuration', async () => {
      const config = { temperature: 2.27, lattice_size: 32 };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(config),
      });

      const result = await getConfig();

      expect(result).toEqual(config);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/config'),
        expect.any(Object)
      );
    });
  });

  describe('sampleMCMC', () => {
    it('should send correct parameters and return samples', async () => {
      const response = {
        samples: [[[1, -1], [-1, 1]]],
        energies: [-8],
        magnetizations: [0],
        temperature: 2.27,
        lattice_size: 2,
      };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(response),
      });

      const params = {
        temperature: 2.27,
        lattice_size: 32,
        n_samples: 100,
        n_sweeps: 10,
      };

      const result = await sampleMCMC(params);

      expect(result).toEqual(response);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/sample/mcmc'),
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(params),
        })
      );
    });

    it('should throw APIError on server error', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        json: () => Promise.resolve({ detail: 'Sampling failed' }),
      });

      await expect(
        sampleMCMC({ temperature: 2.27, lattice_size: 32, n_samples: 100 })
      ).rejects.toThrow(APIError);
    });
  });

  describe('sampleDiffusion', () => {
    it('should send correct parameters', async () => {
      const response = {
        samples: [[[1, 1], [1, 1]]],
        energies: [-8],
        magnetizations: [1],
        temperature: 1.5,
        lattice_size: 2,
      };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(response),
      });

      const params = {
        temperature: 1.5,
        lattice_size: 16,
        n_samples: 50,
        num_steps: 100,
      };

      const result = await sampleDiffusion(params);

      expect(result).toEqual(response);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/sample/diffusion'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify(params),
        })
      );
    });
  });

  describe('getRandomConfiguration', () => {
    it('should request random configuration with lattice size', async () => {
      const response = {
        spins: [[1, -1], [-1, 1]],
        energy: -4,
        magnetization: 0,
      };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(response),
      });

      const result = await getRandomConfiguration(32);

      expect(result).toEqual(response);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/sample/random?lattice_size=32'),
        expect.any(Object)
      );
    });
  });

  describe('compareSamplers', () => {
    it('should send comparison request', async () => {
      const response = {
        mcmc: {
          magnetization: { M: [0, 0.5], P_M: [0.5, 0.5] },
          energy: { E: [-100, -90], P_E: [0.5, 0.5] },
          correlation: { r: [1, 2], C_r: [1, 0.5] },
          autocorrelation_time: 10,
          mean_mag: 0.25,
          var_mag: 0.1,
        },
        diffusion: {
          magnetization: { M: [0, 0.5], P_M: [0.5, 0.5] },
          energy: { E: [-100, -90], P_E: [0.5, 0.5] },
          correlation: { r: [1, 2], C_r: [1, 0.5] },
          mean_mag: 0.25,
          var_mag: 0.1,
        },
        comparison_metrics: { kl_divergence: 0.01 },
        temperature: 2.27,
        lattice_size: 32,
      };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(response),
      });

      const params = { temperature: 2.27, lattice_size: 32, n_samples: 100 };
      const result = await compareSamplers(params);

      expect(result).toEqual(response);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/analysis/compare'),
        expect.objectContaining({ method: 'POST' })
      );
    });
  });

  describe('getPhaseDiagram', () => {
    it('should request phase diagram with parameters', async () => {
      const response = {
        temperatures: [1.0, 2.0, 3.0],
        mean_magnetization: [1.0, 0.5, 0.1],
        std_magnetization: [0.01, 0.1, 0.05],
        T_critical: 2.269,
        lattice_size: 32,
      };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(response),
      });

      const params = { lattice_size: 32, n_temps: 10, n_samples_per_temp: 50 };
      const result = await getPhaseDiagram(params);

      expect(result).toEqual(response);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringMatching(/\/analysis\/phase_diagram\?.*lattice_size=32/),
        expect.any(Object)
      );
    });

    it('should handle empty parameters', async () => {
      const response = {
        temperatures: [1.0, 2.0],
        mean_magnetization: [1.0, 0.5],
        std_magnetization: [0.01, 0.1],
        T_critical: 2.269,
        lattice_size: 16,
      };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(response),
      });

      await getPhaseDiagram({});

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/analysis/phase_diagram?'),
        expect.any(Object)
      );
    });
  });

  describe('error handling', () => {
    it('should extract error detail from response', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        statusText: 'Bad Request',
        json: () => Promise.resolve({ detail: 'Invalid temperature value' }),
      });

      try {
        await checkHealth();
        expect.fail('Should have thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(APIError);
        expect((error as APIError).message).toBe('Invalid temperature value');
      }
    });

    it('should use default message when JSON parsing fails', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        json: () => Promise.reject(new Error('Not JSON')),
      });

      try {
        await checkHealth();
        expect.fail('Should have thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(APIError);
        expect((error as APIError).message).toContain('500');
      }
    });

    it('should throw NetworkError on fetch failure', async () => {
      const fetchError = new TypeError('Failed to fetch');
      mockFetch.mockRejectedValueOnce(fetchError);

      await expect(checkHealth()).rejects.toThrow(NetworkError);
    });
  });

  describe('createSamplingWebSocket', () => {
    it('should create WebSocket with correct URL', () => {
      const onFrame = vi.fn();
      const onDone = vi.fn();
      const onError = vi.fn();

      const ws = createSamplingWebSocket(onFrame, onDone, onError);

      expect(ws).toBeInstanceOf(MockWebSocket);
      expect(ws.url).toContain('/ws/sample');
    });

    it('should call onFrame for frame messages', () => {
      const onFrame = vi.fn();
      const onDone = vi.fn();
      const onError = vi.fn();

      const ws = createSamplingWebSocket(onFrame, onDone, onError);

      const frameData = { type: 'frame', spins: [[1]], energy: -4, magnetization: 1 };
      ws.onmessage?.({ data: JSON.stringify(frameData) } as MessageEvent);

      expect(onFrame).toHaveBeenCalledWith(frameData);
    });

    it('should call onDone for done messages', () => {
      const onFrame = vi.fn();
      const onDone = vi.fn();
      const onError = vi.fn();

      const ws = createSamplingWebSocket(onFrame, onDone, onError);

      ws.onmessage?.({ data: JSON.stringify({ type: 'done' }) } as MessageEvent);

      expect(onDone).toHaveBeenCalled();
    });

    it('should call onError for error messages', () => {
      const onFrame = vi.fn();
      const onDone = vi.fn();
      const onError = vi.fn();

      const ws = createSamplingWebSocket(onFrame, onDone, onError);

      ws.onmessage?.({
        data: JSON.stringify({ type: 'error', message: 'Test error' }),
      } as MessageEvent);

      expect(onError).toHaveBeenCalled();
    });

    it('should call onError for invalid JSON', () => {
      const onFrame = vi.fn();
      const onDone = vi.fn();
      const onError = vi.fn();

      const ws = createSamplingWebSocket(onFrame, onDone, onError);

      ws.onmessage?.({ data: 'invalid json' } as MessageEvent);

      expect(onError).toHaveBeenCalled();
    });

    it('should call onError on connection error', () => {
      const onFrame = vi.fn();
      const onDone = vi.fn();
      const onError = vi.fn();

      const ws = createSamplingWebSocket(onFrame, onDone, onError);

      ws.onerror?.(new Event('error'));

      expect(onError).toHaveBeenCalled();
    });

    it('should call onError on unclean close', () => {
      const onFrame = vi.fn();
      const onDone = vi.fn();
      const onError = vi.fn();

      const ws = createSamplingWebSocket(onFrame, onDone, onError);

      ws.onclose?.({ wasClean: false, code: 1006 } as CloseEvent);

      expect(onError).toHaveBeenCalled();
    });

    it('should not call onError on clean close', () => {
      const onFrame = vi.fn();
      const onDone = vi.fn();
      const onError = vi.fn();

      const ws = createSamplingWebSocket(onFrame, onDone, onError);

      ws.onclose?.({ wasClean: true, code: 1000 } as CloseEvent);

      expect(onError).not.toHaveBeenCalled();
    });
  });
});
