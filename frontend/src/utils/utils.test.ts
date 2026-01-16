/**
 * Unit tests for utility functions.
 */

import { describe, it, expect } from 'vitest';

// Format utilities
import {
  formatNumber,
  formatTemperature,
  formatEnergy,
  formatMagnetization,
  formatLatticeSize,
  formatPercentage,
  formatFrameCount,
  formatDuration,
  formatScientific,
  formatAutocorrelationTime,
  getPhaseDescription,
  getTemperatureColor,
} from './format';

// Validation utilities
import {
  validateTemperature,
  validateLatticeSize,
  validateNumSteps,
  validateNumSamples,
  validateSpinConfiguration,
  validateSamplingParams,
  clamp,
  clampTemperature,
  clampLatticeSize,
  isInRange,
} from './validation';

// Error utilities
import {
  ErrorCode,
  AppError,
  NetworkError,
  TimeoutError,
  APIError,
  WebSocketError,
  parseError,
  isErrorCode,
} from './errors';

describe('format utilities', () => {
  describe('formatNumber', () => {
    it('should format number with default decimals', () => {
      expect(formatNumber(3.14159)).toBe('3.142');
    });

    it('should format number with custom decimals', () => {
      expect(formatNumber(3.14159, 2)).toBe('3.14');
      expect(formatNumber(3.14159, 5)).toBe('3.14159');
    });

    it('should handle integers', () => {
      expect(formatNumber(42, 2)).toBe('42.00');
    });
  });

  describe('formatTemperature', () => {
    it('should format temperature with default options', () => {
      expect(formatTemperature(1.5)).toBe('1.50');
    });

    it('should show critical indicator for T_c', () => {
      expect(formatTemperature(2.27)).toBe('2.27 (T_c)');
      expect(formatTemperature(2.3)).toBe('2.30 (T_c)');
    });

    it('should hide critical indicator when disabled', () => {
      expect(formatTemperature(2.27, { showCritical: false })).toBe('2.27');
    });

    it('should respect custom decimals', () => {
      expect(formatTemperature(2.5, { decimals: 3 })).toBe('2.500');
    });

    it('should not show critical for temperatures far from T_c', () => {
      expect(formatTemperature(1.0)).toBe('1.00');
      expect(formatTemperature(4.0)).toBe('4.00');
    });
  });

  describe('formatEnergy', () => {
    it('should format negative energy', () => {
      expect(formatEnergy(-128)).toBe('-128.000');
    });

    it('should format positive energy with plus sign', () => {
      expect(formatEnergy(50)).toBe('+50.000');
    });

    it('should format zero with plus sign', () => {
      expect(formatEnergy(0)).toBe('+0.000');
    });

    it('should respect custom decimals', () => {
      expect(formatEnergy(-64, 1)).toBe('-64.0');
    });
  });

  describe('formatMagnetization', () => {
    it('should format negative magnetization', () => {
      expect(formatMagnetization(-0.5)).toBe('-0.500');
    });

    it('should format positive magnetization with plus sign', () => {
      expect(formatMagnetization(0.75)).toBe('+0.750');
    });

    it('should format zero with plus sign', () => {
      expect(formatMagnetization(0)).toBe('+0.000');
    });
  });

  describe('formatLatticeSize', () => {
    it('should format as NxN', () => {
      expect(formatLatticeSize(32)).toBe('32 \u00D7 32');
      expect(formatLatticeSize(16)).toBe('16 \u00D7 16');
    });
  });

  describe('formatPercentage', () => {
    it('should format as percentage', () => {
      expect(formatPercentage(0.5)).toBe('50.0%');
      expect(formatPercentage(0.123)).toBe('12.3%');
    });

    it('should respect custom decimals', () => {
      expect(formatPercentage(0.5678, 2)).toBe('56.78%');
    });
  });

  describe('formatFrameCount', () => {
    it('should format as 1-indexed', () => {
      expect(formatFrameCount(0, 100)).toBe('1 / 100');
      expect(formatFrameCount(50, 100)).toBe('51 / 100');
    });
  });

  describe('formatDuration', () => {
    it('should format milliseconds', () => {
      expect(formatDuration(500)).toBe('500ms');
    });

    it('should format seconds', () => {
      expect(formatDuration(1500)).toBe('1.5s');
      expect(formatDuration(30000)).toBe('30.0s');
    });

    it('should format minutes and seconds', () => {
      expect(formatDuration(90000)).toBe('1m 30s');
      expect(formatDuration(120000)).toBe('2m 0s');
    });
  });

  describe('formatScientific', () => {
    it('should format normal numbers', () => {
      expect(formatScientific(0.01)).toBe('0.0100');
      expect(formatScientific(1234)).toBe('1.23e+3');
    });

    it('should format very small numbers', () => {
      expect(formatScientific(0.0001)).toBe('1.00e-4');
    });

    it('should format very large numbers', () => {
      expect(formatScientific(100000)).toBe('1.00e+5');
    });

    it('should handle zero', () => {
      expect(formatScientific(0)).toBe('0');
    });
  });

  describe('formatAutocorrelationTime', () => {
    it('should format tau', () => {
      expect(formatAutocorrelationTime(10.5)).toBe('10.5');
    });

    it('should show speedup when reference is provided', () => {
      expect(formatAutocorrelationTime(5, 100)).toBe('5.0 (20x faster)');
    });

    it('should not show speedup if tau >= reference', () => {
      expect(formatAutocorrelationTime(100, 50)).toBe('100.0');
    });
  });

  describe('getPhaseDescription', () => {
    it('should return ordered for low temperature', () => {
      expect(getPhaseDescription(1.0)).toBe('Ordered (Ferromagnetic)');
      expect(getPhaseDescription(2.0)).toBe('Ordered (Ferromagnetic)');
    });

    it('should return disordered for high temperature', () => {
      expect(getPhaseDescription(3.0)).toBe('Disordered (Paramagnetic)');
      expect(getPhaseDescription(4.0)).toBe('Disordered (Paramagnetic)');
    });

    it('should return critical near T_c', () => {
      expect(getPhaseDescription(2.27)).toBe('Critical (Phase Transition)');
      expect(getPhaseDescription(2.3)).toBe('Critical (Phase Transition)');
    });
  });

  describe('getTemperatureColor', () => {
    it('should return blue for low temperature', () => {
      expect(getTemperatureColor(1.0)).toBe('text-blue-400');
    });

    it('should return red for high temperature', () => {
      expect(getTemperatureColor(4.0)).toBe('text-red-400');
    });

    it('should return yellow near critical', () => {
      expect(getTemperatureColor(2.27)).toBe('text-yellow-400');
    });
  });
});

describe('validation utilities', () => {
  describe('validateTemperature', () => {
    it('should accept valid temperatures', () => {
      expect(validateTemperature(1.0).isValid).toBe(true);
      expect(validateTemperature(2.27).isValid).toBe(true);
      expect(validateTemperature(5.0).isValid).toBe(true);
    });

    it('should reject too low temperature', () => {
      const result = validateTemperature(0);
      expect(result.isValid).toBe(false);
      expect(result.error).toContain('at least');
    });

    it('should reject too high temperature', () => {
      const result = validateTemperature(100);
      expect(result.isValid).toBe(false);
      expect(result.error).toContain('at most');
    });

    it('should reject NaN', () => {
      const result = validateTemperature(NaN);
      expect(result.isValid).toBe(false);
      expect(result.error).toContain('number');
    });
  });

  describe('validateLatticeSize', () => {
    it('should accept valid sizes', () => {
      expect(validateLatticeSize(8).isValid).toBe(true);
      expect(validateLatticeSize(32).isValid).toBe(true);
      expect(validateLatticeSize(64).isValid).toBe(true);
    });

    it('should reject too small size', () => {
      const result = validateLatticeSize(2);
      expect(result.isValid).toBe(false);
      expect(result.error).toContain('at least');
    });

    it('should reject too large size', () => {
      const result = validateLatticeSize(1000);
      expect(result.isValid).toBe(false);
      expect(result.error).toContain('at most');
    });

    it('should reject non-integers', () => {
      const result = validateLatticeSize(32.5);
      expect(result.isValid).toBe(false);
      expect(result.error).toContain('integer');
    });
  });

  describe('validateNumSteps', () => {
    it('should accept valid steps', () => {
      expect(validateNumSteps(1).isValid).toBe(true);
      expect(validateNumSteps(100).isValid).toBe(true);
      expect(validateNumSteps(1000).isValid).toBe(true);
    });

    it('should reject zero or negative', () => {
      expect(validateNumSteps(0).isValid).toBe(false);
      expect(validateNumSteps(-1).isValid).toBe(false);
    });

    it('should reject too many steps', () => {
      const result = validateNumSteps(5000);
      expect(result.isValid).toBe(false);
      expect(result.error).toContain('at most');
    });
  });

  describe('validateNumSamples', () => {
    it('should accept valid samples', () => {
      expect(validateNumSamples(1).isValid).toBe(true);
      expect(validateNumSamples(100).isValid).toBe(true);
    });

    it('should reject too many samples', () => {
      const result = validateNumSamples(50000);
      expect(result.isValid).toBe(false);
    });
  });

  describe('validateSpinConfiguration', () => {
    it('should accept null', () => {
      expect(validateSpinConfiguration(null).isValid).toBe(true);
    });

    it('should accept valid configuration', () => {
      const spins = [
        [1, -1],
        [-1, 1],
      ];
      expect(validateSpinConfiguration(spins).isValid).toBe(true);
    });

    it('should reject empty array', () => {
      const result = validateSpinConfiguration([]);
      expect(result.isValid).toBe(false);
      expect(result.error).toContain('empty');
    });

    it('should reject non-square', () => {
      const spins = [
        [1, -1, 1],
        [-1, 1],
      ];
      const result = validateSpinConfiguration(spins);
      expect(result.isValid).toBe(false);
      expect(result.error).toContain('square');
    });

    it('should reject invalid spin values', () => {
      const spins = [
        [1, 0],
        [-1, 1],
      ];
      const result = validateSpinConfiguration(spins);
      expect(result.isValid).toBe(false);
      expect(result.error).toContain('Invalid spin value');
    });
  });

  describe('validateSamplingParams', () => {
    it('should accept valid params', () => {
      const params = { temperature: 2.27, latticeSize: 32, numSteps: 100 };
      expect(validateSamplingParams(params).isValid).toBe(true);
    });

    it('should return first validation error', () => {
      const params = { temperature: 0, latticeSize: 2, numSteps: 0 };
      const result = validateSamplingParams(params);
      expect(result.isValid).toBe(false);
      expect(result.error).toContain('Temperature');
    });
  });

  describe('clamp', () => {
    it('should clamp to min', () => {
      expect(clamp(0, 1, 10)).toBe(1);
    });

    it('should clamp to max', () => {
      expect(clamp(20, 1, 10)).toBe(10);
    });

    it('should return value if in range', () => {
      expect(clamp(5, 1, 10)).toBe(5);
    });
  });

  describe('clampTemperature', () => {
    it('should clamp temperature', () => {
      expect(clampTemperature(-1)).toBeGreaterThan(0);
      expect(clampTemperature(100)).toBeLessThan(100);
      expect(clampTemperature(2.27)).toBe(2.27);
    });
  });

  describe('clampLatticeSize', () => {
    it('should clamp and round', () => {
      expect(clampLatticeSize(1)).toBeGreaterThanOrEqual(4);
      expect(clampLatticeSize(1000)).toBeLessThanOrEqual(256);
      expect(clampLatticeSize(32.7)).toBe(33);
    });
  });

  describe('isInRange', () => {
    it('should check range', () => {
      expect(isInRange(5, 1, 10)).toBe(true);
      expect(isInRange(0, 1, 10)).toBe(false);
      expect(isInRange(11, 1, 10)).toBe(false);
      expect(isInRange(1, 1, 10)).toBe(true);
      expect(isInRange(10, 1, 10)).toBe(true);
    });
  });
});

describe('error utilities', () => {
  describe('ErrorCode', () => {
    it('should have network error codes', () => {
      expect(ErrorCode.NETWORK_ERROR).toBe('NETWORK_ERROR');
      expect(ErrorCode.TIMEOUT_ERROR).toBe('TIMEOUT_ERROR');
    });

    it('should have API error codes', () => {
      expect(ErrorCode.API_ERROR).toBe('API_ERROR');
      expect(ErrorCode.VALIDATION_ERROR).toBe('VALIDATION_ERROR');
    });

    it('should have WebSocket error codes', () => {
      expect(ErrorCode.WS_CONNECTION_ERROR).toBe('WS_CONNECTION_ERROR');
    });
  });

  describe('AppError', () => {
    it('should create error with message and code', () => {
      const error = new AppError('Test error', ErrorCode.API_ERROR);
      expect(error.message).toBe('Test error');
      expect(error.code).toBe(ErrorCode.API_ERROR);
      expect(error.name).toBe('AppError');
    });

    it('should default to UNKNOWN_ERROR', () => {
      const error = new AppError('Unknown');
      expect(error.code).toBe(ErrorCode.UNKNOWN_ERROR);
    });

    it('should include timestamp', () => {
      const error = new AppError('Test');
      expect(error.timestamp).toBeInstanceOf(Date);
    });

    it('should get user-friendly message', () => {
      const error = new AppError('Original', ErrorCode.NETWORK_ERROR);
      expect(error.getUserMessage()).toContain('connect');
    });

    it('should check if recoverable', () => {
      expect(new AppError('', ErrorCode.NETWORK_ERROR).isRecoverable()).toBe(true);
      expect(new AppError('', ErrorCode.TIMEOUT_ERROR).isRecoverable()).toBe(true);
      expect(new AppError('', ErrorCode.VALIDATION_ERROR).isRecoverable()).toBe(false);
    });

    it('should serialize to JSON', () => {
      const error = new AppError('Test', ErrorCode.API_ERROR, {
        statusCode: 500,
        details: { foo: 'bar' },
      });
      const json = error.toJSON();
      expect(json.message).toBe('Test');
      expect(json.code).toBe(ErrorCode.API_ERROR);
      expect(json.statusCode).toBe(500);
      expect(json.details).toEqual({ foo: 'bar' });
    });
  });

  describe('NetworkError', () => {
    it('should create network error', () => {
      const error = new NetworkError('Connection failed');
      expect(error.code).toBe(ErrorCode.NETWORK_ERROR);
      expect(error.name).toBe('NetworkError');
    });
  });

  describe('TimeoutError', () => {
    it('should create timeout error with timeout value', () => {
      const error = new TimeoutError('Timed out', 5000);
      expect(error.code).toBe(ErrorCode.TIMEOUT_ERROR);
      expect(error.name).toBe('TimeoutError');
      expect(error.timeoutMs).toBe(5000);
    });
  });

  describe('APIError', () => {
    it('should create API error with status code', () => {
      const error = new APIError('Not found', 404);
      expect(error.code).toBe(ErrorCode.NOT_FOUND);
      expect(error.name).toBe('APIError');
      expect(error.statusCode).toBe(404);
    });

    it('should map status codes correctly', () => {
      expect(new APIError('', 400).code).toBe(ErrorCode.VALIDATION_ERROR);
      expect(new APIError('', 500).code).toBe(ErrorCode.SERVER_ERROR);
      expect(new APIError('', 503).code).toBe(ErrorCode.SERVER_ERROR);
    });
  });

  describe('WebSocketError', () => {
    it('should create WebSocket error', () => {
      const error = new WebSocketError('Connection failed', ErrorCode.WS_CONNECTION_ERROR);
      expect(error.code).toBe(ErrorCode.WS_CONNECTION_ERROR);
      expect(error.name).toBe('WebSocketError');
    });
  });

  describe('parseError', () => {
    it('should return AppError unchanged', () => {
      const original = new AppError('Test', ErrorCode.API_ERROR);
      expect(parseError(original)).toBe(original);
    });

    it('should parse Error to AppError', () => {
      const original = new Error('Test error');
      const parsed = parseError(original);
      expect(parsed).toBeInstanceOf(AppError);
      expect(parsed.message).toBe('Test error');
    });

    it('should detect network errors', () => {
      const original = new Error('Failed to fetch');
      const parsed = parseError(original);
      expect(parsed).toBeInstanceOf(NetworkError);
    });

    it('should parse string errors', () => {
      const parsed = parseError('Something went wrong');
      expect(parsed).toBeInstanceOf(AppError);
      expect(parsed.message).toBe('Something went wrong');
    });

    it('should handle unknown types', () => {
      const parsed = parseError({ foo: 'bar' });
      expect(parsed).toBeInstanceOf(AppError);
      expect(parsed.code).toBe(ErrorCode.UNKNOWN_ERROR);
    });
  });

  describe('isErrorCode', () => {
    it('should check error code', () => {
      const error = new AppError('Test', ErrorCode.API_ERROR);
      expect(isErrorCode(error, ErrorCode.API_ERROR)).toBe(true);
      expect(isErrorCode(error, ErrorCode.NETWORK_ERROR)).toBe(false);
    });

    it('should return false for non-AppError', () => {
      expect(isErrorCode(new Error('Test'), ErrorCode.API_ERROR)).toBe(false);
      expect(isErrorCode('string', ErrorCode.API_ERROR)).toBe(false);
    });
  });
});
