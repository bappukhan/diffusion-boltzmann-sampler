/**
 * Custom error classes for typed error handling.
 */

/**
 * Error codes for categorizing errors.
 */
export enum ErrorCode {
  // Network errors
  NETWORK_ERROR = 'NETWORK_ERROR',
  TIMEOUT_ERROR = 'TIMEOUT_ERROR',
  CONNECTION_REFUSED = 'CONNECTION_REFUSED',

  // API errors
  API_ERROR = 'API_ERROR',
  VALIDATION_ERROR = 'VALIDATION_ERROR',
  NOT_FOUND = 'NOT_FOUND',
  SERVER_ERROR = 'SERVER_ERROR',

  // WebSocket errors
  WS_CONNECTION_ERROR = 'WS_CONNECTION_ERROR',
  WS_MESSAGE_ERROR = 'WS_MESSAGE_ERROR',
  WS_CLOSED = 'WS_CLOSED',

  // Application errors
  INVALID_STATE = 'INVALID_STATE',
  UNKNOWN_ERROR = 'UNKNOWN_ERROR',
}

/**
 * Base error class for application errors.
 */
export class AppError extends Error {
  public readonly code: ErrorCode;
  public readonly statusCode?: number;
  public readonly details?: Record<string, unknown>;
  public readonly timestamp: Date;

  constructor(
    message: string,
    code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
    options?: {
      statusCode?: number;
      details?: Record<string, unknown>;
      cause?: Error;
    }
  ) {
    super(message, { cause: options?.cause });
    this.name = 'AppError';
    this.code = code;
    this.statusCode = options?.statusCode;
    this.details = options?.details;
    this.timestamp = new Date();

    // Maintain proper stack trace
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, AppError);
    }
  }

  /**
   * Get user-friendly error message.
   */
  getUserMessage(): string {
    return getUserFriendlyMessage(this.code, this.message);
  }

  /**
   * Check if error is recoverable (can retry).
   */
  isRecoverable(): boolean {
    return [
      ErrorCode.NETWORK_ERROR,
      ErrorCode.TIMEOUT_ERROR,
      ErrorCode.CONNECTION_REFUSED,
      ErrorCode.WS_CONNECTION_ERROR,
    ].includes(this.code);
  }

  /**
   * Serialize error for logging.
   */
  toJSON(): Record<string, unknown> {
    return {
      name: this.name,
      message: this.message,
      code: this.code,
      statusCode: this.statusCode,
      details: this.details,
      timestamp: this.timestamp.toISOString(),
      stack: this.stack,
    };
  }
}

/**
 * Network-related error.
 */
export class NetworkError extends AppError {
  constructor(message: string, options?: { cause?: Error }) {
    super(message, ErrorCode.NETWORK_ERROR, options);
    this.name = 'NetworkError';
  }
}

/**
 * Timeout error.
 */
export class TimeoutError extends AppError {
  public readonly timeoutMs: number;

  constructor(message: string, timeoutMs: number, options?: { cause?: Error }) {
    super(message, ErrorCode.TIMEOUT_ERROR, {
      ...options,
      details: { timeoutMs },
    });
    this.name = 'TimeoutError';
    this.timeoutMs = timeoutMs;
  }
}

/**
 * API error from backend.
 */
export class APIError extends AppError {
  constructor(
    message: string,
    statusCode: number,
    options?: { details?: Record<string, unknown>; cause?: Error }
  ) {
    const code = getErrorCodeFromStatus(statusCode);
    super(message, code, { ...options, statusCode });
    this.name = 'APIError';
  }
}

/**
 * WebSocket error.
 */
export class WebSocketError extends AppError {
  constructor(
    message: string,
    code: ErrorCode.WS_CONNECTION_ERROR | ErrorCode.WS_MESSAGE_ERROR | ErrorCode.WS_CLOSED,
    options?: { cause?: Error }
  ) {
    super(message, code, options);
    this.name = 'WebSocketError';
  }
}

/**
 * Map HTTP status code to error code.
 */
function getErrorCodeFromStatus(status: number): ErrorCode {
  if (status === 400) return ErrorCode.VALIDATION_ERROR;
  if (status === 404) return ErrorCode.NOT_FOUND;
  if (status >= 500) return ErrorCode.SERVER_ERROR;
  return ErrorCode.API_ERROR;
}

/**
 * Get user-friendly error message.
 */
function getUserFriendlyMessage(code: ErrorCode, fallback: string): string {
  const messages: Record<ErrorCode, string> = {
    [ErrorCode.NETWORK_ERROR]: 'Unable to connect to the server. Please check your internet connection.',
    [ErrorCode.TIMEOUT_ERROR]: 'The request took too long. Please try again.',
    [ErrorCode.CONNECTION_REFUSED]: 'Cannot connect to the backend. Is the server running?',
    [ErrorCode.API_ERROR]: 'An error occurred while processing your request.',
    [ErrorCode.VALIDATION_ERROR]: 'Invalid input parameters. Please check your values.',
    [ErrorCode.NOT_FOUND]: 'The requested resource was not found.',
    [ErrorCode.SERVER_ERROR]: 'A server error occurred. Please try again later.',
    [ErrorCode.WS_CONNECTION_ERROR]: 'WebSocket connection failed. Please refresh the page.',
    [ErrorCode.WS_MESSAGE_ERROR]: 'Error receiving data from server.',
    [ErrorCode.WS_CLOSED]: 'Connection to server was closed.',
    [ErrorCode.INVALID_STATE]: 'The application is in an invalid state.',
    [ErrorCode.UNKNOWN_ERROR]: fallback,
  };

  return messages[code] || fallback;
}

/**
 * Parse an unknown error into an AppError.
 */
export function parseError(error: unknown): AppError {
  if (error instanceof AppError) {
    return error;
  }

  if (error instanceof Error) {
    // Check for network-related errors
    if (error.message.includes('fetch') || error.message.includes('network')) {
      return new NetworkError(error.message, { cause: error });
    }

    return new AppError(error.message, ErrorCode.UNKNOWN_ERROR, { cause: error });
  }

  if (typeof error === 'string') {
    return new AppError(error, ErrorCode.UNKNOWN_ERROR);
  }

  return new AppError('An unknown error occurred', ErrorCode.UNKNOWN_ERROR);
}

/**
 * Check if an error is a specific type.
 */
export function isErrorCode(error: unknown, code: ErrorCode): boolean {
  return error instanceof AppError && error.code === code;
}
