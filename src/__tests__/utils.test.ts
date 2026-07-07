import { withRetry } from '../utils';

function errorWithStatus(status: number, message = 'error'): Error & { status: number } {
  const err = new Error(message) as Error & { status: number };
  err.status = status;
  return err;
}

// ─── withRetry: retry vs. fail-fast classification ───────────────────────────

describe('withRetry', () => {
  test('returns the result on first success without retrying', async () => {
    const fn = jest.fn().mockResolvedValue('ok');
    const result = await withRetry(fn, 5, 1);
    expect(result).toBe('ok');
    expect(fn).toHaveBeenCalledTimes(1);
  });

  test('fails fast on a non-retryable 4xx error (no retries attempted)', async () => {
    const fn = jest.fn().mockRejectedValue(errorWithStatus(400));
    await expect(withRetry(fn, 5, 1)).rejects.toThrow();
    expect(fn).toHaveBeenCalledTimes(1);
  });

  test('fails fast on 401 without exhausting attempts', async () => {
    const fn = jest.fn().mockRejectedValue(errorWithStatus(401));
    await expect(withRetry(fn, 5, 1)).rejects.toThrow();
    expect(fn).toHaveBeenCalledTimes(1);
  });

  test('retries on 429 (rate limit) up to maxAttempts', async () => {
    const fn = jest.fn().mockRejectedValue(errorWithStatus(429));
    await expect(withRetry(fn, 3, 1)).rejects.toThrow();
    expect(fn).toHaveBeenCalledTimes(3);
  });

  test('retries on 5xx server errors up to maxAttempts', async () => {
    const fn = jest.fn().mockRejectedValue(errorWithStatus(503));
    await expect(withRetry(fn, 3, 1)).rejects.toThrow();
    expect(fn).toHaveBeenCalledTimes(3);
  });

  test('retries errors with no status (network failures) up to maxAttempts', async () => {
    const fn = jest.fn().mockRejectedValue(new Error('ECONNRESET'));
    await expect(withRetry(fn, 3, 1)).rejects.toThrow('ECONNRESET');
    expect(fn).toHaveBeenCalledTimes(3);
  });

  test('succeeds after transient retryable failures', async () => {
    const fn = jest
      .fn()
      .mockRejectedValueOnce(errorWithStatus(429))
      .mockRejectedValueOnce(errorWithStatus(500))
      .mockResolvedValue('recovered');
    const result = await withRetry(fn, 5, 1);
    expect(result).toBe('recovered');
    expect(fn).toHaveBeenCalledTimes(3);
  });
});
