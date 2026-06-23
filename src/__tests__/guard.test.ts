import { runWithConcurrency } from '../pipeline/guard';

// ─── Fix 4: bounded concurrency ──────────────────────────────────────────────

describe('runWithConcurrency', () => {
  test('returns results in input order', async () => {
    const items = [3, 1, 4, 1, 5];
    const results = await runWithConcurrency(items, async (n) => n * 2, 3);
    expect(results).toEqual([6, 2, 8, 2, 10]);
  });

  test('never exceeds concurrency limit', async () => {
    let inFlight = 0;
    let maxInFlight = 0;

    const items = Array.from({ length: 20 }, (_, i) => i);
    await runWithConcurrency(
      items,
      async () => {
        inFlight++;
        maxInFlight = Math.max(maxInFlight, inFlight);
        await new Promise((r) => setTimeout(r, 5));
        inFlight--;
      },
      5,
    );

    expect(maxInFlight).toBeLessThanOrEqual(5);
  });

  test('concurrency=1 runs tasks sequentially', async () => {
    const order: number[] = [];
    const items = [0, 1, 2, 3];
    await runWithConcurrency(
      items,
      async (n) => {
        order.push(n);
        await new Promise((r) => setTimeout(r, 5));
      },
      1,
    );
    expect(order).toEqual([0, 1, 2, 3]);
  });

  test('handles fewer items than concurrency slots', async () => {
    const results = await runWithConcurrency([10, 20], async (n) => n + 1, 10);
    expect(results).toEqual([11, 21]);
  });

  test('propagates errors from fn', async () => {
    const items = [1, 2, 3];
    await expect(
      runWithConcurrency(items, async (n) => {
        if (n === 2) throw new Error('boom');
        return n;
      }, 2),
    ).rejects.toThrow('boom');
  });
});
