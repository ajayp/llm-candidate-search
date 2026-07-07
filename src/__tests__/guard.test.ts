const mockCreate = jest.fn();
jest.mock('openai', () =>
  jest.fn().mockImplementation(() => ({
    chat: { completions: { create: mockCreate } },
  })),
);

import { runWithConcurrency, isFacepalm, runGuard } from '../pipeline/guard';
import { RerankResult } from '../pipeline/reranking';
import { CandidateProfile, StructuredQuery } from '../types';

function makeProfile(seniority: CandidateProfile['seniority']): CandidateProfile {
  return {
    id: 'p1',
    jobCategory: 'Software Engineering',
    name: 'Candidate',
    title: 'Engineer',
    seniority,
    location: { city: 'San Francisco', country: 'United States' },
    yearsOfExperience: 5,
    skills: [],
    summary: '',
    experience: [],
    education: [],
  };
}

function makeResult(seniority: CandidateProfile['seniority']): RerankResult {
  return {
    profile: makeProfile(seniority),
    l1Score: 0.8,
    l2Score: 0.8,
    embeddingRecord: { profileId: 'p1', fullVector: [], shortVector: [], profileText: '', embeddedAt: '' },
  };
}

function makeQuery(seniority: string | null): StructuredQuery {
  return {
    raw: '',
    title: null,
    seniority,
    location: { city: null, region: null, country: null },
    locationStrict: false,
    qualifications: [],
    requiredQualifications: [],
    ambiguousQualifications: [],
    queryText: '',
  };
}

// ─── Cross-track seniority mismatch (IC ↔ management) ────────────────────────

describe('isFacepalm', () => {
  test('no query seniority: never a facepalm', () => {
    expect(isFacepalm(makeResult('senior'), makeQuery(null))).toBe(false);
  });

  test('same track, within 1 rank: not a facepalm', () => {
    expect(isFacepalm(makeResult('staff'), makeQuery('senior'))).toBe(false);
  });

  test('same track, 2+ rank gap: facepalm', () => {
    expect(isFacepalm(makeResult('junior'), makeQuery('staff'))).toBe(true);
  });

  test('cross-track, 1 rank gap: facepalm (numeric distance alone would miss this)', () => {
    // director (mgmt, rank 6) vs principal (IC, rank 5) — only 1 rank apart, but different tracks
    expect(isFacepalm(makeResult('principal'), makeQuery('director'))).toBe(true);
  });

  test('cross-track, exact numeric rank match: still a facepalm', () => {
    // chief (mgmt, rank 8) vs a hypothetical IC at the same numeric rank would still mismatch on track;
    // using vp (mgmt, rank 7) vs principal (IC, rank 5) to keep within real SENIORITY_RANK values
    expect(isFacepalm(makeResult('principal'), makeQuery('vp'))).toBe(true);
  });

  test('same track, exact match: not a facepalm', () => {
    expect(isFacepalm(makeResult('director'), makeQuery('director'))).toBe(false);
  });
});

// ─── Guard assessment parsing: fail-closed on broken model output ───────────

describe('runGuard — assessment parsing', () => {
  function mockResponse(content: string) {
    mockCreate.mockResolvedValue({ choices: [{ message: { content } }] });
  }

  beforeEach(() => {
    mockCreate.mockReset();
    jest.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  test('valid response passes fit and explanation through unchanged', async () => {
    mockResponse(JSON.stringify({ fit: 'good', explanation: 'meets most requirements', qualificationChecks: [] }));
    const [result] = await runGuard(makeQuery('senior'), [makeResult('senior')]);
    expect(result.fit).toBe('good');
    expect(result.guardExplanation).toBe('meets most requirements');
  });

  test('overrides self-reported fit to poor when 2+ qualifications are unmet', async () => {
    mockResponse(
      JSON.stringify({
        fit: 'good',
        explanation: 'looks fine',
        qualificationChecks: [
          { qualification: 'Python', met: false },
          { qualification: 'NLP', met: false },
        ],
      }),
    );
    const [result] = await runGuard(makeQuery('senior'), [makeResult('senior')]);
    expect(result.fit).toBe('poor');
  });

  test('unparseable JSON fails closed to poor with an auditable explanation', async () => {
    mockResponse('not json at all');
    const [result] = await runGuard(makeQuery('senior'), [makeResult('senior')]);
    expect(result.fit).toBe('poor');
    expect(result.guardExplanation).toMatch(/unparseable model output/i);
    expect(console.error).toHaveBeenCalled();
  });

  test('missing "fit" field fails closed to poor', async () => {
    mockResponse(JSON.stringify({ explanation: 'no fit field here' }));
    const [result] = await runGuard(makeQuery('senior'), [makeResult('senior')]);
    expect(result.fit).toBe('poor');
  });

  test('invalid fit value fails closed to poor', async () => {
    mockResponse(JSON.stringify({ fit: 'amazing', explanation: 'not a real fit level' }));
    const [result] = await runGuard(makeQuery('senior'), [makeResult('senior')]);
    expect(result.fit).toBe('poor');
  });

  test('missing "explanation" field fails closed to poor', async () => {
    mockResponse(JSON.stringify({ fit: 'good' }));
    const [result] = await runGuard(makeQuery('senior'), [makeResult('senior')]);
    expect(result.fit).toBe('poor');
  });
});

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
