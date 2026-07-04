import { matchesLocation, matchesSeniority, buildScoredTiers, RetrievalResult } from '../pipeline/retrieval';
import { CandidateProfile, StructuredQuery } from '../types';

function makeProfile(
  id: string,
  city: string,
  country: string,
  seniority: CandidateProfile['seniority'],
): CandidateProfile {
  return {
    id,
    jobCategory: 'Software Engineering',
    name: `Candidate ${id}`,
    title: 'Engineer',
    seniority,
    location: { city, country },
    yearsOfExperience: 5,
    skills: [],
    summary: '',
    experience: [],
    education: [],
  };
}

function makeResult(profile: CandidateProfile, l1Score: number): RetrievalResult {
  return {
    profile,
    l1Score,
    embeddingRecord: { profileId: profile.id, fullVector: [], shortVector: [], profileText: '', embeddedAt: '' },
  };
}

function makeQuery(overrides: Partial<StructuredQuery> = {}): StructuredQuery {
  return {
    raw: '',
    title: null,
    seniority: null,
    location: { city: null, region: null, country: null },
    locationStrict: false,
    qualifications: [],
    requiredQualifications: [],
    ambiguousQualifications: [],
    queryText: '',
    ...overrides,
  };
}

// ─── Fix 1: city vs metro-area location filtering ────────────────────────────

describe('matchesLocation — city vs region', () => {
  test('metro query (region set): Oakland passes for Bay Area search', () => {
    const query = makeQuery({
      locationStrict: true,
      location: { city: null, region: 'San Francisco Bay Area', country: 'United States' },
    });
    expect(matchesLocation(makeProfile('1', 'Oakland', 'United States', 'senior'), query)).toBe(true);
  });

  test('specific city query (no region): Oakland is filtered for San Francisco search', () => {
    const query = makeQuery({
      locationStrict: true,
      location: { city: 'San Francisco', region: null, country: 'United States' },
    });
    expect(matchesLocation(makeProfile('1', 'Oakland', 'United States', 'senior'), query)).toBe(false);
  });

  test('specific city query: San Francisco passes for San Francisco search', () => {
    const query = makeQuery({
      locationStrict: true,
      location: { city: 'San Francisco', region: null, country: 'United States' },
    });
    expect(matchesLocation(makeProfile('1', 'San Francisco', 'United States', 'senior'), query)).toBe(true);
  });

  test('specific city query: San Jose is filtered for New York search', () => {
    const query = makeQuery({
      locationStrict: true,
      location: { city: 'New York', region: null, country: 'United States' },
    });
    expect(matchesLocation(makeProfile('1', 'San Jose', 'United States', 'senior'), query)).toBe(false);
  });

  test('non-strict query passes everyone through', () => {
    const query = makeQuery({
      locationStrict: false,
      location: { city: 'New York', region: null, country: 'United States' },
    });
    expect(matchesLocation(makeProfile('1', 'Tokyo', 'Japan', 'senior'), query)).toBe(true);
  });

  test('wrong country is always filtered', () => {
    const query = makeQuery({
      locationStrict: true,
      location: { city: null, region: null, country: 'United States' },
    });
    expect(matchesLocation(makeProfile('1', 'London', 'United Kingdom', 'senior'), query)).toBe(false);
  });
});

// ─── matchesSeniority ─────────────────────────────────────────────────────────

describe('matchesSeniority', () => {
  function q(seniority: string | null) {
    return makeQuery({ seniority });
  }

  test('passes when query has no seniority', () => {
    expect(matchesSeniority(makeProfile('1', 'SF', 'US', 'senior'), q(null))).toBe(true);
  });

  test('exact match passes', () => {
    expect(matchesSeniority(makeProfile('1', 'SF', 'US', 'senior'), q('senior'))).toBe(true);
  });

  test('±1 within IC track passes (mid for senior query)', () => {
    expect(matchesSeniority(makeProfile('1', 'SF', 'US', 'mid'), q('senior'))).toBe(true);
  });

  test('±1 within IC track passes (staff for senior query)', () => {
    expect(matchesSeniority(makeProfile('1', 'SF', 'US', 'staff'), q('senior'))).toBe(true);
  });

  test('±2 within IC track fails (junior for senior query)', () => {
    expect(matchesSeniority(makeProfile('1', 'SF', 'US', 'junior'), q('senior'))).toBe(false);
  });

  test('cross-track IC→mgmt fails even when ranks are adjacent (principal vs director)', () => {
    // principal=5 (IC), director=6 (mgmt) — rank diff is 1 but different tracks
    expect(matchesSeniority(makeProfile('1', 'SF', 'US', 'director'), q('principal'))).toBe(false);
  });

  test('cross-track mgmt→IC fails (director query, senior candidate)', () => {
    expect(matchesSeniority(makeProfile('1', 'SF', 'US', 'senior'), q('director'))).toBe(false);
  });

  test('within mgmt track passes (vp for director query)', () => {
    expect(matchesSeniority(makeProfile('1', 'SF', 'US', 'vp'), q('director'))).toBe(true);
  });

  test('unknown seniority on profile passes (fallback)', () => {
    const profile = { ...makeProfile('1', 'SF', 'US', 'senior'), seniority: 'contractor' as any };
    expect(matchesSeniority(profile, q('senior'))).toBe(true);
  });
});

// ─── Fix 2+3: tiered relaxation ───────────────────────────────────────────────

describe('buildScoredTiers', () => {
  const query = makeQuery({
    locationStrict: true,
    location: { city: null, region: null, country: 'United States' },
    seniority: 'senior',
  });

  const MIN_SCORE = 0.40;
  const MIN_SURVIVORS = 10;

  test('no relaxation needed: returns tier1 only when survivors >= abmMinSurvivors', () => {
    const hydrated = Array.from({ length: 12 }, (_, i) =>
      makeResult(makeProfile(`p${i}`, 'San Francisco', 'United States', 'senior'), 0.80),
    );
    const result = buildScoredTiers(hydrated, query, MIN_SCORE, MIN_SURVIVORS);
    expect(result).toHaveLength(12);
  });

  test('Fix 3: relaxation fires when tier1 has enough raw candidates but all score below threshold', () => {
    // 12 candidates pass location+seniority but score < 0.40 → old code would not relax
    const lowScorers = Array.from({ length: 12 }, (_, i) =>
      makeResult(makeProfile(`low${i}`, 'San Francisco', 'United States', 'senior'), 0.30),
    );
    // 8 location-only candidates with good scores
    const locationOnly = Array.from({ length: 8 }, (_, i) =>
      makeResult(makeProfile(`loc${i}`, 'Austin', 'United States', 'junior'), 0.60),
    );
    const hydrated = [...lowScorers, ...locationOnly];
    const result = buildScoredTiers(hydrated, query, MIN_SCORE, MIN_SURVIVORS);
    // After Fix 3, tier1 scored count = 0 (below threshold), so seniority relax fires,
    // then location relax fires; result includes the 8 location-only candidates
    expect(result.length).toBeGreaterThan(0);
    expect(result.every((r) => r.l1Score >= MIN_SCORE)).toBe(true);
  });

  test('Fix 2: perfect matches appear before relaxed candidates', () => {
    // 5 perfect matches (seniority + location + score)
    const perfect = Array.from({ length: 5 }, (_, i) =>
      makeResult(makeProfile(`perfect${i}`, 'San Francisco', 'United States', 'senior'), 0.75),
    );
    // 10 location-only candidates with higher L1 scores
    const locationOnly = Array.from({ length: 10 }, (_, i) =>
      makeResult(makeProfile(`loc${i}`, 'Austin', 'United States', 'junior'), 0.90),
    );
    const hydrated = [...locationOnly, ...perfect]; // locationOnly ranks higher by L1
    const result = buildScoredTiers(hydrated, query, MIN_SCORE, MIN_SURVIVORS);

    const perfectIds = new Set(perfect.map((r) => r.profile.id));
    const firstFive = result.slice(0, 5).map((r) => r.profile.id);
    expect(firstFive.every((id) => perfectIds.has(id))).toBe(true);
  });

  test('no duplicate candidates after relaxation', () => {
    const perfect = Array.from({ length: 3 }, (_, i) =>
      makeResult(makeProfile(`p${i}`, 'San Francisco', 'United States', 'senior'), 0.70),
    );
    const locationOnly = Array.from({ length: 10 }, (_, i) =>
      makeResult(makeProfile(`l${i}`, 'New York', 'United States', 'mid'), 0.65),
    );
    const hydrated = [...perfect, ...locationOnly];
    const result = buildScoredTiers(hydrated, query, MIN_SCORE, MIN_SURVIVORS);
    const ids = result.map((r) => r.profile.id);
    expect(ids.length).toBe(new Set(ids).size);
  });

  test('all results meet minimum score threshold', () => {
    const mixed = [
      makeResult(makeProfile('a', 'San Francisco', 'United States', 'senior'), 0.50),
      makeResult(makeProfile('b', 'San Francisco', 'United States', 'senior'), 0.20), // below threshold
      makeResult(makeProfile('c', 'Austin', 'United States', 'junior'), 0.60),
      makeResult(makeProfile('d', 'Austin', 'United States', 'junior'), 0.15), // below threshold
    ];
    const result = buildScoredTiers(mixed, query, MIN_SCORE, MIN_SURVIVORS);
    expect(result.every((r) => r.l1Score >= MIN_SCORE)).toBe(true);
  });
});
