import { buildBackgroundSkillFrequency, computeSkillEnrichment, topDistinguishingSkills } from '../pipeline/matchStats';
import { CandidateProfile, SkillMatchStat } from '../types';

function makeProfile(id: string, skills: string[]): CandidateProfile {
  return {
    id,
    jobCategory: 'Software Engineering',
    name: `Candidate ${id}`,
    title: 'Engineer',
    seniority: 'senior',
    location: { city: 'San Francisco', country: 'United States' },
    yearsOfExperience: 5,
    skills,
    summary: '',
    experience: [],
    education: [],
  };
}

describe('buildBackgroundSkillFrequency', () => {
  test('computes fraction of profiles containing each skill', () => {
    const profiles = [
      makeProfile('1', ['Python', 'SQL']),
      makeProfile('2', ['Python']),
      makeProfile('3', ['Java']),
      makeProfile('4', ['Java']),
    ];
    const freq = buildBackgroundSkillFrequency(profiles);
    expect(freq.get('Python')).toBeCloseTo(0.5);
    expect(freq.get('SQL')).toBeCloseTo(0.25);
    expect(freq.get('Java')).toBeCloseTo(0.5);
  });

  test('empty profile list returns empty map', () => {
    expect(buildBackgroundSkillFrequency([]).size).toBe(0);
  });
});

describe('computeSkillEnrichment', () => {
  test('skill more common in foreground than background scores > 1', () => {
    const background = new Map([['Rust', 0.1], ['Python', 0.5]]);
    const foreground = [
      makeProfile('1', ['Rust', 'Python']),
      makeProfile('2', ['Rust', 'Python']),
    ];
    const result = computeSkillEnrichment(foreground, background);
    expect(result.get('Rust')!.score).toBeCloseTo(10.0); // fgFreq 1.0 / bgFreq 0.1
    expect(result.get('Python')!.score).toBeCloseTo(2.0); // fgFreq 1.0 / bgFreq 0.5
  });

  test('only includes skills present in the foreground', () => {
    const background = new Map([['Rust', 0.1], ['COBOL', 0.05]]);
    const foreground = [makeProfile('1', ['Rust'])];
    const result = computeSkillEnrichment(foreground, background);
    expect(result.has('COBOL')).toBe(false);
  });

  test('foreground fraction is always finite (background guaranteed > 0)', () => {
    const background = new Map([['Rust', 0.1]]);
    const foreground = [makeProfile('1', ['Rust'])];
    const result = computeSkillEnrichment(foreground, background);
    expect(Number.isFinite(result.get('Rust')!.score)).toBe(true);
  });

  test('carries the raw foreground count alongside the score', () => {
    const background = new Map([['Rust', 0.1]]);
    const foreground = [makeProfile('1', ['Rust']), makeProfile('2', ['Rust']), makeProfile('3', [])];
    const result = computeSkillEnrichment(foreground, background);
    expect(result.get('Rust')!.foregroundCount).toBe(2);
  });
});

describe('topDistinguishingSkills', () => {
  test('keeps only score > 1, sorted descending, capped at topN', () => {
    const enrichment = new Map([
      ['Rust', { score: 5.0, foregroundCount: 3 }],
      ['Python', { score: 1.0, foregroundCount: 10 }],
      ['Kubernetes', { score: 3.0, foregroundCount: 2 }],
      ['SQL', { score: 0.5, foregroundCount: 8 }],
      ['Go', { score: 8.0, foregroundCount: 1 }],
    ]);
    const profile = makeProfile('1', ['Rust', 'Python', 'Kubernetes', 'SQL', 'Go']);
    const result = topDistinguishingSkills(profile, enrichment, 3);
    expect(result).toEqual<SkillMatchStat[]>([
      { skill: 'Go', enrichment: 8.0 },
      { skill: 'Rust', enrichment: 5.0 },
      { skill: 'Kubernetes', enrichment: 3.0 },
    ]);
  });

  test('all-scores-<=1 case returns empty array', () => {
    const enrichment = new Map([
      ['Python', { score: 1.0, foregroundCount: 10 }],
      ['SQL', { score: 0.5, foregroundCount: 8 }],
    ]);
    const profile = makeProfile('1', ['Python', 'SQL']);
    expect(topDistinguishingSkills(profile, enrichment, 3)).toEqual([]);
  });

  test('skill not present in the enrichment map at all is ignored', () => {
    const enrichment = new Map([['Rust', { score: 5.0, foregroundCount: 1 }]]);
    const profile = makeProfile('1', ['Rust', 'UnknownSkill']);
    expect(topDistinguishingSkills(profile, enrichment, 3)).toEqual([{ skill: 'Rust', enrichment: 5.0 }]);
  });

  test('topN smaller than qualifying skill count truncates correctly', () => {
    const enrichment = new Map([
      ['Rust', { score: 5.0, foregroundCount: 1 }],
      ['Kubernetes', { score: 3.0, foregroundCount: 1 }],
      ['Go', { score: 8.0, foregroundCount: 1 }],
    ]);
    const profile = makeProfile('1', ['Rust', 'Kubernetes', 'Go']);
    expect(topDistinguishingSkills(profile, enrichment, 1)).toEqual([{ skill: 'Go', enrichment: 8.0 }]);
  });

  test('tied scores break by higher foreground count first', () => {
    const enrichment = new Map([
      ['RareButShared', { score: 7.24, foregroundCount: 3 }],
      ['RareAndLonely', { score: 7.24, foregroundCount: 1 }],
    ]);
    const profile = makeProfile('1', ['RareAndLonely', 'RareButShared']);
    const result = topDistinguishingSkills(profile, enrichment, 2);
    expect(result.map((s) => s.skill)).toEqual(['RareButShared', 'RareAndLonely']);
  });

  test('tied score and tied foreground count fall back to alphabetical order', () => {
    const enrichment = new Map([
      ['Zebra', { score: 7.24, foregroundCount: 1 }],
      ['Apple', { score: 7.24, foregroundCount: 1 }],
    ]);
    const profile = makeProfile('1', ['Zebra', 'Apple']);
    const result = topDistinguishingSkills(profile, enrichment, 2);
    expect(result.map((s) => s.skill)).toEqual(['Apple', 'Zebra']);
  });
});
