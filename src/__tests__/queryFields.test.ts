import { deriveQueryFields } from '../pipeline/queryFields';

function baseInput(overrides: Partial<Parameters<typeof deriveQueryFields>[0]> = {}) {
  return {
    seniority: null,
    title: null,
    qualifications: [],
    requiredQualifications: [],
    location: { city: null, region: null, country: null },
    ...overrides,
  };
}

describe('deriveQueryFields', () => {
  test('joins seniority and title into role, skipping missing parts', () => {
    const { role } = deriveQueryFields(baseInput({ seniority: 'senior', title: 'Software Engineer' }));
    expect(role).toBe('senior Software Engineer');
  });

  test('role is empty string when both seniority and title are missing', () => {
    const { role } = deriveQueryFields(baseInput());
    expect(role).toBe('');
  });

  test('role falls back to title alone when seniority is missing', () => {
    const { role } = deriveQueryFields(baseInput({ title: 'Recruiter' }));
    expect(role).toBe('Recruiter');
  });

  test('prefers requiredQualifications over qualifications when both are present', () => {
    const { skills } = deriveQueryFields(
      baseInput({ qualifications: ['SQL', 'Python'], requiredQualifications: ['Python'] }),
    );
    expect(skills).toEqual(['Python']);
  });

  test('falls back to qualifications when requiredQualifications is empty', () => {
    const { skills } = deriveQueryFields(baseInput({ qualifications: ['SQL', 'Python'] }));
    expect(skills).toEqual(['SQL', 'Python']);
  });

  test('locationPhrase picks the most specific part: city over region over country', () => {
    const { locationPhrase } = deriveQueryFields(
      baseInput({ location: { city: 'San Francisco', region: 'Bay Area', country: 'United States' } }),
    );
    expect(locationPhrase).toBe('San Francisco');
  });

  test('locationPhrase falls back to region when city is missing', () => {
    const { locationPhrase } = deriveQueryFields(
      baseInput({ location: { city: null, region: 'Bay Area', country: 'United States' } }),
    );
    expect(locationPhrase).toBe('Bay Area');
  });

  test('locationPhrase falls back to country when city and region are missing', () => {
    const { locationPhrase } = deriveQueryFields(
      baseInput({ location: { city: null, region: null, country: 'United States' } }),
    );
    expect(locationPhrase).toBe('United States');
  });

  test('locationPhrase is null when no location parts are present', () => {
    const { locationPhrase } = deriveQueryFields(baseInput());
    expect(locationPhrase).toBeNull();
  });
});
