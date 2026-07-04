import { resolveStructuredQualifications } from '../pipeline/queryUnderstanding';

describe('resolveStructuredQualifications', () => {
  test('leaves known-safe acronyms and plain skills unchanged, no ambiguity flagged', () => {
    const result = resolveStructuredQualifications(
      ['RAG', 'LLM', 'PyTorch', 'semantic search'],
      ['RAG', 'LLM'],
    );
    expect(result.qualifications).toEqual(['RAG', 'LLM', 'PyTorch', 'semantic search']);
    expect(result.requiredQualifications).toEqual(['RAG', 'LLM']);
    expect(result.ambiguousQualifications).toEqual([]);
  });

  test('expands a known, unambiguous acronym in both qualification lists', () => {
    const result = resolveStructuredQualifications(['K8s'], ['K8s']);
    expect(result.qualifications).toEqual(['Kubernetes']);
    expect(result.requiredQualifications).toEqual(['Kubernetes']);
  });

  test('flags an unrecognized acronym in requiredQualifications as ambiguous, left as written', () => {
    const result = resolveStructuredQualifications(
      ['PyTorch', 'TS', 'semantic search'],
      ['RAG', 'LLM', 'PyTorch', 'TS', 'semantic search'],
    );
    expect(result.requiredQualifications).toContain('TS');
    expect(result.ambiguousQualifications).toEqual(['TS']);
  });

  test('only requiredQualifications feed ambiguousQualifications, not the general qualifications list', () => {
    const result = resolveStructuredQualifications(['TS'], []);
    expect(result.qualifications).toEqual(['TS']);
    expect(result.ambiguousQualifications).toEqual([]);
  });

  test('empty inputs produce empty outputs', () => {
    const result = resolveStructuredQualifications([], []);
    expect(result).toEqual({ qualifications: [], requiredQualifications: [], ambiguousQualifications: [] });
  });
});
