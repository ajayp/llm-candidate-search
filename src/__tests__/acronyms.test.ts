import { resolveQualification } from '../acronyms';

describe('resolveQualification', () => {
  test('expands a known, unambiguous acronym', () => {
    expect(resolveQualification('K8s')).toEqual({ text: 'Kubernetes', ambiguous: false });
    expect(resolveQualification('JS')).toEqual({ text: 'JavaScript', ambiguous: false });
  });

  test('leaves known-safe acronyms exactly as written, not ambiguous', () => {
    expect(resolveQualification('NLP')).toEqual({ text: 'NLP', ambiguous: false });
    expect(resolveQualification('RAG')).toEqual({ text: 'RAG', ambiguous: false });
  });

  test('flags an unrecognized acronym-shaped token as ambiguous, left as written', () => {
    expect(resolveQualification('TS')).toEqual({ text: 'TS', ambiguous: true });
  });

  test('leaves plain words/phrases unchanged, not ambiguous', () => {
    expect(resolveQualification('PyTorch')).toEqual({ text: 'PyTorch', ambiguous: false });
    expect(resolveQualification('distributed systems experience')).toEqual({
      text: 'distributed systems experience',
      ambiguous: false,
    });
  });

  test('trims surrounding whitespace', () => {
    expect(resolveQualification('  TS  ')).toEqual({ text: 'TS', ambiguous: true });
  });
});
