// Deterministic acronym handling, shared by Stage 1 (query understanding) and Stage 4 (guard),
// so the two never silently disagree about what an abbreviation means.

// Only genuinely unambiguous mappings belong here — if there's any plausible second meaning,
// it does not belong in this table (see KNOWN_SAFE_ACRONYMS and the ambiguous fallback instead).
export const KNOWN_ACRONYM_EXPANSIONS: Record<string, string> = {
  K8s: 'Kubernetes',
  JS: 'JavaScript',
};

// Abbreviations that are already the standard, professionally-used form — profiles list these
// as-is, so expanding them would actively hurt matching, not help it.
export const KNOWN_SAFE_ACRONYMS = new Set([
  'NLP', 'RAG', 'LLM', 'LLMs', 'SQL', 'AWS', 'GCP', 'API', 'ML', 'AI', 'SDK', 'UI', 'UX',
]);

function looksLikeAcronym(token: string): boolean {
  return /^[A-Za-z0-9]{2,5}$/.test(token) && /[A-Z]/.test(token);
}

export interface ResolvedQualification {
  text: string;
  ambiguous: boolean;
}

// Resolves a single extracted qualification string:
// - a known, unambiguous acronym gets expanded to its full name
// - a known-safe acronym is left exactly as written (no expansion needed)
// - anything else that's acronym-shaped but unrecognized is flagged ambiguous, left as written
// - plain words/phrases pass through unchanged
export function resolveQualification(raw: string): ResolvedQualification {
  const trimmed = raw.trim();
  const expansion = KNOWN_ACRONYM_EXPANSIONS[trimmed];
  if (expansion) return { text: expansion, ambiguous: false };
  if (KNOWN_SAFE_ACRONYMS.has(trimmed)) return { text: trimmed, ambiguous: false };
  if (looksLikeAcronym(trimmed)) return { text: trimmed, ambiguous: true };
  return { text: trimmed, ambiguous: false };
}
