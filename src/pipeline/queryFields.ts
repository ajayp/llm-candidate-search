// Shared by queryUnderstanding.ts (query embedding text) and hyde.ts (HyDE prompt) so the
// two never independently drift on how role/skills/location are derived from a parsed query.
export interface QueryFieldsInput {
  seniority: string | null;
  title: string | null;
  qualifications: string[];
  requiredQualifications: string[];
  location: { city: string | null; region: string | null; country: string | null };
}

export interface QueryFields {
  role: string;
  skills: string[];
  locationPhrase: string | null;
}

export function deriveQueryFields(query: QueryFieldsInput): QueryFields {
  const role = [query.seniority, query.title].filter(Boolean).join(' ');
  const skills = query.requiredQualifications.length > 0
    ? query.requiredQualifications
    : query.qualifications;
  const locationParts = [query.location.city, query.location.region, query.location.country].filter(Boolean);
  const locationPhrase = locationParts.length > 0 ? (locationParts[0] as string) : null;

  return { role, skills, locationPhrase };
}
