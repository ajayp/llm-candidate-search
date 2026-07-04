export interface CandidateProfile {
  id: string;
  jobCategory: string;
  name: string;
  title: string;
  seniority: 'intern' | 'junior' | 'mid' | 'senior' | 'staff' | 'principal' | 'director' | 'vp' | 'chief';
  location: { city: string; country: string };
  yearsOfExperience: number;
  skills: string[];
  summary: string;
  experience: Array<{
    title: string;
    company: string;
    durationYears: number;
    description: string;
  }>;
  education: Array<{
    degree: string;
    field: string;
    institution: string;
  }>;
}

export interface EmbeddingRecord {
  profileId: string;
  fullVector: number[];
  shortVector: number[];
  profileText: string;
  embeddedAt: string;
}

export interface StructuredQuery {
  raw: string;
  title: string | null;
  seniority: string | null;
  location: { city: string | null; region: string | null; country: string | null };
  locationStrict: boolean;
  qualifications: string[];
  requiredQualifications: string[];
  queryText: string;
}

export type FitLevel = 'poor' | 'partial' | 'good' | 'excellent';

export interface SkillMatchStat {
  skill: string;
  enrichment: number;
}

export interface SearchResult {
  profile: CandidateProfile;
  l1Score: number;
  l2Score: number;
  guardExplanation: string;
  fit: FitLevel;
  facepalm: boolean;
  distinguishingSkills: SkillMatchStat[];
}

export interface EvalQuery {
  id: string;
  raw: string;
  relevantProfileIds: string[];
}

export interface EvalQueryRaw {
  id: string;
  raw: string;
}

export interface CohereRerankResult {
  index: number;
  relevance_score: number;
  document?: { text: string };
}

export interface CohereRerankResponse {
  results: CohereRerankResult[];
}
