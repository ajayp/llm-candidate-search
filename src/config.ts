export const CONFIG = {
  openai: {
    embeddings: {
      model: 'text-embedding-3-large',
      fullDims: 3072,
      shortDims: 512,
      batchSize: 100,
    },
    chat: {
      queryUnderstandingModel: 'gpt-4o',
      guardModel: 'gpt-4o-mini',
      generatorModel: 'gpt-4o-mini',
    },
  },
  cohere: {
    rerankingModel: 'rerank-v4.0-pro',
    rerankEndpoint: 'https://api.cohere.com/v2/rerank',
  },
  faiss: {
    nlist: 16,
    nprobe: 16,
  },
  pipeline: {
    l1TopK: 600,
    abmMinSurvivors: 10,
    l2TopN: 100,
    finalTopK: 25,
    guardMaxConcurrency: 5,
    l1MinScore: 0.40,
    l2MinScore: 0.50,
  },
  data: {
    profileCount: 720,
    profilesPath: 'data/profiles.json',
    embeddingsCachePath: 'data/embeddings_cache.json',
    indexPath: 'data/index.faiss',
    profileIdMapPath: 'data/profile_id_map.json',
    evalQueriesRawPath: 'data/eval_queries_raw.json',
    evalQueriesPath: 'data/eval_queries.json',
    evalResultsPath: 'data/eval_results.json',
  },
  retry: {
    maxAttempts: 5,
    baseDelayMs: 1000,
  },
};

export const JOB_CATEGORIES = [
  'Software Engineering',
  'Data Science',
  'Product Management',
  'Marketing',
  'Sales',
  'Finance',
  'Design',
  'Healthcare',
  'Legal',
  'Operations',
] as const;

export const SENIORITY_LEVELS = [
  'intern',
  'junior',
  'mid',
  'senior',
  'staff',
  'principal',
  'director',
  'vp',
  'chief',
] as const;

export const SENIORITY_RANK: Record<string, number> = {
  intern: 0,
  junior: 1,
  mid: 2,
  senior: 3,
  staff: 4,
  principal: 5,
  director: 6,
  vp: 7,
  chief: 8,
};

export const TECH_HUBS = [
  { city: 'San Francisco', country: 'United States' },
  { city: 'New York', country: 'United States' },
  { city: 'Seattle', country: 'United States' },
  { city: 'Austin', country: 'United States' },
  { city: 'Boston', country: 'United States' },
  { city: 'Chicago', country: 'United States' },
  { city: 'Los Angeles', country: 'United States' },
  { city: 'Denver', country: 'United States' },
  { city: 'London', country: 'United Kingdom' },
  { city: 'Berlin', country: 'Germany' },
  { city: 'Amsterdam', country: 'Netherlands' },
  { city: 'Paris', country: 'France' },
  { city: 'Toronto', country: 'Canada' },
  { city: 'Vancouver', country: 'Canada' },
  { city: 'Sydney', country: 'Australia' },
  { city: 'Singapore', country: 'Singapore' },
  { city: 'Tokyo', country: 'Japan' },
  { city: 'Tel Aviv', country: 'Israel' },
  { city: 'Bangalore', country: 'India' },
  { city: 'Dublin', country: 'Ireland' },
] as const;
