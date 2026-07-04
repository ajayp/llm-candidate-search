import OpenAI from 'openai';
import { StructuredQuery } from '../types';
import { CONFIG } from '../config';
import { withRetry } from '../utils';

let _client: OpenAI | null = null;

function getClient(): OpenAI {
  if (!_client) _client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  return _client;
}

const PARSE_QUERY_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    title: { type: ['string', 'null'] },
    seniority: {
      type: ['string', 'null'],
      enum: ['intern', 'junior', 'mid', 'senior', 'staff', 'principal', 'director', 'vp', 'chief', null],
    },
    location: {
      type: 'object',
      additionalProperties: false,
      properties: {
        city: {
          type: ['string', 'null'],
          description:
            'Specific city only. Set to null for metro areas ("Bay Area", "Greater London"), regions, or soft language ("preferably", "ideally").',
        },
        region: {
          type: ['string', 'null'],
          description:
            'Metro area or region when the query names one but not a specific city — e.g. "San Francisco Bay Area", "Greater New York", "Greater London", "GTA". Set to null when a specific city is given or no location is mentioned.',
        },
        country: { type: ['string', 'null'] },
      },
      required: ['city', 'region', 'country'],
    },
    locationStrict: {
      type: 'boolean',
      description:
        'True when location is stated as a requirement without soft language — i.e. the query says "in X", "based in X", "located in X", "X only", "must be in X", or simply lists a location without any qualifier. False only when softened ("preferably", "ideally", "nice to have", "open to remote") or no location is mentioned at all.',
    },
    qualifications: {
      type: 'array',
      items: { type: 'string' },
      description: 'All qualifications mentioned (skills, experience, tools)',
    },
    requiredQualifications: {
      type: 'array',
      items: { type: 'string' },
      description: 'Skills, tools, or experience that are core to the role. Include any qualification not softened by words like "preferably", "ideally", "nice to have", or "preferred". When in doubt, include it.',
    },
  },
  required: ['title', 'seniority', 'location', 'locationStrict', 'qualifications', 'requiredQualifications'],
};

const COUNTRY_ALIASES: Record<string, string> = {
  'US': 'United States',
  'USA': 'United States',
  'U.S.': 'United States',
  'U.S.A.': 'United States',
  'UK': 'United Kingdom',
  'U.K.': 'United Kingdom',
};

function normalizeCountry(country: string | null): string | null {
  if (!country) return null;
  return COUNTRY_ALIASES[country.trim()] ?? country;
}

interface ParsedQueryInput {
  title: string | null;
  seniority: string | null;
  location: { city: string | null; region: string | null; country: string | null };
  locationStrict: boolean;
  qualifications: string[];
  requiredQualifications: string[];
}

function buildQueryText(parsed: ParsedQueryInput): string {
  const role = [parsed.seniority, parsed.title].filter(Boolean).join(' ');
  const skills = parsed.requiredQualifications.length > 0
    ? parsed.requiredQualifications
    : parsed.qualifications;

  const locationParts = [
    parsed.location.city,
    parsed.location.region,
    parsed.location.country,
  ].filter(Boolean);
  const locationPhrase = locationParts.length > 0 ? locationParts[0] : null;

  const parts: string[] = [];
  if (role) parts.push(role);
  if (skills.length > 0) parts.push(`with expertise in ${skills.join(', ')}`);
  if (locationPhrase) parts.push(`based in ${locationPhrase}`);

  return parts.join(', ');
}

export async function understandQuery(rawQuery: string): Promise<StructuredQuery> {
  const client = getClient();

  const response = await withRetry(
    () =>
      client.chat.completions.create({
        model: CONFIG.openai.chat.queryUnderstandingModel,
        max_tokens: 1024,
        response_format: {
          type: 'json_schema',
          json_schema: {
            name: 'parse_query',
            strict: true,
            schema: PARSE_QUERY_SCHEMA,
          },
        },
        messages: [
          {
            role: 'system',
            content:
              'You are a recruiter search assistant. Parse the given recruiter query into structured JSON. ' +
              'Extract role title, seniority level, location, and qualifications. ' +

              'For location.city: set only when a specific city is named as a hard requirement. Set to null for metro areas ("Bay Area", "Greater London"), regions, or soft language. ' +
              'For location.region: set to the metro area or region name when one is mentioned but no specific city is given (e.g. "Bay Area" → "San Francisco Bay Area", "GTA" → "Greater Toronto Area", "Greater London" → "Greater London"). Set to null when a specific city is given or no location is mentioned. ' +
              'For location.country: always infer and set it when the location clearly implies one — a metro area or region is sufficient. Only leave null if the location is genuinely ambiguous or no location is mentioned at all. ' +

              'For locationStrict: set to true whenever location is stated as a requirement without soft language — this includes simply naming a location ("in X", "based in X", "X only") with no qualifier. Set to false ONLY when softened ("preferably", "ideally", "nice to have", "open to remote") or when no location is mentioned at all. ' +

              'For any extracted qualification (in either qualifications or requiredQualifications): extract each distinct skill, tool, or technology as its own separate array entry — never combine multiple technologies into one string (e.g. "RAG and LLM experience" produces two entries, "RAG" and "LLM", not one entry "RAG LLM"). ' +
              'Strip filler verbs and phrases like "knows", "has", "is skilled in", "proficient in", "experience with" — extract just the skill or technology name (e.g. "knows PyTorch" → "PyTorch"). Preserve meaningful qualifying words that narrow the requirement itself, not just restate possession (e.g. "NLP background" not just "NLP", "distributed systems experience" not just "distributed systems", "5+ years Python" not just "Python"). ' +
              'Expand a technical abbreviation to its standard full name only when it is genuinely ambiguous or rarely used in that short form professionally, and you are confident of a single unambiguous meaning (e.g. "TS" → "TypeScript", "K8s" → "Kubernetes"). Do NOT expand widely-used standard abbreviations that professionals already list as-is — keep "NLP", "RAG", "LLM", "SQL", "AWS", "GCP", "API", "ML", "AI" exactly as written. ' +

              'For requiredQualifications: include only qualifications that the recruiter explicitly stated in the query. ' +
              'If the title contains one or more specific technology terms (e.g. "RAG engineer", "React developer", "Kubernetes specialist"), add each of those technologies as its own separate required qualification (e.g. title "AI RAG LLM engineer" adds "RAG" and "LLM" as two entries, not "AI RAG LLM" as one). Do not infer other skills implied by the role title (e.g. do not add "JavaScript" just because the title is "frontend engineer"). ' +
              'Only exclude a qualification if it is explicitly softened ("preferably", "ideally", "nice to have", "preferred", "a plus"). ' +

              'If a field is not mentioned and cannot be inferred, return null or an empty array.',
          },
          { role: 'user', content: rawQuery },
        ],
      }),
    CONFIG.retry.maxAttempts,
    CONFIG.retry.baseDelayMs,
  );

  const content = response.choices[0].message.content;
  if (!content) {
    throw new Error('Query understanding returned empty response');
  }

  let parsed: ParsedQueryInput;
  try {
    parsed = JSON.parse(content) as ParsedQueryInput;
  } catch {
    throw new Error(`Query understanding returned unparseable JSON: ${content.slice(0, 200)}`);
  }
  const queryText = buildQueryText(parsed);

  return {
    raw: rawQuery,
    title: parsed.title,
    seniority: parsed.seniority,
    location: {
      city: parsed.location.city,
      region: parsed.location.region ?? null,
      country: normalizeCountry(parsed.location.country),
    },
    locationStrict: parsed.locationStrict ?? false,
    qualifications: parsed.qualifications,
    requiredQualifications: parsed.requiredQualifications,
    queryText,
  };
}
