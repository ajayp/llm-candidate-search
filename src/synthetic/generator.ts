import fs from 'fs';
import path from 'path';
import OpenAI from 'openai';
import { CandidateProfile } from '../types';
import { CONFIG, JOB_CATEGORIES, TECH_HUBS } from '../config';
import { withRetry } from '../utils';

// 9 tiers × 10 categories × 8 profiles = 720
const SENIORITY_BATCHES = [
  ['intern'],
  ['junior'],
  ['mid'],
  ['senior'],
  ['staff'],
  ['principal'],
  ['director'],
  ['vp'],
  ['chief'],
] as const;

const PROFILES_PER_BATCH = 8;

const SENIORITY_DESCRIPTIONS: Record<string, string> = {
  intern: 'Student or recent graduate in a learning role with limited ownership.',
  junior: 'Early-career engineer who executes defined tasks with guidance and supervision.',
  mid: 'Independent contributor who owns features or projects and delivers with minimal supervision.',
  senior: 'Experienced engineer who drives projects end-to-end, mentors others, and influences technical decisions beyond their immediate work.',
  staff: 'Senior IC track: provides technical leadership across teams, drives architecture and technical strategy, and influences engineering direction without direct management responsibilities.',
  principal: 'Advanced IC track: organization-wide technical strategy, engineering standards, and long-term architecture across products or divisions.',
  director: 'Management track: leads multiple teams or a function, responsible for roadmap, staffing, and execution; maintains sufficient technical depth to make architecture, build-vs-buy, and hiring decisions.',
  vp: 'Executive leader responsible for large engineering organizations, organizational strategy, budgets, and business outcomes; typically reports to the C-suite.',
  chief: 'C-suite executive (e.g., CTO, CEO, CPO, CISO) with company-wide scope, responsible for organizational strategy, leadership, and external stakeholders such as boards and investors.',
};

function loadExisting(filePath: string): CandidateProfile[] {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
  } catch {
    return [];
  }
}

function buildCompletedSet(profiles: CandidateProfile[]): Set<string> {
  const counts = new Map<string, number>();
  for (const p of profiles) {
    if (p.jobCategory && p.seniority) {
      const key = `${p.jobCategory}::${p.seniority}`;
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
  }
  const done = new Set<string>();
  for (const batch of SENIORITY_BATCHES) {
    for (const category of JOB_CATEGORIES) {
      const total = batch.reduce((sum, s) => sum + (counts.get(`${category}::${s}`) ?? 0), 0);
      if (total >= PROFILES_PER_BATCH) done.add(`${category}::${[...batch].join('/')}`);
    }
  }
  return done;
}

function getClient(): OpenAI {
  return new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
}

const PROFILE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    profiles: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          name: { type: 'string' },
          title: { type: 'string' },
          seniority: {
            type: 'string',
            enum: ['intern', 'junior', 'mid', 'senior', 'staff', 'principal', 'director', 'vp', 'chief'],
          },
          location: {
            type: 'object',
            additionalProperties: false,
            properties: {
              city: { type: 'string' },
              country: { type: 'string' },
            },
            required: ['city', 'country'],
          },
          yearsOfExperience: { type: 'number' },
          skills: { type: 'array', items: { type: 'string' } },
          summary: { type: 'string' },
          experience: {
            type: 'array',
            items: {
              type: 'object',
              additionalProperties: false,
              properties: {
                title: { type: 'string' },
                company: { type: 'string' },
                durationYears: { type: 'number' },
                description: { type: 'string' },
              },
              required: ['title', 'company', 'durationYears', 'description'],
            },
          },
          education: {
            type: 'array',
            items: {
              type: 'object',
              additionalProperties: false,
              properties: {
                degree: { type: 'string' },
                field: { type: 'string' },
                institution: { type: 'string' },
              },
              required: ['degree', 'field', 'institution'],
            },
          },
        },
        required: [
          'name', 'title', 'seniority', 'location', 'yearsOfExperience',
          'skills', 'summary', 'experience', 'education',
        ],
      },
    },
  },
  required: ['profiles'],
};

async function generateBatch(
  client: OpenAI,
  category: string,
  seniorityLevels: readonly string[],
  count: number,
): Promise<Omit<CandidateProfile, 'id' | 'jobCategory'>[]> {
  const techHubList = TECH_HUBS.map((h) => `${h.city}, ${h.country}`).join('; ');
  const seniorityStr = seniorityLevels.join(' or ');

  const seniorityDesc = seniorityLevels.map((s) => SENIORITY_DESCRIPTIONS[s]).join(' ');

  const prompt = `Generate ${count} realistic synthetic candidate profiles for the "${category}" job function.

Seniority: ${seniorityStr} — ${seniorityDesc}
Skills: 5–15 relevant skills per profile
Experience: 3–5 roles with realistic company names and 2–4 sentence descriptions
Education: 1–2 degrees
Summary: 2–3 sentences bio
Location: MUST be one of these exact city/country pairs: ${techHubList}
Years of experience: appropriate for the seniority level

Make profiles diverse in background, gender, and company history. Use realistic tech company names.`;

  const response = await withRetry(
    () =>
      client.chat.completions.create({
        model: CONFIG.openai.chat.generatorModel,
        max_tokens: 8192,
        response_format: {
          type: 'json_schema',
          json_schema: {
            name: 'generate_profiles',
            strict: true,
            schema: PROFILE_SCHEMA,
          },
        },
        messages: [{ role: 'user', content: prompt }],
      }),
    CONFIG.retry.maxAttempts,
    CONFIG.retry.baseDelayMs,
  );

  const content = response.choices[0].message.content;
  if (!content) {
    throw new Error('No content in generator response');
  }

  const input = JSON.parse(content) as { profiles: Omit<CandidateProfile, 'id' | 'jobCategory'>[] };
  // Ensure the model returned the expected count. Trim extras and warn if mismatch.
  const returned = Array.isArray(input.profiles) ? input.profiles : [];
  return returned.slice(0, count);
}

export async function generateProfiles(profilesPath: string): Promise<CandidateProfile[]> {
  const client = getClient();
  const allProfiles = loadExisting(profilesPath);
  const completedSet = buildCompletedSet(allProfiles);

  for (const category of JOB_CATEGORIES) {
    console.log(`\nGenerating ${category} profiles...`);
    for (let batchIdx = 0; batchIdx < SENIORITY_BATCHES.length; batchIdx++) {
      const seniorityLevels = SENIORITY_BATCHES[batchIdx];
      const key = `${category}::${[...seniorityLevels].join('/')}`;

      if (completedSet.has(key)) {
        console.log(`  Batch ${batchIdx + 1}/${SENIORITY_BATCHES.length} (${seniorityLevels.join('/')})... skipped (already in profiles.json)`);
        continue;
      }

      process.stdout.write(
        `  Batch ${batchIdx + 1}/${SENIORITY_BATCHES.length} (${seniorityLevels.join('/')})...`,
      );

      const raw = await generateBatch(client, category, seniorityLevels, PROFILES_PER_BATCH);
      const profiles = raw.map((p, i) => ({
        ...p,
        id: crypto.randomUUID(),
        jobCategory: category,
        seniority: seniorityLevels[i % seniorityLevels.length] as CandidateProfile['seniority'],
      }));
      allProfiles.push(...profiles);
      console.log(` ${profiles.length} profiles`);

      fs.mkdirSync(path.dirname(profilesPath), { recursive: true });
      fs.writeFileSync(profilesPath, JSON.stringify(allProfiles, null, 2));
      // Test mode: stop after the first generated batch when env var is set
      if (process.env.GENERATE_ONE_BATCH === 'true') {
        console.log('  Single-batch test mode enabled — stopping after first generated batch.');
        return allProfiles;
      }
    }
  }

  return allProfiles;
}
