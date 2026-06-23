import OpenAI from 'openai';
import { StructuredQuery } from '../types';
import { CONFIG } from '../config';
import { withRetry } from '../utils';

let _client: OpenAI | null = null;

function getClient(): OpenAI {
  if (!_client) _client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  return _client;
}

export async function generateHyDE(query: StructuredQuery): Promise<string> {
  const client = getClient();

  const role = [query.seniority, query.title].filter(Boolean).join(' ');
  const skills = query.requiredQualifications.length > 0
    ? query.requiredQualifications
    : query.qualifications;
  const locationParts = [query.location.city, query.location.region, query.location.country].filter(Boolean);
  const locationPhrase = locationParts.length > 0 ? `based in ${locationParts[0]}` : '';

  const userPrompt = [
    role && `Role: ${role}`,
    skills.length > 0 && `Required skills: ${skills.join(', ')}`,
    locationPhrase && `Location: ${locationPhrase}`,
  ]
    .filter(Boolean)
    .join('\n');

  const response = await withRetry(
    () =>
      client.chat.completions.create({
        model: CONFIG.openai.chat.guardModel,
        max_tokens: 200,
        messages: [
          {
            role: 'system',
            content:
              'You are writing a LinkedIn profile summary for a hypothetical ideal candidate. ' +
              'Given a job description, write a 3-4 sentence first-person summary as if you were that candidate. ' +
              'Include their seniority, core expertise, key skills, and relevant background. ' +
              'Be specific and concrete. Output only the summary — no labels, no preamble.',
          },
          { role: 'user', content: userPrompt },
        ],
      }),
    CONFIG.retry.maxAttempts,
    CONFIG.retry.baseDelayMs,
  );

  const content = response.choices[0].message.content;
  if (!content) throw new Error('HyDE generation returned empty response');
  return content.trim();
}
