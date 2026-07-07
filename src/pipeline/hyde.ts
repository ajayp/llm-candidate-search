import OpenAI from 'openai';
import { StructuredQuery } from '../types';
import { CONFIG } from '../config';
import { withRetry } from '../utils';
import { deriveQueryFields } from './queryFields';

let _client: OpenAI | null = null;

function getClient(): OpenAI {
  if (!_client) _client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  return _client;
}

export async function generateHyDE(query: StructuredQuery): Promise<string> {
  const client = getClient();

  const { role, skills, locationPhrase } = deriveQueryFields(query);

  const userPrompt = [
    role && `Role: ${role}`,
    skills.length > 0 && `Required skills: ${skills.join(', ')}`,
    locationPhrase && `Location: based in ${locationPhrase}`,
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
