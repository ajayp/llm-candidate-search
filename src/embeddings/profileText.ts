import { CandidateProfile } from '../types';

export function profileToText(profile: CandidateProfile): string {
  const experienceText = profile.experience
    .map(
      (e) =>
        `${e.title} at ${e.company} (${e.durationYears} ${e.durationYears === 1 ? 'year' : 'years'}): ${e.description}`,
    )
    .join(' | ');

  const educationText = profile.education
    .map((e) => `${e.degree} in ${e.field} at ${e.institution}`)
    .join(' | ');

  return [
    `[Title]: ${profile.title}`,
    `[Location]: ${profile.location.city}, ${profile.location.country}`,
    `[Seniority]: ${profile.seniority}`,
    `[Years of Experience]: ${profile.yearsOfExperience}`,
    `[Skills]: ${profile.skills.join(', ')}`,
    `[Summary]: ${profile.summary}`,
    `[Experience]: ${experienceText}`,
    `[Education]: ${educationText}`,
  ].join('\n');
}
