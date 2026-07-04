import { CandidateProfile, SkillMatchStat } from '../types';

interface SkillEnrichmentStat {
  score: number; // enrichment ratio = (foreground fraction) / (background fraction)
  foregroundCount: number; // raw count in the foreground set, used only as a tiebreaker
}

function skillCounts(profiles: CandidateProfile[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const profile of profiles) {
    for (const skill of profile.skills) {
      counts.set(skill, (counts.get(skill) ?? 0) + 1);
    }
  }
  return counts;
}

// Background: fraction of the full corpus with each skill. Computed once per process.
export function buildBackgroundSkillFrequency(profiles: CandidateProfile[]): Map<string, number> {
  const counts = skillCounts(profiles);
  const freq = new Map<string, number>();
  for (const [skill, count] of counts) {
    freq.set(skill, count / profiles.length);
  }
  return freq;
}

// Enrichment (lift) per skill: (foreground fraction) / (background fraction), plus the raw
// foreground count for tiebreaking. Only computed for skills present in the foreground —
// since foreground is always a subset of the background corpus, the background fraction for
// those skills is guaranteed > 0, so no smoothing is needed.
//
// Skills that appear only once anywhere in the corpus (and that one appearance is in the
// foreground) all hit the same ceiling ratio (backgroundSize / foregroundSize), regardless of
// which skill it is — the ratio alone can't distinguish "rare and interesting" from "rare and
// coincidental" at n=1. topDistinguishingSkills breaks that tie using foregroundCount.
export function computeSkillEnrichment(
  foregroundProfiles: CandidateProfile[],
  backgroundFreq: Map<string, number>,
): Map<string, SkillEnrichmentStat> {
  const foregroundCounts = skillCounts(foregroundProfiles);
  const enrichment = new Map<string, SkillEnrichmentStat>();
  for (const [skill, count] of foregroundCounts) {
    const fgFreq = count / foregroundProfiles.length;
    const bgFreq = backgroundFreq.get(skill) ?? fgFreq;
    enrichment.set(skill, { score: fgFreq / bgFreq, foregroundCount: count });
  }
  return enrichment;
}

// Per-candidate: intersect the candidate's own skills against the enrichment map, keep only
// skills that are actually enriched (score > 1), sorted descending by score. Ties (common among
// singleton skills, see computeSkillEnrichment) are broken by foreground count — favoring skills
// more of the retrieved pool shares over one-off flukes — then by skill name for a deterministic
// final order.
export function topDistinguishingSkills(
  profile: CandidateProfile,
  enrichment: Map<string, SkillEnrichmentStat>,
  topN: number,
): SkillMatchStat[] {
  return profile.skills
    .map((skill) => ({ skill, stat: enrichment.get(skill) }))
    .filter((s): s is { skill: string; stat: SkillEnrichmentStat } => (s.stat?.score ?? 0) > 1)
    .sort(
      (a, b) =>
        b.stat.score - a.stat.score ||
        b.stat.foregroundCount - a.stat.foregroundCount ||
        a.skill.localeCompare(b.skill),
    )
    .slice(0, topN)
    .map(({ skill, stat }) => ({ skill, enrichment: stat.score }));
}
