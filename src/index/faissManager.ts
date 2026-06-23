import fs from 'fs';
import path from 'path';
import { EmbeddingRecord } from '../types';
import { CONFIG } from '../config';
import { normalizeVector, flattenVectors } from '../utils';

// faiss-node types are incomplete; use require for flexibility
// eslint-disable-next-line @typescript-eslint/no-var-requires
const faiss = require('faiss-node');

export interface FaissSearchResult {
  profileId: string;
  score: number;
}

interface FaissIndexInstance {
  ntotal: number;
  train(data: number[]): void;
  add(data: number[]): void;
  search(query: number[], k: number): { distances: number[]; labels: number[] };
  write(path: string): void;
}

export interface FaissIndex {
  index: FaissIndexInstance;
  profileIdMap: string[];
}

export function buildIndex(records: EmbeddingRecord[]): FaissIndex {
  const d = CONFIG.openai.embeddings.shortDims;

  const profileIdMap: string[] = records.map((r) => r.profileId);
  const normalizedVectors = records.map((r) => normalizeVector(r.shortVector));
  const flatData = flattenVectors(normalizedVectors);

  const index = new faiss.IndexFlatL2(d) as FaissIndexInstance;
  index.add(flatData);
  console.log(`Index built: ${index.ntotal} vectors indexed.`);

  return { index, profileIdMap };
}

export function saveIndex(faissIndex: FaissIndex, indexPath: string, mapPath: string): void {
  const indexDir = path.dirname(indexPath);
  if (!fs.existsSync(indexDir)) fs.mkdirSync(indexDir, { recursive: true });

  faissIndex.index.write(indexPath);
  fs.writeFileSync(mapPath, JSON.stringify(faissIndex.profileIdMap, null, 2));
  console.log(`Index saved to ${indexPath}`);
}

export function loadIndex(indexPath: string, mapPath: string): FaissIndex {
  if (!fs.existsSync(indexPath)) {
    throw new Error(`FAISS index not found at ${indexPath}. Run "npm run index" first.`);
  }
  const index = faiss.Index.read(indexPath) as FaissIndexInstance;
  const profileIdMap = JSON.parse(fs.readFileSync(mapPath, 'utf-8')) as string[];
  return { index, profileIdMap };
}

export function searchIndex(
  faissIndex: FaissIndex,
  queryVector: number[],
  topK: number,
): FaissSearchResult[] {
  const { index, profileIdMap } = faissIndex;
  const normalizedQuery = normalizeVector(queryVector);
  const { distances, labels } = index.search(normalizedQuery, topK) as {
    distances: number[];
    labels: number[];
  };

  const results: FaissSearchResult[] = [];
  for (let i = 0; i < labels.length; i++) {
    const label = Number(labels[i]);
    if (label < 0 || label >= profileIdMap.length) continue;
    // faiss returns squared L2 distance (d_sq). For unit vectors: cos = 1 - d_sq/2
    const cosScore = 1 - distances[i] / 2;
    results.push({ profileId: profileIdMap[label], score: cosScore });
  }

  return results;
}
