/**
 * Turn a four-letter word into t, the network's only "text" input.
 *
 * The network never reads letters. This rule does: compare the word with WAVE
 * (t = 0) and FUJI (t = 1) letter by letter, and land closer to whichever it
 * matches more. WEVE → 0.2 (mostly wave), FERI → 0.67 (mostly Fuji).
 */
export const WORDS = ['WAVE', 'FUJI'] as const;

export interface WordT {
  t: number;
  /** Letters that differ from WAVE and from FUJI, position by position. */
  offWave: number;
  offFuji: number;
}

export function wordToT(word: string): WordT | null {
  const w = word.toUpperCase();
  if (!/^[A-Z]{4}$/.test(w)) return null;
  const differences = (target: string) => [...w].filter((c, i) => c !== target[i]).length;
  const offWave = differences(WORDS[0]);
  const offFuji = differences(WORDS[1]);
  // WAVE and FUJI share no letter in the same position, so the sum is never 0.
  return { t: offWave / (offWave + offFuji), offWave, offFuji };
}
