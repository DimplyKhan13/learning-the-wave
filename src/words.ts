/**
 * Turn a four-letter word into t, the network's only "text" input.
 *
 * The network never reads letters. This rule does, one letter at a time: how
 * far is the letter, in the alphabet, from the WAVE letter and from the FUJI
 * letter in the same position? Closer to W than to F means closer to the wave.
 * The word's t is the average over its four letters.
 *
 *   Z vs W/F: 3 away from W, 20 from F → 3 / (3 + 20) = 0.13 (mostly wave)
 *   WEVE → 0.05 (the wave)    FERI → 0.63 (more Fuji)
 */
export const WORDS = ['WAVE', 'FUJI'] as const;

export interface LetterT {
  letter: string;
  /** Alphabet distance to the WAVE letter and to the FUJI letter in this position. */
  toWave: number;
  toFuji: number;
  t: number;
}

export function wordToT(word: string): { t: number; letters: LetterT[] } | null {
  const w = word.toUpperCase();
  if (!/^[A-Z]{4}$/.test(w)) return null;
  const letters = [...w].map((letter, i) => {
    const toWave = Math.abs(letter.charCodeAt(0) - WORDS[0].charCodeAt(i));
    const toFuji = Math.abs(letter.charCodeAt(0) - WORDS[1].charCodeAt(i));
    // WAVE and FUJI differ in every position, so the two distances never are both 0.
    return { letter, toWave, toFuji, t: toWave / (toWave + toFuji) };
  });
  return { t: letters.reduce((sum, l) => sum + l.t, 0) / letters.length, letters };
}
