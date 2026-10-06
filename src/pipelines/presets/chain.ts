type Dimensions = { width: number; height: number };

/** Какие звенья нужны цепочке Mode C/CA при данных исходном и целевом размере. */
export interface ModeCChain {
  /** Upscale-Denoise ×2 по исходнику. */
  upscale1: boolean;
  /** Уменьшение до цели: она между 1.2× и 2× исходника. */
  downscale2: boolean;
  /** Уменьшение до половины цели перед вторым ×2: цель между 2.4× и 4×. */
  downscale4: boolean;
  /** Второе увеличение CNNx2M. */
  upscale2: boolean;
}

/**
 * Решения те же, что в Anime4K Mode C (условия //!WHEN в GLSL): считаются до
 * постройки, чтобы знать последнее звено — ему отдаётся зажим Clamp
 * Highlights (deRing) вместо отдельного прохода.
 */
export function planModeC(native: Dimensions, target: Dimensions): ModeCChain {
  const above = (k: number) => target.width > k * native.width && target.height > k * native.height;
  const below = (k: number) => target.width < k * native.width && target.height < k * native.height;
  const upscale1 = above(1.2);
  const downscale2 = above(1.2) && below(2);
  const downscale4 = above(2.4) && below(4);
  let current = native;
  if (upscale1) current = { width: native.width * 2, height: native.height * 2 };
  if (downscale2) current = target;
  if (downscale4) current = { width: Math.ceil(target.width / 2), height: Math.ceil(target.height / 2) };
  const upscale2 = target.width > 1.2 * current.width && target.height > 1.2 * current.height;
  return {
    upscale1, downscale2, downscale4, upscale2,
  };
}
