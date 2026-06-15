export type ThemeId =
  | 'terminal'
  | 'bloomberg'
  | 'tradingview'
  | 'light'
  | 'paper'
  | 'oled'
  | 'a11y';

export type ChartTheme = {
  bg: string;
  text: string;
  grid: string;
  border: string;
  candleUp: string;
  candleDown: string;
  emaPrimary: string;
  emaSecondary: string;
  volume: string;
  linkFallback: string;
  accentPurple: string;
};

export type ThemeMeta = {
  id: ThemeId;
  label: string;
  hint: string;
  metaColor: string;
};

export const THEMES: ThemeMeta[] = [
  {
    id: 'terminal',
    label: '极客终端',
    hint: 'Matrix 绿 · 扫描线 · 等宽字',
    metaColor: '#020402',
  },
  {
    id: 'bloomberg',
    label: '彭博终端',
    hint: '深蓝底 · 琥珀强调 · 机构风',
    metaColor: '#0a0e14',
  },
  {
    id: 'tradingview',
    label: 'TradingView',
    hint: '冷灰蓝 · 经典 K 线配色',
    metaColor: '#131722',
  },
  {
    id: 'light',
    label: '日间专业',
    hint: '浅灰白底 · 无衬线 · 低眩光',
    metaColor: '#f5f6f8',
  },
  {
    id: 'paper',
    label: '纸质账本',
    hint: '米黄暖色 · 衬线标题',
    metaColor: '#f4f0e8',
  },
  {
    id: 'oled',
    label: '纯黑 OLED',
    hint: '真黑 #000 · 无扫描线 · 省电',
    metaColor: '#000000',
  },
  {
    id: 'a11y',
    label: '高对比无障碍',
    hint: '大字号 · 强对比 · WCAG 友好',
    metaColor: '#000000',
  },
];

export const CHART_THEMES: Record<ThemeId, ChartTheme> = {
  terminal: {
    bg: '#020402',
    text: '#7ae87a',
    grid: '#143814',
    border: '#2a7a2a',
    candleUp: '#00ff41',
    candleDown: '#ff3366',
    emaPrimary: '#5cffff',
    emaSecondary: '#ffcc44',
    volume: '#2a7a2a',
    linkFallback: '#7ae87a',
    accentPurple: '#d966ff',
  },
  bloomberg: {
    bg: '#0a0e14',
    text: '#9aa0a8',
    grid: '#1a2230',
    border: '#2a3545',
    candleUp: '#3dd68c',
    candleDown: '#f23645',
    emaPrimary: '#4da6ff',
    emaSecondary: '#ff8c00',
    volume: '#2a3545',
    linkFallback: '#c8cdd4',
    accentPurple: '#c77dff',
  },
  tradingview: {
    bg: '#131722',
    text: '#787b86',
    grid: '#1e222d',
    border: '#363a45',
    candleUp: '#26a69a',
    candleDown: '#ef5350',
    emaPrimary: '#2962ff',
    emaSecondary: '#ff9800',
    volume: '#363a45',
    linkFallback: '#787b86',
    accentPurple: '#ab47bc',
  },
  light: {
    bg: '#ffffff',
    text: '#5c6370',
    grid: '#eef0f4',
    border: '#d1d5db',
    candleUp: '#059669',
    candleDown: '#dc2626',
    emaPrimary: '#2563eb',
    emaSecondary: '#d97706',
    volume: '#cbd5e1',
    linkFallback: '#64748b',
    accentPurple: '#7c3aed',
  },
  paper: {
    bg: '#faf8f4',
    text: '#6b5f4f',
    grid: '#ebe6dc',
    border: '#c4b8a8',
    candleUp: '#2d6a4f',
    candleDown: '#9b2226',
    emaPrimary: '#5c4a32',
    emaSecondary: '#b45309',
    volume: '#d6cfc0',
    linkFallback: '#6b5f4f',
    accentPurple: '#7c4a6a',
  },
  oled: {
    bg: '#000000',
    text: '#b0b0b0',
    grid: '#1a1a1a',
    border: '#333333',
    candleUp: '#00e676',
    candleDown: '#ff5252',
    emaPrimary: '#40c4ff',
    emaSecondary: '#ffd740',
    volume: '#2a2a2a',
    linkFallback: '#b0b0b0',
    accentPurple: '#ea80fc',
  },
  a11y: {
    bg: '#000000',
    text: '#ffffff',
    grid: '#333333',
    border: '#ffffff',
    candleUp: '#00ff00',
    candleDown: '#ff4444',
    emaPrimary: '#00d4ff',
    emaSecondary: '#ffff00',
    volume: '#444444',
    linkFallback: '#ffffff',
    accentPurple: '#ff66ff',
  },
};

const STORAGE_KEY = 'cms-theme';

export function isThemeId(v: unknown): v is ThemeId {
  return typeof v === 'string' && THEMES.some((t) => t.id === v);
}

export function getChartTheme(theme: ThemeId): ChartTheme {
  return CHART_THEMES[theme];
}

function updateMetaThemeColor(theme: ThemeId) {
  const meta = document.querySelector('meta[name="theme-color"]');
  const color = THEMES.find((t) => t.id === theme)?.metaColor ?? '#020402';
  if (meta) meta.setAttribute('content', color);
}

export function readStoredTheme(): ThemeId {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (isThemeId(stored)) return stored;
  } catch {
    /* ignore */
  }
  return 'terminal';
}

/** Call before React mount to avoid theme flash. */
export function initTheme(): ThemeId {
  const id = readStoredTheme();
  document.documentElement.dataset.theme = id;
  updateMetaThemeColor(id);
  return id;
}

export function applyTheme(theme: ThemeId) {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    /* ignore */
  }
  updateMetaThemeColor(theme);
  window.dispatchEvent(new CustomEvent('cms-theme-change', { detail: theme }));
}
