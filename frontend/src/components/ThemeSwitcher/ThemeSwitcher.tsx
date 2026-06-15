import { useTheme } from '@/context/ThemeContext.tsx';
import type { ThemeId } from '@/lib/theme.ts';
import styles from './ThemeSwitcher.module.css';

export function ThemeSwitcher() {
  const { theme, setTheme, themes } = useTheme();
  const current = themes.find((t) => t.id === theme);

  return (
    <label className={styles.root} title={current?.hint}>
      <span className={styles.label}>皮肤</span>
      <select
        className={styles.select}
        value={theme}
        onChange={(e) => setTheme(e.target.value as ThemeId)}
        aria-label="切换界面皮肤"
      >
        {themes.map((t) => (
          <option key={t.id} value={t.id}>
            {t.label}
          </option>
        ))}
      </select>
    </label>
  );
}
