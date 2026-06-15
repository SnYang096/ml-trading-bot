import { NavLink, Outlet } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { Suspense, useEffect, useState } from 'react';
import { apiGet } from '@/api/client.ts';
import type { NavLink as NavLinkRow } from '@/api/types.ts';
import { PageFallback } from '@/components/PageFallback.tsx';
import { ThemeSwitcher } from '@/components/ThemeSwitcher/ThemeSwitcher.tsx';
import { useTheme } from '@/context/ThemeContext.tsx';
import { prefetchPage } from '@/lib/pagePrefetch.ts';
import { PAGES, resolveLinkUrl } from '@/lib/shell.ts';
import styles from './AppShell.module.css';

export function AppShell() {
  const [extLinks, setExtLinks] = useState<NavLinkRow[]>([]);
  const { theme } = useTheme();
  const isTerminal = theme === 'terminal';

  useQuery({
    queryKey: ['links'],
    queryFn: async () => {
      const { data } = await apiGet<{ links: NavLinkRow[] }>('/api/links');
      setExtLinks(data.links || []);
      return data;
    },
    staleTime: 60_000,
  });

  useEffect(() => {
    document.title = 'MLBot Console';
  }, []);

  return (
    <div className={styles.root}>
      <header className={styles.toolbar}>
        <h1 className={styles.title}>
          {isTerminal ? (
            <>
              <span className={styles.prompt}>root@mlbot</span>
              <span className={styles.path}>:~/console$ </span>
              <span className={styles.cursor}>_</span>
            </>
          ) : (
            <span className={styles.brand}>MLBot Console</span>
          )}
        </h1>
        <nav className={styles.nav}>
          {PAGES.map((p) => (
            <NavLink
              key={p.id}
              to={p.href}
              className={({ isActive }) =>
                isActive ? `${styles.navLink} ${styles.navActive}` : styles.navLink
              }
              onMouseEnter={() => prefetchPage(p.id)}
              onFocus={() => prefetchPage(p.id)}
            >
              {p.label}
            </NavLink>
          ))}
        </nav>
        <div className={styles.extLinks}>
          <ThemeSwitcher />
          {extLinks.map((link) => (
            <a
              key={link.id}
              href={resolveLinkUrl(link)}
              target="_blank"
              rel="noopener noreferrer"
            >
              {link.label}
            </a>
          ))}
        </div>
      </header>
      <main className={styles.main}>
        <Suspense fallback={<PageFallback />}>
          <Outlet />
        </Suspense>
      </main>
    </div>
  );
}
