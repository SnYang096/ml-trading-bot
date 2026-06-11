import { useEffect, useState } from 'react';
import styles from './TradeMapBusyHud.module.css';

const SPINNER_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'] as const;

export type TradeMapBusyMode = 'full' | 'history';

const MODE_LABEL: Record<TradeMapBusyMode, string> = {
  full: '> STREAM /api/trade-map/bundle',
  history: '> PREFETCH ohlcv_history',
};

export function TradeMapBusyHud({ mode }: { mode: TradeMapBusyMode }) {
  const [frame, setFrame] = useState(0);

  useEffect(() => {
    const t = window.setInterval(() => {
      setFrame((v) => (v + 1) % SPINNER_FRAMES.length);
    }, 90);
    return () => window.clearInterval(t);
  }, []);

  return (
    <div className={styles.overlay} aria-live="polite" aria-busy="true">
      <div className={styles.scanlines} />
      <div className={styles.hud}>
        <span className={styles.spinner}>{SPINNER_FRAMES[frame]}</span>
        <span className={styles.label}>{MODE_LABEL[mode]}</span>
        <span className={styles.cursor}>_</span>
      </div>
    </div>
  );
}

export function TradeMapBusyStatus({ mode }: { mode: TradeMapBusyMode }) {
  const [frame, setFrame] = useState(0);

  useEffect(() => {
    const t = window.setInterval(() => {
      setFrame((v) => (v + 1) % SPINNER_FRAMES.length);
    }, 90);
    return () => window.clearInterval(t);
  }, []);

  return (
    <span className={styles.statusBusy}>
      <span className={styles.statusSpinner}>{SPINNER_FRAMES[frame]}</span>
      {MODE_LABEL[mode]}
    </span>
  );
}
