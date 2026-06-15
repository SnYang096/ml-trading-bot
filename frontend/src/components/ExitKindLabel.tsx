import { EXIT_KIND_GUIDE, exitKindGuideText, exitKindMeta } from '@/lib/exitKind.ts';
import { useId, useState, type CSSProperties, type ReactNode } from 'react';

const helpMarkStyle: CSSProperties = {
  cursor: 'help',
  borderBottom: '1px dotted rgba(255, 255, 255, 0.35)',
};

const headBtnStyle: CSSProperties = {
  background: 'none',
  border: 'none',
  color: 'inherit',
  font: 'inherit',
  padding: 0,
  cursor: 'help',
};

const legendPanelStyle: CSSProperties = {
  margin: '0 0 12px',
  padding: '10px 12px',
  fontSize: '0.82rem',
  lineHeight: 1.45,
  borderRadius: 6,
  border: '1px solid rgba(255, 255, 255, 0.12)',
  background: 'rgba(0, 0, 0, 0.25)',
};

export function ExitKindLabel({ kind }: { kind: unknown }) {
  const { label, tip } = exitKindMeta(kind);
  if (label === '—') return <span>—</span>;
  return (
    <span title={tip} style={helpMarkStyle}>
      {label}
    </span>
  );
}

export function ExitKindColumnHeader({
  legendOpen,
  onToggleLegend,
}: {
  legendOpen?: boolean;
  onToggleLegend?: () => void;
}) {
  const legendId = useId();
  return (
    <th scope="col">
      <button
        type="button"
        style={headBtnStyle}
        title={exitKindGuideText()}
        aria-expanded={legendOpen ?? false}
        aria-controls={onToggleLegend ? legendId : undefined}
        onClick={(e) => {
          e.stopPropagation();
          onToggleLegend?.();
        }}
      >
        平仓方式 <span aria-hidden>ⓘ</span>
      </button>
    </th>
  );
}

export function ExitKindLegendPanel({ open }: { open: boolean }) {
  if (!open) return null;
  return (
    <div className="muted" style={legendPanelStyle} role="note">
      <p style={{ margin: '0 0 8px' }}>
        多种平仓方式来自不同账户层与策略：Spot 卖出、Trend 止损/Regime/结构退出、Multi-leg
        止盈/止损/市价平/跨策略清理等。悬停单元格可看该笔说明；下表为全集。
      </p>
      <dl style={{ margin: 0, display: 'grid', gap: '6px 12px', gridTemplateColumns: 'auto 1fr' }}>
        {EXIT_KIND_GUIDE.map((g) => (
          <div key={g.label} style={{ display: 'contents' }}>
            <dt style={{ margin: 0, fontWeight: 600, whiteSpace: 'nowrap' }}>{g.label}</dt>
            <dd style={{ margin: 0 }}>{g.tip}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

export function useExitKindLegend(): {
  legendOpen: boolean;
  toggleLegend: () => void;
  legendPanel: ReactNode;
} {
  const [legendOpen, setLegendOpen] = useState(false);
  return {
    legendOpen,
    toggleLegend: () => setLegendOpen((v) => !v),
    legendPanel: <ExitKindLegendPanel open={legendOpen} />,
  };
}
