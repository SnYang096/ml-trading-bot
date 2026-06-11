export function ChunkLoadError({ page, detail }: { page: string; detail?: string }) {
  return (
    <div className="page">
      <h2>{page} 加载失败</h2>
      <p className="pnl-neg">
        静态资源版本不一致。请执行 <code>make frontend-build</code> 并重启 business console，再用
        Ctrl+Shift+R 强制刷新。
      </p>
      {detail ? <p className="muted">{detail}</p> : null}
    </div>
  );
}
