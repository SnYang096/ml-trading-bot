import { useEffect, useState } from 'react';

/** True when the document tab is visible (Page Visibility API). */
export function usePageVisible(): boolean {
  const [visible, setVisible] = useState(
    () => typeof document === 'undefined' || document.visibilityState !== 'hidden',
  );

  useEffect(() => {
    const onChange = () => setVisible(document.visibilityState !== 'hidden');
    document.addEventListener('visibilitychange', onChange);
    return () => document.removeEventListener('visibilitychange', onChange);
  }, []);

  return visible;
}

/** React Query refetchInterval helper: pause when tab hidden. */
export function visibleRefetchInterval(visible: boolean, ms: number): number | false {
  return visible ? ms : false;
}
