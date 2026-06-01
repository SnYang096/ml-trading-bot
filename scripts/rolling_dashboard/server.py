"""CLI entrypoint and ``ThreadingHTTPServer`` for the results dashboard."""

from __future__ import annotations

import argparse
import errno
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import List, Optional

from .constants import PROJECT_ROOT
from .handler import build_request_handler


def run_server(
    *,
    bind: str,
    port: int,
    results_root: Path,
) -> None:
    results_root = results_root.resolve()
    if not results_root.is_dir():
        print(f"❌ Not a directory: {results_root}", file=sys.stderr)
        sys.exit(1)
    handler = build_request_handler(results_root)
    try:
        httpd = ThreadingHTTPServer((bind, port), handler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            print(
                f"❌ 端口 {port} 已被占用。\n"
                "   「rolling-dashboard」已包含对 results/ 的静态文件服务（与 mlbot server 同类），\n"
                "   一般只需开一个进程：请先 Ctrl+C 关掉占用端口的进程（例如 mlbot server），\n"
                f"   或换端口：--port 8010 ，或结束占用：mlbot rolling-dashboard --port {port} --force",
                file=sys.stderr,
            )
            sys.exit(1)
        raise
    print("🌐 本地研发服务（results 静态 + /browse + /rd 实验管理）")
    print(f"   root:    {results_root}")
    print(f"   bind:    http://{bind}:{port}/")
    print(f"   R&D 实验: http://{bind}:{port}/rd")
    print(f"   浏览目录: http://{bind}:{port}/browse")
    print(f"   入口:    http://{bind}:{port}/dashboard")
    print("   实盘 CMS（远程）: mlbot console → :8800/trade-map（与本地 /rd 分离）")
    print("   Ctrl+C 停止")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n停止")


def run_from_project(
    project_root: Path,
    *,
    bind: str,
    port: int,
    results_rel: str,
    force: bool = False,
) -> None:
    root = Path(results_rel)
    if not root.is_absolute():
        root = (Path(project_root) / results_rel).resolve()
    if force:
        from src.cli.main import _find_listening_pids, _kill_pids, _port_is_in_use

        if _port_is_in_use(port, bind=bind):
            pids = _find_listening_pids(port)
            if not pids:
                print(
                    f"❌ Port {port} in use but could not find PID (try another port).",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(f"⚠️  Port {port} in use by PID(s) {pids}; terminating (--force)...")
            _kill_pids(pids)
            import time as _time

            for _ in range(30):
                if not _port_is_in_use(port, bind=bind):
                    break
                _time.sleep(0.1)
            if _port_is_in_use(port, bind=bind):
                print(f"❌ Port {port} still busy after kill.", file=sys.stderr)
                sys.exit(1)
    run_server(bind=bind, port=port, results_root=root)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="rolling_sim 汇总看板 + 静态 results")
    ap.add_argument(
        "--root",
        default="results",
        help="相对仓库根或绝对路径（默认 results）",
    )
    ap.add_argument("--bind", default="127.0.0.1", help="绑定地址")
    ap.add_argument("--port", "-p", type=int, default=8008, help="端口")
    ap.add_argument(
        "--force",
        action="store_true",
        help="若端口被占用则尝试结束监听进程（与 mlbot server --force 同类）",
    )
    args = ap.parse_args(argv)
    run_from_project(
        PROJECT_ROOT,
        bind=args.bind,
        port=int(args.port),
        results_rel=args.root,
        force=bool(args.force),
    )


if __name__ == "__main__":
    main()
