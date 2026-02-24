#!/usr/bin/env python3
"""
实验删除脚本（别名）
这是一个别名脚本，实际功能由 scripts/cleanup_old_experiments.py 提供
"""
import sys
import os
import subprocess
from pathlib import Path

def main():
    # 获取脚本目录
    script_dir = Path(__file__).parent.absolute()
    main_script = script_dir.parent / "scripts" / "cleanup_old_experiments.py"
    
    if not main_script.exists():
        print(f"错误: 主脚本不存在 {main_script}")
        sys.exit(1)
    
    # 将所有参数传递给主脚本
    cmd = [sys.executable, str(main_script)] + sys.argv[1:]
    
    # 执行主脚本
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()