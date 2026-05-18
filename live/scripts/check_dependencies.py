#!/usr/bin/env python
"""实盘启动依赖自检程序（策略B 版）

检查项目：
1. 配置文件（constitution, strategies/tpc）
2. warmup ticks 数据（live/{universe}/data/ticks/）
3. Binance API 密钥
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class LiveDependencyChecker:
    """实盘启动依赖检查器"""
    
    def __init__(self, live_root: str = "live"):
        self.live_root = Path(live_root)
        self.config_dir = self.live_root / "config"
        self.ticks_dir = self.live_root / "data" / "ticks"
        self.bars_dir = self.live_root / "data" / "bars"
        
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def check_all(self, symbols: List[str]) -> bool:
        """执行所有检查
        
        Returns:
            True if all checks passed, False otherwise
        """
        logger.info("=" * 60)
        logger.info("🔍 实盘启动依赖自检")
        logger.info("=" * 60)
        
        self._check_config_files()
        self._check_warmup_ticks(symbols)
        self._check_api_key()
        
        # 输出结果
        logger.info("")
        logger.info("=" * 60)
        logger.info("📋 自检结果汇总")
        logger.info("=" * 60)
        
        if self.errors:
            logger.error(f"❌ 发现 {len(self.errors)} 个致命错误：")
            for i, err in enumerate(self.errors, 1):
                logger.error(f"   {i}. {err}")
            logger.error("")
            logger.error("⛔ 启动被拒绝！请修复上述错误后重试。")
            return False
        
        if self.warnings:
            logger.warning(f"⚠️  发现 {len(self.warnings)} 个警告：")
            for i, warn in enumerate(self.warnings, 1):
                logger.warning(f"   {i}. {warn}")
            logger.warning("")
        
        logger.info("✅ 所有依赖检查通过！")
        return True
    
    def _check_config_files(self) -> None:
        """检查配置文件完整性"""
        logger.info("")
        logger.info("📁 检查配置文件...")
        
        # 检查config目录
        if not self.config_dir.exists():
            self.errors.append(f"配置目录缺失: {self.config_dir}")
            return
        
        # 检查constitution
        constitution_dir = self.config_dir / "constitution"
        if not constitution_dir.exists():
            self.errors.append(f"宪法配置缺失: {constitution_dir}")
        else:
            logger.info(f"   ✅ Constitution: {constitution_dir}")
        
        # 检查 strategies/tpc：现行布局是 meta.yaml + archetypes/<layer>.yaml
        # （`holding` 不是单独文件，而是 archetype yaml 内部的 section）
        tpc_dir = self.config_dir / "strategies" / "tpc"
        if not tpc_dir.exists():
            self.errors.append(f"TPC策略配置缺失: {tpc_dir}")
        else:
            arch_dir = tpc_dir / "archetypes"
            # evidence.yaml: optional at runtime (loader returns empty EvidenceConfig if absent)
            required_archetypes = ["gate.yaml", "execution.yaml", "prefilter.yaml"]
            missing_files = []

            if not (tpc_dir / "meta.yaml").exists():
                missing_files.append("meta.yaml")
            if not arch_dir.exists():
                missing_files.append("archetypes/")
            else:
                for filename in required_archetypes:
                    if not (arch_dir / filename).exists():
                        missing_files.append(f"archetypes/{filename}")

            if missing_files:
                self.errors.append(
                    f"TPC配置文件缺失: {', '.join(missing_files)} (路径: {tpc_dir})"
                )
            else:
                logger.info(f"   ✅ TPC Config: {tpc_dir}")
                present = list(required_archetypes)
                evid = arch_dir / "evidence.yaml"
                if evid.exists():
                    present.append("evidence.yaml")
                else:
                    self.warnings.append(
                        f"未找到 {arch_dir / 'evidence.yaml'}（可选；无则 Evidence 层为空配置）"
                    )
                logger.info(
                    "      - meta.yaml, " + ", ".join(f"archetypes/{f}" for f in present)
                )
    
    def _check_warmup_ticks(self, symbols: List[str]) -> None:
        """检查 warmup ticks 数据（策略B：基于历史 ticks 重算特征）"""
        logger.info("")
        logger.info("📊 检查 warmup ticks 数据...")
        
        if not self.ticks_dir.exists():
            self.warnings.append(
                f"ticks 目录缺失: {self.ticks_dir}\n"
                f"      请先运行: bash live/scripts/prepare_warmup_ticks.sh"
            )
            return
        
        for symbol in symbols:
            symbol_dir = self.ticks_dir / symbol
            
            if not symbol_dir.exists():
                self.warnings.append(f"Symbol {symbol} ticks 目录缺失: {symbol_dir}")
                continue
            
            parquet_files = sorted(symbol_dir.glob("*.parquet"))
            if not parquet_files:
                self.warnings.append(
                    f"Symbol {symbol} 无 warmup ticks 数据（系统将从零开始累积，需 4h+）"
                )
            else:
                first = parquet_files[0].stem
                last = parquet_files[-1].stem
                logger.info(
                    f"   ✅ {symbol}: {len(parquet_files)} 天 "
                    f"({first} ~ {last})"
                )
    
    def _check_api_key(self) -> None:
        """检查 Binance API 密钥配置"""
        logger.info("")
        logger.info("🔑 检查 API 密钥...")
        
        import os
        api_key = os.environ.get("BINANCE_API_KEY", "")
        api_secret = os.environ.get("BINANCE_API_SECRET", "")
        
        env_file = self.live_root.parent / "binance_mainnet.env"
        
        if api_key and api_secret:
            logger.info("   ✅ API 密钥已配置（环境变量）")
        elif env_file.exists():
            logger.info(f"   ✅ API 密钥文件存在: {env_file}")
        else:
            self.warnings.append(
                f"未找到 API 密钥（环境变量或 {env_file}）\n"
                f"      观察模式(TRADE_SIZE=0.0)可不配置，实际交易必须配置"
            )


def main():
    """主入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="实盘启动依赖自检")
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTCUSDT",
        help="交易币种（逗号分隔），默认BTCUSDT",
    )
    parser.add_argument(
        "--live-root",
        type=str,
        default="live",
        help="实盘根目录，默认live",
    )
    
    args = parser.parse_args()
    
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    
    checker = LiveDependencyChecker(live_root=args.live_root)
    success = checker.check_all(symbols)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
