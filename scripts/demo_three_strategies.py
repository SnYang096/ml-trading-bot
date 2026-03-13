#!/usr/bin/env python3
"""
三策略实盘启动演示脚本

演示如何同时启动 BPC + ME + FER 三个 archetype 策略
使用 LivePCM 进行多策略仲裁和信号管理

环境变量设置:
MLBOT_LIVE_SYMBOLS=BTCUSDT,ETHUSDT
MLBOT_STRATEGIES_ROOT=live/highcap/config/strategies
MLBOT_LIVE_STORAGE_BASE=data/live_storage
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.time_series_model.live.generic_live_strategy import GenericLiveStrategy
from src.time_series_model.portfolio.live_pcm import LivePCM
from src.live_data_stream import StorageManager, MultiSymbolManager
from src.live_data_stream.order_manager_factory import init_order_manager_from_env


# =============================================================================
# 配置设置
# =============================================================================

# 环境变量配置
os.environ.setdefault("MLBOT_LIVE_SYMBOLS", "BTCUSDT,ETHUSDT")
os.environ.setdefault("MLBOT_STRATEGIES_ROOT", "live/highcap/config/strategies")
os.environ.setdefault("MLBOT_LIVE_STORAGE_BASE", "data/live_storage")
os.environ.setdefault("MLBOT_LIVE_USE_FUTURES", "true")
os.environ.setdefault("MLBOT_LIVE_WARMUP_DAYS", "7")
os.environ.setdefault("MLBOT_LIVE_TRADE_SIZE", "0.0")  # 0 = 不自动交易
os.environ.setdefault("MLBOT_MAX_SLOTS", "2")

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# 策略配置验证
# =============================================================================


def check_strategy_configs():
    """检查三个策略的配置文件完整性"""

    print("=== 策略配置文件检查 ===\n")

    strategies_root = os.environ["MLBOT_STRATEGIES_ROOT"]
    strategies = ["bpc", "me-long", "fer"]

    all_configs_present = True

    for strategy in strategies:
        arch_dir = Path(strategies_root) / strategy / "archetypes"
        print(f"检查策略: {strategy.upper()}")
        print(f"  配置目录: {arch_dir}")

        required_files = [
            "direction.yaml",
            "gate.yaml",
            "evidence.yaml",
            "execution.yaml",
            "entry_filters.yaml",
        ]

        missing_files = []
        for filename in required_files:
            file_path = arch_dir / filename
            if file_path.exists():
                print(f"    ✅ {filename}")
            else:
                print(f"    ❌ {filename} (缺失)")
                missing_files.append(filename)
                all_configs_present = False

        if missing_files:
            print(f"  ⚠️  缺失配置文件: {', '.join(missing_files)}")
        print()

    return all_configs_present


# =============================================================================
# 策略初始化
# =============================================================================


def create_strategies():
    """创建三个策略实例"""

    print("=== 初始化策略实例 ===\n")

    strategies = {}

    # BPC 策略
    print("1. 初始化 BPC 策略...")
    try:
        bpc_strategy = GenericLiveStrategy(
            strategy_name="bpc",
            strategies_root=os.environ["MLBOT_STRATEGIES_ROOT"],
        )
        bpc_strategy.load_configs()
        strategies["bpc"] = bpc_strategy
        print("   ✅ BPC 策略初始化成功")
    except Exception as e:
        print(f"   ❌ BPC 策略初始化失败: {e}")
        return None

    # ME 策略
    print("\n2. 初始化 ME 策略...")
    try:
        me_strategy = GenericLiveStrategy(
            strategy_name="me-long",
            strategies_root=os.environ["MLBOT_STRATEGIES_ROOT"],
        )
        me_strategy.load_configs()
        strategies["me-long"] = me_strategy
        print("   ✅ ME 策略初始化成功")
    except Exception as e:
        print(f"   ❌ ME 策略初始化失败: {e}")
        return None

    # FER 策略
    print("\n3. 初始化 FER 策略...")
    try:
        fer_strategy = GenericLiveStrategy(
            strategy_name="fer",
            strategies_root=os.environ["MLBOT_STRATEGIES_ROOT"],
        )
        fer_strategy.load_configs()
        strategies["fer"] = fer_strategy
        print("   ✅ FER 策略初始化成功")
    except Exception as e:
        print(f"   ❌ FER 策略初始化失败: {e}")
        return None

    return strategies


# =============================================================================
# PCM 仲裁层设置
# =============================================================================


def setup_pcm(strategies):
    """设置 LivePCM 仲裁层"""

    print("\n=== 设置 PCM 仲裁层 ===\n")

    # 创建 LivePCM 实例
    # Regime-Aware: NORMAL(LV>FER>ME>BPC), HIGH_VOL(LV>ME>FER>BPC)
    pcm = LivePCM(
        archetype_priority=["LV", "FER", "ME-LONG", "BPC"],
        max_slots=int(os.environ.get("MLBOT_MAX_SLOTS", "2")),
        regime_config_path=os.environ.get(
            "MLBOT_PCM_REGIME_CONFIG", "config/pcm_regime.yaml"
        ),
    )

    # 注册所有策略
    for name, strategy in strategies.items():
        pcm.register(name, strategy)
        print(f"✅ 注册策略: {name.upper()}")

    print(f"\n📊 PCM 配置:")
    print(f"   优先级顺序: {' > '.join(pcm.archetype_priority)}")
    print(f"   最大 slot 数: {pcm._max_slots}")
    print(f"   已注册策略: {', '.join(pcm.registered_archetypes)}")

    return pcm


# =============================================================================
# 模拟特征数据测试
# =============================================================================


def test_signal_generation(strategies, pcm):
    """测试信号生成功能"""

    print("\n=== 信号生成测试 ===\n")

    # 测试特征数据
    test_features = {
        "close": 50000.0,
        "volume": 100.0,
        "atr": 500.0,
        # BPC 特征
        "bpc_score_breakout": 0.8,
        "bpc_was_in_pullback": 1,
        # ME 特征
        "me_compression_score": 0.7,
        "me_volume_expansion": 1.5,
        # FER 特征
        "impulse_failure_score": -0.6,  # 负值表示多头失败 → 做多
        "aggressor_absorption_ratio": 0.8,
    }

    symbols = os.environ["MLBOT_LIVE_SYMBOLS"].split(",")

    for symbol in symbols:
        print(f"\n📊 测试币种: {symbol}")
        print("-" * 50)

        # 测试单策略信号
        for name, strategy in strategies.items():
            try:
                intents = strategy.decide(features=test_features, symbol=symbol)
                if intents:
                    intent = intents[0]
                    print(
                        f"✅ {name.upper()}: {intent.action} (置信度: {intent.confidence:.3f})"
                    )
                else:
                    print(f"❌ {name.upper()}: 无信号")
            except Exception as e:
                print(f"❌ {name.upper()}: 错误 - {e}")

        # 测试 PCM 仲裁
        print(f"\n⚖️  PCM 仲裁结果:")
        try:
            pcm_intents = pcm.decide(features=test_features, symbol=symbol)
            if pcm_intents:
                intent = pcm_intents[0]
                print(
                    f"✅ 最终信号: {intent.action} via {intent.archetype.upper()} "
                    f"(置信度: {intent.confidence:.3f})"
                )
            else:
                print("❌ 无仲裁信号")
        except Exception as e:
            print(f"❌ 仲裁失败: {e}")


# =============================================================================
# 主函数
# =============================================================================


async def main():
    """主函数 - 演示三策略启动流程"""

    print("=" * 80)
    print("🚀 三策略实盘启动演示 (BPC + ME + FER)")
    print("=" * 80)
    print()

    # 1. 环境信息
    print("=== 环境配置 ===")
    print(f"交易对: {os.environ['MLBOT_LIVE_SYMBOLS']}")
    print(f"策略目录: {os.environ['MLBOT_STRATEGIES_ROOT']}")
    print(f"存储目录: {os.environ['MLBOT_LIVE_STORAGE_BASE']}")
    print(f"交易大小: {os.environ['MLBOT_LIVE_TRADE_SIZE']} (0=观察模式)")
    print()

    # 2. 配置文件检查
    if not check_strategy_configs():
        print("❌ 配置文件不完整，无法继续")
        return 1

    # 3. 策略初始化
    strategies = create_strategies()
    if strategies is None:
        print("❌ 策略初始化失败")
        return 1

    # 4. PCM 设置
    pcm = setup_pcm(strategies)

    # 5. 信号测试
    test_signal_generation(strategies, pcm)

    # 6. 实盘启动准备状态
    print("\n" + "=" * 80)
    print("✅ 启动准备完成")
    print("=" * 80)
    print("\n下一步可执行:")
    print("1. 设置真实交易大小: export MLBOT_LIVE_TRADE_SIZE=0.1")
    print("2. 启动完整实盘: python scripts/run_live.py")
    print("3. 监控日志输出")

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 程序异常: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
