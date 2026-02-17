"""
多策略实盘启动验证测试
验证 BPC/ME/FER 是否能同时启动
"""

import pytest
import os
from unittest.mock import Mock, patch
import sys

# 添加项目路径
sys.path.insert(0, "/home/yin/trading/ml_trading_bot")


def test_current_strategy_status():
    """检查各策略实盘实现状态"""

    print("=== 多策略实盘启动状态检查 ===\n")

    # 1. 检查策略实现文件
    strategies = {
        "BPC": {
            "live_strategy": "src/time_series_model/live/bpc_live_strategy.py",
            "status": "✅ 已实现",
            "class": "BPCLiveStrategy",
        },
        "FER": {
            "live_strategy": "src/time_series_model/live/fer_live_strategy.py",
            "status": "❌ 未实现",
            "class": "FERLiveStrategy",
            "todo": 'z实验_004_fer/todo.md 标注为"待执行"',
        },
        "ME": {
            "live_strategy": "src/time_series_model/live/me_live_strategy.py",
            "status": "❌ 未实现",
            "class": "MELiveStrategy",
            "todo": 'z实验_003_me/todo.md 标注为"待执行"',
        },
    }

    print("1. 策略实现状态:")
    for name, info in strategies.items():
        file_path = info["live_strategy"]
        exists = os.path.exists(file_path)
        status = "✅ 已实现" if exists else "❌ 未实现"
        print(f"   {name}: {status} ({file_path})")

        if not exists:
            print(f"         原因: {info.get('todo', '文件不存在')}")
    print()

    # 2. 检查 run_live.py 集成状态
    print("2. run_live.py 集成状态:")

    # 检查当前 run_live.py 内容
    with open("scripts/run_live.py", "r") as f:
        content = f.read()

    bpc_imported = (
        "from src.time_series_model.live.bpc_live_strategy import BPCLiveStrategy"
        in content
    )
    fer_imported = (
        "from src.time_series_model.live.fer_live_strategy import FERLiveStrategy"
        in content
    )
    me_imported = (
        "from src.time_series_model.live.me_live_strategy import MELiveStrategy"
        in content
    )

    pcm_registered_bpc = 'pcm.register("bpc", bpc)' in content
    pcm_registered_fer = 'pcm.register("fer", fer)' in content
    pcm_registered_me = 'pcm.register("me", me)' in content

    print(f"   BPC 导入: {'✅' if bpc_imported else '❌'}")
    print(f"   FER 导入: {'✅' if fer_imported else '❌'}")
    print(f"   ME 导入: {'✅' if me_imported else '❌'}")
    print()
    print(f"   BPC 注册: {'✅' if pcm_registered_bpc else '❌'}")
    print(f"   FER 注册: {'✅' if pcm_registered_fer else '❌'}")
    print(f"   ME 注册: {'✅' if pcm_registered_me else '❌'}")
    print()

    # 3. 检查配置文件同步状态
    print("3. 配置文件同步状态:")

    config_dirs = {
        "BPC": "config/strategies/bpc/archetypes/",
        "FER": "config/strategies/fer/archetypes/",
        "ME": "config/strategies/me/archetypes/",
    }

    live_dirs = {
        "BPC": "live/highcap/config/strategies/bpc/archetypes/",
        "FER": "live/highcap/config/strategies/fer/archetypes/",
        "ME": "live/highcap/config/strategies/me/archetypes/",
    }

    for strategy in ["BPC", "FER", "ME"]:
        config_exists = os.path.exists(config_dirs[strategy])
        live_exists = os.path.exists(live_dirs[strategy])

        config_count = len(os.listdir(config_dirs[strategy])) if config_exists else 0
        live_count = len(os.listdir(live_dirs[strategy])) if live_exists else 0

        status = (
            "✅"
            if (config_exists and live_exists and config_count == live_count)
            else "❌"
        )
        print(
            f"   {strategy}: {status} (研究:{config_count}文件, 实盘:{live_count}文件)"
        )
    print()

    # 4. 总体结论
    implemented_count = sum(
        [
            os.path.exists(strategies["BPC"]["live_strategy"]),
            os.path.exists(strategies["FER"]["live_strategy"]),
            os.path.exists(strategies["ME"]["live_strategy"]),
        ]
    )

    registered_count = sum([pcm_registered_bpc, pcm_registered_fer, pcm_registered_me])

    print("4. 总体结论:")
    print(f"   ✅ 已实现策略: {implemented_count}/3")
    print(f"   ✅ 已注册策略: {registered_count}/3")
    print()

    if implemented_count == 3 and registered_count == 3:
        print("🎉 结论: 可以同时启动 BPC + FER + ME")
        return True
    else:
        print("❌ 结论: 无法同时启动")
        print("   原因:")
        if implemented_count < 3:
            missing = [
                k
                for k, v in strategies.items()
                if not os.path.exists(v["live_strategy"])
            ]
            print(f"     - 缺少实现实盘策略: {', '.join(missing)}")
        if registered_count < 3:
            unregistered = []
            if not pcm_registered_bpc:
                unregistered.append("BPC")
            if not pcm_registered_fer:
                unregistered.append("FER")
            if not pcm_registered_me:
                unregistered.append("ME")
            print(f"     - 未在 run_live.py 中注册: {', '.join(unregistered)}")
        return False


def test_pcm_priority_configuration():
    """检查 PCM 仲裁优先级配置"""

    print("=== PCM 仲裁优先级检查 ===\n")

    # 检查各策略的优先级配置
    priority_configs = {
        "BPC": "config/strategies/bpc/meta.yaml",
        "FER": "config/strategies/fer/meta.yaml",
        "ME": "config/strategies/me/meta.yaml",
    }

    priorities = {}

    for strategy, config_path in priority_configs.items():
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                content = f.read()
                # 简单解析 yaml
                if "pcm_priority:" in content:
                    import re

                    match = re.search(r"pcm_priority:\s*(\d+)", content)
                    if match:
                        priorities[strategy] = int(match.group(1))
                    else:
                        priorities[strategy] = None
                else:
                    priorities[strategy] = None
        else:
            priorities[strategy] = "配置文件缺失"

    print("策略优先级配置:")
    for strategy, priority in priorities.items():
        if isinstance(priority, int):
            status = "✅" if priority is not None else "❌"
            print(f"   {strategy}: {status} 优先级={priority}")
        else:
            print(f"   {strategy}: ❌ {priority}")

    print()

    # 验证优先级逻辑
    if all(isinstance(p, int) for p in priorities.values()):
        sorted_by_priority = sorted(priorities.items(), key=lambda x: x[1])
        print("仲裁优先级顺序 (数字越小优先级越高):")
        for i, (strategy, priority) in enumerate(sorted_by_priority, 1):
            print(f"   {i}. {strategy} (优先级: {priority})")
        print()

        expected_order = ["FER", "ME", "BPC"]  # FER=0, ME=1, BPC=2
        actual_order = [s for s, _ in sorted_by_priority]

        if actual_order == expected_order:
            print("✅ 优先级配置正确")
            return True
        else:
            print(f"❌ 优先级配置错误，期望: {expected_order}, 实际: {actual_order}")
            return False
    else:
        print("❌ 无法验证优先级配置")
        return False


def test_required_next_steps():
    """列出实现多策略同时启动的必要步骤"""

    print("=== 实现多策略同时启动的必要步骤 ===\n")

    steps = [
        {
            "task": "实现 FERLiveStrategy",
            "status": "❌ 未完成",
            "file": "src/time_series_model/live/fer_live_strategy.py",
            "reference": "参考 BPCLiveStrategy 实现模式",
        },
        {
            "task": "实现 MELiveStrategy",
            "status": "❌ 未完成",
            "file": "src/time_series_model/live/me_live_strategy.py",
            "reference": "参考 BPCLiveStrategy 实现模式",
        },
        {
            "task": "修改 run_live.py 集成所有策略",
            "status": "❌ 未完成",
            "actions": [
                "导入 FERLiveStrategy 和 MELiveStrategy",
                "创建对应的策略实例",
                '注册到 LivePCM: pcm.register("fer", fer_strategy)',
                '注册到 LivePCM: pcm.register("me", me_strategy)',
            ],
        },
        {
            "task": "配置同步检查",
            "status": "✅ 已完成",
            "actions": [
                "FER 配置已同步到 live/highcap/",
                "ME 配置已同步到 live/highcap/",
            ],
        },
        {
            "task": "特征一致性验证",
            "status": "⏳ 待执行",
            "reference": "使用 compare_same_data.py 验证 6 币种一致性",
        },
        {
            "task": "PCM 仲裁测试",
            "status": "⏳ 待执行",
            "reference": "验证 FER > ME > BPC 优先级逻辑",
        },
        {
            "task": "E2E 冒烟测试",
            "status": "⏳ 待执行",
            "reference": "tick → 特征 → 信号 → 开仓完整流程",
        },
    ]

    print("待办任务清单:")
    for i, step in enumerate(steps, 1):
        print(f"{i}. {step['task']}")
        print(f"   状态: {step['status']}")
        if "file" in step:
            print(f"   文件: {step['file']}")
        if "actions" in step:
            print("   操作:")
            for action in step["actions"]:
                print(f"     - {action}")
        if "reference" in step:
            print(f"   参考: {step['reference']}")
        print()


if __name__ == "__main__":
    print("🔍 多策略实盘启动验证测试\n")

    # 执行各项检查
    status_ok = test_current_strategy_status()
    print("-" * 50)

    priority_ok = test_pcm_priority_configuration()
    print("-" * 50)

    test_required_next_steps()

    print("=" * 50)
    print("📊 最终结论:")
    if status_ok and priority_ok:
        print("✅ 系统已准备好，可以同时启动 BPC + FER + ME")
    else:
        print("❌ 系统未准备好，需要完成以下工作:")
        if not status_ok:
            print("   - 实现缺失的实盘策略类")
            print("   - 在 run_live.py 中注册所有策略")
        if not priority_ok:
            print("   - 修正 PCM 仲裁优先级配置")
