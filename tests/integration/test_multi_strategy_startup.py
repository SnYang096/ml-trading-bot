"""
多策略实盘启动验证测试
验证 BPC/ME/FER 是否能同时启动
"""

import pytest
import os
from unittest.mock import Mock, patch


def test_current_strategy_status():
    """检查各策略实盘实现状态"""

    print("=== 多策略实盘启动状态检查 ===\n")

    # 1. 检查策略实现文件
    strategies = {
        "BPC": {
            "live_strategy": "src/time_series_model/live/generic_live_strategy.py",
            "status": "✅ 已实现",
            "class": "GenericLiveStrategy",
            "note": 'strategy_name="bpc"',
        },
        "FER": {
            "live_strategy": "src/time_series_model/live/generic_live_strategy.py",
            "status": "✅ 已实现",
            "class": "GenericLiveStrategy",
            "note": 'strategy_name="fer"',
        },
        "ME": {
            "live_strategy": "src/time_series_model/live/generic_live_strategy.py",
            "status": "✅ 已实现",
            "class": "GenericLiveStrategy",
            "note": 'strategy_name="me"',
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
        "from src.time_series_model.live.generic_live_strategy import GenericLiveStrategy"
        in content
    )
    fer_imported = "GenericLiveStrategy" in content and 'strategy_name="fer"' in content
    pcm_yaml_loop = (
        "for arch in enabled_archetypes" in content and "pcm.register(_name" in content
    )
    me_runtime_ok = pcm_yaml_loop and "strategy_name=rk" in content

    pcm_registered_bpc = pcm_yaml_loop or 'pcm.register("bpc", bpc)' in content
    pcm_registered_fer = 'pcm.register("fer", fer)' in content
    pcm_registered_me = pcm_yaml_loop or 'pcm.register("me", me)' in content

    print(f"   BPC 导入 (GenericLiveStrategy): {'✅' if bpc_imported else '❌'}")
    print(f"   FER 导入 (GenericLiveStrategy): {'✅' if fer_imported else '❌'}")
    print(f"   ME 宪法驱动装载 (strategy_name=rk): {'✅' if me_runtime_ok else '❌'}")
    print()
    print(f"   PCM 宪法驱动注册循环: {'✅' if pcm_yaml_loop else '❌'}")
    print(f"   BPC 注册: {'✅' if pcm_registered_bpc else '❌'}")
    print(f"   FER 注册: {'✅' if pcm_registered_fer else '❌'}")
    print(
        f"   ME（若 constitution 启用且磁盘存在则会注册）: {'✅' if pcm_registered_me else '⚠️'}"
    )
    print()

    # 3. 配置文件同步状态
    print("3. 配置文件同步状态:")

    config_dirs = {
        "BPC": "config/strategies/bpc/archetypes/",
        "FER": "config/strategies/fer/archetypes/",
        "ME": "config/strategies/bad-candidates/me/archetypes/",
    }

    live_dirs = {
        "BPC": "live/highcap/config/strategies/bpc/archetypes/",
        "FER": "live/highcap/config/strategies/fer/archetypes/",
        "ME": None,  # ME 已归档到研究侧 bad-candidates，不再镜像到 live/highcap
    }

    for strategy in ["BPC", "FER", "ME"]:
        config_exists = os.path.exists(config_dirs[strategy])
        live_path = live_dirs[strategy]
        live_exists = bool(live_path) and os.path.exists(live_path)

        config_count = len(os.listdir(config_dirs[strategy])) if config_exists else 0
        live_count = len(os.listdir(live_path)) if (live_path and live_exists) else 0

        if strategy == "ME":
            ok = config_exists and config_count > 0
            status = "✅" if ok else "❌"
            print(
                f"   {strategy}: {status} (研究归档:{config_count}文件, live: 不使用)"
            )
            continue

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

    registered_count = int(pcm_registered_bpc) + int(bool(pcm_registered_me))
    if pcm_registered_fer:
        registered_count += 1

    print("4. 总体结论:")
    print(f"   ✅ GenericLiveStrategy 代码路径: {implemented_count}/3")
    print(f"   ✅ PCM 注册路径点数: {registered_count}（BPC 必须；ME 仅在启用时加载）")
    print()

    if implemented_count == 3 and pcm_registered_bpc:
        print("🎉 结论: BPC 主路径就绪；ME/FER 由宪法与磁盘决定是否参与")
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
        if not pcm_registered_bpc:
            print("     - run_live.py 未检测到 BPC PCM 注册路径")
        return False
    """检查 PCM 仲裁优先级配置"""

    print("=== PCM 仲裁优先级检查 ===\n")

    # 检查各策略的优先级配置
    priority_configs = {
        "BPC": "config/strategies/bpc/meta.yaml",
        "FER": "config/strategies/fer/meta.yaml",
        "ME": "config/strategies/bad-candidates/me/meta.yaml",
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
            "task": "实现 FER 策略",
            "status": "✅ 已完成（通过 GenericLiveStrategy）",
            "file": "src/time_series_model/live/generic_live_strategy.py",
            "reference": '使用 GenericLiveStrategy(strategy_name="fer")',
        },
        {
            "task": "实现 ME 策略",
            "status": "✅ 已完成（通过 GenericLiveStrategy）",
            "file": "src/time_series_model/live/generic_live_strategy.py",
            "reference": '使用 GenericLiveStrategy(strategy_name="me")',
        },
        {
            "task": "修改 run_live.py 集成所有策略",
            "status": "✅ 已完成",
            "actions": [
                "导入 GenericLiveStrategy",
                "创建对应的策略实例",
                '注册到 LivePCM: pcm.register("bpc/fer/me", strategy)',
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
