#!/usr/bin/env python3
"""
批量清理历史实验脚本

功能:
- 根据时间戳删除指定的实验
- 批量删除多个实验
- 删除指定策略的全部历史实验
- 删除错误状态的实验
- 删除指定日期范围内的实验
- dry-run 模式预览删除内容
"""

import argparse
import os
import shutil
from pathlib import Path
from datetime import datetime
import json


def parse_arguments():
    parser = argparse.ArgumentParser(description="批量清理历史实验")
    parser.add_argument("--strategy", required=True, help="策略名称 (bpc, me, fer)")
    parser.add_argument(
        "--timestamp", nargs="+", help="要删除的实验时间戳 (YYYYMMDD_HHMMSS)"
    )
    parser.add_argument("--all", action="store_true", help="删除指定策略的全部历史实验")
    parser.add_argument(
        "--status",
        choices=["error", "alert", "adopt", "keep"],
        help="删除指定状态的实验",
    )
    parser.add_argument(
        "--date-range",
        nargs=2,
        metavar=("START_DATE", "END_DATE"),
        help="删除指定日期范围内的实验 (YYYY-MM-DD 格式)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="dry-run 模式，只显示将要删除的内容，不实际删除",
    )

    return parser.parse_args()


def get_experiment_dirs(strategy):
    """获取指定策略的所有实验目录"""
    base_path = Path("results/research_history") / strategy
    if not base_path.exists():
        print(f"策略 {strategy} 的实验目录不存在: {base_path}")
        return []

    experiment_dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and item.name.startswith(("202", "19")) and "_" in item.name:
            # 验证是否为时间戳格式 YYYYMMDD_HHMMSS
            parts = item.name.split("_")
            if len(parts) == 2 and len(parts[0]) == 8 and len(parts[1]) == 6:
                try:
                    # 尝试解析时间戳
                    timestamp = datetime.strptime(
                        item.name.replace("_", ""), "%Y%m%d%H%M%S"
                    )
                    experiment_dirs.append(item)
                except ValueError:
                    continue

    return sorted(experiment_dirs, key=lambda x: x.name)


def get_experiment_status(exp_dir):
    """获取实验的状态"""
    report_file = exp_dir / "report.json"
    if report_file.exists():
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                report = json.load(f)
            # 从 comparison.decision 获取决策状态
            decision = report.get("comparison", {}).get("decision", "unknown").lower()
            return decision
        except Exception as e:
            print(f"解析报告文件失败 {report_file}: {e}")
            pass

    # 如果没有report.json，尝试从其他方式推断状态
    return "unknown"


def parse_timestamp(timestamp_str):
    """解析时间戳字符串为datetime对象"""
    try:
        return datetime.strptime(timestamp_str.replace("_", ""), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def should_delete_based_on_date(exp_dir, start_date_str, end_date_str):
    """根据日期范围判断是否应该删除实验"""
    exp_timestamp = parse_timestamp(exp_dir.name)
    if exp_timestamp is None:
        return False

    # 将日期字符串转换为datetime对象
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    # 将日期范围转换为当天的起始和结束时间
    start_datetime = datetime.combine(start_date.date(), datetime.min.time())
    end_datetime = datetime.combine(end_date.date(), datetime.max.time())

    return start_datetime <= exp_timestamp <= end_datetime


def main():
    args = parse_arguments()

    if not args.dry_run:
        confirm = input(f"确定要删除 {args.strategy} 策略的实验吗? (y/N): ")
        if confirm.lower() != "y":
            print("取消操作")
            return

    experiment_dirs = get_experiment_dirs(args.strategy)
    if not experiment_dirs:
        print(f"没有找到 {args.strategy} 策略的实验目录")
        return

    dirs_to_delete = []

    if args.all:
        # 删除全部历史实验
        dirs_to_delete = experiment_dirs
    elif args.timestamp:
        # 删除指定时间戳的实验
        for ts in args.timestamp:
            for exp_dir in experiment_dirs:
                if exp_dir.name == ts:
                    dirs_to_delete.append(exp_dir)
                    break
            else:
                print(f"警告: 找不到时间戳为 {ts} 的实验")
    elif args.status:
        # 删除指定状态的实验
        for exp_dir in experiment_dirs:
            status = get_experiment_status(exp_dir)
            if status == args.status.lower():
                dirs_to_delete.append(exp_dir)
    elif args.date_range:
        # 删除指定日期范围内的实验
        start_dt = datetime.strptime(args.date_range[0], "%Y-%m-%d")
        end_dt = datetime.strptime(args.date_range[1], "%Y-%m-%d")

        for exp_dir in experiment_dirs:
            if should_delete_based_on_date(
                exp_dir, args.date_range[0], args.date_range[1]
            ):
                dirs_to_delete.append(exp_dir)
    else:
        print("请指定要删除的实验类型: --timestamp, --all, --status, 或 --date-range")
        return

    if not dirs_to_delete:
        print("没有找到匹配条件的实验")
        return

    print(f"将要删除 {len(dirs_to_delete)} 个实验:")
    for exp_dir in dirs_to_delete:
        status = get_experiment_status(exp_dir)
        print(f"  - {exp_dir.name} (状态: {status})")

    if args.dry_run:
        print("\n这是 dry-run 模式，实际文件不会被删除")
        return

    # 确认删除
    confirm = input(f"\n确认删除这 {len(dirs_to_delete)} 个实验? (y/N): ")
    if confirm.lower() != "y":
        print("取消操作")
        return

    # 执行删除
    deleted_count = 0
    for exp_dir in dirs_to_delete:
        try:
            shutil.rmtree(exp_dir)
            print(f"已删除: {exp_dir}")
            deleted_count += 1
        except Exception as e:
            print(f"删除失败 {exp_dir}: {e}")

    print(f"\n完成: 成功删除 {deleted_count} 个实验目录")


if __name__ == "__main__":
    main()
