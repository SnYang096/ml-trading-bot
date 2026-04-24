# BPC 实盘就绪 TODO

## ✅ 已完成

| 任务 | 完成日期 | 说明 |
|------|----------|------|
| 端到端冒烟测试 | 2026-02-12 | run_live.py 启动 → 收 tick → 计算特征 → 出信号 → 下单，完整跑通 |
| 特征一致性验证（6币种） | 2026-02-15 | compare_same_data.py 6币种 99.8-100% 信号一致 |
| Evidence quantiles 修复 | 2026-02-13 | 消除 look-ahead bug，研究/实盘统一 |
| CVD 命名重构 | 2026-02-15 | cvd_short/medium/long → cvd_roll20/roll60/roll288 + 旧数据兼容 |
| check_dependencies.sh 修正 | 2026-02-15 | Feature Store 废弃提示 + 实盘目录路径 + 16/16 全绿 |
| warmup 数据覆盖 | 2026-02-15 | 6币种 × 6个月（2025-08 ~ 2026-02）全覆盖 |
| 实盘启动文档 | 2026-02-15 | 实盘启动命令.md 含可信验证章节 |

## 🔜 上线后持续做（不阻塞推代码）

| 任务 | 优先级 | 说明 |
|------|--------|------|
| SignalRouter 接入 | P2 | BPC 单 archetype 暂不需要，多 archetype 时必须接 |
| Edge 统计脚本验证 | P2 | 暂不需要 Edge_archetype，AOS = Evidence Score 就够 |
| Hyperliquid 兼容 | P2 | 未来可能接，目前还没做抽象层 |
| 阶段三：长期运行验证 | P1 | 7天模拟回放、内存泄漏、断连恢复 |
| 阶段四：监控模板 | P1 | Prometheus + 健康检查 + 告警阈值 |