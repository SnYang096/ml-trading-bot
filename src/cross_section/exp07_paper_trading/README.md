# exp07 — Offline Walk-Forward Paper Trading

离线 dry-run 模拟器。**不连接交易所**，只在文件上记账和记录决策。目的：
1. 把 exp04/exp05 的研究结果变成可持续跟踪的"虚拟账户"
2. 实盘前在真实时间轴上验证 regime 判别 + 换仓节奏
3. 积累 trade log 供日后与真实成交 benchmark

## 文件结构

- `paper_engine.py` — `init` / `rebalance` / `status` / **`mid_stop`** 子命令
- `update_pnl.py` — 读 account_state，用最新价结算未实现 PnL（不写 state）
- `price_source.py` — 价源：`parquet`（历史重放）或 **`binance_futures`**（公开 ticker，可轮询）

## 账户状态文件 (per-account)

`reports/cross_section/exp07_paper/<name>/`:
- `account_state.json` — 虚拟持仓 + equity（rebalance 时覆盖）
- `trade_log.jsonl` — 每次 rebalance 的完整决策记录（append-only）
- `equity_history.parquet` — 每次 rebalance 后权益时间序列
- `latest_decision.txt` — 最新一次决策人眼可读
- `current_status.md` — `update_pnl.py` 生成的 MTM 快照

## 用法

```bash
# 1. 初始化（选 preset 或启用 regime-switch）
python -m src.cross_section.exp07_paper_trading.paper_engine init \
    --name live_mom --preset mom_only --account-size 10000

# 或启用 exp05 的 regime-switch 权重
python -m src.cross_section.exp07_paper_trading.paper_engine init \
    --name live_regime --use-regime-switch --account-size 10000

# 2. 触发一次 rebalance（cron 每 14 天）
python -m src.cross_section.exp07_paper_trading.paper_engine rebalance \
    --name live_regime --as-of 2026-04-24

# 3. 查看当前 MTM 状态（不改 account_state）
python -m src.cross_section.exp07_paper_trading.update_pnl \
    --name live_regime --as-of 2026-04-24

# 3b. 用 Binance 实时价 + 轮询 3 次（间隔 2s）
python -m src.cross_section.exp07_paper_trading.update_pnl \
    --name live_regime --price-source binance_futures --poll-sec 2 --poll-max 3

# 4. 期中止损（cron 每小时）：先 dry-run，再加 --apply 平仓记账
python -m src.cross_section.exp07_paper_trading.paper_engine mid_stop --name live_regime \
    --price-source binance_futures --poll-sec 1 --poll-max 2
python -m src.cross_section.exp07_paper_trading.paper_engine mid_stop --name live_regime \
    --price-source binance_futures --apply
```

## 语义与约束

- `rebalance` 的流程（walk-forward）：
  1. 加载截至 `as_of` 的所有历史数据
  2. **结算上一期持仓**：用 `as_of` 时最新价，side_sign × (last/entry − 1) × notional，扣 2 边 fee
  3. **识别 regime**（若启用）：`compute_regime_labels()` 取最后一行 `collapsed`
  4. 构建 composite score，选 `MAX_LONGS` / `MAX_SHORTS`（默认 2/2）
  5. 按当前 equity 50/50 均摊 notional 建新仓
  6. 写 trade_log / account_state / equity_history / latest_decision

- 执行假设（简化）：
  - 即时成交在 `as_of` 收盘价
  - fee = FEE_BPS_PER_SIDE (5 bps) × 两边
  - 无滑点
  - 无 funding cost 结算（回测里也没加，保持一致）
  - 实盘需要额外：slippage model、订单簿影响、funding fee、实时价查询

- **期中止损**：`paper_engine mid_stop` 按当前价检查每腿是否低于 `-stop_loss_per_leg`（`init` 可改，默认与回测一致 0.15）。`--apply` 时移除该腿并记 realized PnL + `trade_log`。

## 与 exp05 的集成

`--use-regime-switch` 读 `reports/cross_section/exp05_regime_ic/regime_ic/regime_weights.yaml`，每次 rebalance 时识别当前 regime，用该 regime 的 factor weights 构建 score。

## 下一步 (v2)

1. 期中止损（per-bar check）
2. walk-forward OOS 权重：用过去 N 天 IC 算权重，只用于未来 M 天
3. CSV export for streamlit 可视化
4. 对接真实交易所 CCXT/Binance API（此时本 module 的 trade_log 成为 ground truth）
