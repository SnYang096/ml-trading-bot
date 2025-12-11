# 回测方法对比分析

## 1. run_vectorbt_backtest（统一回测方法）

### 位置
`scripts/train_strategy_pipeline.py` 中的 `run_vectorbt_backtest` 函数

### 特点
- **通用性**：适用于所有策略的统一回测框架
- **基于 vectorbt**：使用 vectorbt 库进行回测
- **配置驱动**：通过 `backtest.yaml` 配置文件控制回测参数
- **支持多种任务类型**：binary classification、multiclass、regression

### 核心逻辑

#### 1.1 信号生成
- **Binary/Multiclass**：根据预测概率和阈值生成入场/出场信号
- **支持信号方向**：`use_signal_direction=True` 时，方向由 `signal` 列决定，预测值只控制是否参与交易
- **RR 退出逻辑**：`use_rr_exit=True` 时，使用 `simulate_rr_exits` 计算基于 ATR 的止损/止盈退出

#### 1.2 退出机制
- **概率退出**：根据预测概率阈值退出（默认）
- **RR 退出**（可选）：基于 ATR 的止损/止盈退出，与标签生成逻辑一致
  - 止损：入场价 ± 1×ATR（反向）
  - 止盈：入场价 ± 2×ATR（同向）
  - 最大持仓时间：可配置

#### 1.3 支持的配置参数
```yaml
enabled: true
params:
  price_col: "close"
  fee: 0.0004
  slippage: 0.0
  initial_cash: 10000.0
  freq: "4H"  # 必须配置，用于计算 Sharpe 等指标
  long_entry_threshold: 0.6
  long_exit_threshold: 0.4
  short_entry_threshold: 0.4
  short_exit_threshold: 0.6
  use_signal_direction: true  # SR 反转策略使用
  signal_col: "signal"
  use_rr_exit: true  # 使用 RR 退出逻辑
  rr:
    max_holding_bars: 24
    stop_loss_r: 1.0
    take_profit_r: 2.0
    atr_window: 14
    entry_offset: 1
```

#### 1.4 返回指标
- `total_return_pct`: 总收益率
- `sharpe`: Sharpe 比率
- `max_drawdown_pct`: 最大回撤
- `win_rate`: 胜率
- `debug`: 调试信息（可选，包含交易记录、信号等）

### 优点
- ✅ 统一框架，易于维护
- ✅ 配置驱动，灵活可调
- ✅ 使用成熟的 vectorbt 库，指标计算准确
- ✅ 支持 RR 退出逻辑，与标签生成一致

### 局限性
- ⚠️ 不支持策略特定的复杂逻辑（如加仓、减仓、trailing stop）
- ⚠️ 不支持 CVD 相位判断等高级退出条件
- ⚠️ 仓位大小固定或简单比例，不支持动态仓位管理

---

## 2. 策略特定回测脚本

### 2.1 sr_reversal_backtest.py

#### 特点
- **策略特定逻辑**：专门为 SR 反转策略设计
- **手动实现**：不依赖 vectorbt，手动实现回测逻辑
- **详细交易记录**：记录每笔交易的详细信息

#### 核心逻辑

**入场条件**：
- 预测概率 >= `min_confidence`（默认 0.3）
- 方向由 SR 类型决定（支撑区 → 做多，阻力区 → 做空）

**退出条件**：
1. **止盈**：价格触及 `entry_price ± 2×ATR`
2. **止损**：价格触及 `entry_price ± 1×ATR`
3. **超时**：持仓时间 >= `max_holding_bars`（默认 50）
4. **减仓**（可选）：CVD 相位转负且预测概率 < 0.7 时，减半仓

**仓位管理**：
- 仓位大小 = `base_position_size × predictions[i]`（与预测概率成正比）
- 支持减仓逻辑（部分止盈）

**返回指标**：
- `total_trades`: 总交易数
- `win_rate`: 胜率
- `avg_rr`: 平均 R/R
- `avg_win_rr`: 盈利交易平均 R/R
- `avg_loss_rr`: 亏损交易平均 R/R
- `profit_factor`: 盈亏比
- `sharpe_ratio`: Sharpe 比率
- `max_drawdown`: 最大回撤
- `trades`: 详细交易记录列表

#### 优点
- ✅ 支持减仓逻辑（CVD 相位判断）
- ✅ 详细的交易记录（每笔交易的 R/R、退出原因等）
- ✅ 更丰富的指标（avg_win_rr、avg_loss_rr、profit_factor）

#### 局限性
- ⚠️ 手动实现，代码复杂
- ⚠️ 指标计算可能不如 vectorbt 准确
- ⚠️ 不支持更复杂的仓位管理（如加仓）

---

### 2.2 sr_breakout_backtest.py

#### 特点
- **突破策略特定逻辑**：专门为 SR 突破策略设计
- **支持加仓和减仓**：更复杂的仓位管理
- **Trailing Stop**：动态止盈

#### 核心逻辑

**入场条件**：
- 预测 R/R >= `min_predicted_rr`（默认 1.0）
- 方向由突破方向决定（向上突破 → 做多，向下突破 → 做空）

**退出条件**：
1. **初始止盈**：`entry_price ± 2×ATR`
2. **Trailing Stop**（可选）：如果预测 R/R > `trailing_stop_threshold`（默认 2.0），启用动态止盈
3. **止损**：突破点反向 1×ATR 或区间边界
4. **超时**：持仓时间 >= `max_holding_bars`
5. **减仓**（可选）：当实际 R/R 达到预测值的 80% 时，减半仓锁定利润

**仓位管理**：
- 初始仓位 = `base_position_size × predicted_R_R`
- **加仓**（可选）：价格回踩不破且 CVD 持续流入，在 0.5×ATR 处加仓
- **减仓**（可选）：实际 R/R 达到预测值的 80% 时，减半仓

**返回指标**：
- 与 `sr_reversal_backtest.py` 类似，但增加了：
  - `mfe`: Maximum Favorable Excursion（最大有利偏移）
  - `mae`: Maximum Adverse Excursion（最大不利偏移）

#### 优点
- ✅ 支持加仓逻辑
- ✅ 支持 Trailing Stop
- ✅ 支持动态减仓（基于实际 R/R vs 预测 R/R）
- ✅ 更丰富的交易分析指标（MFE、MAE）

#### 局限性
- ⚠️ 代码更复杂
- ⚠️ 手动实现，维护成本高

---

## 3. 对比总结

| 特性 | run_vectorbt_backtest | sr_reversal_backtest | sr_breakout_backtest |
|------|----------------------|---------------------|---------------------|
| **通用性** | ✅ 通用框架 | ❌ 策略特定 | ❌ 策略特定 |
| **配置驱动** | ✅ 是 | ❌ 否 | ❌ 否 |
| **RR 退出逻辑** | ✅ 支持 | ✅ 支持 | ✅ 支持 |
| **减仓逻辑** | ❌ 不支持 | ✅ 支持（CVD 判断） | ✅ 支持（R/R 判断） |
| **加仓逻辑** | ❌ 不支持 | ❌ 不支持 | ✅ 支持 |
| **Trailing Stop** | ❌ 不支持 | ❌ 不支持 | ✅ 支持 |
| **详细交易记录** | ⚠️ 可选（debug 模式） | ✅ 完整记录 | ✅ 完整记录 |
| **指标丰富度** | ⚠️ 基础指标 | ✅ 丰富（R/R 分析） | ✅ 丰富（R/R + MFE/MAE） |
| **代码维护** | ✅ 统一维护 | ⚠️ 分散维护 | ⚠️ 分散维护 |
| **准确性** | ✅ vectorbt 计算 | ⚠️ 手动计算 | ⚠️ 手动计算 |

---

## 4. 建议

### 当前状态
- **统一回测**（`run_vectorbt_backtest`）已经支持基本的 RR 退出逻辑，与标签生成一致
- **策略特定回测**提供了更复杂的仓位管理和退出逻辑

### 是否需要保留策略特定回测？

**建议保留**，原因：
1. **复杂仓位管理**：加仓、减仓、trailing stop 等逻辑在统一回测中不支持
2. **策略特定退出条件**：CVD 相位判断、动态减仓等需要策略特定的逻辑
3. **详细分析**：策略特定回测提供更详细的交易记录和分析指标

### 改进方向
1. **增强统一回测**：在 `run_vectorbt_backtest` 中增加对加仓、减仓、trailing stop 的支持
2. **配置化策略特定逻辑**：将策略特定的回测逻辑也配置化，通过配置文件控制
3. **统一接口**：让策略特定回测也返回与统一回测相同的指标格式，便于对比

