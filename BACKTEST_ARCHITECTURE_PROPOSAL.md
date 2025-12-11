# 回测架构设计方案

## 两种方案对比

### 方案1：统一回测机制 + YAML 配置

#### 实现方式
```yaml
# backtest.yaml
backtest:
  enabled: true
  type: "vectorbt"  # 统一使用 vectorbt
  params:
    # 通用参数
    price_col: close
    fee: 0.0004
    
    # 策略特定逻辑通过配置表达
    entry_logic:
      type: "signal_direction"  # 或 "prediction_threshold"
      signal_col: signal
      threshold: 0.6
    
    exit_logic:
      type: "rr_exit"  # 或 "probability_exit", "trailing_stop"
      rr:
        stop_loss_r: 1.0
        take_profit_r: 2.0
        max_holding_bars: 50
    
    position_management:
      type: "fixed"  # 或 "proportional", "add_reduce"
      base_size: 0.1
      # 如果需要加仓/减仓
      add_position:
        enabled: true
        condition: "cvd_flow"
        atr_distance: 0.5
      reduce_position:
        enabled: true
        condition: "rr_threshold"
        threshold: 0.8
```

#### 优点
- ✅ **代码统一**：只有一个回测实现，维护简单
- ✅ **配置灵活**：通过配置控制不同策略的行为
- ✅ **易于测试**：统一的测试框架

#### 缺点
- ❌ **配置复杂**：复杂逻辑（如 CVD 相位判断、动态减仓）难以用配置表达
- ❌ **扩展性差**：新增策略特定逻辑需要修改统一代码
- ❌ **可读性差**：复杂的配置不如代码直观
- ❌ **性能问题**：统一代码需要处理所有情况，可能有性能开销

---

### 方案2：分离回测类 + YAML 指定（推荐）

#### 实现方式

**1. 定义统一接口**
```python
# src/time_series_model/strategies/backtesting/base_backtest.py
from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
import numpy as np

class BaseBacktest(ABC):
    """统一回测接口"""
    
    @abstractmethod
    def run(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行回测
        
        Returns:
            统一格式的回测结果字典，包含：
            - total_return_pct: 总收益率
            - sharpe: Sharpe 比率
            - max_drawdown_pct: 最大回撤
            - win_rate: 胜率
            - total_trades: 总交易数
            - trades: 交易记录列表（可选）
        """
        pass
```

**2. 实现策略特定回测类**
```python
# src/time_series_model/strategies/backtesting/sr_reversal_backtest.py
from .base_backtest import BaseBacktest

class SRReversalBacktest(BaseBacktest):
    """SR 反转策略回测"""
    
    def run(self, df, predictions, **kwargs):
        # 使用现有的 sr_reversal_backtest 逻辑
        # 但返回统一格式
        result = backtest_sr_reversal(df, predictions, ...)
        return self._normalize_result(result)

# src/time_series_model/strategies/backtesting/sr_breakout_backtest.py
class SRBreakoutBacktest(BaseBacktest):
    """SR 突破策略回测"""
    
    def run(self, df, predictions, **kwargs):
        # 使用现有的 sr_breakout_backtest 逻辑
        result = backtest_sr_breakout(df, predictions, ...)
        return self._normalize_result(result)

# src/time_series_model/strategies/backtesting/vectorbt_backtest.py
class VectorBTBacktest(BaseBacktest):
    """通用 vectorbt 回测（用于简单策略）"""
    
    def run(self, df, predictions, **kwargs):
        # 使用现有的 run_vectorbt_backtest 逻辑
        return run_vectorbt_backtest(df, predictions, ...)
```

**3. YAML 配置指定回测类**
```yaml
# config/strategies/sr_reversal/backtest.yaml
backtest:
  enabled: true
  class: "src.time_series_model.strategies.backtesting.sr_reversal_backtest.SRReversalBacktest"
  params:
    min_confidence: 0.3
    base_position_size: 0.1
    stop_loss_r: 1.0
    take_profit_r: 2.0
    max_holding_bars: 50
    enable_reduce_position: true

# config/strategies/sr_breakout/backtest.yaml
backtest:
  enabled: true
  class: "src.time_series_model.strategies.backtesting.sr_breakout_backtest.SRBreakoutBacktest"
  params:
    min_predicted_rr: 1.0
    base_position_size: 0.1
    enable_add_position: true
    enable_reduce_position: true
    trailing_stop_threshold: 2.0

# config/strategies/trend_following/backtest.yaml
backtest:
  enabled: true
  class: "src.time_series_model.strategies.backtesting.vectorbt_backtest.VectorBTBacktest"
  params:
    long_entry_threshold: 0.6
    long_exit_threshold: 0.4
    # ... 简单策略使用通用回测
```

**4. 在 train_strategy_pipeline.py 中动态加载**
```python
def run_backtest(
    df: pd.DataFrame,
    preds: np.ndarray,
    backtest_cfg,
    task_type: str,
) -> Optional[Dict[str, float]]:
    if not backtest_cfg.enabled:
        return None
    
    # 如果指定了回测类，使用策略特定回测
    if hasattr(backtest_cfg, 'class') and backtest_cfg.class:
        backtest_class = import_callable(
            backtest_cfg.class.rsplit('.', 1)[0],  # module
            backtest_cfg.class.rsplit('.', 1)[1]   # class name
        )
        backtest_instance = backtest_class()
        return backtest_instance.run(
            df=df,
            predictions=preds,
            task_type=task_type,
            **backtest_cfg.params
        )
    else:
        # 默认使用统一回测
        return run_vectorbt_backtest(df, preds, backtest_cfg, task_type)
```

#### 优点
- ✅ **灵活性高**：每个策略可以有完全独立的回测逻辑
- ✅ **代码清晰**：策略特定逻辑集中在各自的回测类中
- ✅ **易于扩展**：新增策略只需实现新的回测类
- ✅ **可读性好**：代码比配置更直观
- ✅ **性能优化**：每个回测类只处理自己的逻辑，无额外开销
- ✅ **接口统一**：通过基类保证返回格式一致

#### 缺点
- ⚠️ **代码分散**：回测逻辑分布在多个文件中
- ⚠️ **可能有重复**：不同策略间可能有相似的逻辑（但可以通过基类或工具函数复用）

---

## 推荐方案：方案2（分离回测类 + YAML 指定）

### 理由

1. **策略差异大**：
   - SR 反转：需要 CVD 相位判断、减仓逻辑
   - SR 突破：需要加仓、trailing stop、动态减仓
   - 趋势跟随：可能只需要简单的概率退出
   
   这些差异很难用统一的配置表达。

2. **复杂逻辑难以配置化**：
   - CVD 相位判断需要代码逻辑
   - 动态减仓（基于实际 R/R vs 预测 R/R）需要计算
   - Trailing stop 需要实时更新止盈价
   
   这些逻辑用代码实现更清晰、更易维护。

3. **已有代码基础**：
   - 策略特定回测脚本已经存在且功能完整
   - 只需重构为类，并统一接口

4. **未来扩展性**：
   - 新策略可以轻松添加新的回测类
   - 不需要修改统一代码

### 实施步骤

1. **创建基类接口**：
   - 定义 `BaseBacktest` 抽象类
   - 定义统一的返回格式

2. **重构现有回测脚本**：
   - 将 `sr_reversal_backtest.py` 重构为 `SRReversalBacktest` 类
   - 将 `sr_breakout_backtest.py` 重构为 `SRBreakoutBacktest` 类
   - 将 `run_vectorbt_backtest` 封装为 `VectorBTBacktest` 类

3. **更新配置加载器**：
   - 在 `StrategyConfigLoader` 中支持 `backtest.class` 字段
   - 支持动态加载回测类

4. **更新训练脚本**：
   - 在 `train_strategy_pipeline.py` 中支持动态调用回测类

5. **统一返回格式**：
   - 确保所有回测类返回相同格式的结果
   - 便于后续分析和对比

### 统一返回格式示例

```python
{
    "total_return_pct": float,      # 总收益率
    "sharpe": float,                # Sharpe 比率
    "max_drawdown_pct": float,       # 最大回撤
    "win_rate": float,              # 胜率
    "total_trades": int,            # 总交易数
    "winning_trades": int,          # 盈利交易数
    "losing_trades": int,           # 亏损交易数
    "avg_rr": float,                # 平均 R/R（可选）
    "profit_factor": float,         # 盈亏比（可选）
    "trades": List[Dict],           # 交易记录（可选，debug 模式）
}
```

---

## 总结

**推荐使用方案2（分离回测类 + YAML 指定）**，因为：
- 更灵活，支持策略特定的复杂逻辑
- 代码更清晰，易于维护和扩展
- 已有代码基础，重构成本低
- 通过统一接口保证一致性

