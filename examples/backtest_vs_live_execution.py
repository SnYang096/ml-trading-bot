"""
回测与实盘执行层调用示例
展示Execution层在两种场景下的不同使用方式
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple

from src.time_series_model.strategies.bpc_strategy_v2 import BPCStrategyV2
from src.time_series_model.execution.execution_controller import ExecutionController
from src.time_series_model.execution.noise_penalty import ExecutionNoisePenalty, NoisePenaltyConfig
from src.time_series_model.execution.tier import create_default_bpc_tiers, ExecutionParams


class BacktestExecutionEngine:
    """回测执行引擎"""
    
    def __init__(self, config: Dict):
        self.strategy = BPCStrategyV2()
        self.config = config
        self.results = []
        
    def run_backtest(self, historical_data: pd.DataFrame) -> Dict:
        """
        运行回测
        
        Args:
            historical_data: 历史数据，包含所有特征
            
        Returns:
            回测结果
        """
        print("🚀 开始回测执行...")
        
        for i in range(len(historical_data)):
            # 获取当前时间点的数据
            current_data = historical_data.iloc[:i+1]  # 从开始到当前点的所有数据
            
            if len(current_data) < 50:  # 需要足够的历史数据
                continue
                
            # 评估交易机会
            approved, params = self.strategy.evaluate_trade_opportunity(current_data)
            
            if approved:
                # 记录交易决策和参数调整
                record = {
                    'timestamp': current_data.index[-1],
                    'evidence_score': params['evidence_score'],
                    'noise_penalty': params['noise_penalty'],
                    'sl_r': params['execution_params'].sl_r,
                    'tp_r': params['execution_params'].tp_r,
                    'size_multiplier': params['execution_params'].size_multiplier,
                    'decision': 'TRADE_APPROVED'
                }
                self.results.append(record)
                
                print(f"📊 Bar {i}: 交易获批 | 证据分数: {params['evidence_score']:.3f} | "
                      f"噪声惩罚: {params['noise_penalty']:.3f} | "
                      f"参数: SL={params['execution_params'].sl_r:.2f}R, "
                      f"TP={params['execution_params'].tp_r:.2f}R, "
                      f"Size={params['execution_params'].size_multiplier:.2f}x")
            else:
                record = {
                    'timestamp': current_data.index[-1],
                    'decision': 'TRADE_DENIED'
                }
                self.results.append(record)
                
                if i % 100 == 0:  # 每100个bar打印一次进度
                    print(f"⏳ Bar {i}: 交易被拒")
        
        # 计算回测统计
        trade_approved_count = sum(1 for r in self.results if r['decision'] == 'TRADE_APPROVED')
        
        stats = {
            'total_bars': len(historical_data),
            'trade_opportunities': len(self.results),
            'approved_trades': trade_approved_count,
            'approval_rate': trade_approved_count / len(self.results) if self.results else 0,
            'results': self.results
        }
        
        print(f"✅ 回测完成! 总共处理 {len(historical_data)} 个Bar")
        print(f"📈 交易获批: {trade_approved_count}/{len(self.results)} ({stats['approval_rate']*100:.1f}%)")
        
        return stats


class LiveExecutionEngine:
    """实盘执行引擎"""
    
    def __init__(self, config: Dict):
        self.strategy = BPCStrategyV2()
        self.config = config
        self.state_buffer = pd.DataFrame()  # 维护状态缓冲区
        self.active_positions = []  # 活跃头寸
        self.last_update_time = None
        
    def process_realtime_bar(self, current_bar: pd.Series) -> Optional[Dict]:
        """
        处理实时K线数据
        
        Args:
            current_bar: 当前K线数据（包含特征）
            
        Returns:
            交易决策或None
        """
        # 将当前K线添加到状态缓冲区
        current_df = pd.DataFrame([current_bar])
        current_df.index = [current_bar.name]  # 使用时间戳作为索引
        
        if self.state_buffer.empty:
            self.state_buffer = current_df
        else:
            self.state_buffer = pd.concat([self.state_buffer, current_df])
        
        # 保持足够的历史数据（例如最近200个bar）
        if len(self.state_buffer) > 200:
            self.state_buffer = self.state_buffer.iloc[-200:]
        
        # 如果历史数据不足，跳过
        if len(self.state_buffer) < 50:
            return None
        
        print(f"⚡ 处理实时数据: {current_bar.name}")
        
        # 评估交易机会
        approved, params = self.strategy.evaluate_trade_opportunity(self.state_buffer)
        
        if approved:
            decision = {
                'timestamp': current_bar.name,
                'action': 'PLACE_ORDER',
                'evidence_score': params['evidence_score'],
                'noise_penalty': params['noise_penalty'],
                'execution_params': {
                    'sl_r': params['execution_params'].sl_r,
                    'tp_r': params['execution_params'].tp_r,
                    'size_multiplier': params['execution_params'].size_multiplier
                },
                'raw_data': current_bar.to_dict()
            }
            
            print(f"🎯 实时决策: 交易获批 | 证据分数: {params['evidence_score']:.3f} | "
                  f"噪声惩罚: {params['noise_penalty']:.3f} | "
                  f"参数: SL={params['execution_params'].sl_r:.2f}R, "
                  f"TP={params['execution_params'].tp_r:.2f}R")
            
            return decision
        else:
            print(f"🚫 实时决策: 交易被拒")
            return None
    
    def get_current_status(self) -> Dict:
        """获取当前执行状态"""
        return {
            'buffer_size': len(self.state_buffer),
            'active_positions': len(self.active_positions),
            'last_update': self.last_update_time,
            'ready_for_trading': len(self.state_buffer) >= 50
        }


def simulate_historical_data(n_bars: int = 1000) -> pd.DataFrame:
    """模拟历史数据用于测试"""
    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='4H')
    
    data = {
        'timestamp': dates,
        'open': 100 + np.cumsum(np.random.randn(n_bars) * 0.1),
        'high': 100 + np.cumsum(np.random.randn(n_bars) * 0.1) + 0.15,
        'low': 100 + np.cumsum(np.random.randn(n_bars) * 0.1) - 0.15,
        'close': 100 + np.cumsum(np.random.randn(n_bars) * 0.1),
        'volume': np.random.randint(1000, 5000, n_bars)
    }
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    # 添加BPC相关特征
    df['bpc_score_breakout'] = np.random.rand(n_bars)
    df['bpc_score_pullback'] = np.random.rand(n_bars)
    df['bpc_score_continuation'] = np.random.rand(n_bars)
    df['bpc_pullback_depth_pct'] = np.random.rand(n_bars) * 0.1
    df['bpc_impulse_return_atr'] = np.random.randn(n_bars) * 0.1
    df['bpc_dir_consistency_short'] = np.random.rand(n_bars)
    df['bpc_dir_consistency_mid'] = np.random.rand(n_bars)
    df['bpc_dir_consistency_long'] = np.random.rand(n_bars)
    df['cvd_divergence_score'] = np.random.randn(n_bars) * 0.1
    df['price_momentum_divergence'] = np.random.randn(n_bars) * 0.1
    df['bpc_pullback_delta_absorption'] = np.random.rand(n_bars)
    df['cvd_change_5_pct'] = np.random.randn(n_bars) * 0.1
    df['trend_r2_20'] = np.random.rand(n_bars)
    df['path_efficiency_pct'] = np.random.rand(n_bars)
    df['price_dir_consistency_pct'] = np.random.rand(n_bars)
    df['macd_atr'] = np.random.randn(n_bars) * 0.1
    df['rsi_normalized'] = np.random.rand(n_bars)
    df['atr_percentile'] = np.random.rand(n_bars)
    df['bb_width_normalized_pct'] = np.random.rand(n_bars)
    df['vpin_score'] = np.random.rand(n_bars)
    df['volume_ratio_pct'] = np.random.rand(n_bars)
    df['ofci_pct'] = np.random.rand(n_bars)
    df['shd_pct'] = np.random.rand(n_bars)
    df['vol_regime_score'] = np.random.rand(n_bars)
    df['vol_trend_score'] = np.random.rand(n_bars)
    df['sr_strength_max'] = np.random.rand(n_bars)
    
    # 添加数学特征（用于Execution层）
    df['wpt_price_fluctuation'] = np.random.rand(n_bars)
    df['spectrum_price_entropy'] = np.random.rand(n_bars)
    df['hilbert_price_env'] = np.random.rand(n_bars)
    df['hurst_price_rolling'] = np.random.rand(n_bars)
    df['evt_tail_risk'] = np.random.rand(n_bars)  # EVT特征
    
    return df


def demonstrate_execution_differences():
    """演示回测和实盘执行层的差异"""
    
    print("=" * 80)
    print("🔄 回测 vs 实盘 Execution层使用对比演示")
    print("=" * 80)
    
    # 生成模拟历史数据
    print("\n📊 生成模拟历史数据...")
    historical_data = simulate_historical_data(200)  # 较少的数据用于演示
    print(f"✅ 生成了 {len(historical_data)} 个历史数据点")
    
    # 1. 回测执行演示
    print("\n" + "="*50)
    print("🎯 回测执行演示")
    print("="*50)
    
    backtest_config = {
        'mode': 'backtest',
        'sensitivity_analysis': True,
        'record_details': True
    }
    
    backtest_engine = BacktestExecutionEngine(backtest_config)
    backtest_results = backtest_engine.run_backtest(historical_data)
    
    # 2. 实盘执行演示（使用相同数据模拟实时流）
    print("\n" + "="*50)
    print("⚡ 实盘执行演示（模拟）")
    print("="*50)
    
    live_config = {
        'mode': 'live',
        'risk_limits': {'max_drawdown': 0.15},
        'state_persistence': True
    }
    
    live_engine = LiveExecutionEngine(live_config)
    
    # 模拟实时数据流
    trade_decisions = []
    for idx, (_, row) in enumerate(historical_data.iterrows()):
        if idx < 50:  # 跳过前面的数据积累期
            continue
            
        decision = live_engine.process_realtime_bar(row)
        if decision:
            trade_decisions.append(decision)
        
        if idx % 50 == 0:  # 每50个bar显示一次状态
            status = live_engine.get_current_status()
            print(f"   状态: 缓冲区大小={status['buffer_size']}, "
                  f"准备就绪={status['ready_for_trading']}")
    
    print(f"\n✅ 实盘模拟完成! 产生 {len(trade_decisions)} 个交易决策")
    
    # 3. 对比总结
    print("\n" + "="*80)
    print("📋 回测 vs 实盘 Execution层对比总结")
    print("="*80)
    
    comparison = {
        '维度': ['数据处理', '决策频率', '参数调整', '状态管理', '风险控制', '性能要求'],
        '回测': [
            '批量处理历史数据', 
            '对所有历史点进行决策', 
            '记录所有参数调整用于分析', 
            '从零状态开始', 
            '事后分析和参数优化', 
            '吞吐量优先，资源充足'
        ],
        '实盘': [
            '流式处理实时数据', 
            '实时响应当前市场', 
            '即时应用参数调整', 
            '维持历史状态', 
            '实时监控和硬性限制', 
            '延迟优先，稳定性要求高'
        ]
    }
    
    import tabulate
    print(tabulate.tabulate(comparison, headers='keys', tablefmt='grid'))
    
    print("\n💡 核心相同点:")
    print("  • 分层架构一致 (Gate/Evidence/Execution)")
    print("  • 数学特征使用原则一致 (仅在Execution层)")
    print("  • Noise Penalty计算逻辑一致")
    print("  • Tier选择和参数调整规则一致")
    
    print("\n🎯 差异关键点:")
    print("  • 回测: 事后分析，参数优化，完整历史")
    print("  • 实盘: 实时决策，风险控制，状态维持")


if __name__ == "__main__":
    demonstrate_execution_differences()