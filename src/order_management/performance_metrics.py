"""
性能指标计算
包括胜率、盈亏比、最大回撤等
"""
import logging
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any
from collections import defaultdict

from .models import PerformanceMetrics, Position, PositionStatus
from .storage import Storage

logger = logging.getLogger(__name__)


class PerformanceMetricsCalculator:
    """性能指标计算器"""
    
    def __init__(self, storage: Storage):
        """
        初始化性能指标计算器
        
        Args:
            storage: 存储层实例
        """
        self.storage = storage
    
    def calculate_daily_metrics(
        self,
        target_date: Optional[date] = None,
        symbol: Optional[str] = None
    ) -> PerformanceMetrics:
        """
        计算每日性能指标
        
        Args:
            target_date: 目标日期，None表示今天
            symbol: 交易对符号，None表示所有交易对
        
        Returns:
            性能指标对象
        """
        if target_date is None:
            target_date = date.today()
        
        # 获取该日期的所有已平仓仓位
        # 注意：这里简化实现，实际应该从数据库查询指定日期的仓位
        all_positions = self._get_closed_positions_for_date(target_date, symbol)
        
        # 计算指标
        total_trades = len(all_positions)
        winning_trades = sum(1 for p in all_positions if (p.realized_pnl or 0) > 0)
        losing_trades = sum(1 for p in all_positions if (p.realized_pnl or 0) < 0)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        total_pnl = sum(p.realized_pnl or 0 for p in all_positions)
        total_profit = sum(p.realized_pnl or 0 for p in all_positions if (p.realized_pnl or 0) > 0)
        total_loss = abs(sum(p.realized_pnl or 0 for p in all_positions if (p.realized_pnl or 0) < 0))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else (float('inf') if total_profit > 0 else 0.0)
        
        # 计算平均盈亏
        wins = [p.realized_pnl for p in all_positions if (p.realized_pnl or 0) > 0]
        losses = [p.realized_pnl for p in all_positions if (p.realized_pnl or 0) < 0]
        
        average_win = sum(wins) / len(wins) if wins else 0.0
        average_loss = sum(losses) / len(losses) if losses else 0.0
        
        largest_win = max(wins) if wins else 0.0
        largest_loss = min(losses) if losses else 0.0
        
        # 计算最大回撤（简化实现）
        max_drawdown, max_drawdown_period = self._calculate_max_drawdown(all_positions)
        
        # 计算Sharpe比率（简化实现，需要收益率序列）
        sharpe_ratio = self._calculate_sharpe_ratio(all_positions)
        
        # 创建指标对象
        metric_id = f"metrics_{target_date.isoformat()}_{symbol or 'all'}"
        metrics = PerformanceMetrics(
            metric_id=metric_id,
            date=target_date,
            symbol=symbol,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_profit=total_profit,
            total_loss=total_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            max_drawdown_period=max_drawdown_period,
            sharpe_ratio=sharpe_ratio,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss
        )
        
        return metrics
    
    def _get_closed_positions_for_date(
        self,
        target_date: date,
        symbol: Optional[str] = None
    ) -> List[Position]:
        """
        获取指定日期的已平仓仓位
        
        Args:
            target_date: 目标日期
            symbol: 交易对符号
        
        Returns:
            仓位列表
        """
        # 简化实现：获取所有已平仓仓位，然后过滤日期
        # 实际应该优化数据库查询
        all_positions = []
        
        # 这里需要从数据库查询，简化实现
        # 实际应该添加一个查询方法来获取指定日期范围的仓位
        return all_positions
    
    def _calculate_max_drawdown(
        self,
        positions: List[Position]
    ) -> tuple[Optional[float], Optional[str]]:
        """
        计算最大回撤
        
        Args:
            positions: 仓位列表
        
        Returns:
            (最大回撤, 回撤期间)
        """
        if not positions:
            return None, None
        
        # 按时间排序
        sorted_positions = sorted(
            positions,
            key=lambda p: p.exit_time or p.entry_time
        )
        
        # 计算累计盈亏
        cumulative_pnl = 0.0
        peak = 0.0
        max_dd = 0.0
        dd_start = None
        dd_end = None
        
        for position in sorted_positions:
            cumulative_pnl += position.realized_pnl or 0
            
            if cumulative_pnl > peak:
                peak = cumulative_pnl
            
            drawdown = peak - cumulative_pnl
            if drawdown > max_dd:
                max_dd = drawdown
                # 简化：不记录具体期间
        
        return max_dd if max_dd > 0 else None, None
    
    def _calculate_sharpe_ratio(
        self,
        positions: List[Position]
    ) -> Optional[float]:
        """
        计算Sharpe比率（简化实现）
        
        Args:
            positions: 仓位列表
        
        Returns:
            Sharpe比率
        """
        if not positions or len(positions) < 2:
            return None
        
        # 计算收益率序列（简化：使用盈亏作为收益率）
        returns = [p.realized_pnl or 0 for p in positions]
        
        if not returns:
            return None
        
        mean_return = sum(returns) / len(returns)
        
        # 计算标准差
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return None
        
        # 假设无风险利率为0
        sharpe = mean_return / std_dev
        
        return sharpe
    
    def save_daily_metrics(
        self,
        target_date: Optional[date] = None,
        symbol: Optional[str] = None
    ) -> bool:
        """
        计算并保存每日性能指标
        
        Args:
            target_date: 目标日期
            symbol: 交易对符号
        
        Returns:
            是否成功
        """
        try:
            metrics = self.calculate_daily_metrics(target_date, symbol)
            return self.storage.create_performance_metrics(metrics)
        except Exception as e:
            logger.error(f"保存性能指标失败: {e}")
            return False
    
    def get_performance_summary(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取性能摘要
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            symbol: 交易对符号
        
        Returns:
            性能摘要
        """
        # 从数据库获取性能指标
        metrics_list = self.storage.get_performance_metrics(symbol=symbol)
        
        if start_date or end_date:
            filtered_metrics = []
            for m in metrics_list:
                if isinstance(m.date, date):
                    m_date = m.date
                else:
                    m_date = datetime.fromisoformat(str(m.date)).date() if isinstance(m.date, str) else m.date
                
                if start_date and m_date < start_date:
                    continue
                if end_date and m_date > end_date:
                    continue
                filtered_metrics.append(m)
            metrics_list = filtered_metrics
        
        if not metrics_list:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'profit_factor': 0.0
            }
        
        # 汇总指标
        total_trades = sum(m.total_trades for m in metrics_list)
        total_winning = sum(m.winning_trades for m in metrics_list)
        total_pnl = sum(m.total_pnl for m in metrics_list)
        total_profit = sum(m.total_profit for m in metrics_list)
        total_loss = sum(m.total_loss for m in metrics_list)
        
        win_rate = total_winning / total_trades if total_trades > 0 else 0.0
        profit_factor = total_profit / total_loss if total_loss > 0 else (float('inf') if total_profit > 0 else 0.0)
        
        return {
            'total_trades': total_trades,
            'winning_trades': total_winning,
            'losing_trades': total_trades - total_winning,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'profit_factor': profit_factor,
            'period': {
                'start': start_date.isoformat() if start_date else None,
                'end': end_date.isoformat() if end_date else None
            }
        }
