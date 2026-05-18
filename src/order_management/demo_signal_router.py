"""
实盘使用示例：SignalRouter 与 PositionManager 集成

流程：
1. 收集多个 symbol + archetype 的候选信号
2. SignalRouter 排序选出前 N 个
3. PositionManager 执行开仓/加仓
"""

from time_series_model.portfolio.signal_router import (
    CandidateSignal,
    SignalRouter,
    load_archetype_edges_from_config,
)
from order_management.position_manager import PositionManager
from order_management.models import PositionSide


def demo_signal_routing_and_execution(
    position_manager: PositionManager,
):
    """
    演示信号路由 + 仓位管理
    
    场景：
    - BTC 同时触发 BPC 和 ME → 选 AOS 最高的
    - ETH 触发 Reversal
    - SOL 触发 BPC
    → 最终选前 2 个执行
    """
    
    # 1. 从配置文件加载 archetype edges
    try:
        archetype_edges = load_archetype_edges_from_config(
            'config/strategies/bad-candidates/bpc/archetypes/archetype_edges.yaml'
        )
        print(f"加载 archetype edges: {archetype_edges}")
    except FileNotFoundError:
        # 如果配置文件不存在，使用硬编码的默认值
        print("配置文件不存在，使用默认 edges")
        archetype_edges = {
            'BPC': 0.62,
            'ME': 0.85,
            'Reversal': 0.55,
        }
    
    # 2. 初始化信号路由器
    router = SignalRouter(archetype_edges=archetype_edges, max_slots=2)
    
    # 3. 收集候选信号（来自你的策略系统）
    candidates = [
        # BTC 同时触发 BPC 和 ME
        CandidateSignal(
            symbol='BTCUSDT',
            archetype='BPC',
            evidence_score=0.75,  # AOS = 0.62 * 0.75 = 0.465
            side='LONG',
            entry_price=50000.0,
            stop_loss_price=49000.0,
            take_profit_price=52000.0,
            notes='BPC: Breakout confirmed',
        ),
        CandidateSignal(
            symbol='BTCUSDT',
            archetype='ME',
            evidence_score=0.65,  # AOS = 0.85 * 0.65 = 0.5525（更高）
            side='LONG',
            entry_price=50000.0,
            stop_loss_price=49000.0,
            take_profit_price=53000.0,
            notes='ME: Strong momentum',
        ),
        # ETH 触发 Reversal
        CandidateSignal(
            symbol='ETHUSDT',
            archetype='Reversal',
            evidence_score=0.9,  # AOS = 0.55 * 0.9 = 0.495
            side='SHORT',
            entry_price=3000.0,
            stop_loss_price=3100.0,
            take_profit_price=2800.0,
            notes='Reversal: Exhaustion detected',
        ),
        # SOL 触发 BPC
        CandidateSignal(
            symbol='SOLUSDT',
            archetype='BPC',
            evidence_score=0.6,  # AOS = 0.62 * 0.6 = 0.372
            side='LONG',
            entry_price=100.0,
            stop_loss_price=95.0,
            take_profit_price=110.0,
            notes='BPC: Support breakout',
        ),
    ]
    
    # 4. 信号路由：处理冲突并排序
    ranked_signals = router.route_signals(candidates)
    
    print(f"\n=== 信号路由结果 ===")
    print(f"候选信号数: {len(candidates)}")
    print(f"筛选后信号数: {len(ranked_signals)}")
    
    for ranked in ranked_signals:
        print(f"\nRank {ranked.rank}: {ranked.signal.symbol} - {ranked.signal.archetype}")
        print(f"  AOS: {ranked.aos:.3f} (Edge: {ranked.edge:.2f} × Evidence: {ranked.signal.evidence_score:.2f})")
        print(f"  Side: {ranked.signal.side}")
        print(f"  Entry: {ranked.signal.entry_price}")
        print(f"  Notes: {ranked.signal.notes}")
    
    # 5. 执行开仓（按 rank 顺序）
    print(f"\n=== 执行开仓 ===")
    for ranked in ranked_signals:
        signal = ranked.signal
        
        # 转换 side
        position_side = PositionSide.LONG if signal.side == 'LONG' else PositionSide.SHORT
        
        # 调用 PositionManager 的智能开仓/加仓
        result = position_manager.open_or_add_position(
            symbol=signal.symbol,
            side=position_side,
            entry_price=signal.entry_price,
            size=50.0,  # 仓位大小（实际应根据风险管理计算）
            archetype=signal.archetype,
            stop_loss_price=signal.stop_loss_price,
            take_profit_price=signal.take_profit_price,
            strategy_id=f'{signal.archetype}_strategy',
            max_slots=2,
            max_add_count=3,
        )
        
        if result['success']:
            action = result['action']
            position = result['position']
            print(f"\n✅ {signal.symbol} {signal.archetype} {action.upper()} 成功")
            print(f"   Position ID: {position.position_id}")
            print(f"   Size: {position.current_size}")
        else:
            reason = result['reason']
            print(f"\n❌ {signal.symbol} {signal.archetype} 失败: {reason}")
    
    # 6. 查看仓位摘要
    summary = position_manager.get_position_summary()
    print(f"\n=== 仓位摘要 ===")
    print(f"已用 slot: {summary['used_slots']} / {summary['max_slots']}")
    print(f"\n按 archetype 统计:")
    for arch, stat in summary['by_archetype'].items():
        print(f"  {arch}: {stat['count']} 个仓位, 总大小 {stat['total_size']:.2f}")


if __name__ == '__main__':
    # 需要先初始化 BinanceAPI 和 PositionManager
    print("请在实盘环境中调用 demo_signal_routing_and_execution()")
    print("需要传入已初始化的 PositionManager 实例")
