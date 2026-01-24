#!/usr/bin/env python3
"""
验证层级化配置的脚本
"""

import yaml
import os
from yin_bot.intraday_sniper.config import IntradaySniperConfig

def validate_config():
    """验证配置文件是否能正确加载"""
    try:
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.yaml")
        
        print(f"尝试加载配置文件: {config_path}")
        
        # 从YAML文件加载配置
        config = IntradaySniperConfig.from_yaml(config_path)
        print("配置文件加载成功！")
        print(f"交易品种: {config.instrument_id}")
        print(f"Bar类型: {config.bar_type}")
        
        # 打印技术指标配置
        indicators = config.indicators
        print("\n=== 技术指标配置 ===")
        print(f"布林带周期: {indicators['bollinger_bands']['period']}")
        print(f"布林带标准差: {indicators['bollinger_bands']['stddev']}")
        
        # 打印压缩指标配置
        mdc_config = indicators['adaptive_multi_dim_compression']
        print("\n=== 压缩指标配置 ===")
        print("Indicator Config:")
        for key, value in mdc_config['indicator_config'].items():
            print(f"  {key}: {value}")
            
        print("\nWeight Config:")
        for key, value in mdc_config['weight_config'].items():
            print(f"  {key}: {value}")
            
        print("\nThreshold Config:")
        for key, value in mdc_config['threshold_config'].items():
            print(f"  {key}: {value}")
        
        # 打印其他配置
        print(f"\n=== 风险管理 ===")
        print(f"每笔风险: {config.risk_management['risk_per_trade']}")
        print(f"目标盈亏比: {config.risk_management['target_r_ratio']}")
        
        print(f"\n=== 会话时间 ===")
        print(f"开始时间: {config.session['start']}")
        print(f"结束时间: {config.session['end']}")
        
        return True
    except Exception as e:
        print(f"配置验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    validate_config()