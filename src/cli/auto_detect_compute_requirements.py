"""
自动推导计算需求：从gate rules和regime配置中提取特征并映射到optional blocks

实施方案3：自动推导计算需求（最智能）
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, Any
import yaml
import sys

# 获取项目根目录
def get_project_root() -> Path:
    """获取项目根目录"""
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists() or (current / ".git").exists():
            return current
        current = current.parent
    return Path(__file__).parent.parent.parent

PROJECT_ROOT = get_project_root()


def extract_required_features_from_execution_archetypes(
    execution_archetypes_path: str | Path = "config/nnmultihead/execution_archetypes.yaml",
) -> Set[str]:
    """
    从execution_archetypes.yaml中提取所有gate rules和evidence rules需要的特征
    
    Returns:
        需要的特征名称集合
    """
    path = Path(execution_archetypes_path)
    if not path.is_absolute():
        # 假设相对于项目根目录
        path = PROJECT_ROOT / path
    
    if not path.exists():
        return set()
    
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    
    required_features: Set[str] = set()
    
    # 扫描所有archetypes
    for regime_name, regime_data in config.get('regimes', {}).items():
        for arch_name, arch_data in regime_data.get('archetypes', {}).items():
            # Gate rules
            gate_rules = arch_data.get('gate_rules', {})
            for rule in gate_rules.get('rules', []):
                if 'key' in rule:
                    required_features.add(str(rule['key']))
            
            # Evidence rules
            evidence_rules = arch_data.get('evidence_rules', [])
            for rule in evidence_rules:
                if 'key' in rule:
                    required_features.add(str(rule['key']))
                # any_key_contains是模式匹配，需要提取模式字符串用于匹配
                if rule.get('kind') == 'any_key_contains' and 'any_key_contains' in rule:
                    patterns = rule['any_key_contains']
                    if isinstance(patterns, list):
                        # 将模式添加到特征集合中（用于后续的模式匹配）
                        for pattern in patterns:
                            required_features.add(str(pattern))
                    elif isinstance(patterns, str):
                        required_features.add(str(patterns))
    
    return required_features


def map_features_to_optional_blocks(
    features: Set[str],
    optional_blocks_library: Dict[str, List[str]],
    feature_dependencies: Dict[str, any] | None = None,
) -> Set[str]:
    """
    将特征映射到optional blocks
    
    Args:
        features: 需要的特征名称集合
        optional_blocks_library: optional blocks定义（block_name -> feature_nodes）
        feature_dependencies: feature dependencies配置（用于查找特征属于哪个node）
    
    Returns:
        需要的optional blocks集合
    """
    required_blocks: Set[str] = set()
    
    # 特征到block的简单映射规则
    feature_to_block_patterns = {
        'vpin_block': ['vpin'],
        'volume_profile_block': ['vp_', 'volume_profile', 'vpvr_', 'vpvr'],
        'trade_cluster_block': ['trade_cluster'],
    }
    
    # 直接模式匹配
    for block_name, patterns in feature_to_block_patterns.items():
        if block_name not in optional_blocks_library:
            continue
        for feat in features:
            for pattern in patterns:
                if pattern.lower() in feat.lower():
                    required_blocks.add(block_name)
                    break
    
    # 通过feature_dependencies查找（如果提供）
    if feature_dependencies:
        feats = feature_dependencies.get('features', {}) or {}
        
        # 构建特征到node的映射
        feature_to_nodes: Dict[str, List[str]] = {}
        for node_name, node_cfg in feats.items():
            if not isinstance(node_cfg, dict):
                continue
            output_cols = node_cfg.get('output_columns', [])
            if isinstance(output_cols, list):
                for col in output_cols:
                    if col not in feature_to_nodes:
                        feature_to_nodes[col] = []
                    feature_to_nodes[col].append(node_name)
        
        # 查找特征属于哪个node，然后查找node属于哪个block
        for feat in features:
            nodes = feature_to_nodes.get(feat, [])
            for node in nodes:
                for block_name, block_nodes in optional_blocks_library.items():
                    if node in block_nodes:
                        required_blocks.add(block_name)
                        break
    
    return required_blocks


def auto_detect_compute_requirements(
    task_spec_path: str | Path,
    execution_archetypes_path: str | Path = "config/nnmultihead/execution_archetypes.yaml",
    feature_dependencies_path: str | Path = "config/feature_dependencies.yaml",
) -> Set[str]:
    """
    自动推导计算需求：从gate rules和regime配置中提取特征并映射到optional blocks
    
    Args:
        task_spec_path: TaskSpec文件路径
        execution_archetypes_path: execution_archetypes.yaml路径
        feature_dependencies_path: feature_dependencies.yaml路径
    
    Returns:
        需要的optional blocks集合
    """
    # 1. 提取需要的特征
    required_features = extract_required_features_from_execution_archetypes(
        execution_archetypes_path
    )
    
    if not required_features:
        return set()
    
    # 2. 读取TaskSpec获取optional_blocks_library
    ts_path = Path(task_spec_path)
    if not ts_path.is_absolute():
        ts_path = PROJECT_ROOT / ts_path
    
    if not ts_path.exists():
        return set()
    
    with open(ts_path, 'r', encoding='utf-8') as f:
        ts_obj = yaml.safe_load(f) or {}
    
    # 获取feature_plan
    fp: Dict[str, any] = {}
    plan_ref = str(ts_obj.get("feature_plan_ref") or "").strip()
    if plan_ref:
        p = Path(plan_ref)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists():
            objp = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            if isinstance(objp, dict):
                fp = objp.get("feature_plan") or {}
    
    # Apply overrides
    fp_over = ts_obj.get("feature_plan_overrides") or {}
    if isinstance(fp_over, dict) and fp_over:
        fp = {**fp, **fp_over}
    
    optional_blocks_library = fp.get("optional_blocks_library", {}) or {}
    
    # 3. 读取feature_dependencies（可选）
    feature_dependencies = None
    deps_path = Path(feature_dependencies_path)
    if not deps_path.is_absolute():
        deps_path = PROJECT_ROOT / deps_path
    if deps_path.exists():
        with open(deps_path, 'r', encoding='utf-8') as f:
            feature_dependencies = yaml.safe_load(f) or {}
    
    # 4. 映射特征到blocks
    required_blocks = map_features_to_optional_blocks(
        required_features,
        optional_blocks_library,
        feature_dependencies,
    )
    
    return required_blocks


if __name__ == "__main__":
    # 测试
    required_blocks = auto_detect_compute_requirements(
        "config/tasks/task_spec_highcap6_2024_202510.yaml"
    )
    print(f"自动推导的required blocks: {sorted(required_blocks)}")
