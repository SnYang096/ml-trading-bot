# DECISION — TPC trend_pool_guard sweep (2026-06-12)

## 待填（Phase 3 grid 跑完后）

| Variant | total_r | max_dd_r | trades | unprot_reject | post_unlock_reject | 备注 |
|---------|---------|----------|--------|---------------|-------------------|------|
| G0_prod_1_2 | | | | | | 生产 baseline |
| G1_be1_3 | | | | | | |
| G2_be3_3 | | | | | | 3 裸仓但总 cap 3 |
| G3_be3_6 | | | | | | 用户提案近似 |
| G4_guard_off | | | | | | 无上限对照 |

## 结论（draft）

- [ ] 维持 prod **1/2**
- [ ] 改为 **1/3**（G1）
- [ ] 改为 **3/3**（G2）
- [ ] 改为 **3/6**（G3）
- [ ] 需要新代码：显式 `base_symbols + protected_count` 动态 cap

## 理由

（填写：DD vs R 权衡、reject 漏斗是否说明「错过入场」、与 correlation_guard 交互）
