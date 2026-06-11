# 准备第三个变体策略树（G1 + vol_leverage < 0.03 bull-only）

如果你想在本 grid 里包含“最后一次单边单调机会”，需要先准备下面这个策略树。

## 快速创建步骤

```bash
# 1. 基于当前最好的 G1 树复制
cp -a config/experiments/20260601_1130_tpc_gate_final_lock/variants/tpc_gate_ablate_G1_no_bull_vol_strategies \
   config/experiments/20260601_1130_tpc_gate_final_lock/variants/tpc_gate_G10_vla_lt003_bull_strategies

# 2. 修改 gate.yaml 里的 vol_leverage 规则
#    把原来的中间带改成极低尾单边：
#
#    - id: gate_tpc_vol_leverage_asymmetry_mid_bull_only
#      ...
#      when:
#        all_of:
#        - vol_leverage_asymmetry:
#            value_lt: 0.03          # ← 改成 0.03（或你最终想试的阈值）
#        - ema_1200_position:
#            value_gt: 0.10
#      then:
#        action: deny
#      disabled: false               # 关键：打开
#      disabled_reason: "20260601 final candidate: vla < 0.03 bull only"
#
#    同时把 comment / reason 改成单边版本说明即可。

# 3. （可选但推荐）把其他不需要的规则保持 disabled 状态
```

修改完成后，本 grid 里的 `G1_plus_vla_lt003_bull` 就能正常使用了。

---

**注意**：这个变体**大概率不会比 G1 更好**（G6 <0.05 已经很差了），加它的主要目的是“把单调路线的最后一点可能性彻底试完”，然后可以心安理得地删除所有 vol 相关 gate 规则。

跑完 grid 后，你就可以非常干净地决定：

- 只保留 G1 形态（推荐）
- 还是把 vol_leverage 极低尾作为最终规则之一
- 并把 gate.yaml 里所有 disabled 的历史规则全部删除
