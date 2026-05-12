# ADR（草案）：策略配置 `research/` + `archetypes/` 分层与 adopt / deploy / 运维统一

**状态**：草案，待 review  
**范围**：以 `bpc` 为叙述主例，原则推广到 `tpc` / `me` 等 **事件链主线策略**（与 **SRB**、**FER** 等 **独立 slug** 并列，勿把 SRB 当作 BPC+ME+TPC 的总称）及 `chop_grid` / `dual_add_trend` 多腿；**不**在本文档中落地代码。  
**相关现状**：多腿已部分采用「根 canonical yaml + `archetypes/` + `live/highcap/...` 镜像」；BPC 仍以「整树实验目录 + adopt 写回多路径」为主，与本 ADR 目标对齐后需一次中等规模重构。

---

## 1. 背景与要解决的问题

- **职责混杂**：训练目标、sweep 矩阵、一次性实验配置与「已上线旋钮」混在 `config/strategies/<slug>/` 根下，adopt / deploy 时难以一眼区分「git 长期意图」与「rolling 产出」。
- **adopt 面过大**：若 adopt 等同「整目录覆盖」，易误伤 `meta.yaml`、`features.yaml` 或模型路径声明。
- **运维心智不统一**：`results/` 时间戳实验、`mlbot pipeline list/adopt` 的 `history_dir`、`deploy_config_to_live.py` 的 live 镜像三条线若命名与边界不统一，操作文档与脚本易分叉。

---

## 2. 决策目标（采纳后应达成的样子）

| 维度 | 目标 |
|------|------|
| **目录** | `research/`：可调、可丢、可不进 live；`archetypes/`：仅采纳后的薄层/旋钮；**根**：稳定入口（`meta.yaml`、`features.yaml`、及引擎级默认 yaml 若保留）。 |
| **adopt** | **默认只**将实验侧 `archetypes/*` 写回生产 `config/strategies/<slug>/archetypes/`（可选：显式开关再写根引擎片段）。 |
| **deploy** | **默认只**将 `config/strategies/<slug>/archetypes/`（+ `TOP_LEVEL_CONFIGS` 允许的根文件，如 `meta.yaml`）同步到 `live/highcap/config/strategies/<slug>/`。多腿 pipeline 指向 `research/calibrate_roll.default.yaml`。 |
| **实验** | 所有 rolling / 对比产物在 `output.history_dir`（通常 `results/...`）；`list` / `diff` / `adopt` 只依赖该树 + **与 run 相同的 `--config`**。 |

**非目标（首版可不做的）**：把 **全局多策略** 编排（如 `config/prod_train_pipeline_2h.yaml`）搬进某一策略子树；重写 PCM/宪法全局 deploy 策略。

---

## 3. 建议目录树（示意）

以 `bpc` 为例（`tpc`/`me` 同构；多腿 **`research/`** 内 **`calibrate_roll.default.yaml`** 为 rolling 运行时入口，`grid_backtest`/`dual_add_backtest` 在 YAML 编排层）：

```text
config/strategies/bpc/
  meta.yaml                 # 稳定：周期、side、策略身份
  features.yaml             # 稳定：特征管线入口
  research/                 # 可调：训练目标、sweep、研究专用 yaml、README（不进 live）
    calibrate_roll.default.yaml      # rolling 快：`rolling.mode` 常为 `turbo_fixed_features`
    research_roll.features_on.yaml   # 慢：`slow_realistic` + 更重结构节拍
    validate_static.full_study.yaml    # static holdout：`rolling.mode: non_rolling`
    # 可选：`pipeline.yaml`、`threshold_search.yaml`、`experiment_*.yaml`（并列 / extends，§3.3）
    ...
  archetypes/               # 采纳层：prefilter/gate/execution/direction 等「旋钮」
    prefilter.yaml
    gate.yaml
    ...
```

**`live/highcap/config/strategies/bpc/`**：与 research **无关**；镜像 **根稳定文件 + `archetypes/`**（与当前 deploy 哲学一致，仅缩小「默认复制集合」时需改脚本白名单）。

### 3.1 策略专用 `prod_train_pipeline_*.yaml` 放进 `research/` 是否合理？

**结论：合理（推荐作为可选共位方案），尤其适合「单策略 only」类 yaml。**

| 优点 | 说明 |
|------|------|
| **所有权清晰** | 「与 bpc 相关、只给 bpc 跑 rolling 的编排」与研究声明、搜索空间 **一并落在 `research/`**（推荐并入 packaged 管线 YAML 的同层 `pipeline`/`study`/`threshold_search` 分块，见 §3.3），新人路径短。 |
| **与 live 边界一致** | `deploy_config_to_live` **不**部署 `research/`；该文件留在研究侧，不会污染 `live/highcap/...`。 |
| **现有字段多为根相对路径** | 例如 `strategies.bpc.config: config/strategies/bpc`、`turbo_fixed_features.fixed_strategies_root: config/strategies` —— 路径相对 **仓库根 / 进程 cwd**，与「本 yaml 磁盘在 `bpc/research/` 下」**无冲突**，一般 **不必**因搬家而改 yaml 内部字符串。 |

| 注意 | 说明 |
|------|------|
| **引用全仓要改** | README、`A快速启动命令.md`、CI、个人脚本里所有 `--config config/prod_train_pipeline_...bpc_only.yaml` 需改为新路径；建议 **一期迁移 + 一期删旧路径**（或保留根目录 **薄 shim** 仅 `!!include` / 注释指向新路径，兼容一个 release）。 |
| **全局多策略 yaml 仍放根** | `config/prod_train_pipeline_2h.yaml` 等 **多策略 + PCM** 编排继续放在 `config/`，避免「打开 bpc 目录却驱动全宇宙」的错觉。 |
| **命名** | 可保留长文件名便于 `rg`；**§3.2 已定**：`calibrate_roll.default.yaml`、`research_roll.features_on.yaml`、`validate_static.*.yaml`；不再有 `turbo`/`slow` 短文件名约定。 |

**命令示例（迁移后）**：

```bash
BPC_PIPE=config/strategies/bpc/research/calibrate_roll.default.yaml

mlbot pipeline run --strategy bpc --config "$BPC_PIPE" --stage rolling_sim --skip-shap
mlbot pipeline list --strategy bpc --config "$BPC_PIPE"
mlbot pipeline adopt <timestamp> --strategy bpc --config "$BPC_PIPE"
```

**实现时自检**：对 yaml 做一次 `grep` 是否存在依赖 **相对本文件目录** 的路径（如 `./foo`）；若有则改为从仓库根起的绝对风格（与现有一致）。

### 3.2 显式 `--config` 与 `research/` 下「约定文件名」（双模式）

编排文件既允许 **写全路径**，也推荐在策略侧固定 **少量约定名**，减少文档与肌肉记忆负担；**实现上**由 CLI / `auto_research_pipeline` 在「未传 `--config`」或传 **策略相对短名** 时做解析（本文档只定契约，不写具体代码行号）。

**约定（建议默认，可按策略复制）**

| 路径（相对仓库根） | 含义 |
|--------------------|------|
| `config/strategies/<slug>/research/calibrate_roll.default.yaml` | **快速节奏入口**：`pipeline` 段与当前 `rolling.mode: turbo_fixed_features` 类 prod_train 对齐；可同文件含 `study` / `threshold_search`（§3.3 变体 B）。 |
| `config/strategies/<slug>/research/research_roll.features_on.yaml` | **慢节奏入口**：`pipeline` 段与 `slow_realistic` 类 prod_train 对齐；可选同文件 `study` / `threshold_search`。 |
| `config/strategies/<slug>/research/validate_static.full_study.yaml` | **一次性划分入口（simple）**：仅用顶层 `dates` + `validation_months` 的静态 Train \| Val \| Test 合同；用于 idea / 回归 / 定稿小窗；**不**替代月度 rolling 主路径（`calibrate_roll`/`research_roll`）。详见 **§14**。 |

可选扩展：`research/pipeline.yaml` 参与默认探测，可用 `extends:` 指向 packaged 入口（见 `strategy_layout.resolve_default_pipeline_config`）；`experiment_<name>.yaml` 存放一次性分支。

**解析优先级（建议）**

1. 若用户传入 **`--config <path>`** 且文件存在 → **始终使用该文件**（显式优先）。
2. 若未传 `--config` 但传了 **`--strategy <slug>`** → 依次探测：  
   `research/calibrate_roll.default.yaml` → `research/research_roll.features_on.yaml` →（可选）`research/pipeline.yaml`；**均不存在**则回退到仓库既有默认（如 `config/research_pipeline.yaml`），并 **打印一行警告**，避免 silent 用错 `history_dir`。
3. ~~短写 `--config turbo`~~（已弃用）；一律使用 **`--profile`/`--config`** 指向 `research/` 下的 packaged stem 或完整路径。
4. （待实现，见 **§14**）若存在 **`research/validate_static.full_study.yaml`** 且用户显式 `--config` 指向该文件 → 走 **静态划分** 管线；**不**参与 `calibrate_roll`/`research_roll` 的默认探测顺序，避免 silent 切换 `history_dir`。

**与长文件名并存的方式（二选一）**

- **已定**：`research/calibrate_roll.default.yaml`、`research/research_roll.features_on.yaml`、`validate_static.*.yaml` 为 **真源**；历史 `turbo.yaml`/`slow.yaml`、`config/prod_train_*_only.yaml` 仅存档案时不得再被 loader / multileg 兜底解析。
- **归档**：如需对照旧树，仅从 git / 备份读取。

**命令示例（约定模式，实现后）**：

```bash
# 等价意图：使用 config/strategies/bpc/research/calibrate_roll.default.yaml（若已实现 §3.2 解析）
mlbot pipeline run --strategy bpc --stage rolling_sim --skip-shap

# 显式仍可用（审计 / CI / 多版本并排）
mlbot pipeline run --strategy bpc \
  --config config/strategies/bpc/research/calibrate_roll.default.yaml \
  --stage rolling_sim --skip-shap
```

**运维注意**：CI 与生产手册建议 **至少一条 job 写显式 `--config`**，避免「默认文件被误删仍 green」；本地日常可用约定名提速。

### 3.3 `research/calibrate_roll.default.yaml` / `research_roll.features_on.yaml` 与 BPC 管线语言对齐

**结论**：`research/calibrate_roll.default.yaml` / `research/research_roll.features_on.yaml` 就是管线入口，顶层使用 BPC 已有语言，不再混入 `study` / `threshold_search` / `calibration_profiles` 这类额外 DSL。

多腿策略差异由 `strategy_type` 分发到各自代码实现；YAML 只表达通用编排和 KPI 契约：

```yaml
# research/calibrate_roll.default.yaml（示意）
dates: ...
rolling: ...
threshold_calibration: ...
strategies:
  chop_grid:
    strategy_type: grid
    kpi_gates:
      prefilter: ...
      backtest: ...
grid_backtest: ...
output: ...
```

**禁止**：在多腿研究入口重新发明独立搜索语言。候选阈值、网格 spacing、TP、腿数上限等由多腿 calibration dispatcher 根据 `strategy_type` 生成并写入对应 archetype。

**迁移 grep 点**：凡引用 `config/strategies/<slug>/research.yaml` 或 `threshold_search.yaml` 的脚本/文档，随目录迁入 `research/` 后更新路径；loader 须同时支持 **变体 B 的切块解析** 与（若保留）**变体 A 的多文件合并**。

---

## 4. 运行时语义：谁覆盖谁（必须写进实现与测试）

1. **基线**：根目录 + `features.yaml` / `meta.yaml` 定义的长期默认（可版本化、可 code review）。
2. **叠加**：`archetypes/*.yaml` 对基线做 **结构化 deep merge**（或按文件粒度的明确 overlay 规则），禁止「半份 yaml 导致缺键静默回退到错误默认」。
3. **research/**：仅被 **研究管线 / 本地实验** 读取；**生产路径（live、run_live、feature-store 若读策略目录）不得依赖 `research/`**。

> **未决问题（review 时请拍板）**：BPC 的 `labels_gate`、模型路径等是继续放根，还是部分迁入 `research/` 且仅训练读——影响面最大。

---

## 5. 命令与流程（统一心智）

### 5.1 研究跑 rolling / fast_month

```bash
# 与「产出根目录」绑定的 yaml 必须与 list/adopt 使用同一份编排（显式 `--config`，或省略时由 §3.2 解析到 `calibrate_roll.default.yaml`）
mlbot pipeline run --strategy bpc \
  --config config/strategies/bpc/research/calibrate_roll.default.yaml \
  --stage rolling_sim --skip-shap

# 实现 §3.2 后，可与上式等价（当存在 research/calibrate_roll.default.yaml 且未传 --config 时由 CLI 解析）
# mlbot pipeline run --strategy bpc --stage rolling_sim --skip-shap
```

- **产物**：`{output.history_dir}/bpc/<timestamp>/...`（含 `report.json`、`strategies/bpc/` 或既有树；与现实现保持兼容或提供迁移 shim）。
- **约定**：**`list`/`adopt`/`diff` 使用的编排必须与本次 run 一致**（同一 `--config` 路径，或同一套 §3.2 约定解析结果），否则 `history_dir` 错位（已踩坑）。

### 5.2 列出与对比

```bash
mlbot pipeline list --strategy bpc --config <同上 yaml>
mlbot pipeline diff --strategy bpc <ts_a> <ts_b> --config <同上 yaml>
```

### 5.3 采纳（本 ADR 后）

```bash
mlbot pipeline adopt <timestamp> --strategy bpc --config <同上 yaml>
```

- **语义**：默认 **仅** `cp` 实验目录下 `archetypes/*` → `config/strategies/bpc/archetypes/`（可选：`--also-root` 类 flag 写根引擎，默认关闭）。
- **校验**：BPC 保留 **locked prefilter/gate** 语义校验（仅针对写入 `archetypes/` 的规则）；多腿保持「无 prefilter 时 copy-only」分支，避免两套逻辑漂移——建议实现为「策略类型注册表 + 校验插件」。

### 5.4 同步到 live

```bash
python scripts/deploy_config_to_live.py --diff --strategy bpc
python scripts/deploy_config_to_live.py --deploy --yes --strategy bpc   # 不加 --git-commit 除非运维规范要求
```

- **语义**：默认 **仅** diff/deploy `archetypes/` + 白名单根文件（与现有多腿 `MULTI_LEG_*` 规则对齐后可抽象为「策略 deploy profile」）。
- **全局配置**：`constitution.yaml` 等 GLOBAL 与策略 deploy **分列**（避免「deploy bpc 顺带改宪法」的惊讶）；可用 `--strategy-only` 类 flag 或文档写清默认行为。

---

## 6. 运维与协作方式

| 主题 | 建议 |
|------|------|
| **Git** | 跟踪：`meta.yaml`、`features.yaml`、`research/*`、`archetypes/*`；**不**跟踪 `results/`。大体积实验用 `.gitignore` + 对象存储可选。 |
| **回滚** | `archetypes/` 回滚 = `git revert` 单次 adopt 提交；live 回滚 = 再 deploy 上一 tag 或从 git 检出 `live/` 子树。 |
| **审计** | adopt 前强制 `pipeline diff` 或 CI 生成「archetypes 文件级 diff」artifact；生产变更走 PR（含 live 镜像 diff）。 |
| **密钥** | 策略目录内 **禁止** API key；环境变量与 `live/` 外密钥管理不变。 |
| **监控** | 部署后健康检查：`run_live` / dry-run 加载策略、或最小化 `mlbot diagnose` 子集（与现有一致即可）。 |

---

## 7. 迁移建议（分阶段，降低风险）

1. **文档与契约**：冻结本 ADR + merge 规则；更新 README「活跃策略 & adopt/deploy」一节与 `A快速启动命令.md` 对齐。
2. **目录空壳**：在 `bpc`（及他策略）下增加 `research/`，把**明确属于研究**的文件移入并 grep 全仓改路径；根与 `archetypes/` 暂不变。
2b. **pipeline 共位**：快慢真源为 `research/calibrate_roll.default.yaml`、`research/research_roll.features_on.yaml`；`load_pipeline_config` 支持顶层 ``extends`` 链（供其他实验 yaml 复用）。
3. **adopt 收窄**：改 `_adopt_experiment_config`：BPC 默认只写 `archetypes/`；保留一周 `--full-tree` 逃生阀或 config flag。
4. **deploy 收窄**：`deploy_config_to_live.py` 与 BPC `ARCHETYPE_FILES` 对齐为「profile」表；多腿与经典共用抽象。
5. **删除逃生阀**：确认无调用方后移除全树 adopt。

---

## 8. 风险与开放问题（请 review 时勾选/批注）

**合并审阅**：全部 checkbox 的 **编号 + 优先级视图** 见 **§13 Master TODO**（可与本节同步勾选，避免双份漂移则只维护一处）。

- [ ] **Merge 实现复杂度**：deep merge 与 YAML anchor、多文件顺序；需要单元测试 +  golden fixture。
- [ ] **历史实验 adopt**：旧目录结构是否保留 `_cmd_adopt_experiment` 的 legacy 分支多久。
- [ ] **feature-store / train 入口**：是否所有读 `config/strategies/bpc` 的代码路径都经过统一 loader。
- [ ] **CI**：是否增加「adopt dry-run + deploy --diff」门禁，避免 silent 行为变化。
- [ ] **Pipeline 共位**：移动 `prod_train_pipeline_*_bpc_only.yaml` 后，CI / 文档 / 个人脚本中 `--config` 是否全部更新；是否需根目录 shim 过渡期。
- [ ] **约定名解析**：`mlbot pipeline` 对「仅 `--strategy`」、`research/calibrate_roll.default.yaml` / `research/research_roll.features_on.yaml` 探测顺序、与 `config/research_pipeline.yaml` 回退的日志与测试是否完备。
- [ ] **研究 yaml 迁入**：根目录 `research.yaml` / `threshold_search.yaml` 迁入 `research/` 后，采用 **§3.3 变体 B（单入口 YAML 内含 `pipeline`/`study`/`threshold_search` 分块）** 或 **变体 A（并列文件 + include）** 之一并写死；全仓引用与 loader 一致。
- [ ] **多腿 stage profile**：每策略 `archetypes/README.md`（或等价）是否列出 **自定义文件名 ↔ pipeline 序位**；loader / adopt / CI 是否按 profile 解析而非硬编码 `prefilter.yaml`。

---

## 9. 结论（供拍板）

在 **merge 语义与加载器单一入口** 写清楚的前提下，「`research/` + `archetypes/` + 根稳定文件 + 实验只落 `results/` + adopt/deploy 只碰采纳层」能显著降低误操作与文档分叉，**长期更好**；代价是一次 **跨脚本、跨加载路径** 的重构，建议按 §7 分阶段并与多腿现有约定 **收敛成一套 deploy profile**。

Review 通过后，可将本文档重命名为带日期的 ADR 编号（例如 `docs/architecture/adr/0001-*.md`）并开独立实现 PR。

---

## 10. 附录：多腿与经典「信号层」及 packaged research 对齐（范式草案）

**动机**：多腿当前以 **regime + 网格/双腿逻辑 + 独立 backtest** 为主，与经典链 **Prefilter → Gate → EntryFilter → Execution** 及 **快 (`turbo_fixed_features`) / 慢 (`slow_realistic`)** 分工 **看起来不统一**；若一开始未明确要求「同一范式」，后续运维、CI、文档和 **adopt/deploy 心智** 都会分叉。**对齐是合理的**。

**原则**：优先 **宏观流程一致**（特征筛选 → 阈值/旋钮 → 分层决策 → 执行参数）与 **落点一致**（`results/` → adopt → `archetypes/` → deploy、`research/calibrate_roll.default.yaml`、`research/research_roll.features_on.yaml`）。**各层 yaml 可用策略贴切命名**（如 Regime 对应经典 Prefilter **那一环的功能**）；须在 **`README.md` 或 `archetypes/README.md`** 写明 **文件名 ↔ pipeline 序位 / profile**，避免 adopt 后无人知晓映射。

### 10.1 两档对齐（建议采纳顺序）

| 档位 | 做什么 | 风险 |
|------|--------|------|
| **A. 流程 / 落点对齐** | 多腿走 **同一套宏观 stage 顺序**（名称可与经典一致：`prefilter` / `gate` / `entry_filter` / `execution`；对内可为 no-op）。rolling、`list/adopt`、两份 packaged rolling 入口与经典一致；**采纳结果落在 `archetypes/`**（及约定根 yaml），deploy 一致。**各 stage 读哪份文件** 由 **策略 profile** 声明（允许 `regime.yaml` 等自定义名）。 | 需 **profile 映射** 与 **`enabled: false`**，避免空跑。 |
| **B. 指标与 schema 深度对齐** | 在 A 上逐步具备与经典 **可比的** KPI；能复用则复用。 | 高；需新指标定义，勿硬抄 BPC 列名。 |

建议：**先做 A**；维护 **宏观流程 ↔ 文件名 / profile** 映射表；**B** 走 roadmap。

### 10.2 信号层：功能类比 + 落点（文件名可策略自定义）

**左列为经典「功能环」**；右列为多腿示例——可用 **`archetypes/regime.yaml`** 等命名，由 profile 标明其挂在 **prefilter 序位**（或团队统一序位名）即可。

| 经典功能环 | 多腿侧常见承载（示例） | 说明 |
|------------|------------------------|------|
| **特征筛选 / mask** | Regime、chop、box、trend_confidence 等 | 功能类似第一环；可 **`regime.yaml`**，不必强称 `prefilter.yaml`。 |
| **硬风控** | 腿数、净/总暴露、回撤、交易所约束 | **`gate.yaml`** 或 **`risk_gates.yaml`**。 |
| **入场 refine** | 段内再确认、排除盒等 | **`entry_filters.yaml`**（或等价块落在 packaged `research/calibrate_roll.default.yaml`）。 |
| **执行参数** | spacing、TP、网格档等 | **`execution.yaml`**；**execution_opt 阶段名** 与经典对齐。 |

### 10.2a 讨论结论（相对旧稿）

- **不要求** `archetypes/` 物理文件名与 BPC **完全一致**；**要求** **宏观流程 + 落点 + adopt/deploy** 一致，且 **每策略文档化** 文件名 ↔ stage 映射。  
- **Regime vs Prefilter**：允许叫 **Regime**；共识上即「第一环：特征/状态筛选 + 阈值」。

若某层在多腿 **确实不存在**，在管线配置中显式 **`enabled: false`** + 文档一句 **「N/A 原因」**，优于造空文件。

### 10.3 `calibrate_roll` / `research_roll` 与「挑特征 + 特征阈值 + 执行搜索」

| 节奏 | 经典含义（概括） | 多腿对齐方式 |
|------|------------------|--------------|
| **calibrate_roll（月度快）** | 特征集 **相对固定**，主打 **阈值链 + execution 网格**；文件名 `research/calibrate_roll.default.yaml` | 多腿已有 `turbo_fixed_features` 类 prod_train：**对齐同一 `research/calibrate_roll.default.yaml` 入口**；特征侧以 **`features.yaml` + feature_dependencies** 定死为主，**threshold_search** 映射到「regime/网格阈值 sweep」而非强行 Pool-B。 |
| **research_roll（节奏慢）** | 结构快照 + 元算法 + 更保守的阈值/EF；文件名 `research/research_roll.features_on.yaml` | 多腿 **`research/research_roll.features_on.yaml`** 存在时：拉长标定窗、打开/加强 **结构类** 校验（若将来与 chop 结构特征挂钩）；**无 slow 文件则跳过**，不要求与 BPC 同月数。 |

**结论**：多腿 **不必** 在第一版就具备与 BPC 完全同构的「特征搜索 + 全量阈值」；但 **应在同一套 packaged profile 文件名与 pipeline 宏观序位下**，把「已有能力」挂上去（**层可自建名**，见 §10.2），并把缺口标为 **roadmap**——与经典在 **流程与落点** 上对齐，在 **术语与文件命名** 上允许诚实差异。

### 10.4 与本文档其它章节的关系

- §3：研究声明与搜索空间 **默认并入** `research/calibrate_roll.default.yaml`、`research/research_roll.features_on.yaml` 的 **`study` / `threshold_search` 块**（变体 B）；若拆文件（变体 A）则仍可与 **月度 rolling sweep（快路径）** 绑定；**research_roll** 可无 `threshold_search` 块直至需要。  
- §7 迁移：「范式对齐」可单独开 **Phase：多腿 pipeline 壳 + archetype 落点与 stage profile（含自定义文件名映射）**，与目录搬迁可并行但 **不同 PR** 更易 review。

---

## 11. 多腿 packaged rolling 与「跨 history 列举」：事实核对与产品缺口（共识）

本节回应：**(A)** 多腿是否必须同时有「快管线 + 慢管线」叙事；**(B)** 各策略 packaged `research/calibrate_roll.default.yaml` 是否已覆盖「阈值/执行网格、回测报告、交易地图」；**(C)** `mlbot pipeline list` 只能看到「当前配置下的 history」的问题，以及 **列举全部管线历史 + 网站化 adopt/deploy** 的方向。

### 11.1 方向判断（与用户表述对齐）

- **快 / 慢在多腿都应存在**：与 §10.3 一致——**`calibrate_roll.default.yaml`**（`turbo_fixed_features`）侧重 **阈值链 + execution 类搜索 + 月度滚动**；**`research_roll.features_on.yaml`** 侧重更长窗、结构/元算法与更保守的标定。
- **「慢线特征挑选与快线合并」作为默认**：可接受——即 **默认不在 rolling 内开 Pool-B 式特征搜索**，而以各策略根目录 **`features.yaml` + `feature_dependencies`** 定稿（例如 dual_add 侧 `trend_confidence_f`、chop 侧由网格/诊断路径锁定的 chop 语义）。这与 `turbo_fixed_features.disable_feature_search: true` 的 prod 形态一致；**缺口**在于：若产品仍要求「快线显式可调阈值 YAML + execution 网格块」，应与 **`research_roll.features_on.yaml` 顶层 `threshold_calibration:` / `rolling_calibration:`** 对照（见下表）。
- **`pipeline list` 只扫一个 `output.history_dir`**：实现上 `_cmd_list_experiments(history_dir, strategy)` 只读 **当前加载 pipeline 配置里的** `history_dir / <strategy>`（见 `scripts/auto_research_pipeline.py`）。因此 **换一份 `--config`（例如默认 `research_pipeline.yaml` vs `config/strategies/chop_grid/research/calibrate_roll.default.yaml`）就会「看不到」** `results/chop_grid/calibrate_roll.default`、`results/dual_add_trend/calibrate_roll.default` 等目录下的时间戳——**不是 bug，而是单根设计**。要「看到所有研究成果」，需要 **新命令或索引层**（见 §11.4），静态站点是合理延伸（扫描多根下的 `report.json` / `report.html` / `trading_map*.html` / `stitched_summary.json` 等）。

### 11.2 事实核对：`calibrate_roll.default.yaml` vs `research_roll.features_on.yaml`

| 能力 | `config/strategies/chop_grid/research/calibrate_roll.default.yaml` | `config/strategies/dual_add_trend/research/calibrate_roll.default.yaml` | `config/strategies/*/research/research_roll.features_on.yaml`（对照） |
|------|-----------------------------------------------------|-----------------------------------------------------------|-----------------------------------------------------|
| Rolling 模式 | `rolling.mode: turbo_fixed_features`，`disable_feature_search: true` | 同上 | `slow_realistic`；仍可带 `turbo_fixed_features` 子块 |
| **YAML 顶层 `rolling_calibration:`**（`threshold_calibration` / `execution_opt` 等） | **无** | **无** | **有**（例如 `config/strategies/dual_add_trend/research/research_roll.features_on.yaml` 中 `threshold_calibration` + `execution_opt` 均为 `enabled: true`） |
| 全窗 backtest + 地图类配置 | `grid_backtest.enabled: true`，含 `map_*` / `continuous_map_*` | `dual_add_backtest.enabled: true`，含 `map_*` / `continuous_map_*` | slow chop 侧同样有 `grid_backtest` 等（以各文件为准） |
| 产物根目录 | `output.history_dir: results/chop_grid/calibrate_roll.default` | `results/dual_add_trend/calibrate_roll.default` | 一般为 `results/.../research_roll.features_on` 等（以各文件为准） |

**结论（精炼）**：

1. **两份 packaged 快 YAML 已支持**：独立策略管线、`turbo_fixed_features`、**全窗** `grid_backtest` / `dual_add_backtest` 及其 **交易地图相关字段**；rolling 运行中仍由各策略的 **月度多腿回测** 产出 `report` / stitched / continuous 类产物（具体文件名以运行日志与 `_rolling_sim` 目录为准）。
2. 对照 **`research_roll.features_on.yaml`** 里显式声明的 **`threshold_calibration` / `rolling_calibration`**：`calibrate_roll.default` 快变体常常在顶层 **不交**整块 `rolling_calibration.threshold_calibration` / `execution_opt`。若团队把「执行网格搜索」语义绑定到 **`research_roll` YAML** 的开关字段，需在 runbook 写明 **`calibrate_roll` = rolling 内建路径；网格类显式开关以 `research_roll` YAML 为准**，或在快线 packaged YAML 上补同源 `rolling_calibration` 块（并确认 runner 是否会消费）。

### 11.3 与 README / 快速命令的交叉引用

操作员已习惯 **同一 `--config` 下 list/adopt**；§11.1 的缺口应在 **`A快速启动命令.md` 或部署 runbook** 中用一句话强调：**列举 chop/dual 的快/慢历史时 `--config` 必须指向对应的 packaged `research/` YAML（通常为 `calibrate_roll.default.yaml` 或 `research_roll.features_on.yaml`）**，直至 §11.4 的跨根命令落地。

### 11.4 建议能力（Roadmap，非实现承诺）

**CLI（命名待定）**

- **`pipeline list-runs` 扩展**：`--all-history-roots` 或 `--from-prod-configs`：从登记文件或 `config/prod_train*.yaml` 解析所有 `output.history_dir`，对每个 `(history_dir, strategy)` 复用现有列表逻辑；或 **`pipeline list-configs`** 先列出已知 prod 管线名与默认 `history_dir`。
- **登记来源**：优先 **显式 manifest**（避免全盘扫描 `results/`）；次选 **glob `config/prod_train*.yaml` 抽取 `output.history_dir`**。

**网站（可选）**

- 静态页：多根索引 → 每次 run 的摘要指标、报告链接、地图 HTML、**adopt 候选**；与 **deploy** 文档链（例如 live 镜像目录）交叉引用。  
- **adopt / deploy**：保持「人读报告 → 选 run_id → `pipeline adopt` → `deploy_config_to_live`」的闸门；网站只做 **发现与导航**，不替代签名与审计。

---

## 12. 稳定流水线标准：TODO 总表（BPC / TPC / ME + 多腿 parity）

**合并审阅**：分主题 checkbox 见下；**一张表总览（T01–T33 + P0–P3）** 见 **§13**。

> **用语**：**SRB** 与 **BPC、ME、TPC** 一样，是 **独立策略 slug**（见仓库 `bad-candidates/srb` 等路径），**不是** BPC+ME+TPC 的统称。本节「与经典对齐」指 **以 Prefilter→Gate→Entry→Execution + `rolling_calibration` + 事件回测为主线的这类策略**（文档里以 **bpc / tpc / me** 为主例）；多腿 parity 是与 **这条主线** 对齐，而非与「SRB」这一策略名混写。

本节把 **§2–§8、§10–§11** 里分散的勾选项 **按「稳定流水线」维度重排**，便于你逐项对照「是否符合可运维、可复盘、可收窄 adopt」的标准。  
**共识（多腿）**：多腿 **同样有第一环「体制 / 特征 mask」**——在范式上与经典 **Prefilter** 对齐即可命名为 **Regime**（见 §10.2）；**特征列数通常少于 BPC**，但不等于「不能挑特征 / 不能调参」：仍可在 **`features.yaml` + `feature_dependencies`** 内演进列，并在 **`threshold_search`（或等价块）** 中对 **regime 阈值、网格/执行旋钮** 做 sweep；与经典差异主要在 **统计量是否够开 Pool-B 式宽搜索**，而非能力上禁止。

### 12.0 多腿：研究 vs 实盘 — **当前已分哪里、仍混哪里**（团队最高优先级）

**结论先说**：**`deploy_config_to_live.py` 已经在「同步到 live」这一刀上把研究与实盘分开了**；你感觉「卸在一起」主要来自 **`config/strategies/chop_grid|dual_add_trend/` 根目录仍把研究 yaml 与引擎 yaml 放在同一棵树里**，心智上不像 BPC 那样一眼分清。

| 维度 | 现状（以 `chop_grid` / `dual_add_trend` 为例） |
|------|-----------------------------------------------|
| **同步到 `live/highcap/...` 的内容** | **`archetypes/` 下全部 yaml** + **`TOP_LEVEL_CONFIGS` 中出现的根文件（如 `meta.yaml`）**。`research/**` **从不**镜像到 live（以 `deploy_config_to_live.py` 为准）。 |
| **默认不会进 live 的** | **`research.yaml`、`threshold_search.yaml`** 不在 `TOP_LEVEL_CONFIGS` 列表内 → **deploy 不会复制**；`config/strategies/<slug>/research/` 子目录（若仅有文档）也 **不在** deploy 拷贝范围内。因此 **`live/.../chop_grid` 里通常看不到 `research.yaml`**，与 `config/strategies/chop_grid/` 根下那份 **可以并存而不污染实盘树**。 |
| **仍「混」的点（要优先改的）** | 研究与搜索声明分散在策略根时，会与 §2「`research/` 可调、不进 live」冲突 —— **持续迁入 `research/`** 并删掉根上冗余 `research.yaml`/`threshold_search.yaml`。**多腿运行时**仅以 `research/calibrate_roll.default.yaml`（及 `extends` 链）为准。 |

**因此本 ADR 将下列事项标为「稳定流水线最高优先级」（多腿先行亦可）**：

- [ ] **物理拆分**：把 **`research.yaml`、`threshold_search.yaml`**（及未来 **`research/calibrate_roll.default.yaml`** 等）迁入 **`config/strategies/<slug>/research/`**；策略根只保留 **deploy 白名单**内的稳定文件（`meta.yaml`、`features.yaml`、`archetypes/`）；**不把**滚动研究入口放在根命名 `grid.yaml`/`dual_add.yaml`。  
- [ ] **文档一句**：在策略 `README.md` 或 runbook 写明 **「`live/highcap` 仅含 …；`research*` 仅研究侧读」**，消除「以为实盘会吃 research.yaml」的误解。  
- [ ] **（可选加固）** 在 `deploy_config_to_live.py` 增加 **显式忽略列表**（如 `research.yaml`、`threshold_search.yaml`、`research/**`），即使将来误把文件名加进 `TOP_LEVEL_CONFIGS` 也不会上 live —— 与 **deploy profile**（§12.2）同一 PR 评审更稳。

### 12.1 目录、契约与加载（全策略）

- [ ] **`research/` + `archetypes/` + 根稳定文件** 边界写进 README / runbook；`live/` **不**含 `research/`。
- [ ] **§3.3 检查**：packaged `calibrate_roll` / `research_roll` / `validate_static` YAML 在变体 **A/B**（并列 vs 单文件分块）上 **只用一套 loader**，禁止分叉解析。
- [ ] **`mlbot pipeline` 约定名解析**（§3.2）：仅 `--strategy` 时的探测顺序、回退 `config/research_pipeline.yaml` 的 **警告日志**；单元测试覆盖。
- [ ] **feature-store / train / diagnose** 凡读策略目录，是否 **统一经 loader**（或明确例外列表），避免直读路径绕过 `archetypes/` merge。

### 12.2 BPC / TPC / ME（事件链主线）稳定项

- [ ] **adopt 默认收窄**：仅写 `archetypes/`（+ 白名单根文件）；`--full-tree` 逃生阀期限与告警。
- [ ] **deploy profile 表**：`deploy_config_to_live.py` 与 `ARCHETYPE_FILES` / 策略白名单 **单一来源**；与多腿共用抽象（§7.4）。
- [ ] **deep merge**：顺序、anchor、golden fixture；禁止半份 yaml 静默回退。
- [ ] **CI 门禁**：至少一条 job **显式 `--config`**；可选 `adopt --dry-run` + `deploy --diff` artifact。
- [ ] **Pipeline 共位**（若执行 §7.2b）：全仓 `--config` 替换 + 根目录 shim 过渡期文档。

### 12.3 多腿与 fast/slow「parity」（流程对齐，实现可专用）

**目标**：与经典 **同一宏观序位**（特征/体制 → 阈值与旋钮 → 分层决策 → 执行），**不要求**物理文件名与 BPC 完全一致；**要求** 每策略 **profile 文档化** + 管线里 **`enabled: false` + N/A 原因** 无歧义。

- [ ] **`archetypes/README.md`（或等价）**：`chop_grid` / `dual_add_trend` 各列 **文件名 ↔ pipeline 序位**（例：`regime.yaml` ↔ prefilter 位、`regime_thresholds.yaml` ↔ 可调阈值载体）；与 `docs/z实验_011_*` 交叉引用。
- [ ] **Regime = Prefilter 位（共识落地）**：rolling / adopt / 文档三处用语一致；禁止「口头叫 prefilter、文件却无任何映射」。
- [ ] **特征与搜索（你关心的「仍能挑选」）**  
  - [ ] **列演进**：`features.yaml` 变更走 **feature_dependencies** 与 review；在 **`turbo_fixed_features`** 且 **`disable_feature_search: true`** 时，明确 **「谁负责提议新列」**（离线 Pool-B/诊断 vs rolling 内开关）。  
  - [ ] **阈值 / 超参搜索**：`threshold_search`（或并入 `calibrate_roll.default.yaml`）中声明 **regime、网格 spacing、TP、腿数上限**；**research_roll** 路径可拉长窗或收紧 KPI，与 §10.3 一致。  
  - [ ] **样本量闸门**：多腿每层 sweep 前 **最小行数 / 最小成交** 不满足则 **skip 并写报告**，避免 silent 过拟合。
- [ ] **`fast_month` 与 BPC/TPC/ME 主线路径对齐（代码或文档二选一必须拍板）**  
  - [ ] **现状文档化**：多腿在 `_run_fast_month_stage` 中 **跳过** `run_strategy_pipeline` 式全层标定、走 `_run_multileg_month_strategy` 的行为，写入 runbook（避免误以为「多腿偷工」）。  
  - [ ] **若要靠拢**：定义多腿版 **「轻量 rolling_calibration」**（例：regime/execution 候选网格 + 全窗校验），**禁止**无映射地打开 BPC 的 `prefilter.optimize` / `entry_filter` 链；实现单独 PR。  
  - [ ] **research_roll（多腿）**：`research/research_roll.features_on.yaml` 存在时的 **额外结构快照 / 保守 KPI** 是否跑、哪几个月跑，写进 **`research_roll` YAML** 与文档。
- [ ] **`calibrate_roll` packaged 快 YAML** 与 `rolling_calibration:`：§11.2 缺口是否通过 **补 yaml 块** 或 **仅文档解释「内建标定」** 关闭；与团队对「执行网格」定义对齐。

### 12.4 产物、list/adopt、跨根发现

- [ ] **`list`/`adopt`/`diff` 与 run 使用同一份编排**（同一 `--config` 或同一套 §3.2 解析）；`history_dir` 错位已在 §11 说明——操作文档 **加粗提醒**。
- [ ] **§11.4 Roadmap**（可选）：`pipeline list-runs --all-history-roots` 或 manifest；静态成果站不与闸门抢权。

### 12.5 观测、回滚与发布

- [ ] **rolling 报告最低集**：每月/每 run **可机读指标**（Sharpe、笔数、拒因分布）路径稳定；多腿与经典 **schema 可比性**（§10.1 档位 B）是否要做、分几期。
- [ ] **回滚剧本**：`git revert` adopt；live 回滚 tag；与 **监控 / dry-run**（§6）一致。
- [ ] **大文件与密钥**：`results/` 不进 git；策略目录无密钥。

### 12.6 Review 时建议勾选顺序

0. **§12.0**（多腿研究/实盘 **物理目录** 拆分 + 文档 + 可选 deploy 忽略加固）— **团队最高优先级**，先解决「卸在一起」的心智与误同步风险。  
1. **12.1 + 12.4**（契约与 list/adopt）— 防「跑完找不到 / adopt 错树」。  
2. **12.3**（多腿 parity + regime/特征/阈值叙事）— 防「多腿与 **BPC/TPC/ME 主线** 心智分叉」。  
3. **12.2 + 12.5**（BPC/TPC/ME 的 adopt 与观测）— 防「上线不可审计」。

---

## 13. Master TODO 总清单（合并 §8 + §12，便于一次审阅）

> **说明**：下列与 **§8 风险与开放问题**、**§12 各小节** 的 checkbox **语义重复**；**以本节为「合并视图」** 做 review / 排期时，勾选可在 §8/§12 同步打勾，或只维护本节（团队自行定一种，避免双份长期漂移）。**优先级**：**P0** = 不做则易误 deploy / 误用 history；**P1** = 契约与 adopt/deploy 正确性；**P2** = parity 与体验；**P3** = roadmap / 可选加固。

### P0 — 多腿研究 / 实盘与操作安全（最高）

| ID | 事项 | 详见 |
|----|------|------|
| T01 | 多腿：`research.yaml` / `threshold_search.yaml`（及后续 `research/calibrate_roll.default.yaml` 等）**迁入** `config/strategies/<slug>/research/`；根目录仅保留引擎 yaml + deploy 白名单 | §12.0 |
| T02 | 策略 README 或 runbook **一句写清**：`live/highcap` 同步范围 vs 研究侧只读路径 | §12.0 |
| T03 | （可选）`deploy_config_to_live.py` **显式忽略列表**，防误把研究文件加入 `TOP_LEVEL_CONFIGS` | §12.0 |
| T04 | `list` / `adopt` / `diff` 与 run **同一编排**（`--config` 或 §3.2 解析）；文档 **加粗** `history_dir` 与 `--config` 绑定 | §12.4、§11.1 |

### P1 — 目录契约、loader、CI、adopt/deploy

| ID | 事项 | 详见 |
|----|------|------|
| T10 | `research/` + `archetypes/` + 根稳定边界 → README/runbook；**`live/` 不含 `research/`** | §12.1、§2 |
| T11 | §3.3 **定稿**变体 A 或 B；**单一 loader**；全仓引用一致 | §12.1、§8 |
| T12 | `mlbot pipeline` **约定名解析** + 回退警告 + 测试 | §12.1、§8 |
| T13 | feature-store / train / diagnose：**统一 loader 或例外列表** | §12.1、§8 |
| T14 | adopt **默认仅 `archetypes/`**（+ 白名单）；`--full-tree` 期限与告警 | §12.2、§7 |
| T15 | **deploy profile 单一来源**（`ARCHETYPE_FILES`、多腿 `MULTI_LEG_*`、白名单合一） | §12.2、§7 |
| T16 | **deep merge**：顺序、anchor、**golden fixture** + 单测 | §12.2、§8 |
| T17 | CI：**至少一条显式 `--config`**；可选 adopt dry-run + deploy diff artifact | §12.2、§8 |
| T18 | **Merge / adopt legacy**：旧 adopt 分支保留多久、何时删 | §8 |
| T19 | prod train **共位迁移**（若做）：全仓 `--config` + shim 过渡期 | §12.2、§7.2b、§8 |

### P2 — 多腿 parity、叙事、fast_month

| ID | 事项 | 详见 |
|----|------|------|
| T20 | `chop_grid` / `dual_add_trend`：**`archetypes/README.md`** 文件名 ↔ stage 映射 + 链到 z实验文档 | §12.3、§8 |
| T21 | **Regime = Prefilter 位** 在 rolling / adopt / 文档三处一致 | §12.3 |
| T22 | 多腿特征：**列演进责任**、**threshold_search 空间**、**样本量闸门**（三子项） | §12.3 |
| T23 | **fast_month**：多腿 vs BPC/TPC/ME — **文档化现状** 或 **轻量 rolling_calibration 设计**（二选一拍板 + PR 边界） | §12.3 |
| T24 | **research_roll（多腿）**：是否有额外快照/KPI、写进 `research_roll.features_on.yaml` | §12.3 |
| T25 | packaged **`calibrate_roll.default.yaml`**（快线）vs 显式 **`rolling_calibration`**：补 YAML 或用文档固定「内建标定 vs 明示块」的语义 | §12.3、§11.2 |

### P3 — 观测、roadmap、卫生

| ID | 事项 | 详见 |
|----|------|------|
| T30 | rolling **可机读报告最低集**；多腿 vs 事件链 **KPI schema** 档位 B 是否分期 | §12.5、§10.1 |
| T31 | **回滚剧本** + 监控/dry-run 对齐 | §12.5、§6 |
| T32 | **`results/` 不进 git**；策略目录 **无密钥** | §12.5 |
| T33 | **§11.4**：跨 `history_dir` list / manifest / 静态站（可选） | §11.4、§12.4 |

### 建议排期（与 §12.6 一致，可贴到 issue epic）

1. **P0（T01–T04）** — 多腿物理拆分 + deploy 心智 + list/adopt 文档。  
2. **P1（T10–T19）** — loader、CI、adopt/deploy、merge 测试。  
3. **P2（T20–T25）** — parity 与 fast_month 叙事/实现拍板。  
4. **P3（T30–T33）** — 观测、卫生、跨根发现 roadmap。

---

## 14. 时间轴、`non_rolling` 与 PCM 各层跑频（2026-04 讨论定稿，待实现）

本节吸收 **rolling vs 固定 Val/Test**、**`validation_months` vs `calibration_months`**、**Gate / Prefilter / EntryFilter 的 cutoff** 等讨论结论，作为后续实现 PR 的合同依据（**尚未**改 `auto_research_pipeline` 默认行为）。

### 14.1 两条研究入口（产品语义）

| 入口 | 时间语义 | 用途 |
|------|-----------|------|
| **`research/calibrate_roll.default.yaml` / `research/research_roll.features_on.yaml`** | **Walk-forward**：自然月（或 `rolling_calibration.step_months`）为步长推进；标定窗由 `rolling.windows.calibration_months` 等定义 **每个目标月 M** 的 `[M−K, M−1]`（与 `_calib_and_test_windows` 一致）。 | 主研究、rolling_sim / fast_month。 |
| **`research/validate_static.full_study.yaml`（新增）** | **一次性划分**：`Train [start, holdout_start)` → `Val [holdout_start, test_start)` → `Test [test_start, end]`，由顶层 **`dates` + `holdout_months` + `validation_months`** 固定算出 `test_start`（与 `resolve_strategy_dates` 一致）；**不**跑按月 rolling 或仅跑等价 minimal stage。 | idea 验证、黄金回归、上线前「最后一窗」类短实验；**不**替代 `calibrate_roll`/`research_roll` 月度 rolling。 |

**`rolling.mode` 建议**：为 `non_rolling` 增加显式枚举（**勿**将「simple」与现有 `legacy` 混用——`legacy` 在 fast_month 路径上另有「不跑逐层标定」等语义，见代码注释）。

### 14.2 跑频（已拍板）

| 层 / 能力 | 触发节奏 | 说明 |
|-----------|-----------|------|
| **特征筛选、结构类重活**（含 SHAP / 宽特征搜索等，依策略是否启用） | **`slow_realistic.cadence_months`**（或等价「结构月」合同） | **非**每个自然月必跑；与「季结构 + 月快变量」设计一致。 |
| **阈值链 / rolling_calibration 内各标定**（threshold、prefilter、execution 等开关允许的组合） | **`rolling_calibration.step_months`**（通常为 **1**） | 按 **每个 rolling 月 M** 推进，与 `calibration_months` 定义的拟合窗配合。 |

多腿（chop_grid / dual_add_trend）在 **无 SHAP 式特征搜索** 时，仍以 **`rolling.mode`** + **`cadence_months`**（slow 节拍）对照 **`turbo_fixed_features` 月度快节拍**区分快慢；阈值/regime 类调整仍跟 **step_month** 叙事对齐（见 §10.3、§11.2）。

### 14.3 `cutoff` 与 `calibration_months`：为何曾有 cutoff、目标态如何统一

- **历史原因**：CLI / 管线通过 **`--cutoff-date`**（及元数据里的 `validation_end`）把 **「优化 / 调参」** 限制在 **Val 日历段**，避免 **Test（纯 OOS）段** 的标签或统计 **泄漏进** Gate、Prefilter、EntryFilter 的优化目标。  
- **现状痛点**：同一配置里 **fast_month** 已按 **`calibration_months`** **逐月滚动** 拟合窗，而 **Gate 优化** 仍可能仅用 **整条实验固定的 `test_start`** 切段，**两套时间边界** 并存，读者易误认为 bug。  
- **目标态（Prefilter / Gate / EntryFilter 同一逻辑）**：  
  - **特征筛选 / 结构路径**（仅在 **cadence 触发的结构月**跑或等价阶段）：使用的数据窗与 **train 合同一致** —— **`[start_date, val_boundary)`** 语义上的「训练+早 holdout 中允许进模型的部分」由实现 PR 精确定义边界（可与 `holdout_start`、`test_start`、结构锚点对齐表）。  
  - **阈值调整路径**（**每个 step_month**）：在 **Val 段**（或与 walk-forward 合同一致的 **当月拟合窗内的 val 语义**）上调阈值；**允许使用 pre-holdout（warmup / train）** 作为拟合输入，与「`calibration_months` 窗可跨过 `holdout_start`」一致。  
  - **实现要求**：为 rolling 模式引入显式 **`time_split_policy`**（命名待定）或等价开关；将 **分散的 `--cutoff-date`** 拼装收敛到 **单一解析函数**（按 **M + calibration_months + cadence** 推导），**Prefilter / Gate / EntryFilter** 共用，避免三层三套口径。

### 14.4 全窗回测与可视化

- **`full_cycle` / `grid_backtest` 等全窗产物**：时间轴 **可包含 warmup**（`start_date` 至 `holdout_start` 前）；**竖线** 标注 **`holdout_start`**，以及（若启用 Val/Test 分离）**`test_start`**，与 §14.1 表格一致，避免读者以为「全窗 = 仅 OOS」。  
- **多腿**：`grid_backtest` / `dual_add_backtest` 的日期 **默认继承** 顶层 `dates`（或未来 `pipeline_calendar`），**禁止**与主实验日历静默分叉（与前期「单真相源」讨论一致）。

### 14.5 与 §7 / Master TODO 的关系

建议在 **P1 稳定 loader / deploy 之后** 单独立项（避免与 T01–T19 混 PR）：

- [ ] 新增 `non_rolling.yaml` + `rolling.mode: non_rolling`（或最终命名）+ 文档与最小单测。  
- [ ] **`time_split_policy` + 统一 cutoff** + 全仓调用点收敛 + golden 日期窗测试。  
- [ ] Runbook：`validate_static.*.yaml`（`rolling.mode: non_rolling`）与月度 rolling YAML 的 **stage 对照表**。

**详见**：`docs/architecture/ADR_research_p3_roadmap.md` 中 **Time contract** 小节（与 P3 可选工具链并列，可排 P2/P4）。

---

## 15. 实施决议（执行记录，随 PR 更新）

| 日期 | 决议 | 说明 |
|------|------|------|
| **2026-04-28** | **§14 时间合同与 `non_rolling`（讨论定稿，待实现 PR）** | （1）新增 **`research/validate_static.full_study.yaml`** 与显式 **`rolling.mode: non_rolling`**（勿与 `legacy` 混用）。（2）**跑频**：特征筛选 / 结构类重活按 **`cadence_months`**；阈值链 / rolling_calibration 按 **`step_months`**。（3）**cutoff**：Prefilter / Gate / EntryFilter **同一规则**；rolling 下与 **`calibration_months`** 推导的窗对齐；特征筛选用 **start→val** 与 train 合同一致；阈值拟合用 **Val**，**允许 pre-holdout**。（4）**full_cycle** 可含 warmup，**竖线**标 `holdout_start` / `test_start`。详见 **§14**。 |
| **2026-04-28** | **§14.3 第一步：`rolling.time_split_policy` + `pcm_cutoff`（已落地）** | `load_pipeline_config` 写入 **`time_split_policy`**（默认 `static_holdout`；可选 `walk_forward_monthly`）。**`scripts/pipeline/calibration_window.py`** + **`pcm_cutoff.py`**；**`run_strategy_pipeline`** 中 SHAP / Gate / EntryFilter 的 **`--cutoff-date`** 与 **`validation_end`** 元数据统一经 **`resolve_pcm_cutoff_date`**；**`fast_month`** 嵌套标定传入 **`month_token`** + **`pcm_rolling_calibration_months`**。单测：`tests/unit/test_pcm_cutoff.py`、`test_pipeline_config_extends`。后续：`non_rolling.yaml`、`filter_stages` 等余下调用点。 |
| **2026-04-28** | **§14.3 第二步：对齐续** | **`research/validate_static.full_study.yaml`**（BPC 系 `extends: research_roll.features_on.yaml`；多腿系 `extends: calibrate_roll.default.yaml`，`rolling.mode: non_rolling`，`results/bpc/validate_static.full_study`）；**`load_pipeline_config`** 允许 **`non_rolling`**；**`rolling_sim` / `fast_month`** 遇 **`non_rolling`** 显式 **`p.error`**；**`fast_month`** 内 **`_run_fast_month_stage`** 双保险 **`ValueError`**；**多腿** `grid_backtest` / `dual_add_backtest` 与顶层 **`dates` 不一致则报错**，YAML 已删重复日期；**`run_strategy_pipeline`** 内 Prefilter 对比向量回测与 EF mini-backtest 的 **`--test-end`** 统一为 **`_val_segment_end`**（与 `pcm_cutoff` 一致）；**`filter_stages.run_entry_filter_stage`** 增加 **`pcm_cutoff_date`** 与 **`_pcm_ef_*`** 辅助；`_resolve_stage_strategies_root` 将 **`non_rolling`** 与月度 rolling **`history_dir` 合同**对齐到同一产物根语义。单测：`test_filter_stages_pcm_cutoff.py`、`test_load_bpc_non_rolling_extends_turbo`、`test_multileg_backtest_dates_mismatch_raises`。 |
| **2026-04-30** | **§3.3 采用变体 A（首段交付）** | 研究声明与搜索空间以 **并列文件** 落在 `config/strategies/<slug>/research/`（`research.yaml`、`threshold_search.yaml` 文件名可保留）；**首段交付**以 packaged `research/*.yaml` 为准；不把全局 prod_train 塞进单文件的 `turbo.yaml`/`slow.yaml` 历史形态（变体 B 单列 PR）。 |
| **2026-04-30** | **§7.2b / T19 prod_train 共位 — Deferred** | `prod_train_pipeline_*_only.yaml` 仍放 `config/` 根；后续若共位，再开迁移 PR 并更新 CI/文档。 |
