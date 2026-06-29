# C 系统零基础学习与验证指南

> **写给谁**：有仓库代码、想理解 chop / trend 在做什么，并在自己机器上验证的人；**不必**先搞懂全部部署术语。  
> **和专家服务器的关系**：代码同源；专家机 = 同一套规则 + 连交易所 + 可能真钱。你可以从「看电影重播」开始，不必复制他的 API Key。

---

## 1. 先记住一句话

这个仓库里，和你最相关的主要叫 **C 系统**（多腿、震荡 + 趋势）：

| 策略 | 人话 | 什么时候做 |
|------|------|------------|
| **chop_grid** | 震荡市里摆「网格」，赚价格来回扫 | 行情 **震荡**（chop 高） |
| **trend_scalp** | 趋势市里 **顺着方向**开腿、有限加仓 | 行情 **有趋势、低震荡、非箱体** |

专家服务器上跑的是：**行情 → 算特征 → 判断能不能做 →（可选）自动下单**。  
你手里的代码 = **同一套判断规则**；差别只在有没有连交易所、是不是真钱。

---

## 2. 只学 8 个词（够撑过前几周）

| 词 | 人话 | 类比 |
|----|------|------|
| **K 线 / 2h** | 每 2 小时一根蜡烛图 | 策略每隔 2 小时看一眼市场 |
| **回测** | 用历史数据假装当时下单 | 看电影重播，检验规则过不过得去 |
| **特征** | 从 K 线算出的数字（趋势强不强、是否震荡） | 体温计读数，不是水晶球 |
| **Regime / 段** | 「这段时间适不适合做某策略」 | 绿灯路段可以开，红灯别进 |
| **Shadow** | 模拟实盘循环，**不真实下单** | 演习 |
| **Testnet** | 连交易所 **测试环境**，假钱 | 驾校 |
| **Mainnet** | **真钱**实盘 | 正式上路（不急） |
| **Feature Bus** | 一台进程收行情、算特征，别的进程来读 | 中央厨房 |

其它名词（PCM、User Stream、Hedge Mode、systemd…）**等你要上 Testnet 再学**。

---

## 3. 一张总图（贴在脑子里）

```text
历史 parquet 数据（或实盘行情）
       │
       ▼
  算特征（chop 高不高？趋势强不强？是不是箱体？）
       │
       ▼
  够不够开一段？（regime / segment）
       │
       ├── 回测：在历史里模拟买卖 → 看赚亏、看图
       │
       └── Live：Shadow → Testnet → Mainnet
                 （越来越接近真下单）
```

**你现在适合站在左边**（理解 + 回测）；专家生产机在 **最右边**。

---

## 4. 学习路线：四阶段，不要跳步

```text
阶段 1  看懂策略在干什么        ← 只读文档 + 看图
阶段 2  本机回测               ← 一条命令，有数字就行
阶段 3  Shadow                 ← 不下单，看 live 能否转一圈
阶段 4  自己的服务器 + Testnet   ← 需要自己的测试网 Key
```

**阶段 4 之前：不需要币安 Key，不需要 AWS。**

---

## 5. 阶段 1：只读这些（每份约 15～30 分钟）

按顺序读，**不要并行啃源码**。

| 顺序 | 文件 | 内容 |
|------|------|------|
| 1 | [`config/strategies/chop_grid/README.md`](../../config/strategies/chop_grid/README.md) | chop 是什么、何时做、何时不做 |
| 2 | [`config/strategies/trend_scalp/TREND_SCALP_逻辑导读_CN.md`](../../config/strategies/trend_scalp/TREND_SCALP_逻辑导读_CN.md) | trend 分层、三开仓条件、段内怎么交易 |
| 3 | 浏览器打开下图 | 对照 K 线理解概念 |

**图示（浏览器 `open` 或双击）：**

| 文件 | 说明 |
|------|------|
| [`config/strategies/trend_scalp/regime_concepts_annotated.html`](../../config/strategies/trend_scalp/regime_concepts_annotated.html) | 三条件 **示意图**（合成 K 线） |
| [`config/strategies/trend_scalp/BTCUSDT_2025_regime_annotated.html`](../../config/strategies/trend_scalp/BTCUSDT_2025_regime_annotated.html) | **真实** BTC 2025 全年 2h 标注 |

**读图只问三件事：**

- 浅绿/浅红竖条 = trend 段（做多/做空）
- 紫色竖条 = 稳定箱体（trend 不做）
- 空白 = 条件不满足，故意不做

**trend 新开段三条件（复习）：**

| 条件 | 在问什么 |
|------|----------|
| `trend_confidence ≥ 0.70` | 有没有方向？（3/5/10 根 2h 是否同向） |
| `semantic_chop ≤ 0.25` | 是不是来回扫？（震荡则交给 chop） |
| `box_prefilter = false` | 是不是关在箱子里？（箱体交给 CRF/BPC） |

三样 **同时** 满足才新开 trend 段。详见逻辑导读第 3 节。

---

## 6. 阶段 2：本机回测（一条命令）

在项目根目录（Mac 示例）：

```bash
cd /path/to/ml-trading-bot

# chop：2025 年 BTC（几分钟内跑完）
.venv/bin/python scripts/chop_grid_backtest.py \
  --start 2025-01-01 --end 2025-12-31 \
  --symbols BTCUSDT --timeframe 2h \
  --out-dir results/my_first/chop_btc_2025
```

跑完后看 `results/my_first/chop_btc_2025/summary.csv`，先只看三列：

| 列 | 含义 |
|----|------|
| `return_pct` | 这段赚了多少（%） |
| `segments` | 开了几段「震荡网格」 |
| `trades` | 成交多少笔 |

**不必懂每个字段**；先确认：命令能跑、有数字、有报告。

可选（理解 trend 用）：

```bash
.venv/bin/python scripts/diagnose_dual_add_trend.py \
  --config config/experiments/20260618_multileg_param_tune/variants/trend_hold_scaled.yaml \
  --symbols BTCUSDT --start 2025-01-01 --end 2025-12-31 \
  --timeframe 2h --execution-timeframe 1min \
  --no-initial-hedge --scale-max-loser-hold-to-signal \
  --out-dir results/my_first/trend_btc_2025
```

---

## 7. 阶段 3：Shadow（仍不下单）

```bash
.venv/bin/python scripts/run_multi_leg_live.py \
  --mode shadow \
  --bar-source parquet \
  --strategies chop_grid \
  --symbols BTCUSDT \
  --data-dir data/parquet_data \
  --once
```

- **没报错** = live 那套「读数据 → 算信号 → 决定动作」在你电脑上能转一圈。
- 与专家服务器 **逻辑同路**，只是没连真交易所。
- `--once` = 跑一轮就停，适合第一次试。

建议 **先只玩 chop_grid**；trend 在仓库 constitution 里当前偏研究态，见下文「和专家机的差异」。

---

## 8. 阶段 4：自己的服务器 + Testnet（进阶）

**完整逐步命令**见：[`自建服务器部署_chop_grid_CN.md`](自建服务器部署_chop_grid_CN.md)（Shadow → Feature Bus → Testnet → Mainnet）。

快捷脚本（Docker 已构建后）：

```bash
DEPLOY_ROOT=/opt/quant-engine ./scripts/ops/start_self_hosted_chop.sh shadow
DEPLOY_ROOT=/opt/quant-engine ./scripts/ops/start_self_hosted_chop.sh feature-bus   # 终端1
DEPLOY_ROOT=/opt/quant-engine ./scripts/ops/start_self_hosted_chop.sh multileg-testnet --no-orders  # 终端2
```

### 8.1 你需要什么

| 需要 | 不需要 |
|------|--------|
| 自己的 VPS（4GB+ 内存较稳） | 专家的 **主网 API Key** |
| 自己申请的 **Testnet** 合约 Key | 专家的 `binance_mainnet.env` |
| `git clone` 本仓库 + 历史 `data/parquet_data` | 1:1 复制专家 mainnet |

### 8.2 服务器最小步骤（不用 Docker 也行）

```bash
git clone <你的仓库> ml-trading-bot && cd ml-trading-bot
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .

# 从本机同步历史数据（示例）
# rsync -avz data/parquet_data/ user@你的IP:~/ml-trading-bot/data/parquet_data/
```

先 Shadow：

```bash
PYTHONPATH=src .venv/bin/python scripts/run_multi_leg_live.py \
  --mode shadow --bar-source parquet \
  --strategies chop_grid --symbols BTCUSDT \
  --state-dir data/multi_leg_live/state
```

### 8.3 Testnet（假钱，需自己的 Key）

1. 在币安合约 **Testnet** 创建 API Key（与专家主网无关）。
2. 建本地 env 文件（**勿提交 git**）：

```bash
MULTI_LEG_BINANCE_FUTURES_TESTNET_API_KEY=你的key
MULTI_LEG_BINANCE_FUTURES_TESTNET_API_SECRET=你的secret
```

3. **先只观察、不下单**：

```bash
export $(grep -v '^#' live/testnet.env | xargs)

.venv/bin/python scripts/run_multi_leg_live.py \
  --mode testnet \
  --no-orders \
  --strategies chop_grid \
  --symbols BTCUSDT \
  --bar-source parquet \
  --data-dir data/parquet_data
```

4. 日志正常后，再去掉 `--no-orders` 小仓位试单。  
5. 合约账户需 **双向持仓（Hedge Mode）**（testnet/mainnet 会检查）。

更完整的生产架构（Feature Bus 三进程）见：  
[`docs/deployment/LIVE_PRODUCTION_RUNBOOK_CN.md`](../deployment/LIVE_PRODUCTION_RUNBOOK_CN.md)  
[`docs/architecture/live_stream/multi_leg_live_daemon.md`](../architecture/live_stream/multi_leg_live_daemon.md)

---

## 9. 和专家已部署环境的差异（心里有数）

| 项 | 说明 |
|----|------|
| **chop_grid** | 研究/回测与 live 较接近；constitution 里 **已启用** |
| **trend_scalp** | constitution **当前注释禁用**（2026-06-17，bear/range DD 等）；你先当 **研究对象** |
| **B 系统**（BPC/TPC） | 走 `run_live.py`，与 chop/trend **不同进程**；初学可忽略 |
| **同币互斥** | chop 与 trend 不能同币同时占槽；中间 chop 刻度有「留白区」 |

运维分工说明：[`docs/strategy/C系统运维心智梳理.md`](C系统运维心智梳理.md)  
策略合理性（偏深）：[`config/experiments/20260618_multileg_param_tune/TREND_SCALP_策略合理性分析_CN.md`](../../config/experiments/20260618_multileg_param_tune/TREND_SCALP_策略合理性分析_CN.md)

---

## 10. 听不懂某个词时怎么办

1. **归类**：是「行情 / 规则 / 执行 / 部署」哪一类？  
   - 行情：K 线、2h、Feature Bus  
   - 规则：特征、regime、阈值  
   - 执行：下单、手续费、Testnet  
   - 部署：服务器、Docker、API Key  

2. **一次只啃一个策略**：先 **chop_grid**，再 trend。

3. **对照本指南 + 逻辑导读**，或把 **单个词** 拿出来问「白话版」——比硬背文档快。

---

## 11. 建议的第一周

| 天 | 做什么 |
|----|--------|
| 1～2 | 读 chop README + trend 逻辑导读；打开 `BTCUSDT_2025_regime_annotated.html` |
| 3 | 跑 §6 chop 回测，看 `summary.csv` 三列 |
| 4 | 跑 §7 Shadow `--once` |
| 5～7 | （可选）租 VPS、Testnet Key、`--no-orders` 观察几天 |

---

## 12. 相关命令与脚本速查

| 用途 | 命令 / 路径 |
|------|-------------|
| chop 回测 | `scripts/chop_grid_backtest.py` |
| trend 回测 | `scripts/diagnose_dual_add_trend.py` |
| live 演习 | `scripts/run_multi_leg_live.py --mode shadow` |
| 生成 regime 教学图 | `scripts/research/plot_trend_regime_concepts.py` |
| 生成 BTC 年度标注图 | `scripts/research/plot_trend_regime_btc_annotated.py` |
| 研究配置 → live 镜像 | `scripts/deploy_config_to_live.py` |
| 实盘依赖检查 | `live/scripts/check_dependencies.sh` |

---

## 13. 文档索引（由浅入深）

| 文档 | 适合阶段 |
|------|----------|
| 本文 | 零基础总览 |
| **[本地Docker与Testnet命令手册_CN.md](本地Docker与Testnet命令手册_CN.md)** | **复制粘贴命令（Docker / Shadow / Testnet）** |
| [`config/strategies/chop_grid/README.md`](../../config/strategies/chop_grid/README.md) | chop 策略说明 |
| [`config/strategies/trend_scalp/TREND_SCALP_逻辑导读_CN.md`](../../config/strategies/trend_scalp/TREND_SCALP_逻辑导读_CN.md) | trend 逻辑 |
| [`docs/strategy/C系统.md`](C系统.md) | C 系统架构（稍深） |
| [`docs/deployment/LIVE_PRODUCTION_RUNBOOK_CN.md`](../deployment/LIVE_PRODUCTION_RUNBOOK_CN.md) | 生产部署（上 Testnet/Mainnet 时） |

---

*最后更新：与对话中零基础路径、2025 BTC 标注图、自建服务器验证说明对齐。*
