# 📥 历史数据下载指南

## 🎯 快速开始

### Windows 用户（推荐）

```powershell
# 运行下载脚本
.\download_data.ps1
```

然后选择你需要的选项即可。

### 命令行用户

```bash
# 查看帮助
python scripts/download_training_data.py --help

# 查看已下载的数据
python scripts/download_training_data.py --summary
```

---

## 📊 数据说明

### 币种

- **BTCUSDT** - Bitcoin/USDT
- **ETHUSDT** - Ethereum/USDT
- **SOLUSDT** - Solana/USDT

### 时间范围

- **开始**: 2021年1月
- **结束**: 2025年9月
- **总计**: 57个月

### 文件数量

- 每个币种: 57个文件
- 所有币种: 177个文件

### 预计大小

每个月的数据大小约为：
- BTC: 800MB - 2GB
- ETH: 600MB - 1.5GB  
- SOL: 200MB - 800MB

**总计约**: 100GB - 250GB（所有币种全部数据）

---

## 🚀 使用方法

### 方式1：交互式菜单（推荐新手）

```powershell
.\download_data.ps1
```

提供7个选项：
1. 查看已下载的数据摘要
2. 下载所有币种数据
3. 只下载BTC
4. 只下载ETH
5. 只下载SOL
6. 自定义下载
7. 退出

### 方式2：命令行（推荐高级用户）

#### 下载所有币种的全部数据

```bash
python scripts/download_training_data.py
```

#### 只下载BTC

```bash
python scripts/download_training_data.py --symbols BTCUSDT
```

#### 下载BTC和ETH

```bash
python scripts/download_training_data.py --symbols BTCUSDT ETHUSDT
```

#### 指定时间范围

```bash
# 只下载2024年的数据
python scripts/download_training_data.py --start-year 2024 --start-month 1 --end-year 2024 --end-month 12
```

#### 组合使用

```bash
# 下载2023-2024年的BTC和ETH数据
python scripts/download_training_data.py \
  --symbols BTCUSDT ETHUSDT \
  --start-year 2023 --start-month 1 \
  --end-year 2024 --end-month 12
```

---

## 📁 文件组织

### 下载位置

所有数据保存在：`data/raw/`

### 文件命名格式

```
{SYMBOL}-aggTrades-{YEAR}-{MONTH}.zip
```

例如：
- `BTCUSDT-aggTrades-2021-01.zip`
- `ETHUSDT-aggTrades-2024-05.zip`
- `SOLUSDT-aggTrades-2025-09.zip`

### 目录结构

```
ml_project/
└── data/
    └── raw/
        ├── BTCUSDT-aggTrades-2021-01.zip
        ├── BTCUSDT-aggTrades-2021-02.zip
        ├── ...
        ├── ETHUSDT-aggTrades-2021-01.zip
        ├── ETHUSDT-aggTrades-2021-02.zip
        ├── ...
        ├── SOLUSDT-aggTrades-2021-01.zip
        └── SOLUSDT-aggTrades-2021-02.zip
```

---

## ⚙️ 功能特性

### ✅ 智能跳过

已存在的文件会自动跳过，不会重复下载。

### ✅ 断点续传

如果下载中断，重新运行脚本会从中断处继续。

### ✅ 文件验证

- 检查文件大小
- 删除损坏的文件
- 自动重试

### ✅ 进度显示

实时显示：
- 当前下载进度
- 文件大小
- 成功/失败状态

### ✅ 错误处理

- 404错误不重试（文件不存在）
- 网络错误自动重试（最多3次）
- 超时处理（10分钟超时）

---

## 📊 使用示例

### 示例1：首次下载所有数据

```powershell
# 1. 运行脚本
.\download_data.ps1

# 2. 选择选项2（下载所有币种）
# 3. 输入 'y' 确认
# 4. 等待下载完成（可能需要数小时）
```

### 示例2：只下载最近的数据

```bash
# 下载2024-2025年的BTC数据
python scripts/download_training_data.py \
  --symbols BTCUSDT \
  --start-year 2024 --start-month 1 \
  --end-year 2025 --end-month 9
```

### 示例3：补充缺失的数据

```bash
# 1. 查看已有数据
python scripts/download_training_data.py --summary

# 2. 下载缺失的月份
python scripts/download_training_data.py --symbols BTCUSDT
# 已存在的会自动跳过
```

### 示例4：下载单个月份

```bash
# 下载2024年5月的BTC数据
python scripts/download_training_data.py \
  --symbols BTCUSDT \
  --start-year 2024 --start-month 5 \
  --end-year 2024 --end-month 5
```

---

## ⏱️ 预计下载时间

根据网络速度：

| 网络速度 | 单个月 | 单个币种全部 | 全部币种 |
|---------|--------|-------------|---------|
| 10 MB/s | 1-3分钟 | 1-3小时 | 3-9小时 |
| 5 MB/s  | 2-6分钟 | 2-6小时 | 6-18小时 |
| 1 MB/s  | 10-30分钟 | 10-30小时 | 30-90小时 |

**建议**：
- 使用有线网络
- 选择网络空闲时段
- 可以分批下载（先下一个币种，再下另一个）

---

## 💡 使用建议

### 1. 磁盘空间

确保有足够的磁盘空间：
- **单个币种**: 至少 100GB
- **三个币种**: 至少 300GB

### 2. 分批下载

如果磁盘空间或时间有限：

```bash
# 先下载BTC
python scripts/download_training_data.py --symbols BTCUSDT

# 等完成后再下载ETH
python scripts/download_training_data.py --symbols ETHUSDT

# 最后下载SOL
python scripts/download_training_data.py --symbols SOLUSDT
```

### 3. 选择时间范围

如果只需要最近的数据：

```bash
# 只下载2024年至今
python scripts/download_training_data.py \
  --start-year 2024 --start-month 1
```

### 4. 后台运行

Windows PowerShell后台运行：

```powershell
# 方法1：使用 Start-Job
Start-Job -ScriptBlock { 
    cd D:\GitHub\trading\rlbot\ml_project
    python scripts/download_training_data.py --symbols BTCUSDT
}

# 查看任务
Get-Job

# 查看输出
Receive-Job -Id 1
```

---

## 🔍 数据验证

### 查看下载摘要

```bash
python scripts/download_training_data.py --summary
```

输出示例：
```
📁 本地数据摘要
══════════════════════════════════════════════════════════

Bitcoin (BTCUSDT): 57 个月
  最早: 2021-01
  最新: 2025-09
  大小: 85432.5 MB

Ethereum (ETHUSDT): 57 个月
  最早: 2021-01
  最新: 2025-09
  大小: 64328.2 MB
  
...
```

### 检查文件完整性

下载脚本会自动：
- ✅ 验证文件大小
- ✅ 检测损坏文件
- ✅ 标记缺失月份

---

## ⚠️ 常见问题

### Q1: 下载速度很慢怎么办？

**A**: 
- 检查网络连接
- 尝试其他时段（避开高峰期）
- 使用有线网络
- 关闭其他下载任务

### Q2: 下载中断了怎么办？

**A**: 直接重新运行脚本，已下载的文件会自动跳过。

```bash
# 重新运行，自动继续
python scripts/download_training_data.py --symbols BTCUSDT
```

### Q3: 某些月份404找不到？

**A**: 这是正常的，可能原因：
- 该币种在该时期未上线
- Binance未提供该月数据
- 数据尚未发布

脚本会自动跳过404错误，继续下载其他月份。

### Q4: 磁盘空间不足怎么办？

**A**: 
1. 清理旧数据
2. 使用更大的硬盘
3. 分批下载（一次下一个币种）
4. 只下载需要的时间范围

### Q5: 如何删除已下载的数据？

**A**: 直接删除 `data/raw/` 目录下的文件即可。

```powershell
# 删除所有BTC数据
Remove-Item data\raw\BTCUSDT-*.zip

# 删除2021年的数据
Remove-Item data\raw\*-2021-*.zip
```

---

## 📈 下一步

下载完成后：

### 1. 训练模型

```powershell
# 使用下载的数据训练
.\quick_gpu_train.ps1
```

### 2. 自定义训练脚本

参考 `scripts/train_model_wavelet.py`，修改：
- 数据路径
- 训练参数
- 特征工程

### 3. 多币种训练

创建新的训练脚本，同时使用BTC、ETH、SOL数据。

---

## 🆘 需要帮助？

如果遇到问题：

1. 📖 查看错误信息
2. 🔍 运行 `--summary` 检查数据状态
3. 🧹 删除损坏的文件后重试
4. 📝 检查磁盘空间和网络连接

---

## 📝 命令参考

```bash
# 完整参数列表
python scripts/download_training_data.py \
  --data-dir <目录>              # 数据保存目录（默认：data/raw）
  --symbols <币种列表>            # 币种（BTCUSDT ETHUSDT SOLUSDT）
  --start-year <年>              # 开始年份（默认：2021）
  --start-month <月>             # 开始月份（默认：1）
  --end-year <年>                # 结束年份（默认：2025）
  --end-month <月>               # 结束月份（默认：9）
  --summary                      # 只显示摘要，不下载
```

---

**祝下载顺利！📥🚀**

数据是训练的基础，拥有完整的历史数据将大大提升模型性能！


