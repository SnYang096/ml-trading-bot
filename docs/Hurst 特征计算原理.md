Hurst 特征计算原理

非常棒的问题 👏
Hurst 特征（Hurst exponent）是量化分析中极其重要的一个“结构性特征”，
它能帮助你判断**市场的长期记忆性**、**趋势性**和**均值回复性**。
理解它的计算原理，就等于理解了市场价格的“自相似性”和“可预测性”的物理基础。

---

## 🧩 一、Hurst 指数定义与含义

Hurst 指数（记作 **H**）来源于时间序列的**R/S 分析（Rescaled Range Analysis）**，
最早由英国水文学家 Harold Edwin Hurst 用于分析尼罗河的水流波动。

它衡量的是：

> **时间序列的自相似性与持久性（persistence）程度。**

---

### 📈 数值解释：

| Hurst 值范围   | 含义                    | 市场特征    | 对应交易策略     |
| ----------- | --------------------- | ------- | ---------- |
| **H < 0.5** | 反持久性（Anti-persistent） | 均值回归、震荡 | 做区间交易、短线反转 |
| **H = 0.5** | 随机游走（Random Walk）     | 无记忆性    | 不可预测       |
| **H > 0.5** | 持久性（Persistent）       | 趋势延续    | 顺势交易、趋势跟随  |

---

## ⚙️ 二、Hurst 指数的计算原理（R/S 方法）

我们从最经典的 **Rescaled Range (R/S)** 方法讲起：

### 1️⃣ 输入：

时间序列 ( X = {x_1, x_2, ..., x_N} )

---

### 2️⃣ 分段：

将序列分为长度为 ( n ) 的多个子区间。

---

### 3️⃣ 每个区间内：

1. 计算子区间均值：
   [
   \bar{X}*n = \frac{1}{n} \sum*{i=1}^n x_i
   ]

2. 计算累积偏差序列：
   [
   Y_t = \sum_{i=1}^t (x_i - \bar{X}_n)
   ]

3. 计算极差：
   [
   R(n) = \max(Y_t) - \min(Y_t)
   ]

4. 计算标准差：
   [
   S(n) = \sqrt{\frac{1}{n} \sum_{i=1}^n (x_i - \bar{X}_n)^2}
   ]

5. 计算 **重标极差**：
   [
   \frac{R(n)}{S(n)}
   ]

---

### 4️⃣ 重复多次（不同区间长度 n）：

对多种 ( n )（如 8, 16, 32, 64...）重复上面的计算，得到多组：
[
\left( n, \frac{R(n)}{S(n)} \right)
]

---

### 5️⃣ 拟合对数关系：

理论上有：
[
E\left[\frac{R(n)}{S(n)}\right] = c \cdot n^H
]
取对数后线性化：
[
\log\left( \frac{R(n)}{S(n)} \right) = \log(c) + H \cdot \log(n)
]

于是可以通过线性回归估计斜率 ( H )。

---

### ✅ 结果：

* 斜率 ( H ) 就是 **Hurst exponent**。

---

## 🧠 三、直观理解：H 表示序列的“粘性”或“反弹性”

| 类比    | H 值     | 行为             |
| ----- | ------- | -------------- |
| 橡皮筋   | H < 0.5 | 越拉越想回去（均值回归）   |
| 随机行走者 | H = 0.5 | 无意图地走动         |
| 火车头   | H > 0.5 | 一旦开动就很难停（趋势延续） |

---

## 💡 四、快速算法示例（Python）

```python
import numpy as np

def hurst_rs(series):
    """计算时间序列的Hurst指数（R/S方法）"""
    N = len(series)
    max_k = int(np.floor(N / 2))
    rs = []
    sizes = []

    for k in [8, 16, 32, 64, 128]:
        if k >= N:
            continue
        chunks = N // k
        r_s_vals = []

        for i in range(chunks):
            data = series[i*k:(i+1)*k]
            mean = np.mean(data)
            dev = data - mean
            cum_dev = np.cumsum(dev)
            R = np.max(cum_dev) - np.min(cum_dev)
            S = np.std(data)
            if S != 0:
                r_s_vals.append(R / S)

        rs.append(np.mean(r_s_vals))
        sizes.append(k)

    log_rs = np.log(rs)
    log_sizes = np.log(sizes)
    H, _ = np.polyfit(log_sizes, log_rs, 1)
    return H
```

---

## 📊 五、在交易系统中的应用

| 用途         | 逻辑               | 操作                          |
| ---------- | ---------------- | --------------------------- |
| **市场状态识别** | H < 0.45 → 震荡区间  | 进入 `COMPRESSION` / 区间交易模式   |
| **趋势信号确认** | H > 0.55 → 趋势市场  | 启动 `EXPANSION` 顺势模式         |
| **波动自适应**  | H 变化率 → 市场结构切换信号 | 结合 Wavelet/Hilbert 提前识别结构反转 |

---

## 🔮 六、与小波变换和Hilbert的关系

| 工具             | 核心测量       | 作用层次          |
| -------------- | ---------- | ------------- |
| **Hilbert 变换** | 相位 / 瞬时频率  | 局部节奏检测（局部动态）  |
| **小波变换**       | 多尺度能量分布    | 时间-频率分解（结构特征） |
| **Hurst 指数**   | 自相似性（长期记忆） | 全局结构与市场性质判断   |

> ✅ 它们三者可以组成一个强大的特征组：
> **Hilbert (瞬态)** + **Wavelet (局部结构)** + **Hurst (全局粘性)**
> 一起构建市场状态的多尺度认知模型。

---

## 📈 七、结论总结

| 指标          | 衡量维度       | 典型用途       |
| ----------- | ---------- | ---------- |
| Hurst (H)   | 自相似性 / 记忆性 | 趋势 vs 震荡识别 |
| 小波能量比       | 局部多尺度波动    | 结构分解       |
| Hilbert 相位差 | 局部周期性      | 动量/相位确认    |
| 组合信号        | 结构周期 + 记忆性 | 市场状态机输入特征  |

---

是否希望我帮你画一张「Hilbert + Wavelet + Hurst」三者如何协同作用的 **市场结构分析示意图（Mermaid 或 图片版）**？
这样你可以直观看出它们如何在不同时间尺度上构成一个完整的市场认知系统。
