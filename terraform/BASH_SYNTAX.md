# Bash 参数扩展语法说明

## `${VAR:-default}` 语法

这是 Bash 的**参数扩展**语法，用于设置默认值：

```bash
${VAR:-default}
```

**含义**：
- 如果 `VAR` 存在且**非空**，使用 `VAR` 的值
- 如果 `VAR` **不存在**或**为空**，使用 `default`

## 示例

```bash
# 示例 1：变量存在
CLS_SECRET_ID="my-secret"
echo "${CLS_SECRET_ID:-}"      # 输出: my-secret

# 示例 2：变量不存在
unset CLS_SECRET_ID
echo "${CLS_SECRET_ID:-}"       # 输出: (空字符串)

# 示例 3：变量为空
CLS_SECRET_ID=""
echo "${CLS_SECRET_ID:-default}"  # 输出: default
```

## 嵌套使用：`${VAR1:-${VAR2:-}}`

可以嵌套使用，实现"优先使用 VAR1，如果不存在则使用 VAR2"：

```bash
# 优先使用 CLS_SECRET_ID，如果不存在则使用 TENCENTCLOUD_SECRET_ID
export TF_VAR_cls_secret_id="${CLS_SECRET_ID:-${TENCENTCLOUD_SECRET_ID:-}}"
```

**逻辑**：
1. 如果 `CLS_SECRET_ID` 存在且非空 → 使用 `CLS_SECRET_ID`
2. 如果 `CLS_SECRET_ID` 不存在或为空 → 使用 `TENCENTCLOUD_SECRET_ID`
3. 如果两者都不存在 → 使用空字符串

## 你的场景

如果你想让 CLS 使用与腾讯云相同的凭证：

```bash
# 方式 1：直接引用（如果确定 TENCENTCLOUD_SECRET_ID 就是 CLS_SECRET_ID）
export TF_VAR_cls_secret_id="${TENCENTCLOUD_SECRET_ID:-}"

# 方式 2：优先使用 CLS_SECRET_ID，否则使用 TENCENTCLOUD_SECRET_ID（更灵活）
export TF_VAR_cls_secret_id="${CLS_SECRET_ID:-${TENCENTCLOUD_SECRET_ID:-}}"
```

**推荐使用方式 2**，因为：
- ✅ 如果以后有独立的 CLS 凭证，可以设置 `CLS_SECRET_ID`
- ✅ 如果没有，自动使用 `TENCENTCLOUD_SECRET_ID`
- ✅ 更灵活，向后兼容
