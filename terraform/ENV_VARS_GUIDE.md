# Terraform 环境变量使用指南

## terraform.tfvars 不能直接引用环境变量

`terraform.tfvars` 是**静态文件**，不支持函数或环境变量引用。但有以下几种替代方案：

## 方案对比

### ✅ 方案 1：使用 TF_VAR_ 环境变量（推荐，最简单）

Terraform **自动读取** `TF_VAR_` 前缀的环境变量：

```bash
# 设置环境变量
export TF_VAR_cls_secret_id="your-value"
export TF_VAR_cls_secret_key="your-value"
export TF_VAR_cls_topic_id="your-value"

# 直接运行，无需 terraform.tfvars
terraform plan
```

**优点**：
- ✅ 最简单，直接使用环境变量
- ✅ 不需要 terraform.tfvars 文件
- ✅ 可以动态设置

### ✅ 方案 2：使用环境变量文件（推荐，统一管理）

创建 `config/local/terraform.env`，统一管理所有 Terraform 变量：

```bash
# 1. 编辑 config/local/terraform.env（只需一次）
vim config/local/terraform.env

# 2. 运行前加载
source config/local/qclould.env      # 腾讯云凭证
source config/local/terraform.env    # CLS 配置

# 3. 运行 Terraform
cd terraform
terraform plan
```

**优点**：
- ✅ 统一管理所有环境变量
- ✅ 可以引用其他环境变量（如 `${CLS_SECRET_ID:-}`）
- ✅ 与项目其他配置保持一致

### ✅ 方案 3：使用 terraform.tfvars（静态配置）

如果值固定不变，使用 `terraform.tfvars`：

```hcl
# terraform/terraform.tfvars
cls_secret_id  = "your-value"
cls_secret_key = "your-value"
cls_topic_id   = "your-value"
```

**优点**：
- ✅ 配置清晰，一目了然
- ✅ 适合固定配置

**缺点**：
- ❌ 不能引用环境变量
- ❌ 需要手动编辑文件

## 推荐工作流

### 方式 A：纯环境变量（最灵活）

```bash
# 1. 加载所有配置
source config/local/qclould.env
source config/local/terraform.env

# 2. 运行 Terraform（无需 terraform.tfvars）
cd terraform
terraform plan
terraform apply
```

### 方式 B：混合方式（环境变量 + tfvars）

```bash
# 1. 加载腾讯云凭证
source config/local/qclould.env

# 2. 使用 terraform.tfvars 设置 CLS（如果值固定）
# 编辑 terraform/terraform.tfvars

# 3. 运行 Terraform
cd terraform
terraform plan
```

## 当前配置状态

我已经为你配置了：

1. **变量默认值**：CLS 变量现在有默认空值，可以不设置
2. **环境变量文件**：`config/local/terraform.env` 可以引用其他环境变量
3. **terraform.tfvars**：静态配置文件（可选）

## 最佳实践

**推荐使用环境变量方式**，因为：
- ✅ 更灵活，可以动态设置
- ✅ 可以引用其他环境变量
- ✅ 与项目其他配置（如 `qclould.env`）保持一致
- ✅ 不需要维护额外的 tfvars 文件
