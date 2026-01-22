# Terraform 变量设置指南

## 推荐方案：使用 terraform.tfvars 文件（最简单）

我已经为你创建了 `terraform/terraform.tfvars` 文件，**直接编辑这个文件**即可：

```bash
cd terraform
# 编辑 terraform.tfvars，填入 CLS 配置（如果不需要，保持为空字符串）
vim terraform.tfvars
```

**优点**：
- ✅ 一次设置，永久使用
- ✅ 不需要每次运行前设置环境变量
- ✅ 文件已加入 .gitignore，不会提交到 Git

## 方案 2：使用环境变量文件（类似 qclould.env）

如果你更喜欢环境变量的方式，可以使用：

```bash
# 1. 编辑配置文件
vim config/local/terraform.env

# 2. 运行前加载
source config/local/qclould.env      # 腾讯云凭证
source config/local/terraform.env    # CLS 配置

# 3. 运行 Terraform
cd terraform
terraform plan
```

## 当前配置状态

我已经创建了两个文件：

1. **`terraform/terraform.tfvars`** - Terraform 变量文件（推荐使用）
   - 直接编辑这个文件，填入 CLS 配置
   - 如果不需要日志服务，保持为空字符串即可

2. **`config/local/terraform.env`** - 环境变量文件（备选方案）
   - 如果喜欢环境变量方式，可以编辑这个文件
   - 运行前 `source config/local/terraform.env`

## 使用示例

### 使用 terraform.tfvars（推荐）

```bash
# 1. 编辑配置文件（只需一次）
cd terraform
vim terraform.tfvars
# 填入：
# cls_secret_id  = "your-value"
# cls_secret_key = "your-value"
# cls_topic_id   = "your-value"

# 2. 加载腾讯云凭证
source ../config/local/qclould.env

# 3. 直接运行，无需设置其他环境变量
terraform plan
```

### 使用环境变量文件

```bash
# 1. 编辑环境变量文件（只需一次）
vim config/local/terraform.env

# 2. 运行前加载所有配置
source config/local/qclould.env
source config/local/terraform.env

# 3. 运行 Terraform
cd terraform
terraform plan
```

## 总结

**推荐使用 `terraform.tfvars`**，因为：
- 更符合 Terraform 的标准做法
- 不需要每次运行前 source
- 配置更清晰
