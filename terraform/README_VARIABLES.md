# Terraform 变量配置说明

## 问题说明

运行 `terraform plan` 或 `terraform apply` 时，Terraform 会提示输入以下变量：
- `var.secret_id` - 腾讯云 Secret ID
- `var.secret_key` - 腾讯云 Secret Key  
- `var.cls_secret_id` - CLS 日志服务 Secret ID
- `var.cls_secret_key` - CLS 日志服务 Secret Key
- `var.cls_topic_id` - CLS 日志服务 Topic ID

## 解决方案（3种方法）

### 方法 1：使用环境变量（推荐）

```bash
# 从 qclould.env 加载腾讯云凭证
source ../config/local/qclould.env

# 设置 Terraform 变量（TF_VAR_ 前缀）
export TF_VAR_secret_id="$TENCENTCLOUD_SECRET_ID"
export TF_VAR_secret_key="$TENCENTCLOUD_SECRET_KEY"

# CLS 配置（如果不需要日志服务，可以设置为空）
export TF_VAR_cls_secret_id="your-cls-secret-id"
export TF_VAR_cls_secret_key="your-cls-secret-key"
export TF_VAR_cls_topic_id="your-cls-topic-id"

# 然后运行 terraform
cd terraform
terraform plan
```

### 方法 2：创建 terraform.tfvars 文件（推荐用于本地开发）

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# 编辑 terraform.tfvars，填入实际值
# 注意：不要提交 terraform.tfvars 到 Git！
```

然后在 `terraform.tfvars` 中填入：
```hcl
secret_id  = "your-tencent-secret-id"
secret_key = "your-tencent-secret-key"
cls_secret_id  = "your-cls-secret-id"
cls_secret_key = "your-cls-secret-key"
cls_topic_id   = "your-cls-topic-id"
```

### 方法 3：命令行参数（不推荐，太麻烦）

```bash
terraform plan \
  -var="secret_id=xxx" \
  -var="secret_key=xxx" \
  -var="cls_secret_id=xxx" \
  -var="cls_secret_key=xxx" \
  -var="cls_topic_id=xxx"
```

## 如果不需要 CLS 日志服务

如果暂时不需要日志服务，可以：
1. 在 `terraform.tfvars` 中设置为空字符串
2. 或者修改 `main.tf`，让 CLS 相关配置变为可选

## 快速开始

```bash
# 1. 加载腾讯云凭证
source config/local/qclould.env

# 2. 设置 Terraform 变量
export TF_VAR_secret_id="$TENCENTCLOUD_SECRET_ID"
export TF_VAR_secret_key="$TENCENTCLOUD_SECRET_KEY"

# 3. 如果不需要 CLS，可以设置空值
export TF_VAR_cls_secret_id=""
export TF_VAR_cls_secret_key=""
export TF_VAR_cls_topic_id=""

# 4. 运行 terraform
cd terraform
terraform plan
```
