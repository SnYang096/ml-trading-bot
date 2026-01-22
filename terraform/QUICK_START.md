# Terraform 快速开始指南

## 环境变量配置

### 1. 腾讯云凭证（Provider 自动读取）

腾讯云 provider **自动读取**以下环境变量，**无需**设置 `TF_VAR_` 前缀：

```bash
# 从你的配置文件加载
source config/local/qclould.env

# 这些环境变量会被 provider 自动使用：
# - TENCENTCLOUD_SECRET_ID
# - TENCENTCLOUD_SECRET_KEY
```

### 2. CLS 日志服务配置（Terraform 变量）

CLS 配置需要通过 `TF_VAR_` 前缀设置，或者使用 `terraform.tfvars`：

**方式 A：环境变量（推荐）**
```bash
export TF_VAR_cls_secret_id="your-cls-secret-id"
export TF_VAR_cls_secret_key="your-cls-secret-key"
export TF_VAR_cls_topic_id="your-cls-topic-id"
```

**方式 B：terraform.tfvars 文件**
```bash
cd terraform
cat > terraform.tfvars << 'EOF'
cls_secret_id  = "your-cls-secret-id"
cls_secret_key = "your-cls-secret-key"
cls_topic_id   = "your-cls-topic-id"
EOF
```

**方式 C：如果不需要日志服务，设置为空**
```bash
export TF_VAR_cls_secret_id=""
export TF_VAR_cls_secret_key=""
export TF_VAR_cls_topic_id=""
```

## 完整使用流程

```bash
# 1. 加载腾讯云凭证（provider 自动读取）
source config/local/qclould.env

# 2. 设置 CLS 配置（如果需要）
export TF_VAR_cls_secret_id=""
export TF_VAR_cls_secret_key=""
export TF_VAR_cls_topic_id=""

# 3. 运行 Terraform
cd terraform
terraform plan
terraform apply
```

## 总结

- ✅ **TENCENTCLOUD_SECRET_ID/KEY**：Provider 自动读取，**不需要** `TF_VAR_` 前缀
- ✅ **CLS 配置**：需要 `TF_VAR_` 前缀，或使用 `terraform.tfvars`
- ✅ **其他变量**（region, instance_type 等）：有默认值，无需设置
