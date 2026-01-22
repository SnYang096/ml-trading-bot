# Terraform 部署说明

## 网络问题解决方案

如果 `terraform init` 卡住或下载失败，可以尝试以下方法：

### 方法 1：重试（推荐）
```bash
# 清理缓存后重试
rm -rf .terraform .terraform.lock.hcl
terraform init
```

### 方法 2：使用更稳定的版本
已配置使用 `~> 1.80` 版本范围，避免下载最新版本可能遇到的网络问题。

### 方法 3：手动下载 Provider（如果网络持续有问题）
```bash
# 1. 创建 provider 目录
mkdir -p ~/.terraform.d/plugins/registry.terraform.io/tencentcloudstack/tencentcloud/1.80.0/linux_amd64

# 2. 手动下载（需要根据你的系统架构调整）
wget https://github.com/tencentcloudstack/terraform-provider-tencentcloud/releases/download/v1.80.0/terraform-provider-tencentcloud_1.80.0_linux_amd64.zip
unzip terraform-provider-tencentcloud_1.80.0_linux_amd64.zip -d ~/.terraform.d/plugins/registry.terraform.io/tencentcloudstack/tencentcloud/1.80.0/linux_amd64/

# 3. 重新运行 init
terraform init
```

### 方法 4：配置代理（如果有）
```bash
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
terraform init
```

## 变量配置

在运行 `terraform apply` 前，需要设置以下变量：

```bash
export TF_VAR_secret_id="your-tencent-secret-id"
export TF_VAR_secret_key="your-tencent-secret-key"
export TF_VAR_cls_secret_id="your-cls-secret-id"
export TF_VAR_cls_secret_key="your-cls-secret-key"
export TF_VAR_cls_topic_id="your-cls-topic-id"
```

或者创建 `terraform.tfvars` 文件（不要提交到 Git）。
