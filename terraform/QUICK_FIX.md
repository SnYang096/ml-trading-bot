# Terraform 网络问题快速修复

## 问题诊断

✅ 代理服务运行正常（可以连接 Google）
❌ GitHub/registry.terraform.io 连接不稳定

## 立即尝试的解决方案

### 1. 多次重试（最简单）

网络问题可能是间歇性的，多试几次：

```bash
cd terraform
for i in {1..5}; do
  echo "尝试 $i/5..."
  terraform init && break
  sleep 5
done
```

### 2. 检查代理规则

确保代理配置允许访问：
- `registry.terraform.io`
- `github.com`
- `releases.hashicorp.com`

### 3. 临时禁用代理（如果直连可用）

```bash
unset HTTP_PROXY HTTPS_PROXY
cd terraform
terraform init
```

### 4. 使用固定版本（已配置）

当前已配置使用固定版本 `1.80.0`，避免下载最新版本。

## 如果以上都不行

可以暂时跳过 Terraform 部署，直接手动配置服务器：

1. 手动创建 CVM 实例
2. 运行 `terraform/init.sh` 脚本
3. 配置 systemd 服务

或者等待网络恢复后再运行 `terraform init`。
