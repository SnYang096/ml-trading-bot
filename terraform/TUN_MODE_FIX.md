# TUN 模式下的 Terraform 修复方案

## 问题分析

TUN 模式 + GitHub 直连导致：
- ✅ GitHub 可以直连（但可能不稳定）
- ❌ `registry.terraform.io` 直连超时（需要代理）
- ❌ Terraform 无法下载 provider

## 解决方案

### 方案 1：修改代理规则（推荐）

在代理配置中，确保以下域名**通过代理**（不要直连）：
- `registry.terraform.io`
- `*.terraform.io`
- `releases.hashicorp.com`
- `github.com`（如果直连不稳定，也改为代理）

**Clash 配置示例**：
```yaml
rules:
  - DOMAIN-SUFFIX,terraform.io,PROXY
  - DOMAIN-SUFFIX,hashicorp.com,PROXY
  - DOMAIN-SUFFIX,github.com,PROXY  # 如果直连不稳定
  # 其他规则...
```

### 方案 2：临时使用 HTTP 代理模式

如果 TUN 模式配置复杂，可以临时切换：

```bash
# 1. 禁用 TUN 模式，改用 HTTP 代理模式
# 2. 设置环境变量
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897

# 3. 运行 terraform
cd terraform
terraform init
```

### 方案 3：混合模式（TUN + 特定域名代理）

保持 TUN 模式，但为 Terraform 相关域名配置代理规则。

## 快速测试

测试域名是否通过代理：

```bash
# 测试 registry.terraform.io
curl -I https://registry.terraform.io

# 测试 GitHub
curl -I https://github.com

# 如果都返回 200，说明可以访问
```
