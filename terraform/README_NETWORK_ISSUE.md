# Terraform 网络问题解决方案

## 当前问题

Terraform 在下载 provider 时遇到网络连接问题：
- 直接连接 registry.terraform.io 超时
- 通过代理 (127.0.0.1:7897) 连接被重置

## 解决方案

### 方案 1：检查并重启代理服务

```bash
# 检查代理是否运行
ps aux | grep -E "(clash|v2ray|proxy)"

# 如果代理未运行，启动它
# 例如 Clash:
# clash -d ~/.config/clash

# 然后重试
cd terraform
terraform init
```

### 方案 2：使用其他代理端口或配置

如果代理在其他端口，设置环境变量：

```bash
export HTTP_PROXY=http://127.0.0.1:YOUR_PROXY_PORT
export HTTPS_PROXY=http://127.0.0.1:YOUR_PROXY_PORT
cd terraform
terraform init
```

### 方案 3：等待网络恢复后重试

网络问题可能是临时的，可以稍后重试：

```bash
cd terraform
terraform init
```

### 方案 4：使用 VPN 或更换网络环境

如果当前网络环境不稳定，可以：
- 使用 VPN
- 更换网络（如使用手机热点）
- 在另一台网络稳定的机器上运行

### 方案 5：手动下载 Provider（如果网络恢复）

如果网络恢复，可以运行手动安装脚本：

```bash
cd terraform
bash manual_provider_install.sh
terraform init
```

## 临时解决方案

如果急需使用，可以：
1. 在另一台可以正常访问 GitHub 的机器上下载 provider
2. 将 provider 文件复制到 `~/.terraform.d/plugins/` 目录
3. 然后运行 `terraform init`

## 检查网络连接

```bash
# 测试 GitHub 连接
curl -I https://github.com

# 测试 Terraform Registry
curl -I https://registry.terraform.io

# 测试代理
curl -x http://127.0.0.1:7897 -I https://www.google.com
```
