# SSH 私钥配置

## 问题

Terraform provisioner 需要 SSH 私钥来连接新创建的 CVM 实例。

## 解决方案

### 方案 1：创建 SSH 密钥对（如果还没有）

```bash
# 生成新的 SSH 密钥对
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""

# 或者使用 ed25519（更安全）
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
```

### 方案 2：使用现有的 SSH 私钥

如果私钥在其他位置，设置环境变量：

```bash
export TF_VAR_ssh_private_key="/path/to/your/private/key"
```

### 方案 3：临时禁用 provisioner（如果不需要自动配置）

如果暂时不需要通过 provisioner 自动配置，可以注释掉 provisioner 部分，手动 SSH 连接后配置。

## 重要提示

1. **私钥权限**：确保私钥文件权限正确（600）
   ```bash
   chmod 600 ~/.ssh/id_rsa
   ```

2. **公钥上传**：创建实例后，需要将公钥添加到服务器的 `~/.ssh/authorized_keys`
   - 可以通过腾讯云控制台在创建实例时添加
   - 或者通过 provisioner 自动添加

3. **首次连接**：首次 SSH 连接时可能需要接受主机密钥

## 当前配置

- 默认路径：`~/.ssh/id_rsa`
- 使用 `pathexpand()` 函数自动扩展 `~` 路径
- 如果私钥不存在，provisioner 会失败
