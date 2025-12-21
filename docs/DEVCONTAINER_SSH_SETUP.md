# DevContainer SSH 配置说明

本文档说明如何在 DevContainer 中配置 SSH，以便 git 可以使用 SSH 密钥进行身份验证。

## 配置说明

### 1. SSH Agent Forwarding

DevContainer 已配置 SSH Agent Forwarding，通过以下方式实现：

- **mounts**: 将主机的 `SSH_AUTH_SOCK` 挂载到容器的 `/ssh-agent`
- **containerEnv**: 设置 `SSH_AUTH_SOCK=/ssh-agent`
- **SSH Config**: 在容器内创建 `~/.ssh/config`，配置 `IdentityAgent` 指向 `/ssh-agent`

### 2. SSH 密钥文件（可选）

如果需要使用 SSH 密钥文件（而不是仅依赖 SSH Agent），可以：

- **mounts**: 将主机的 `~/.ssh` 目录挂载到容器的 `/home/yin/.ssh-host`
- **postCreateCommand**: 自动复制密钥文件到容器内的 `~/.ssh`，并设置正确的权限

## 使用方法

### 在 VS Code 中打开 DevContainer

1. 打开项目
2. 按 `F1` 或 `Ctrl+Shift+P`
3. 选择 `Dev Containers: Reopen in Container`

### 验证 SSH 配置

进入容器后，运行以下命令验证：

```bash
# 检查 SSH Agent 是否工作
echo $SSH_AUTH_SOCK
ssh-add -l  # 应该列出你的 SSH 密钥

# 测试 git 连接
git ls-remote git@github.com:your-username/your-repo.git
```

### 常见问题

#### 问题 1: `ssh-add -l` 显示 "Could not open a connection to your authentication agent"

**原因**: SSH Agent 未正确转发

**解决方案**:
1. 确保主机上的 SSH Agent 正在运行：
   ```bash
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/your_key
   ```
2. 在 VS Code 中重新打开容器

#### 问题 2: Git 仍然要求输入密码

**原因**: SSH 配置未正确设置

**解决方案**:
1. 检查 `~/.ssh/config` 文件是否存在：
   ```bash
   cat ~/.ssh/config
   ```
2. 确保 `IdentityAgent` 指向 `/ssh-agent`
3. 如果需要使用密钥文件，确保 `~/.ssh` 目录中有正确的密钥文件

#### 问题 3: 权限错误

**原因**: SSH 密钥文件权限不正确

**解决方案**:
```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/*
chmod 644 ~/.ssh/config
```

## 配置细节

### devcontainer.json 配置

```json
{
  "mounts": [
    "source=${localEnv:SSH_AUTH_SOCK},target=/ssh-agent,type=bind",
    "source=${localEnv:HOME}${localEnv:USERPROFILE}/.ssh,target=/home/yin/.ssh-host,type=bind,consistency=cached"
  ],
  "containerEnv": {
    "SSH_AUTH_SOCK": "/ssh-agent"
  },
  "postCreateCommand": "... 配置 SSH ..."
}
```

### 生成的 SSH Config

容器内的 `~/.ssh/config` 会自动配置为：

```
Host *
    ForwardAgent yes
    IdentityAgent /ssh-agent
    StrictHostKeyChecking no
```

这个配置确保：
- 启用 Agent Forwarding
- 使用挂载的 SSH Agent
- 不严格检查主机密钥（适合开发环境）

## 安全注意事项

- `StrictHostKeyChecking no` 适合开发环境，但在生产环境中应设置为 `yes`
- SSH 密钥文件仅在容器内可见，不会影响主机
- 使用 SSH Agent Forwarding 比直接挂载密钥文件更安全

