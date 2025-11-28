# WSL 默认用户设置说明

## 问题
WSL 默认以 root 用户启动，而不是 yin 用户。

## 解决方案

### 方案 1：在 Windows 端设置默认用户（推荐）

在 Windows PowerShell 或 CMD 中执行：

```powershell
# 设置默认用户为 yin
wsl --set-default-user yin

# 如果需要为特定的发行版设置（例如 Ubuntu）
wsl --distribution Ubuntu --set-default-user yin
```

### 方案 2：通过 /root/.bashrc 自动切换（已配置）

已经在 `/root/.bashrc` 中添加了自动切换逻辑：
- 当检测到是直接 root 登录（非 sudo）时，会自动切换到 yin 用户
- 如果是通过 `sudo` 获得的 root 权限，则只提示，不自动切换

## 验证

打开新的 WSL 终端，应该自动以 yin 用户身份登录，而不是 root。

## 注意事项

- 如果需要以 root 身份启动，可以在 Windows 端使用：`wsl -u root`
- 自动切换功能只在交互式 shell 中生效

