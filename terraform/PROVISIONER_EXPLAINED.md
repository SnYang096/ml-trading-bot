# Terraform Provisioner 说明

## 什么是 Provisioner？

**Provisioner** 是 Terraform 中用于在资源创建后执行额外配置的工具。

### 简单理解

```
资源创建（如 CVM 实例） → Provisioner 执行 → 完成配置
```

## 两种类型的 Provisioner

### 1. **local-exec** - 在本地执行命令

```hcl
provisioner "local-exec" {
  command = "echo '资源已创建'"
}
```

### 2. **remote-exec** - 在远程服务器执行命令（通过 SSH）

```hcl
provisioner "remote-exec" {
  inline = [
    "sudo apt-get update",
    "sudo apt-get install -y nginx"
  ]
}
```

## 在你的配置中的 Provisioner

### 1. **file provisioner** - 上传文件到服务器

```hcl
provisioner "file" {
  source      = "${path.module}/systemd/"      # 本地路径
  destination = "/tmp/systemd/"                 # 服务器路径
}
```

**作用**：将本地的 systemd 服务文件上传到服务器的 `/tmp/systemd/` 目录

### 2. **remote-exec provisioner** - 在服务器执行命令

```hcl
provisioner "remote-exec" {
  inline = [
    "sudo cp /tmp/filebeat.env /etc/default/filebeat",
    "sudo chmod 600 /etc/default/filebeat",
    "sudo /tmp/init.sh"
  ]
}
```

**作用**：在服务器上执行命令，完成配置

## 完整流程

```
1. Terraform 创建 CVM 实例
   ↓
2. 等待实例启动（SSH 可用）
   ↓
3. file provisioner 上传文件：
   - systemd/ 服务文件 → /tmp/systemd/
   - monitoring/ 监控配置 → /tmp/monitoring/
   - logging/ 日志配置 → /tmp/logging/
   - filebeat.env → /tmp/filebeat.env
   ↓
4. remote-exec provisioner 执行命令：
   - 复制 filebeat.env 到正确位置
   - 设置权限
   - 运行 init.sh 完成初始化
   ↓
5. 服务器配置完成
```

## 为什么需要 Provisioner？

### 方案对比

**方案 A：只用 user_data（云初始化脚本）**
- ✅ 实例创建时自动执行
- ❌ 无法上传本地文件
- ❌ 无法从 Terraform 获取动态信息

**方案 B：user_data + provisioner（当前方案）**
- ✅ user_data：基础初始化（挂载磁盘、安装软件）
- ✅ provisioner：上传配置文件、执行复杂配置
- ✅ 可以从 Terraform 获取变量和资源信息

## 当前配置的 Provisioner 做了什么？

1. **上传 systemd 服务文件** → 让 systemd 管理服务
2. **上传监控配置** → Prometheus、Grafana 配置
3. **上传日志配置** → Filebeat 配置
4. **上传环境变量文件** → Filebeat 的密钥配置
5. **执行初始化脚本** → 完成所有配置

## 注意事项

1. **需要 SSH 访问**：provisioner 通过 SSH 连接服务器
2. **需要公钥**：服务器必须已有你的 SSH 公钥（已通过 user_data 自动添加）
3. **执行顺序**：file → remote-exec（按顺序执行）
4. **失败处理**：如果 provisioner 失败，资源仍会创建，但配置可能不完整

## 如果不需要 Provisioner？

如果不想使用 provisioner，可以：
1. 注释掉所有 provisioner 块
2. 手动 SSH 连接服务器
3. 手动上传文件和执行命令

但使用 provisioner 可以**自动化整个部署过程**，更高效。
