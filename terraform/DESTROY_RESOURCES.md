# 删除云资源指南

本文档说明如何使用 Terraform 删除腾讯云资源。

## ⚠️ 警告

**删除操作不可逆！** 执行 `terraform destroy` 将删除所有通过 Terraform 创建的资源，包括：
- CVM 实例（及所有数据）
- VPC 和子网
- 安全组
- 其他相关资源

**请确保：**
1. 已备份重要数据
2. 确认不再需要这些资源
3. 了解删除操作的后果

## 🗑️ 删除所有资源

### 方法 1：使用 run.sh 脚本（推荐）

```bash
cd terraform
./run.sh destroy
```

### 方法 2：直接使用 Terraform

```bash
cd terraform
source ../config/local/qclould.env
source ../config/local/terraform.env
export TENCENTCLOUD_SECRET_ID
export TENCENTCLOUD_SECRET_KEY
export TF_VAR_secret_id="$TENCENTCLOUD_SECRET_ID"
export TF_VAR_secret_key="$TENCENTCLOUD_SECRET_KEY"

terraform destroy
```

## 📋 删除前检查清单

执行删除前，请确认：

- [ ] 已备份 CVM 实例上的重要数据
- [ ] 已导出 SQLite 数据库（如果存在）
- [ ] 已保存配置文件
- [ ] 已记录重要的 IP 地址或资源 ID
- [ ] 确认不再需要这些资源

## 🔍 预览删除计划

在删除前，可以先查看将要删除的资源：

```bash
cd terraform
./run.sh plan -destroy
```

这会显示所有将被删除的资源，但**不会实际删除**。

## 🎯 选择性删除

如果需要只删除特定资源，可以：

### 1. 使用 `-target` 参数

```bash
# 只删除 CVM 实例
./run.sh destroy -target=tencentcloud_instance.quant_server

# 只删除 VPC（会同时删除依赖的子网）
./run.sh destroy -target=tencentcloud_vpc.main
```

### 2. 临时注释资源

在 `main.tf` 中注释掉不需要删除的资源，然后运行：

```bash
./run.sh apply  # 这会删除未在配置中的资源
```

**注意：** 这种方法需要小心处理依赖关系。

## 🚨 常见问题

### Q: 删除失败怎么办？

**A:** 可能的原因：
1. 资源被其他服务占用（如安全组被其他实例使用）
2. 资源有依赖关系未正确清理
3. 权限不足

**解决方法：**
- 检查错误信息，手动删除依赖资源
- 在腾讯云控制台手动删除
- 使用 `terraform destroy -force` 强制删除（谨慎使用）

### Q: 删除后如何恢复？

**A:** Terraform 删除的资源无法自动恢复。需要：
1. 重新运行 `terraform apply` 创建新资源
2. 从备份恢复数据

### Q: 如何只删除数据但保留配置？

**A:** 无法直接实现。可以：
1. 手动在 CVM 上删除数据文件
2. 或者删除实例后重新创建（会丢失所有数据）

## 📝 删除步骤示例

```bash
# 1. 进入 terraform 目录
cd terraform

# 2. 预览删除计划（可选，但强烈推荐）
./run.sh plan -destroy

# 3. 确认要删除的资源列表

# 4. 执行删除（需要手动输入 'yes' 确认）
./run.sh destroy

# 5. 等待删除完成（通常需要几分钟）

# 6. 验证删除结果
./run.sh show  # 应该显示没有资源
```

## 🔐 安全建议

1. **生产环境删除前**：先在测试环境验证删除流程
2. **备份优先**：删除前务必备份所有重要数据
3. **分步删除**：对于复杂环境，考虑分步删除资源
4. **保留配置**：删除资源后，保留 Terraform 配置文件以便将来重建

## 📚 相关文档

- [Terraform Destroy 官方文档](https://developer.hashicorp.com/terraform/cli/commands/destroy)
- [腾讯云资源删除说明](https://cloud.tencent.com/document/product/213)
