# Terraform 认证问题排查

## 问题：AuthFailure.SecretIdNotFound

如果遇到 `AuthFailure.SecretIdNotFound` 错误，请按以下步骤排查：

### 1. 检查 SecretId 是否有效

```bash
# 检查 SecretId 格式（应该是 32-128 位字符串）
echo "$TENCENTCLOUD_SECRET_ID" | wc -c

# 检查是否包含特殊字符或空格
echo "$TENCENTCLOUD_SECRET_ID" | grep -E "[^A-Za-z0-9]"
```

### 2. 验证 SecretId 是否有效

登录腾讯云控制台：
- 访问：https://console.cloud.tencent.com/cam/capi
- 检查 SecretId 是否：
  - ✅ 存在且启用
  - ✅ 未过期
  - ✅ 有足够的权限（需要 VPC、CVM、安全组等权限）

### 3. 检查环境变量传递

```bash
# 使用 run.sh 脚本（推荐）
cd terraform
./run.sh plan

# 或手动验证
source ../config/local/qclould.env
echo "SecretId: ${TENCENTCLOUD_SECRET_ID:0:20}..."
terraform plan
```

### 4. 如果 SecretId 无效

1. 在腾讯云控制台创建新的 SecretId/SecretKey
2. 更新 `config/local/qclould.env`：
   ```bash
   export TENCENTCLOUD_SECRET_ID="新的SecretId"
   export TENCENTCLOUD_SECRET_KEY="新的SecretKey"
   ```
3. 重新运行 `./run.sh apply`

### 5. 检查权限

确保 SecretId 有以下权限：
- `QcloudVPCFullAccess` - VPC 全读写权限
- `QcloudCVMFullAccess` - CVM 全读写权限
- `QcloudCLBFullAccess` - 负载均衡全读写权限（如果使用）

### 6. 使用显式变量（备选方案）

如果环境变量不工作，可以在 `terraform.tfvars` 中显式指定：

```hcl
secret_id  = "your-secret-id"
secret_key = "your-secret-key"
```

**注意**：不要提交包含真实凭证的 `terraform.tfvars` 到 Git！
