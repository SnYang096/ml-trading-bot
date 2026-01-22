# 支付权限问题

## 错误信息

```
Code=UnauthorizedOperation
Message=Since you have no payment rights and cannot complete the payment, 
please try again after applying payment rights.
```

## 问题说明

这不是 Terraform 配置问题，而是**腾讯云账户缺少支付权限**。

## 解决方案

### 1. 申请支付权限

1. 登录腾讯云控制台：https://console.cloud.tencent.com
2. 进入 **费用中心** → **账户信息**
3. 申请开通**支付权限**
4. 完成实名认证（如果未完成）

### 2. 检查账户余额

确保账户有足够余额或已绑定支付方式（信用卡/微信支付等）

### 3. 检查资源配额

确认账户有创建 CVM 实例的配额：
- 进入 **云服务器 CVM** → **配额管理**
- 检查是否有限制

## Terraform 配置状态

✅ Terraform 配置正确，已验证通过
✅ 实例类型和可用区已固定为可用组合
✅ 所有资源定义正确

**问题解决后，直接运行：**
```bash
cd terraform
./run.sh apply
```

## 当前固定配置

- 可用区：`ap-tokyo-2`
- 实例类型：`S5.MEDIUM4` (2核4G)
- 配置已验证通过
